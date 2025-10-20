from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import TypedDict

import numpy as np
import uproot
from uproot.behaviors.TBranch import HasBranches
from uproot.reading import ReadOnlyDirectory
from loguru import logger

from gluex_ksks_paper_analysis.environment import (
    CCDB_CONNECTION,
    CCDB_PATH,
    POL_HIST_PATHS,
    POLARIZED_RUN_NUMBERS_PATH,
    PSFLUX_PATH,
    RCDB_CONNECTION,
    RCDB_PATH,
)
from gluex_ksks_paper_analysis.utilities import Histogram

RCDB_SELECTION_PREFIX = """
WITH status AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'status' AND c.int_value = 1
), run_type AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'run_type' AND c.text_value IN ('hd_all.tsg', 'hd_all.tsg_ps', 'hd_all.bcal_fcal_st.tsg')
OR NOT run_number BETWEEN 30000 AND 39999
), beam_current AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'beam_current' AND c.float_value > 2.0
), event_count AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'event_count' AND ((c.int_value > 500000 AND run_number BETWEEN 30000 AND 39999) OR (c.int_value > 10000000 AND run_number BETWEEN 40000 AND 59999) OR (c.int_value > 5000000 AND run_number BETWEEN 71275 AND 79999))
), solenoid_current AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'solenoid_current' AND c.float_value > 100.0
), collimator_diameter AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'collimator_diameter' AND c.text_value != 'Blocking'
), polarized AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'polarization_angle' AND c.float_value >= 0.0
), daq_run AS (
SELECT run_number
FROM conditions c
JOIN condition_types ct ON c.condition_type_id = ct.id
WHERE ct.name = 'daq_run' AND ((run_number BETWEEN 30000 AND 39999) OR (c.text_value = 'PHYSICS' AND run_number BETWEEN 40000 AND 59999) OR (c.text_value = 'PHYSICS_DIRC' AND run_number BETWEEN 71275 AND 79999))
)
"""

RCDB_SELECTION_SUFFIX = """
AND r.number IN (SELECT run_number FROM status)
AND r.number IN (SELECT run_number from run_type)
AND r.number IN (SELECT run_number from beam_current)
AND r.number IN (SELECT run_number from event_count)
AND r.number IN (SELECT run_number from solenoid_current)
AND r.number IN (SELECT run_number from collimator_diameter)
AND r.number IN (SELECT run_number from polarized)
AND r.number IN (SELECT run_number from daq_run)
"""

TRUE_POL_ANGLES = {
    's17': {'0.0': 1.8, '45.0': 47.9, '90.0': 94.5, '135.0': -41.6},
    's18': {'0.0': 4.1, '45.0': 48.5, '90.0': 94.2, '135.0': -42.4},
    'f18': {'0.0': 3.3, '45.0': 48.3, '90.0': 92.9, '135.0': -42.1},
    's20': {'0.0': 1.4, '45.0': 47.1, '90.0': 93.4, '135.0': -42.2},
}

RUN_RANGES = {
    's17': (30000, 39999),
    's18': (40000, 49999),
    'f18': (50000, 59999),
    's20': (71275, 79999),
}

# These are dates in the month after the last calibration update for each REST version
REST_VERSION_TIMESTAMPS = {
    's17': datetime(2018, 12, 1),  # noqa: DTZ001
    's18': datetime(2019, 8, 1),  # noqa: DTZ001
    'f18': datetime(2019, 11, 1),  # noqa: DTZ001
    's20': datetime(2022, 6, 1),  # noqa: DTZ001
}


def build_caches() -> None:
    _ = RCDBData()
    _ = CCDBData()
    _ = PSFlux()
    _ = get_all_polarized_run_numbers()
    RCDB_CONNECTION.unlink(missing_ok=True)
    CCDB_CONNECTION.unlink(missing_ok=True)


def get_run_period(run_number: int) -> str | None:
    for rp, (lo, hi) in RUN_RANGES.items():
        if lo <= run_number <= hi:
            return rp
    return None


def get_run_period_bound(run_number: int) -> str | None:
    for rp, (_, hi) in RUN_RANGES.items():
        if run_number <= hi:
            return rp
    return None


def get_pol_angle(run_period: str | None, angle_deg: str) -> float | None:
    if run_period is None:
        return None
    pol_angle_deg = TRUE_POL_ANGLES[run_period].get(angle_deg)
    if pol_angle_deg is None:
        return None
    return pol_angle_deg * np.pi / 180.0


def get_all_polarized_run_numbers() -> set[int]:
    if POLARIZED_RUN_NUMBERS_PATH.exists():
        return pickle.load(POLARIZED_RUN_NUMBERS_PATH.open('rb'))  # noqa: S301
    logger.info('Building polarized run numbers cache...')
    with sqlite3.connect(RCDB_CONNECTION) as rcdb:
        cursor = rcdb.cursor()
        query = f"""
        {RCDB_SELECTION_PREFIX}
        SELECT r.number
        FROM runs r
        WHERE r.number > 0
        {RCDB_SELECTION_SUFFIX}
        ORDER BY r.number
        """  # noqa: S608
        cursor.execute(query)
        res = {r[0] for r in cursor.fetchall()}
        pickle.dump(res, POLARIZED_RUN_NUMBERS_PATH.open('wb'))
        return res


def get_ccdb_table(
    table_path: str, *, use_timestamp: bool = True
) -> dict[int, list[list[str]]]:
    with sqlite3.connect(CCDB_CONNECTION) as ccdb:
        path_parts = table_path.split('/')
        cursor = ccdb.cursor()
        query = """
            SELECT tt.nColumns, rr.runMin, rr.runMax, cs.vault, a.created
            FROM directories d0
            """
        for i, path_part in enumerate(path_parts[1:-1]):
            query += f"JOIN directories d{i + 1} ON d{i + 1}.parentId = d{i}.id AND d{i + 1}.name = '{path_part}'"
        query += f"""
            JOIN typeTables tt ON d{len(path_parts) - 2}.id = tt.directoryId AND tt.name = '{path_parts[-1]}'
            JOIN constantSets cs ON tt.id = cs.constantTypeId
            JOIN assignments a ON cs.id = a.constantSetId
            JOIN runRanges rr ON a.runRangeId = rr.id
            LEFT JOIN variations v ON a.variationId = v.id
            WHERE d0.name = '{path_parts[0]}'
            AND v.name IS 'default'
            ORDER BY rr.runMin, a.created
            """
        cursor.execute(query)
        res = cursor.fetchall()
        res_table = {}
        for n_columns, run_min, run_max, vault, timestamp in res:
            # We do this because some of the tables have ridiculous run ranges
            run_min_in_range = max(run_min, RUN_RANGES['s17'][0])
            run_max_in_range = min(run_max, RUN_RANGES['s20'][1])
            ts = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')  # noqa: DTZ007
            run_period = get_run_period_bound(run_min_in_range)
            if run_period is None:
                continue
            max_timestamp = REST_VERSION_TIMESTAMPS[run_period]
            if ts <= max_timestamp or use_timestamp is False:
                data = vault.split('|')
                table = [
                    data[i : i + n_columns] for i in range(0, len(data), n_columns)
                ]
                for run in range(run_min_in_range, run_max_in_range + 1):
                    res_table[run] = table
        return res_table


class PSFluxState(TypedDict):
    df_scale: dict[int, float]
    df_ps_accept: dict[int, tuple[float, float, float]]
    df_photon_endpoint: dict[int, float]
    df_tagm_tagged_flux: dict[int, list[tuple[float, float, float]]]
    df_tagm_scaled_energy: dict[int, list[tuple[float, float]]]
    df_tagh_tagged_flux: dict[int, list[tuple[float, float, float]]]
    df_tagh_scaled_energy: dict[int, list[tuple[float, float]]]
    df_photon_endpoint_calib: dict[int, float]
    df_target_scattering_centers: dict[int, tuple[float, float]]


class PSFluxHistograms(TypedDict):
    tagged_flux_histogram: Histogram
    tagm_e_flux_histogram: Histogram
    tagm_flux_histogram: Histogram
    tagh_e_flux_histogram: Histogram
    tagh_flux_histogram: Histogram
    tagged_lumi_histogram: Histogram


class PSFlux:
    # NOTE: luminosity is in inverse picobarns
    def __init__(self) -> None:
        self.df_scale: dict[int, float] = {}
        self.df_ps_accept: dict[int, tuple[float, float, float]] = {}
        self.df_photon_endpoint: dict[int, float] = {}
        self.df_tagm_tagged_flux: dict[int, list[tuple[float, float, float]]] = {}
        self.df_tagm_scaled_energy: dict[int, list[tuple[float, float]]] = {}
        self.df_tagh_tagged_flux: dict[int, list[tuple[float, float, float]]] = {}
        self.df_tagh_scaled_energy: dict[int, list[tuple[float, float]]] = {}
        self.df_photon_endpoint_calib: dict[int, float] = {}
        self.df_target_scattering_centers: dict[int, tuple[float, float]] = {}
        if PSFLUX_PATH.exists():
            with PSFLUX_PATH.open('rb') as f:
                state = pickle.load(f)  # noqa: S301
                self.df_scale = state['df_scale']
                self.df_ps_accept = state['df_ps_accept']
                self.df_photon_endpoint = state['df_photon_endpoint']
                self.df_tagm_tagged_flux = state['df_tagm_tagged_flux']
                self.df_tagm_scaled_energy = state['df_tagm_scaled_energy']
                self.df_tagh_tagged_flux = state['df_tagh_tagged_flux']
                self.df_tagh_scaled_energy = state['df_tagh_scaled_energy']
                self.df_photon_endpoint_calib = state['df_photon_endpoint_calib']
                self.df_target_scattering_centers = state[
                    'df_target_scattering_centers'
                ]
        else:
            logger.info('Building Pair Spectrometer Flux cache...')
            state = self._build_from_db()
            self.df_scale = state['df_scale']
            self.df_ps_accept = state['df_ps_accept']
            self.df_photon_endpoint = state['df_photon_endpoint']
            self.df_tagm_tagged_flux = state['df_tagm_tagged_flux']
            self.df_tagm_scaled_energy = state['df_tagm_scaled_energy']
            self.df_tagh_tagged_flux = state['df_tagh_tagged_flux']
            self.df_tagh_scaled_energy = state['df_tagh_scaled_energy']
            self.df_photon_endpoint_calib = state['df_photon_endpoint_calib']
            self.df_target_scattering_centers = state['df_target_scattering_centers']
            PSFLUX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with PSFLUX_PATH.open('wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _build_from_db() -> PSFluxState:
        berillium_radiation_length = 35.28e-2
        target_length = 29.5
        target_length_conversion_factor = 6.02214e23 * 1e-24 * 1e-3
        target_factor = target_length * target_length_conversion_factor
        df_livetime_ratio_table = get_ccdb_table(
            'PHOTON_BEAM/pair_spectrometer/lumi/trig_live'
        )
        df_livetime_ratio: dict[int, float] = {}
        for k, v in df_livetime_ratio_table.items():
            df_livetime_ratio[k] = (
                (float(v[0][1]) / float(v[3][1])) if float(v[3][1]) > 0.0 else 1.0
            )
        converter_thickness = get_rcdb_text_condition('polarimeter_converter')
        df_radiation_length: dict[int, float] = {
            k: ((75e-6 if v == 'Be 75um' else 750e-6) / berillium_radiation_length)
            for k, v in converter_thickness.items()
        }
        df_scale: dict[int, float] = {
            k: df_livetime_ratio.get(k, 1.0) * 1.0 / (7.0 / 9.0 * v)
            for k, v in df_radiation_length.items()
        }
        df_ps_accept_table = get_ccdb_table(
            'PHOTON_BEAM/pair_spectrometer/lumi/PS_accept'
        )
        df_ps_accept: dict[int, tuple[float, float, float]] = {
            k: (float(v[0][0]), float(v[0][1]), float(v[0][2]))
            for k, v in df_ps_accept_table.items()
        }
        df_photon_endpoint: dict[int, float] = {
            k: float(v[0][0])
            for k, v in get_ccdb_table('PHOTON_BEAM/endpoint_energy').items()
        }
        df_tagm_tagged_flux: dict[int, list[tuple[float, float, float]]] = {
            k: [(float(r[0]), float(r[1]), float(r[2])) for r in v]
            for k, v in get_ccdb_table(
                'PHOTON_BEAM/pair_spectrometer/lumi/tagm/tagged'
            ).items()
        }
        df_tagm_scaled_energy: dict[int, list[tuple[float, float]]] = {
            k: [(float(r[1]), float(r[2])) for r in v]
            for k, v in get_ccdb_table(
                'PHOTON_BEAM/microscope/scaled_energy_range'
            ).items()
        }
        df_tagh_tagged_flux: dict[int, list[tuple[float, float, float]]] = {
            k: [(float(r[0]), float(r[1]), float(r[2])) for r in v]
            for k, v in get_ccdb_table(
                'PHOTON_BEAM/pair_spectrometer/lumi/tagh/tagged'
            ).items()
        }
        df_tagh_scaled_energy: dict[int, list[tuple[float, float]]] = {
            k: [(float(r[1]), float(r[2])) for r in v]
            for k, v in get_ccdb_table(
                'PHOTON_BEAM/hodoscope/scaled_energy_range'
            ).items()
        }
        df_photon_endpoint_calib: dict[int, float] = {
            k: float(v[0][0])
            for k, v in get_ccdb_table('PHOTON_BEAM/hodoscope/endpoint_calib').items()
        }
        df_target_scattering_centers: dict[int, tuple[float, float]] = {
            k: (float(v[0][0]) * target_factor, float(v[0][1]) * target_factor)
            for k, v in get_ccdb_table('TARGET/density', use_timestamp=False).items()
        }

        return PSFluxState(
            df_scale=df_scale,
            df_ps_accept=df_ps_accept,
            df_photon_endpoint=df_photon_endpoint,
            df_tagm_tagged_flux=df_tagm_tagged_flux,
            df_tagm_scaled_energy=df_tagm_scaled_energy,
            df_tagh_tagged_flux=df_tagh_tagged_flux,
            df_tagh_scaled_energy=df_tagh_scaled_energy,
            df_photon_endpoint_calib=df_photon_endpoint_calib,
            df_target_scattering_centers=df_target_scattering_centers,
        )

    @staticmethod
    def ps_accept(x: float, p0: float, p1: float, p2: float) -> float:
        if x > 2.0 * p1 and x < p1 + p2:
            return p0 * (1.0 - 2.0 * p1 / x)
        if x >= p1 + p2:
            return p0 * (2.0 * p2 / x - 1.0)
        return 0.0

    def _run_ctx(self, run_number: int) -> SimpleNamespace:
        endpoint = self.df_photon_endpoint[run_number]
        delta_e = (
            endpoint - self.df_photon_endpoint_calib[run_number]
            if run_number > RUN_RANGES['f18'][1]
            else 0.0
        )
        return SimpleNamespace(
            run=run_number,
            coherent_peak=get_coherent_peak(run_number),
            scale=self.df_scale[run_number],
            ps_pars=self.df_ps_accept[run_number],
            endpoint=endpoint,
            delta_e=delta_e,
            target_sc=self.df_target_scattering_centers[run_number],
        )

    def _accumulate_detector(
        self,
        *,
        ctx: SimpleNamespace,
        flux_rows: list[tuple[float, float, float]],
        scaled_energy_rows: list[tuple[float, float]],
        tagged_flux_histogram: Histogram,
        tag_e_flux_histogram: Histogram,
        tag_flux_histogram: Histogram,
    ) -> None:
        cp_lo, cp_hi = ctx.coherent_peak
        p0, p1, p2 = ctx.ps_pars

        for flux, scaled in zip(flux_rows, scaled_energy_rows):
            energy = (
                ctx.endpoint * (float(scaled[0]) + float(scaled[1])) / 2.0 + ctx.delta_e
            )
            if not (cp_lo < energy < cp_hi):
                continue

            psa = PSFlux.ps_accept(energy, p0, p1, p2)
            if psa <= 0.0:
                continue

            ibin = tagged_flux_histogram.get_index(energy)
            if ibin is None:
                continue

            c = float(flux[1]) * ctx.scale / psa
            e = float(flux[2]) * ctx.scale / psa
            tagged_flux_histogram.counts[ibin] += c
            tagged_flux_histogram.errors[ibin] = np.sqrt(
                tagged_flux_histogram.errors[ibin] ** 2 + e**2
            )
            tag_e_flux_histogram.counts[ibin] += c
            tag_e_flux_histogram.errors[ibin] = np.sqrt(
                tag_e_flux_histogram.errors[ibin] ** 2 + e**2
            )
            ibin_id = tag_flux_histogram.get_index(float(flux[0]))
            if ibin_id is not None:
                tag_flux_histogram.counts[ibin_id] += c

    def _fill_lumi(
        self,
        *,
        tagged_flux_histogram: Histogram,
        tagged_lumi_histogram: Histogram,
        target_sc: tuple[float, float],
    ):
        n_sc, n_sc_err = target_sc
        for i in range(tagged_flux_histogram.bins):
            cnt = tagged_flux_histogram.counts[i]
            if cnt <= 0.0:
                continue
            lumi = cnt * n_sc / 1e12
            flux_rel = tagged_flux_histogram.errors[i] / cnt
            target_rel = n_sc_err / n_sc
            tagged_lumi_histogram.counts[i] = lumi
            tagged_lumi_histogram.errors[i] = lumi * np.sqrt(
                flux_rel**2 + target_rel**2
            )

    def get_histograms(
        self, bins: int = 120, limits: tuple[float, float] = (7.8, 9.0)
    ) -> PSFluxHistograms:
        tagged_flux_histogram = Histogram.empty(bins, limits)
        tagm_e_flux_histogram = Histogram.empty(bins, limits)
        tagm_flux_histogram = Histogram.empty(102, (1, 103))
        tagh_e_flux_histogram = Histogram.empty(bins, limits)
        tagh_flux_histogram = Histogram.empty(274, (1, 275))
        tagged_lumi_histogram = Histogram.empty(bins, limits)

        for run in get_all_polarized_run_numbers():
            ctx = self._run_ctx(run)

            # TAGM
            self._accumulate_detector(
                ctx=ctx,
                flux_rows=self.df_tagm_tagged_flux[run],
                scaled_energy_rows=self.df_tagm_scaled_energy[run],
                tagged_flux_histogram=tagged_flux_histogram,
                tag_e_flux_histogram=tagm_e_flux_histogram,
                tag_flux_histogram=tagm_flux_histogram,
            )

            # TAGH
            self._accumulate_detector(
                ctx=ctx,
                flux_rows=self.df_tagh_tagged_flux[run],
                scaled_energy_rows=self.df_tagh_scaled_energy[run],
                tagged_flux_histogram=tagged_flux_histogram,
                tag_e_flux_histogram=tagh_e_flux_histogram,
                tag_flux_histogram=tagh_flux_histogram,
            )

            # luminosity from total tagged flux
            self._fill_lumi(
                tagged_flux_histogram=tagged_flux_histogram,
                tagged_lumi_histogram=tagged_lumi_histogram,
                target_sc=ctx.target_sc,
            )

        return {
            'tagged_flux_histogram': tagged_flux_histogram,
            'tagm_e_flux_histogram': tagm_e_flux_histogram,
            'tagm_flux_histogram': tagm_flux_histogram,
            'tagh_e_flux_histogram': tagh_e_flux_histogram,
            'tagh_flux_histogram': tagh_flux_histogram,
            'tagged_lumi_histogram': tagged_lumi_histogram,
        }


def get_coherent_peak(run_number: int) -> tuple[float, float]:
    if get_run_period(run_number) != 's20':
        return (8.2, 8.8)
    return (8.0, 8.6)


def get_rcdb_text_condition(condition: str) -> dict[int, str]:
    with sqlite3.connect(RCDB_CONNECTION) as rcdb:
        cursor = rcdb.cursor()
        query = f"""
        {RCDB_SELECTION_PREFIX}
        SELECT r.number, c.text_value
        FROM conditions c
        JOIN condition_types ct ON c.condition_type_id = ct.id
        JOIN runs r ON c.run_number = r.number
        WHERE ct.name = '{condition}'
        {RCDB_SELECTION_SUFFIX}
        ORDER BY r.number
        """  # noqa: S608
        cursor.execute(query)
        return dict(cursor.fetchall())


@dataclass
class ScalingFactors:
    hodoscope_hi_factor: float
    hodoscope_lo_factor: float
    microscope_factor: float
    energy_bound_hi: float
    energy_bound_lo: float


class CCDBData:
    def __init__(self) -> None:
        self.accidental_scaling_factors: dict[int, ScalingFactors] = {}
        if CCDB_PATH.exists():
            with CCDB_PATH.open('rb') as f:
                self.accidental_scaling_factors = pickle.load(f)  # noqa: S301
        else:
            logger.info('Building CCDB cache...')
            self.accidental_scaling_factors = self._build_from_db()
            CCDB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CCDB_PATH.open('wb') as f:
                pickle.dump(
                    self.accidental_scaling_factors, f, protocol=pickle.HIGHEST_PROTOCOL
                )

    @staticmethod
    def _build_from_db() -> dict[int, ScalingFactors]:
        accidental_scaling_factors: dict[int, ScalingFactors] = {}
        with sqlite3.connect(CCDB_CONNECTION) as ccdb:
            cursor = ccdb.cursor()
            query = """
            SELECT rr.runMin, rr.runMax, cs.vault
            FROM directories d
            JOIN typeTables tt ON d.id = tt.directoryId
            JOIN constantSets cs ON tt.id = cs.constantTypeId
            JOIN assignments a ON cs.id = a.constantSetId
            JOIN runRanges rr ON a.runRangeId = rr.id
            LEFT JOIN variations v ON a.variationId = v.id
            WHERE d.name = 'ANALYSIS'
            AND tt.name = 'accidental_scaling_factor'
            AND v.name IS 'default'
            ORDER BY rr.runMin, a.created DESC
            """
            cursor.execute(query)
            asf_results = cursor.fetchall()
            for run_min, run_max, vault in asf_results:
                data = [float(v) for v in vault.split('|')]
                fb = tuple(data[:8])
                scale_factors = ScalingFactors(fb[0], fb[2], fb[4], fb[6], fb[7])
                for run in range(run_min, run_max + 1):
                    accidental_scaling_factors[run] = scale_factors
        return accidental_scaling_factors

    def get_scaling(
        self,
        run_number: int,
        beam_energy: float,
    ) -> float:
        if (factors := self.accidental_scaling_factors.get(run_number)) is None:
            return 1.0
        if beam_energy > factors.energy_bound_hi:
            return float(factors.hodoscope_hi_factor)
        if beam_energy > factors.energy_bound_lo:
            return float(factors.microscope_factor)
        return float(factors.hodoscope_lo_factor)

    def get_accidental_weight(
        self,
        run_number: int,
        beam_energy: float,
        rf: float,
        *,
        is_mc: bool,
        n_out_of_time_peaks: int = 8,
    ) -> float:
        relative_beam_bucket = int(np.floor(rf / 4.008016032 + 0.5))
        if abs(relative_beam_bucket) == 1:
            return 0.0
        if abs(relative_beam_bucket) == 0:
            return 1.0
        scale = (
            1.0
            if is_mc
            else self.get_scaling(
                run_number,
                beam_energy,
            )
        )
        return float(-scale / (n_out_of_time_peaks - 2))


class RCDBData:
    def __init__(self) -> None:
        self.pol_angle_data: dict[int, tuple[str, str, float]] = {}
        self.pol_magnitudes: dict[str, dict[str, Histogram]] = {}
        if RCDB_PATH.exists():
            with RCDB_PATH.open('rb') as f:
                self.pol_angle_data, self.pol_magnitudes = pickle.load(f)  # noqa: S301
        else:
            logger.info('Building RCDB cache...')
            self.pol_angle_data, self.pol_magnitudes = self._build_from_db()
            RCDB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with RCDB_PATH.open('wb') as f:
                pickle.dump(
                    (self.pol_angle_data, self.pol_magnitudes),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    @staticmethod
    def _build_from_db() -> tuple[
        dict[int, tuple[str, str, float]], dict[str, dict[str, Histogram]]
    ]:
        pol_angle_data: dict[int, tuple[str, str, float]] = {}
        pol_magnitudes: dict[str, dict[str, Histogram]] = {}
        with sqlite3.connect(RCDB_CONNECTION) as rcdb:
            cursor = rcdb.cursor()
            query = f"""
            {RCDB_SELECTION_PREFIX}
            SELECT r.number, c.float_value
            FROM conditions c
            JOIN condition_types ct ON c.condition_type_id = ct.id
            JOIN runs r ON c.run_number = r.number
            WHERE ct.name = 'polarization_angle'
            {RCDB_SELECTION_SUFFIX}
            ORDER BY r.number
            """  # noqa: S608
            cursor.execute(query)
            pol_angle_results = cursor.fetchall()
            for run_number, angle_deg in pol_angle_results:
                run_period = get_run_period(run_number)
                pol_angle = get_pol_angle(run_period, str(angle_deg))
                if pol_angle and run_period:
                    pol_angle_data[run_number] = (
                        run_period,
                        str(angle_deg).split('.')[0],
                        pol_angle,
                    )
        for rp, hist_path in POL_HIST_PATHS.items():
            hists = {}
            tfile = uproot.open(hist_path)  # ty: ignore TODO:
            for pol in ['0', '45', '90', '135']:
                hist = tfile[f'hPol{pol}']
                if isinstance(hist, HasBranches | ReadOnlyDirectory):
                    msg = f'Error reading histograms from {hist_path}'
                    raise OSError(msg)
                hists[pol] = Histogram(*hist.to_numpy())
            pol_magnitudes[rp] = hists
        return pol_angle_data, pol_magnitudes

    def get_pol_data(
        self,
        run_number: int,
        beam_energy: float,
    ) -> dict[str, float] | None:
        pol_angle_data = self.pol_angle_data.get(run_number)
        if pol_angle_data is None:
            return None
        run_period, pol_name, angle = pol_angle_data
        if angle is None:
            return None
        pol_hist = self.pol_magnitudes[run_period][pol_name]
        energy_index = pol_hist.get_index(beam_energy)
        if energy_index is None:
            return None
        magnitude = pol_hist.counts[energy_index]
        return {
            'x': float(magnitude * np.cos(angle)),
            'y': float(magnitude * np.sin(angle)),
            'is_polarized': 1.0,
        }
