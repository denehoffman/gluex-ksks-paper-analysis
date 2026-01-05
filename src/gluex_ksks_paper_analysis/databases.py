from __future__ import annotations
import gluex_lumi
from gluex_rcdb import RCDB, aliases
from gluex_ccdb import CCDB

import pickle
from dataclasses import dataclass

import numpy as np
import uproot
from uproot.behaviors.TBranch import HasBranches
from uproot.reading import ReadOnlyDirectory
from loguru import logger

from gluex_ksks_paper_analysis.environment import (
    CCDB_CONNECTION,
    POL_HIST_PATHS,
    POLARIZED_RUN_NUMBERS_PATH,
    PSFLUX_DATA_PATH,
    RCDB_CONNECTION,
    ACCIDENTAL_SCALING_FACTORS_PATH,
    POLARIZATION_DATA_PATH,
)
from gluex_ksks_paper_analysis.utilities import Histogram

TRUE_POL_ANGLES = {
    's17': {'0': 1.8, '45': 47.9, '90': 94.5, '135': -41.6},
    's18': {'0': 4.1, '45': 48.5, '90': 94.2, '135': -42.4},
    'f18': {'0': 3.3, '45': 48.3, '90': 92.9, '135': -42.1},
    's20': {'0': 1.4, '45': 47.1, '90': 93.4, '135': -42.2},
}

REST_VERSIONS = {'s17': 3, 's18': 2, 'f18': 2, 's20': 1}


def build_caches() -> None:
    _ = AccidentalScalingFactors()
    _ = PolarizationData()
    _ = get_all_polarized_run_numbers()
    _ = PSFluxData()
    RCDB_CONNECTION.unlink(missing_ok=True)
    CCDB_CONNECTION.unlink(missing_ok=True)


def get_all_polarized_run_numbers() -> set[int]:
    if POLARIZED_RUN_NUMBERS_PATH.exists():
        return pickle.load(POLARIZED_RUN_NUMBERS_PATH.open('rb'))  # noqa: S301
    logger.info('Building polarized run numbers cache...')
    rcdb = RCDB(str(RCDB_CONNECTION))
    run_numbers = set()
    for run_period in REST_VERSIONS.keys():
        run_numbers |= set(
            rcdb.fetch_runs(
                run_period=run_period,
                filters=[
                    aliases.approved_production(run_period),
                    aliases.is_coherent_beam,
                ],
            )
        )
    POLARIZED_RUN_NUMBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POLARIZED_RUN_NUMBERS_PATH.open('wb') as f:
        pickle.dump(run_numbers, f, protocol=pickle.HIGHEST_PROTOCOL)
    return run_numbers


@dataclass
class ScalingFactors:
    hodoscope_hi_factor: float
    hodoscope_lo_factor: float
    microscope_factor: float
    microscope_energy_hi: float
    microscope_energy_lo: float


class AccidentalScalingFactors:
    def __init__(self) -> None:
        self.accidental_scaling_factors: dict[int, ScalingFactors] = {}

        if ACCIDENTAL_SCALING_FACTORS_PATH.exists():
            self.accidental_scaling_factors = pickle.load(
                ACCIDENTAL_SCALING_FACTORS_PATH.open('rb')
            )  # noqa: S301
            return
        logger.info('Building accidental scaling factors cache...')
        self.accidental_scaling_factors = self._build_from_db()

        ACCIDENTAL_SCALING_FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ACCIDENTAL_SCALING_FACTORS_PATH.open('wb') as f:
            pickle.dump(
                self.accidental_scaling_factors, f, protocol=pickle.HIGHEST_PROTOCOL
            )

    @staticmethod
    def _build_from_db() -> dict[int, ScalingFactors]:
        ccdb = CCDB(str(CCDB_CONNECTION))
        scaling_factors: dict[int, ScalingFactors] = {}
        for run_period in REST_VERSIONS.keys():
            data = ccdb.fetch_run_period(
                'ANALYSIS/accidental_scaling_factor', run_period=run_period
            )
            for run_number, d in data.items():
                r = d.row(0)
                scaling_factors[run_number] = ScalingFactors(
                    hodoscope_hi_factor=r.value('HODOSCOPE_HI_FACTOR'),  # ty:ignore[invalid-argument-type]
                    hodoscope_lo_factor=r.value('HODOSCOPE_LO_FACTOR'),  # ty:ignore[invalid-argument-type]
                    microscope_factor=r.value('MICROSCOPE_FACTOR'),  # ty:ignore[invalid-argument-type]
                    microscope_energy_hi=r.value('MICROSCOPE_ENERGY_HI'),  # ty:ignore[invalid-argument-type]
                    microscope_energy_lo=r.value('MICROSCOPE_ENERGY_LO'),  # ty:ignore[invalid-argument-type]
                )
        return scaling_factors

    def get_scaling(
        self,
        run_number: int,
        beam_energy: float,
    ) -> float:
        if (factors := self.accidental_scaling_factors.get(run_number)) is None:
            return 1.0
        if beam_energy > factors.microscope_energy_hi:
            return float(factors.hodoscope_hi_factor)
        if beam_energy > factors.microscope_energy_lo:
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


class PSFluxData:
    def __init__(self) -> None:
        self.tagged_flux: Histogram
        self.tagged_luminosity: Histogram
        self.tagh_flux: Histogram
        self.tagm_flux: Histogram
        self.tagged_luminosity_by_run_period: dict[str, Histogram]

        if PSFLUX_DATA_PATH.exists():
            payload = pickle.load(PSFLUX_DATA_PATH.open('rb'))  # noqa: S301
            if isinstance(payload, dict):
                self.tagged_flux = payload['tagged_flux']
                self.tagged_luminosity = payload['tagged_luminosity']
                self.tagh_flux = payload['tagh_flux']
                self.tagm_flux = payload['tagm_flux']
                self.tagged_luminosity_by_run_period = payload.get(
                    'tagged_luminosity_by_run_period',
                    payload.get('tagged_luminosity_by_run'),
                )
                if self.tagged_luminosity_by_run_period is None:
                    logger.info('Recomputing per-run-period luminosity cache...')
                    self.tagged_luminosity_by_run_period = (
                        self._build_per_run_period_luminosity()
                    )
                    self._write_cache()
                return
            (
                self.tagged_flux,
                self.tagged_luminosity,
                self.tagh_flux,
                self.tagm_flux,
            ) = payload
            logger.info('Upgrading flux cache with per-run-period luminosities...')
            self.tagged_luminosity_by_run_period = (
                self._build_per_run_period_luminosity()
            )
            self._write_cache()
            return
        logger.info('Building flux cache...')
        (
            self.tagged_flux,
            self.tagged_luminosity,
            self.tagh_flux,
            self.tagm_flux,
            self.tagged_luminosity_by_run_period,
        ) = self._build_from_db()
        self._write_cache()

    def _write_cache(self) -> None:
        PSFLUX_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PSFLUX_DATA_PATH.open('wb') as f:
            pickle.dump(
                {
                    'tagged_flux': self.tagged_flux,
                    'tagged_luminosity': self.tagged_luminosity,
                    'tagh_flux': self.tagh_flux,
                    'tagm_flux': self.tagm_flux,
                    'tagged_luminosity_by_run_period': self.tagged_luminosity_by_run_period,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def _histogram_from_dict(data: dict[str, list[float]]) -> Histogram:
        return Histogram(
            data['counts'],
            data['edges'],
            data['errors'],
        )

    def _build_per_run_period_luminosity(self) -> dict[str, Histogram]:
        per_run: dict[str, Histogram] = {}
        for run_period, rest_version in REST_VERSIONS.items():
            lumi_data = gluex_lumi.get_flux_histograms(
                run_periods={run_period: rest_version},
                edges=list(np.histogram_bin_edges([], 80, range=(8.0, 8.8))),
                coherent_peak=True,
                polarized=True,
                rcdb=str(RCDB_CONNECTION),
                ccdb=str(CCDB_CONNECTION),
            )
            per_run[run_period] = self._histogram_from_dict(
                lumi_data.tagged_luminosity.as_dict()
            )
        return per_run

    def _build_from_db(
        self,
    ) -> tuple[Histogram, Histogram, Histogram, Histogram, dict[str, Histogram]]:
        lumi_data = gluex_lumi.get_flux_histograms(
            run_periods=REST_VERSIONS,
            edges=list(np.histogram_bin_edges([], 80, range=(8.0, 8.8))),
            coherent_peak=True,
            polarized=True,
            rcdb=str(RCDB_CONNECTION),
            ccdb=str(CCDB_CONNECTION),
        )
        tagged_flux_data = lumi_data.tagged_flux.as_dict()
        tagged_luminosity_data = lumi_data.tagged_luminosity.as_dict()
        tagh_flux_data = lumi_data.tagh_flux.as_dict()
        tagm_flux_data = lumi_data.tagm_flux.as_dict()
        return (
            self._histogram_from_dict(tagged_flux_data),
            self._histogram_from_dict(tagged_luminosity_data),
            self._histogram_from_dict(tagh_flux_data),
            self._histogram_from_dict(tagm_flux_data),
            self._build_per_run_period_luminosity(),
        )


class PolarizationData:
    def __init__(self) -> None:
        self.pol_angle_data: dict[int, tuple[str, str, float]] = {}
        self.pol_magnitudes: dict[str, dict[str, Histogram]] = {}
        if POLARIZATION_DATA_PATH.exists():
            self.pol_angle_data, self.pol_magnitudes = pickle.load(
                POLARIZATION_DATA_PATH.open('rb')
            )  # noqa: S301
            return
        logger.info('Building polarization data cache...')
        self.pol_angle_data, self.pol_magnitudes = self._build_from_db()
        POLARIZATION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with POLARIZATION_DATA_PATH.open('wb') as f:
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
        rcdb = RCDB(str(RCDB_CONNECTION))
        for run_period in REST_VERSIONS.keys():
            data = rcdb.fetch(
                ['polarization_angle'],
                run_period=run_period,
                filters=[
                    aliases.approved_production(run_period),
                    aliases.is_coherent_beam,
                ],
            )
            for run_number, d in data.items():
                pol_angle_deg = d['polarization_angle']
                pol_angle_str = str(pol_angle_deg).split('.')[0]
                pol_angle = TRUE_POL_ANGLES[run_period][pol_angle_str] * np.pi / 180
                pol_angle_data[run_number] = (run_period, pol_angle_str, pol_angle)
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
        pol_hist = self.pol_magnitudes[run_period][pol_name]
        energy_index = pol_hist.get_index(beam_energy)
        if energy_index is None:
            return None
        magnitude = pol_hist.counts[energy_index]
        return {
            'pol_magnitude': float(magnitude),
            'pol_angle': float(angle),
        }
