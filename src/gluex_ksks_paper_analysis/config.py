from __future__ import annotations

import json
import operator
import pickle
import re
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from functools import reduce
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import laddu as ld
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator
from loguru import logger
from numpy.typing import NDArray
from pint import Quantity, Unit, UnitRegistry
from tqdm.auto import trange, tqdm

from gluex_ksks_paper_analysis.databases import (
    AccidentalScalingFactors,
    get_all_polarized_run_numbers,
    PSFluxData,
)
from gluex_ksks_paper_analysis.environment import (
    BLACK,
    CMAP,
    DATASET_PATH,
    FITS_PATH,
    BLUE,
    RED,
    NORM,
    PLOTS_PATH,
    REPORTS_PATH,
    RUN_PERIODS,
)
from gluex_ksks_paper_analysis.fit import (
    AngularMomentum,
    Reflectivity,
    Wave,
    build_model,
)
from gluex_ksks_paper_analysis.splot import (
    REGISTRY,
    fit_mixture,
)
from gluex_ksks_paper_analysis.utilities import Histogram
from gluex_ksks_paper_analysis.variables import add_variable

if TYPE_CHECKING:
    from gluex_ksks_paper_analysis.splot import ComponentPDF


_JSON_SCHEMA = json.loads(
    (
        Path(str(resources.files(__package__))) / 'ksks_analysis_config.schema.json'
    ).read_text()
)

_RULE_REGEX = re.compile(
    r"""(?x)
    ^\s*
    (?P<var>[A-Za-z_]\w*)
    \s*(?P<op><=|>=|<|>)\s*
    (?P<value>-?\d+(?:\.\d+)?)\s*$
    """
)

_VARIABLES: list[str] = _JSON_SCHEMA['$defs']['VariableType']['enum']
_WEIGHTS: list[str] = _JSON_SCHEMA['$defs']['WeightType']['enum']

_BASE_COLUMNS: list[str] = [
    'RunNumber',
    'EventNumber',
    'weight',
    'beam_e',
    'beam_px',
    'beam_py',
    'beam_pz',
    'proton_e',
    'proton_px',
    'proton_py',
    'proton_pz',
    'kshort1_e',
    'kshort1_px',
    'kshort1_py',
    'kshort1_pz',
    'kshort2_e',
    'kshort2_px',
    'kshort2_py',
    'kshort2_pz',
    'piplus1_e',
    'piplus1_px',
    'piplus1_py',
    'piplus1_pz',
    'piminus1_e',
    'piminus1_px',
    'piminus1_py',
    'piminus1_pz',
    'piplus2_e',
    'piplus2_px',
    'piplus2_py',
    'piplus2_pz',
    'piminus2_e',
    'piminus2_px',
    'piminus2_py',
    'piminus2_pz',
    'RFL1',
    'RFL2',
    'ChiSqDOF',
    'RF',
    'Proton_Z',
]


@dataclass(frozen=True)
class Cut:
    rules: list[str]
    cache: bool = False

    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame:
        logger.info(f'Applying cuts: {self.rules}')
        exprs: list[pl.Expr] = []
        for rule in self.rules:
            m = _RULE_REGEX.fullmatch(rule)
            if m is None:
                msg = f"Failed to parse cut rule '{rule}'"
                raise ConfigError(msg)
            variable_name = m.group('var')
            df = add_variable(variable_name, df)
            variable = pl.col(variable_name)
            op = m.group('op')
            value = float(m.group('value'))
            match op:
                case '<':
                    exprs.append(variable >= value)
                case '<=':
                    exprs.append(variable > value)
                case '>':
                    exprs.append(variable <= value)
                case '>=':
                    exprs.append(variable < value)
        return df.filter(reduce(operator.and_, exprs, pl.lit(True)))


@dataclass(frozen=True)
class Weight:
    weight_type: str
    sig: str | None
    bkg: str | None
    cache: bool = False

    def apply(
        self,
        df: pl.LazyFrame,
        df_sigmc: pl.LazyFrame,
        df_bkgmc: pl.LazyFrame,
        is_mc: bool = False,
    ) -> pl.LazyFrame:
        logger.info(
            f'Applying weight: {self.weight_type} (sig={self.sig}, bkg={self.bkg})'
        )
        match self.weight_type:
            case 'accidental':
                ccdb = AccidentalScalingFactors()
                polarized_runs = get_all_polarized_run_numbers()
                return Weight.set_accidental_weights(
                    df, ccdb=ccdb, polarized_runs=polarized_runs, is_mc=is_mc
                )
            case 'splot':
                if self.sig is None:
                    raise ConfigError('Missing sig for splot weight')
                sig_fn = REGISTRY.get(self.sig)
                if sig_fn is None:
                    msg = f'Unsupported sig {self.sig}'
                    raise ConfigError(msg)
                signal = sig_fn(df_sigmc, 'sig')
                if self.bkg is None:
                    raise ConfigError('Missing bkg for splot weight')
                bkg_fn = REGISTRY.get(self.bkg)
                if bkg_fn is None:
                    msg = f'Unsupported bkg {self.bkg}'
                    raise ConfigError(msg)
                background = bkg_fn(df_bkgmc, 'bkg')
                return Weight.set_splot_weights(
                    df, signal=signal, background=background
                )
            case _:
                msg = f'Unsupported weight type {self.weight_type}'
                raise ConfigError(msg)

    @staticmethod
    def set_accidental_weights(
        df: pl.LazyFrame,
        *,
        ccdb: AccidentalScalingFactors,
        polarized_runs: set[int],
        is_mc: bool,
    ) -> pl.LazyFrame:
        return (
            df.filter(pl.col('RunNumber').is_in(polarized_runs))
            .sort(['RunNumber', 'EventNumber', 'ChiSqDOF'])
            .group_by(['RunNumber', 'EventNumber'])
            .first()
            .with_columns(
                pl.struct('RunNumber', 'beam_e', 'RF', 'weight')
                .map_elements(
                    lambda s: s['weight']
                    * ccdb.get_accidental_weight(
                        s['RunNumber'], s['beam_e'], s['RF'], is_mc=is_mc
                    ),
                    return_dtype=pl.Float64,
                )
                .alias('weight')
            )
            .filter(pl.col('weight').ne(0.0))
            .with_columns(weight=pl.col('weight').cast(pl.Float32))
        )

    @staticmethod
    def set_splot_weights(
        df: pl.LazyFrame, *, signal: ComponentPDF, background: ComponentPDF
    ) -> pl.LazyFrame:
        data = df.select(
            ['RunNumber', 'EventNumber', 'RFL1', 'RFL2', 'weight']
        ).collect()
        rfls = data[['RFL1', 'RFL2']].to_numpy()
        weights = data['weight'].to_numpy()
        fit_result = fit_mixture(rfls, weights, signal, background, p0_yields=None)
        # TODO: make plots and save fit result
        fit_result.plot_projection(rfls, weights)
        sweights = fit_result.get_sweights(rfls)
        new_weights = weights * sweights
        weights_df = pl.DataFrame(
            {
                'RunNumber': data['RunNumber'],
                'EventNumber': data['EventNumber'],
                'sweight': new_weights,
            }
        )

        return (
            df.join(weights_df.lazy(), on=['RunNumber', 'EventNumber'], how='left')
            .with_columns(weight=pl.col('sweight').cast(pl.Float32))
            .drop('sweight')
            .filter(pl.col('weight') != 0.0)
        )


@dataclass(frozen=True)
class Plot1D:
    variable: str
    label: str
    bins: int
    limits: tuple[float, float]
    units: str = ''

    @property
    def unit(self) -> Unit:
        ureg = UnitRegistry()
        return ureg.parse_units(self.units)

    @property
    def edges(self) -> NDArray:
        return np.histogram_bin_edges([], self.bins, range=self.limits)

    @property
    def xlabel(self) -> str:
        if self.units == '':
            return self.label
        else:
            return f'{self.label} (${self.unit:~L}$)'

    @property
    def ylabel(self) -> str:
        width = float(np.diff(self.edges)[0])
        if self.units == '':
            return f'Counts / ${round(width, 2)}$'
        else:
            bin_width = Quantity(width, self.unit).to_compact()
            bin_width = Quantity(round(bin_width.m, 2), bin_width.u)
            return f'Counts / ${bin_width:~L}$'

    def write(self, lfs: list[pl.LazyFrame], output_path: Path) -> None:
        logger.info(f'Writing plot: {output_path}')
        plt.style.use('gluex_ksks_paper_analysis.style')
        _, ax = plt.subplots()
        lfs = [add_variable(self.variable, df) for df in lfs]
        dfs = [df.select([self.variable, 'weight']).collect() for df in lfs]
        variable = np.concatenate([df[self.variable].to_numpy() for df in dfs])
        weights = np.concatenate([df['weight'].to_numpy() for df in dfs])
        errors = np.sqrt(np.histogram(variable, bins=self.edges, weights=weights**2)[0])
        hist = Histogram(
            *np.histogram(variable, bins=self.edges, weights=weights), errors=errors
        )
        ax.stairs(hist.counts, hist.edges, color='C0')
        ax.errorbar(
            hist.centers,
            hist.counts,
            yerr=hist.errors,
            ls='none',
            color='C0',
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        logger.info(f'Saving plot: {output_path}')
        plt.savefig(output_path)
        plt.close()


@dataclass(frozen=True)
class Plot2D:
    x: str
    y: str

    def write(
        self, lfs: list[pl.LazyFrame], output_path: Path, plots1d: dict[str, Plot1D]
    ) -> None:
        logger.info(f'Writing plot: {output_path}')
        hist_plot_x = plots1d[self.x]
        hist_plot_y = plots1d[self.y]
        plt.style.use('gluex_ksks_paper_analysis.style')
        _, ax = plt.subplots()
        lfs = [add_variable(hist_plot_x.variable, lf) for lf in lfs]
        lfs = [add_variable(hist_plot_y.variable, lf) for lf in lfs]
        dfs = [
            lf.select([hist_plot_x.variable, hist_plot_y.variable, 'weight']).collect()
            for lf in lfs
        ]
        data_x = np.concatenate([df[hist_plot_x.variable].to_numpy() for df in dfs])
        data_y = np.concatenate([df[hist_plot_y.variable].to_numpy() for df in dfs])
        weights = np.concatenate([df['weight'].to_numpy() for df in dfs])
        ax.hist2d(
            data_x,
            data_y,
            bins=[hist_plot_x.bins, hist_plot_y.bins],
            range=[hist_plot_x.limits, hist_plot_y.limits],
            weights=weights,
            cmap=CMAP,
            norm=NORM,
        )
        ax.set_xlabel(hist_plot_x.xlabel)
        ax.set_ylabel(hist_plot_y.xlabel)
        logger.info(f'Saving plot: {output_path}')
        plt.savefig(output_path)
        plt.close()


@dataclass(frozen=True)
class Dataset:
    source: str
    steps: list[str]

    def dataset(
        self,
        run_period: str,
        config: Config,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
    ) -> ld.Dataset:
        return ld.io.read_parquet(
            self._generate_path_from_steps(run_period, self.steps, config),
            p4s=p4s,
            aux=aux,
        )

    def generate_path(self, run_period: str, config: Config) -> Path:
        return self._generate_path_from_steps(run_period, self.steps, config)

    def _get_path(self, run_period: str, steps: list[str]) -> Path:
        p = DATASET_PATH
        for step in steps:
            p /= step
        p /= f'{self.source}_{run_period}.parquet'
        return p

    def _generate_path_from_steps(
        self, run_period: str, steps: list[str], config: Config
    ) -> Path:
        _ = self._get_lazyframe_from_steps(run_period, steps, config)
        return self._get_path(run_period, steps)

    def _get_lazyframe_from_steps(
        self, run_period: str, steps: list[str], config: Config
    ) -> pl.LazyFrame:
        path = self._get_path(run_period, steps)
        logger.debug(f'Searching for dataset: {path}')
        if path.exists():
            return pl.scan_parquet(path)
        df = self._get_lazyframe_from_steps(run_period, steps[:-1], config)
        last_step = steps[-1]
        if last_step in config.cuts and last_step in config.weights:
            msg = f'Ambiguous step {last_step} (both a cut and a weight)'
            raise ConfigError(msg)
        if cut := config.cuts.get(last_step):
            df = cut.apply(df)
            all_columns = set(df.collect_schema().names())
            columns = [c for c in _BASE_COLUMNS if c in all_columns]
            if {'pol_magnitude', 'pol_angle'} <= all_columns:
                columns += ['pol_magnitude', 'pol_angle']

            if cut.cache and ld.mpi.is_root():
                path.parent.mkdir(parents=True, exist_ok=True)
                df.select(columns).sink_parquet(path)
            return df
        if weight := config.weights.get(last_step):
            ds_sigmc = Dataset('sigmc', steps[:-1])
            ds_bkgmc = Dataset('bkgmc', steps[:-1])
            is_mc = self.source != 'data'
            df = weight.apply(
                df,
                ds_sigmc.lazyframe(run_period, config),
                ds_bkgmc.lazyframe(run_period, config),
                is_mc,
            )
            all_columns = set(df.collect_schema().names())
            columns = [c for c in _BASE_COLUMNS if c in all_columns]
            if {'pol_magnitude', 'pol_angle'} <= all_columns:
                columns += ['pol_magnitude', 'pol_angle']
            if weight.cache and ld.mpi.is_root():
                path.parent.mkdir(parents=True, exist_ok=True)
                df.select(columns).sink_parquet(path)
            return df
        else:
            msg = f'Failed to find {last_step} in either cuts or weights'
            raise ConfigError(msg)

    def lazyframe(self, run_period: str, config: Config) -> pl.LazyFrame:
        logger.debug(f'Fetching dataset: {self.source} [{run_period}] ({self.steps})')
        return self._get_lazyframe_from_steps(run_period, self.steps, config)


@dataclass
class FitResult:
    data: str
    accmc: str
    genmc: str
    waves: list[Wave]
    fits: list[ld.MinimizationSummary]
    bootstraps: list[list[ld.MinimizationSummary]]
    bin_edges: NDArray
    data_projection: dict[str, Histogram]
    projections: dict[str, dict[str, Histogram]]
    genmc_projections: dict[str, dict[str, Histogram]]
    genmc_counts: dict[str, Histogram]
    accmc_counts: dict[str, Histogram]

    @property
    def bins(self) -> int:
        return len(self.bin_edges) - 1

    @property
    def limits(self) -> tuple[float, float]:
        return self.bin_edges[0], self.bin_edges[-1]

    @cached_property
    def data_projection_total(self) -> Histogram:
        tot = Histogram.empty(
            len(self.bin_edges) - 1, (self.bin_edges[0], self.bin_edges[-1])
        )
        for proj in self.data_projection.values():
            tot += proj
        return tot

    @cached_property
    def projections_total(self) -> dict[str, Histogram]:
        out = {}
        for wave, wave_proj in self.projections.items():
            tot = Histogram.empty(
                len(self.bin_edges) - 1, (self.bin_edges[0], self.bin_edges[-1])
            )
            for proj in wave_proj.values():
                tot += proj
            out[wave] = tot
        return out

    @cached_property
    def genmc_projections_total(self) -> dict[str, Histogram]:
        out = {}
        for wave, wave_proj in self.genmc_projections.items():
            tot = Histogram.empty(
                len(self.bin_edges) - 1, (self.bin_edges[0], self.bin_edges[-1])
            )
            for proj in wave_proj.values():
                tot += proj
            out[wave] = tot
        return out

    @cached_property
    def accmc_total(self) -> Histogram:
        return Histogram.sum(list(self.accmc_counts.values()))

    @cached_property
    def genmc_total(self) -> Histogram:
        return Histogram.sum(list(self.genmc_counts.values()))

    @cached_property
    def acceptance(self) -> Histogram:
        return self.accmc_total / self.genmc_total


@dataclass
class FitSummary:
    data: str
    accmc: str
    genmc: str
    waves: list[Wave]
    fits: list[ld.MinimizationSummary]
    bootstraps: list[list[ld.MinimizationSummary]]
    bin_edges: NDArray
    data_projection: dict[str, Histogram]
    projections: dict[str, dict[str, Histogram]]


@dataclass
class GeneratedMCProjections:
    genmc_projections: dict[str, dict[str, Histogram]]
    genmc_counts: dict[str, Histogram]
    accmc_counts: dict[str, Histogram]


@dataclass
class BinnedDatasets:
    data: dict[str, ld.BinnedDataset]
    accmc: dict[str, ld.BinnedDataset]


@dataclass(frozen=True)
class Fit:
    waves: list[str]
    data: str
    accmc: str
    bins: int
    limits: tuple[float, float] = (1.0, 2.0)
    genmc: str = 'genmc'
    n_iterations: int = 20
    n_bootstraps: int = 100
    required_columns: list[str] = field(
        default_factory=lambda: [
            'beam_e',
            'beam_px',
            'beam_py',
            'beam_pz',
            'proton_e',
            'proton_px',
            'proton_py',
            'proton_pz',
            'kshort1_e',
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort2_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'pol_magnitude',
            'pol_angle',
            'weight',
        ]
    )
    required_columns_genmc: list[str] = field(
        default_factory=lambda: [
            'beam_e',
            'beam_px',
            'beam_py',
            'beam_pz',
            'proton_e',
            'proton_px',
            'proton_py',
            'proton_pz',
            'kshort1_e',
            'kshort1_px',
            'kshort1_py',
            'kshort1_pz',
            'kshort2_e',
            'kshort2_px',
            'kshort2_py',
            'kshort2_pz',
            'pol_magnitude',
            'pol_angle',
        ]
    )
    genmc_batch_size: int = 10_000_000
    reference_parameter: str | None = 'S+0+ real'
    reference_value: float = 100.0

    @staticmethod
    def _align_parameters(
        parameters: NDArray[np.float64],
        parameter_names: Sequence[str] | None,
        target_order: Sequence[str],
    ) -> NDArray[np.float64]:
        if parameter_names is None:
            msg = 'Missing parameter names for fit result; rerun the fit with the current code.'
            raise ValueError(msg)
        values = np.asarray(parameters, dtype=float)
        lookup = {name: values[i] for i, name in enumerate(parameter_names)}
        aligned: list[float] = []
        missing: list[str] = []
        for name in target_order:
            if name not in lookup:
                missing.append(name)
                continue
            aligned.append(lookup[name])
        if missing:
            msg = f'Missing parameter values for: {", ".join(missing)}'
            raise ValueError(msg)
        return np.asarray(aligned, dtype=float)

    def fit(
        self,
        name: str,
        config: Config,
    ) -> None:
        summary, preloaded_binned = self._ensure_fit_summary(
            name,
            config,
        )
        genmc_projections = self._ensure_genmc_projections(
            name,
            summary,
            config,
            preloaded_binned,
        )
        fit_result = self._build_fit_result(summary, genmc_projections)
        self._render_plots(name, config, fit_result)

    def fit_waves(
        self,
        config: Config,
        seed: int = 0,
    ) -> tuple[FitSummary, BinnedDatasets]:
        binned_datasets = self._load_binned_datasets(config)
        ds_data_binned = binned_datasets.data
        ds_accmc_binned = binned_datasets.accmc
        waves = [Wave(w) for w in self.waves]
        model = build_model(waves)
        rng = np.random.default_rng(seed)
        best_fits: list[ld.MinimizationSummary] = []
        bootstraps: list[list[ld.MinimizationSummary]] = []
        for ibin in trange(self.bins, desc='Bins', position=0, leave=True):
            nlls_ibin = [
                ld.NLL(
                    model
                    * ld.Scalar(
                        f'scale {run_period}', ld.parameter(f'scale {run_period}')
                    ),
                    ds_data_binned[run_period][ibin],
                    ds_accmc_binned[run_period][ibin],
                ).to_expression()
                for run_period in RUN_PERIODS
            ]
            nll_ibin = ld.likelihood_sum(nlls_ibin).fix('S+0+ real', 100.0).load()
            best_fit_ibin = None
            for _ in trange(
                self.n_iterations, desc='Iterations', position=1, leave=False
            ):
                free_params = nll_ibin.free_parameters
                p0 = rng.uniform(-100.0, 100.0, len(free_params))
                bounds = [(None, None)] * len(free_params)
                for i_param, name in enumerate(free_params):
                    if name.startswith('scale'):
                        p0[i_param] = rng.uniform(1_000_000.0, 6_000_000.0)
                        bounds[i_param] = (0.0, None)
                res = nll_ibin.minimize(
                    p0, settings={'skip_hessian': True}, bounds=bounds
                )
                if best_fit_ibin is None or res.fx < best_fit_ibin.fx:
                    best_fit_ibin = res
            if best_fit_ibin is None:
                msg = f'Failed to fit bin {ibin}!'
                raise RuntimeError(msg)
            best_fits.append(best_fit_ibin)
            bootstraps_ibin: list[ld.MinimizationSummary] = []
            for iboot in trange(
                self.n_bootstraps, desc='Bootstraps', position=1, leave=False
            ):
                nlls_iboot = [
                    ld.NLL(
                        model
                        * ld.Scalar(
                            f'scale {run_period}', ld.parameter(f'scale {run_period}')
                        ),
                        ds_data_binned[run_period][ibin].bootstrap(iboot),
                        ds_accmc_binned[run_period][ibin],
                    ).to_expression()
                    for run_period in RUN_PERIODS
                ]
                nll_iboot = ld.likelihood_sum(nlls_iboot).fix('S+0+ real', 100.0).load()
                free_params = nll_iboot.free_parameters
                bounds = [(None, None)] * len(free_params)
                for i_param, name in enumerate(free_params):
                    if name.startswith('scale '):
                        bounds[i_param] = (0.0, None)
                bootstrap_res = nll_iboot.minimize(
                    best_fit_ibin.x, settings={'skip_hessian': True}, bounds=bounds
                )
                bootstraps_ibin.append(bootstrap_res)
            bootstraps.append(bootstraps_ibin)
        bin_edges = np.histogram_bin_edges([], self.bins, range=self.limits)
        logger.info('Projecting fit results (data & accepted MC)...')
        projections: dict[str, dict[str, Histogram]] = {
            str(wave): {
                run_period: Histogram.empty(self.bins, self.limits)
                for run_period in RUN_PERIODS
            }
            for wave in waves
        }
        projections['total'] = {
            run_period: Histogram.empty(self.bins, self.limits)
            for run_period in RUN_PERIODS
        }
        data_projection = {
            run_period: Histogram.empty(self.bins, self.limits)
            for run_period in RUN_PERIODS
        }
        for ibin in trange(self.bins, desc='Bins'):
            for run_period in RUN_PERIODS:
                nll_ibin = ld.NLL(
                    model
                    * ld.Scalar(
                        f'scale {run_period}', ld.parameter(f'scale {run_period}')
                    ),
                    ds_data_binned[run_period][ibin],
                    ds_accmc_binned[run_period][ibin],
                ).fix('S+0+ real', 100.0)
                result_ibin = best_fits[ibin]
                result_params = self._align_parameters(
                    result_ibin.x,
                    result_ibin.parameter_names,
                    nll_ibin.free_parameters,
                )
                bootstraps_ibin = bootstraps[ibin]
                projections['total'][run_period].counts[ibin] = np.sum(
                    nll_ibin.project(result_params)
                )
                boot_nlls_ibin = [
                    ld.NLL(
                        model
                        * ld.Scalar(
                            f'scale {run_period}', ld.parameter(f'scale {run_period}')
                        ),
                        ds_data_binned[run_period][ibin].bootstrap(iboot),
                        ds_accmc_binned[run_period][ibin],
                    ).fix('S+0+ real', 100.0)
                    for iboot in range(len(bootstraps_ibin))
                ]
                projections['total'][run_period].errors[ibin] = np.std(
                    [
                        np.sum(
                            boot_nlls_ibin[iboot].project(
                                self._align_parameters(
                                    bootstrap.x,
                                    bootstrap.parameter_names,
                                    nll_ibin.free_parameters,
                                )
                            )
                        )
                        for iboot, bootstrap in enumerate(bootstraps_ibin)
                    ],
                    ddof=1,
                )
                data_projection[run_period].counts[ibin] = ds_data_binned[run_period][
                    ibin
                ].n_events_weighted
                data_projection[run_period].errors[ibin] = np.sqrt(
                    np.sum(np.power(ds_data_binned[run_period][ibin].weights, 2))
                )
                scale_name = f'scale {run_period}'
                for wave in waves:
                    projections[str(wave)][run_period].counts[ibin] = np.sum(
                        nll_ibin.project_with(
                            result_params,
                            wave.amplitude_names + [scale_name],
                        )
                    )
                    projections[str(wave)][run_period].errors[ibin] = np.std(
                        [
                            np.sum(
                                boot_nlls_ibin[iboot].project_with(
                                    self._align_parameters(
                                        bootstrap.x,
                                        bootstrap.parameter_names,
                                        nll_ibin.free_parameters,
                                    ),
                                    wave.amplitude_names + [scale_name],
                                )
                            )
                            for iboot, bootstrap in enumerate(bootstraps_ibin)
                        ],
                        ddof=1,
                    )
        return (
            FitSummary(
                data=self.data,
                accmc=self.accmc,
                genmc=self.genmc,
                waves=waves,
                fits=best_fits,
                bootstraps=bootstraps,
                bin_edges=bin_edges,
                data_projection=data_projection,
                projections=projections,
            ),
            binned_datasets,
        )

    def _ensure_fit_summary(
        self,
        name: str,
        config: Config,
    ) -> tuple[FitSummary, BinnedDatasets | None]:
        summary_path = self._summary_path(name, config.datasets)
        if summary_path.exists():
            logger.info(f'Loading fit summary: {name}')
            return pickle.load(summary_path.open('rb')), None
        logger.info(f'Running fit: {name}')
        summary, binned = self.fit_waves(config)
        if ld.mpi.is_root():
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(summary, summary_path.open('wb'))
        return summary, binned

    def _ensure_genmc_projections(
        self,
        name: str,
        summary: FitSummary,
        config: Config,
        preloaded_binned: BinnedDatasets | None,
    ) -> GeneratedMCProjections:
        projection_path = self._genmc_projection_path(name, config.datasets)
        if projection_path.exists():
            logger.info(f'Loading generated MC projections: {name}')
            return pickle.load(projection_path.open('rb'))
        logger.info(f'Projecting generated MC: {name}')
        projections = self.project_generated_mc(summary, config, preloaded_binned)
        if ld.mpi.is_root():
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(projections, projection_path.open('wb'))
        return projections

    def project_generated_mc(
        self,
        summary: FitSummary,
        config: Config,
        preloaded_binned: BinnedDatasets | None,
    ) -> GeneratedMCProjections:
        if preloaded_binned is None:
            binned = self._load_binned_datasets(config)
        else:
            binned = preloaded_binned
        ds_data_binned = binned.data
        ds_accmc_binned = binned.accmc
        waves = summary.waves
        shape_model = build_model(waves)
        scaled_models = {
            run_period: (
                shape_model
                * ld.Scalar(
                    f'scale {run_period}',
                    ld.parameter(f'scale {run_period}'),
                )
            ).fix('S+0+ real', 100.0)
            for run_period in RUN_PERIODS
        }
        mass = ld.Mass(['kshort1', 'kshort2'])
        genmc_projections: dict[str, dict[str, Histogram]] = {
            str(wave): {
                run_period: Histogram.empty(self.bins, self.limits)
                for run_period in RUN_PERIODS
            }
            for wave in waves
        }
        genmc_projections['total'] = {
            run_period: Histogram.empty(self.bins, self.limits)
            for run_period in RUN_PERIODS
        }
        genmc_counts = {
            run_period: Histogram.empty(self.bins, self.limits)
            for run_period in RUN_PERIODS
        }
        accmc_counts = {
            run_period: Histogram.empty(self.bins, self.limits)
            for run_period in RUN_PERIODS
        }
        n_bootstraps = len(summary.bootstraps[0]) if summary.bootstraps else 0
        bootstrap_totals: dict[str, dict[str, NDArray]] | None = None
        if n_bootstraps > 0:
            bootstrap_totals = {
                label: {
                    run_period: np.zeros((self.bins, n_bootstraps), dtype=float)
                    for run_period in RUN_PERIODS
                }
                for label in ['total'] + [str(wave) for wave in waves]
            }
        nll_cache = {
            run_period: [
                ld.NLL(
                    shape_model
                    * ld.Scalar(
                        f'scale {run_period}', ld.parameter(f'scale {run_period}')
                    ),
                    ds_data_binned[run_period][ibin],
                    ds_accmc_binned[run_period][ibin],
                ).fix('S+0+ real', 100.0)
                for ibin in range(self.bins)
            ]
            for run_period in RUN_PERIODS
        }
        bootstrap_nlls_cache = {
            run_period: [
                [
                    ld.NLL(
                        shape_model
                        * ld.Scalar(
                            f'scale {run_period}', ld.parameter(f'scale {run_period}')
                        ),
                        ds_data_binned[run_period][ibin].bootstrap(iboot),
                        ds_accmc_binned[run_period][ibin],
                    ).fix('S+0+ real', 100.0)
                    for iboot in range(n_bootstraps)
                ]
                for ibin in range(self.bins)
            ]
            for run_period in RUN_PERIODS
        }
        for run_period in RUN_PERIODS:
            for ibin in range(self.bins):
                acc_bin = ds_accmc_binned[run_period][ibin]
                accmc_counts[run_period].counts[ibin] = acc_bin.n_events_weighted
                accmc_counts[run_period].errors[ibin] = np.sqrt(
                    np.sum(np.power(acc_bin.weights, 2))
                )
        genmc_dataset = config.datasets[self.genmc]
        for run_period in RUN_PERIODS:
            genmc_path = genmc_dataset.generate_path(run_period, config)
            if not genmc_path.exists():
                logger.warning(f'Missing generated MC file: {genmc_path}')
                continue
            logger.info(f'Streaming generated MC for {run_period}: {genmc_path}')
            pq_file = pq.ParquetFile(genmc_path)
            total_rows = pq_file.metadata.num_rows if pq_file.metadata else None
            row_bar = tqdm(
                total=total_rows,
                desc=f'{run_period} generated MC rows',
                unit='rows',
                disable=not ld.mpi.is_root(),
            )
            chunk_idx = 0
            for batch in pq_file.iter_batches(batch_size=self.genmc_batch_size):
                if batch.num_rows == 0:
                    continue
                df = pl.from_arrow(batch)
                assert isinstance(df, pl.DataFrame)
                if df.is_empty():
                    continue
                ds_chunk = ld.io.from_polars(
                    df,
                    p4s=['beam', 'proton', 'kshort1', 'kshort2'],
                    aux=['pol_magnitude', 'pol_angle'],
                )
                chunk_binned = ds_chunk.bin_by(mass, self.bins, self.limits)
                filled_bins = 0
                for ibin, ds_genmc_bin in enumerate(chunk_binned):  # ty:ignore[invalid-argument-type]
                    if ds_genmc_bin.n_events == 0:
                        continue
                    filled_bins += 1
                    nll_ibin = nll_cache[run_period][ibin]
                    bootstrap_nlls_ibin = bootstrap_nlls_cache[run_period][ibin]
                    result_ibin = summary.fits[ibin]
                    result_params = self._align_parameters(
                        result_ibin.x,
                        result_ibin.parameter_names,
                        nll_ibin.free_parameters,
                    )
                    scale_name = f'scale {run_period}'
                    genmc_evaluator = scaled_models[run_period].load(ds_genmc_bin)
                    genmc_counts[run_period].counts[ibin] += (
                        ds_genmc_bin.n_events_weighted
                    )
                    genmc_err = np.sqrt(np.sum(np.power(ds_genmc_bin.weights, 2)))
                    genmc_counts[run_period].errors[ibin] = np.sqrt(
                        np.power(genmc_counts[run_period].errors[ibin], 2)
                        + np.power(genmc_err, 2)
                    )
                    genmc_projections['total'][run_period].counts[ibin] += np.sum(
                        nll_ibin.project(result_params, mc_evaluator=genmc_evaluator)
                    )
                    if bootstrap_totals is not None:
                        for iboot, bootstrap in enumerate(summary.bootstraps[ibin]):
                            bootstrap_params = self._align_parameters(
                                bootstrap.x,
                                bootstrap.parameter_names,
                                nll_ibin.free_parameters,
                            )
                            bootstrap_totals['total'][run_period][ibin, iboot] += (
                                np.sum(
                                    nll_ibin.project(
                                        bootstrap_params,
                                        mc_evaluator=genmc_evaluator,
                                    )
                                )
                            )
                    for wave in waves:
                        genmc_projections[str(wave)][run_period].counts[ibin] += np.sum(
                            nll_ibin.project_with(
                                result_params,
                                wave.amplitude_names + [scale_name],
                                mc_evaluator=genmc_evaluator,
                            )
                        )
                        if bootstrap_totals is not None:
                            for iboot, bootstrap in enumerate(summary.bootstraps[ibin]):
                                bootstrap_params = self._align_parameters(
                                    bootstrap.x,
                                    bootstrap.parameter_names,
                                    nll_ibin.free_parameters,
                                )
                                bootstrap_totals[str(wave)][run_period][
                                    ibin, iboot
                                ] += np.sum(
                                    bootstrap_nlls_ibin[iboot].project_with(
                                        bootstrap_params,
                                        wave.amplitude_names + [scale_name],
                                        mc_evaluator=genmc_evaluator,
                                    )
                                )
                chunk_idx += 1
                row_bar.update(batch.num_rows)
                row_bar.set_postfix(
                    {
                        'chunks': chunk_idx,
                        'rows': df.height,
                        'bins': filled_bins,
                    }
                )
            row_bar.close()
        if bootstrap_totals is not None:
            for label, per_run in bootstrap_totals.items():
                for run_period, arr in per_run.items():
                    for ibin in range(self.bins):
                        if n_bootstraps > 1:
                            err = np.std(arr[ibin], ddof=1)
                        else:
                            err = 0.0
                        genmc_projections[label][run_period].errors[ibin] = err
        return GeneratedMCProjections(
            genmc_projections=genmc_projections,
            genmc_counts=genmc_counts,
            accmc_counts=accmc_counts,
        )

    def _build_fit_result(
        self, summary: FitSummary, projections: GeneratedMCProjections
    ) -> FitResult:
        return FitResult(
            data=summary.data,
            accmc=summary.accmc,
            genmc=self.genmc,
            waves=summary.waves,
            fits=summary.fits,
            bootstraps=summary.bootstraps,
            bin_edges=summary.bin_edges,
            data_projection=summary.data_projection,
            projections=summary.projections,
            genmc_projections=projections.genmc_projections,
            genmc_counts=projections.genmc_counts,
            accmc_counts=projections.accmc_counts,
        )

    def _render_plots(self, name: str, config: Config, fit_result: FitResult) -> None:
        plot_dir = PLOTS_PATH
        for step in config.datasets[self.data].steps:
            plot_dir /= step
        plot_dir.mkdir(parents=True, exist_ok=True)
        fit_plot = plot_dir / f'{name}.png'
        cross_section_plot = plot_dir / f'{name}_xsec.png'
        if not fit_plot.exists() and ld.mpi.is_root():
            logger.info(f'Plotting fit: {name}')
            self.plot_fit(fit_plot, fit_result)
        else:
            logger.info(f'Skipping fit plot: {name}')
        if ld.mpi.is_root():
            logger.info(f'Plotting cross-sections: {name}')
            self.plot_cross_sections(cross_section_plot, fit_result)

    def _fit_directory(self, name: str, datasets: dict[str, Dataset]) -> Path:
        fit_dir = FITS_PATH
        for step in datasets[self.data].steps:
            fit_dir /= step
        return fit_dir / name

    def _summary_path(self, name: str, datasets: dict[str, Dataset]) -> Path:
        return self._fit_directory(name, datasets) / 'fit_summary.pkl'

    def _genmc_projection_path(self, name: str, datasets: dict[str, Dataset]) -> Path:
        return self._fit_directory(name, datasets) / 'genmc_projections.pkl'

    def _load_binned_datasets(
        self,
        config: Config,
    ) -> BinnedDatasets:
        mass = ld.Mass(['kshort1', 'kshort2'])
        ds_data = {
            run_period: config.datasets[self.data].dataset(
                run_period,
                config,
                p4s=['beam', 'proton', 'kshort1', 'kshort2'],
                aux=['pol_magnitude', 'pol_angle'],
            )
            for run_period in RUN_PERIODS
        }
        ds_accmc = {
            run_period: config.datasets[self.accmc].dataset(
                run_period,
                config,
                p4s=['beam', 'proton', 'kshort1', 'kshort2'],
                aux=['pol_magnitude', 'pol_angle'],
            )
            for run_period in RUN_PERIODS
        }
        ds_data_binned = {
            run_period: ds_data_rp.bin_by(mass, self.bins, self.limits)
            for run_period, ds_data_rp in ds_data.items()
        }
        ds_accmc_binned = {
            run_period: ds_accmc_rp.bin_by(mass, self.bins, self.limits)
            for run_period, ds_accmc_rp in ds_accmc.items()
        }
        return BinnedDatasets(data=ds_data_binned, accmc=ds_accmc_binned)

    def plot_fit(self, plot_path: Path, fit_result: FitResult) -> None:
        ureg = UnitRegistry()
        unit = ureg.parse_units('GeV/c^2')
        width = float(np.diff(fit_result.bin_edges)[0])
        bin_width = Quantity(width, unit).to_compact()
        bin_width = Quantity(round(bin_width.m, 2), bin_width.u)
        plt.style.use('gluex_ksks_paper_analysis.style')
        _, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        s_waves = [wave for wave in fit_result.waves if wave.j == AngularMomentum.S]
        d_waves = [wave for wave in fit_result.waves if wave.j == AngularMomentum.D]
        ax_waves = [s_waves, d_waves]
        for i in [0, 1]:
            ax[i].stairs(
                fit_result.data_projection_total.counts,
                fit_result.data_projection_total.edges,
                color=BLACK,
                label='Data',
            )
            ax[i].errorbar(
                fit_result.data_projection_total.centers,
                fit_result.data_projection_total.counts,
                yerr=fit_result.data_projection_total.errors,
                ls='none',
                color=BLACK,
            )
            # NOTE: we omit the totals since they are equal to the data by construction
            # Additionally, there is no information gained from the error bars on totals
            # since they come from the variance in the sum of weights in bootstrapped datasets
            for wave in ax_waves[i]:
                color = (
                    RED if wave.r is None or wave.r == Reflectivity.Positive else BLUE
                )
                ax[i].errorbar(
                    fit_result.projections_total[str(wave)].centers,
                    fit_result.projections_total[str(wave)].counts,
                    yerr=fit_result.projections_total[str(wave)].errors,
                    linestyle='none',
                    marker='o',
                    markersize=2,
                    color=color,
                    label=wave.latex,
                )
            ax[i].legend()
            ax[i].set_ylim(0)
            ax[i].set_xlabel(f'Invariant Mass of $K_S^0K_S^0$ (${unit:~L}$)')
        ax[0].set_ylabel(f'Counts / ${bin_width:~L}$')
        logger.info(f'Saving plot: {plot_path}')
        plt.savefig(plot_path)

    def plot_cross_sections(self, plot_path: Path, fit_result: FitResult) -> None:
        ureg = UnitRegistry()
        mass_unit = ureg.parse_units('GeV/c^2')
        cross_section_unit = ureg.parse_units('nb/GeV')
        pb_to_nb = Quantity(1.0, 'pb').to('nb').m
        plt.style.use('gluex_ksks_paper_analysis.style')
        s_waves = [wave for wave in fit_result.waves if wave.j == AngularMomentum.S]
        d_waves = [wave for wave in fit_result.waves if wave.j == AngularMomentum.D]
        ax_waves = [s_waves, d_waves]
        flux_data = PSFluxData()
        total_luminosity = np.sum(flux_data.tagged_luminosity.counts)
        total_luminosity_error = np.sqrt(
            np.sum(np.power(flux_data.tagged_luminosity.errors, 2))
        )
        luminosity_by_run_period = {
            run_period: (
                np.sum(flux_data.tagged_luminosity_by_run_period[run_period].counts),
                np.sqrt(
                    np.sum(
                        np.power(
                            flux_data.tagged_luminosity_by_run_period[
                                run_period
                            ].errors,
                            2,
                        )
                    )
                ),
            )
            for run_period in RUN_PERIODS
        }
        width = float(np.diff(fit_result.bin_edges)[0])
        corrected_yields: dict[str, Histogram] = {}
        corrected_yields_by_run: dict[str, dict[str, Histogram]] = {}
        for wave in fit_result.waves + ['total']:
            wave_key = str(wave) if isinstance(wave, Wave) else wave
            per_run = {}
            for run_period in RUN_PERIODS:
                per_run[run_period] = (
                    fit_result.data_projection[run_period]
                    * fit_result.genmc_projections[wave_key][run_period]
                    / fit_result.projections['total'][run_period]
                )
            corrected_yields_by_run[wave_key] = per_run
            corrected_yields[wave_key] = Histogram.sum(list(per_run.values()))
        cross_sections_total: dict[str, Histogram] = {}
        cross_sections_by_run: dict[str, dict[str, Histogram]] = {
            run_period: {} for run_period in RUN_PERIODS
        }
        for wave_key, hist in corrected_yields.items():
            cross_sections_total[wave_key] = (
                hist.scalar_div(total_luminosity).scalar_div(width).scalar_mul(pb_to_nb)
            )
            for run_period in RUN_PERIODS:
                lumi, _ = luminosity_by_run_period[run_period]
                cross_sections_by_run[run_period][wave_key] = (
                    corrected_yields_by_run[wave_key][run_period]
                    .scalar_div(lumi)
                    .scalar_div(width)
                    .scalar_mul(pb_to_nb)
                )

        def render_cross_section_axes(
            ax: NDArray, cross_sections: dict[str, Histogram]
        ):
            for i in [0, 1]:
                ax[i].stairs(
                    cross_sections['total'].counts,
                    cross_sections['total'].edges,
                    color=BLACK,
                    label='Total',
                )
                ax[i].errorbar(
                    cross_sections['total'].centers,
                    cross_sections['total'].counts,
                    yerr=cross_sections['total'].errors,
                    ls='none',
                    color=BLACK,
                )
                for wave in ax_waves[i]:
                    color = (
                        RED
                        if wave.r is None or wave.r == Reflectivity.Positive
                        else BLUE
                    )
                    ax[i].errorbar(
                        cross_sections[str(wave)].centers,
                        cross_sections[str(wave)].counts,
                        yerr=cross_sections[str(wave)].errors,
                        linestyle='none',
                        marker='o',
                        markersize=2,
                        color=color,
                        label=wave.latex,
                    )
                ax[i].legend()
                ax[i].set_ylim(0)
                ax[i].set_xlabel(f'Invariant Mass of $K_S^0K_S^0$ (${mass_unit:~L}$)')

        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        render_cross_section_axes(ax, cross_sections_total)
        ax[0].set_ylabel(f'${cross_section_unit:~L}$')
        logger.info(f'Saving plot: {plot_path}')
        fig.savefig(plot_path)
        plt.close(fig)

        acceptance_hist = fit_result.acceptance
        acceptance_counts = np.nan_to_num(
            acceptance_hist.counts, nan=0.0, posinf=0.0, neginf=0.0
        )
        acceptance_max = np.max(acceptance_counts) if acceptance_counts.size else 0.0
        acceptance_ylim = acceptance_max * 1.1 if acceptance_max > 0 else 1.0
        acceptance_plot = plot_path.with_name(f'{plot_path.stem}_acceptance.png')
        fig_acc, ax_acc = plt.subplots(nrows=1, ncols=2, sharey=True)
        render_cross_section_axes(ax_acc, cross_sections_total)
        acceptance_handles = []
        acceptance_axes = []
        for axi in ax_acc:
            twin = axi.twinx()
            handle = twin.stairs(
                acceptance_counts,
                acceptance_hist.edges,
                color=RED,
                linewidth=1.5,
                label='Acceptance',
            )
            twin.set_ylabel('Acceptance', color=RED)
            twin.tick_params(axis='y', colors=RED)
            twin.set_ylim(0, acceptance_ylim)
            acceptance_handles.append(handle)
            acceptance_axes.append(twin)
        ax_acc[0].set_ylabel(f'${cross_section_unit:~L}$')
        for twin, handle in zip(acceptance_axes, acceptance_handles):
            twin.legend(handles=[handle], loc='upper right')
        logger.info(f'Saving plot: {acceptance_plot}')
        fig_acc.savefig(acceptance_plot)
        plt.close(fig_acc)

        for run_period in RUN_PERIODS:
            run_plot = plot_path.with_name(f'{plot_path.stem}_{run_period}.png')
            fig_run, ax_run = plt.subplots(nrows=1, ncols=2, sharey=True)
            render_cross_section_axes(ax_run, cross_sections_by_run[run_period])
            ax_run[0].set_ylabel(f'${cross_section_unit:~L}$')
            logger.info(f'Saving plot: {run_plot}')
            fig_run.savefig(run_plot)
            plt.close(fig_run)

        self._write_luminosity_table(
            luminosity_by_run_period, total_luminosity, total_luminosity_error
        )
        self._write_cross_section_table(
            plot_path,
            cross_sections_by_run,
            cross_sections_total,
            width,
        )

    @staticmethod
    def _integrated_cross_section(
        hist: Histogram, bin_width: float
    ) -> tuple[float, float]:
        total = float(np.sum(hist.counts * bin_width))
        error = float(
            np.sqrt(
                np.sum(
                    np.power(hist.errors * bin_width, 2),
                )
            )
        )
        return total, error

    @staticmethod
    def _write_latex_table(
        path: Path,
        headers: tuple[str, str, str],
        rows: list[tuple[str, float, float]],
        value_fmt: str,
    ) -> None:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            f.write('\\begin{tabular}{lrr}\n')
            f.write(f'{headers[0]} & {headers[1]} & {headers[2]} \\\\\n')
            f.write('\\hline\n')
            for label, value, error in rows:
                f.write(f'{label} & {value:{value_fmt}} & {error:{value_fmt}} \\\\\n')
            f.write('\\end{tabular}\n')

    def _write_luminosity_table(
        self,
        luminosity_by_run_period: dict[str, tuple[float, float]],
        total_luminosity: float,
        total_luminosity_error: float,
    ) -> None:
        rows = [
            (run_period, lumi, err)
            for run_period, (lumi, err) in luminosity_by_run_period.items()
        ]
        rows.append(('All', total_luminosity, total_luminosity_error))
        headers = (
            'Run Period',
            r'Luminosity (pb$^{-1}$)',
            r'Uncertainty (pb$^{-1}$)',
        )
        path = REPORTS_PATH / 'luminosity_summary.tex'
        self._write_latex_table(path, headers, rows, value_fmt='.3e')

    def _write_cross_section_table(
        self,
        plot_path: Path,
        cross_sections_by_run: dict[str, dict[str, Histogram]],
        cross_sections_total: dict[str, Histogram],
        bin_width: float,
    ) -> None:
        rows: list[tuple[str, float, float]] = []
        for run_period in RUN_PERIODS:
            total, error = self._integrated_cross_section(
                cross_sections_by_run[run_period]['total'], bin_width
            )
            rows.append((run_period, total, error))
        total_all, error_all = self._integrated_cross_section(
            cross_sections_total['total'], bin_width
        )
        rows.append(('All', total_all, error_all))
        headers = (
            'Run Period',
            r'$\sigma_{\mathrm{tot}}$ (nb)',
            'Uncertainty (nb)',
        )
        path = REPORTS_PATH / f'{plot_path.stem}_totals.tex'
        self._write_latex_table(path, headers, rows, value_fmt='.3f')


@dataclass(frozen=True)
class Plots:
    dataset: str
    items: list[str]

    def make_plots(self, config: Config) -> None:
        ds = config.datasets[self.dataset]
        plots1d = {k: v for k, v in config.plots.items() if isinstance(v, Plot1D)}

        def build_path(steps: list[str], source: str, plot_name: str) -> Path:
            path = PLOTS_PATH
            for step in steps:
                path /= step
            return path / f'{source}_{plot_name}.png'

        def render_for(dataset: Dataset) -> None:
            dataframes = [
                dataset.lazyframe(run_period, config) for run_period in RUN_PERIODS
            ]
            for name, plot in config.plots.items():
                plot_path = build_path(dataset.steps, dataset.source, name)
                if plot_path.exists():
                    logger.info(f'Plot already exists: {plot_path}, skipping')
                    continue
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(plot, Plot1D):
                    plot.write(dataframes, plot_path)
                else:
                    plot.write(dataframes, plot_path, plots1d)

        for i in range(len(ds.steps) + 1):
            ds_i = Dataset(ds.source, ds.steps[:i])
            render_for(ds_i)


@dataclass(frozen=True)
class Variation:
    plots: list[Plots]
    fits: list[str]

    def process_variation(self, config: Config) -> None:
        for plot_set in self.plots:
            plot_set.make_plots(config)
        for fit in self.fits:
            config.fits[fit].fit(fit, config)


@dataclass(frozen=True)
class Config:
    cuts: dict[str, Cut]
    weights: dict[str, Weight]
    datasets: dict[str, Dataset]
    plots: dict[str, Plot1D | Plot2D]
    fits: dict[str, Fit]
    variations: dict[str, Variation]

    def run(self, variation: str) -> None:
        self.variations[variation].process_variation(self)


class ConfigError(Exception):
    pass


def _check_semantics(raw: dict) -> None:  # noqa: PLR0912
    variations = raw['variations']
    cuts = raw['cuts']
    cut_names = set(cuts)
    weights = raw['weights']
    weight_names = set(weights)
    datasets = raw['datasets']
    dataset_names = set(datasets)
    plots = raw['plots']
    plot_names = set(plots)
    fits = raw['fits']
    fit_names = set(fits)

    known_steps = cut_names | weight_names
    for dataset_name, dataset in datasets.items():
        for step in dataset.get('steps', []):
            if step not in known_steps:
                msg = f"Unknown step '{step}' in dataset '{dataset_name}'"
                raise ConfigError(msg)

    plot1d_names = {name for name, spec in plots.items() if 'variable' in spec}
    for plot_name, plot in plots.items():
        if 'x' in plot or 'y' in plot:
            x, y = plot.get('x'), plot.get('y')
            if x not in plot1d_names:
                msg = (
                    f"2D plot '{plot_name}' must reference 1D plots; missing x = '{x}'"
                )
                raise ConfigError(msg)
            if y not in plot1d_names:
                msg = (
                    f"2D plot '{plot_name}' must reference 1D plots; missing y = '{y}'"
                )
                raise ConfigError(msg)

    for fit_name, fit in fits.items():
        for ref in ('data', 'accmc'):
            if fit[ref] not in dataset_names:
                msg = f"Unknown dataset '{fit[ref]}' in fit '{fit_name}'"
                raise ConfigError(msg)

    for variation_name, variation in variations.items():
        for psel in variation.get('plots', []):
            ds = psel['dataset']
            if ds not in dataset_names:
                msg = f"Unknown dataset '{ds}' in variation '{variation_name}'"
                raise ConfigError(msg)
            for item in psel.get('items', []):
                if item not in plot_names:
                    msg = f"Unknown plot '{item}' in variation '{variation_name}'"
                    raise ConfigError(msg)
        for fref in variation.get('fits', []):
            if fref not in fit_names:
                msg = f"Unknown fit '{fref}' in variation '{variation_name}'"
                raise ConfigError(msg)

    for cut_name, cut in cuts.items():
        for rule in cut.get('rules', []):
            m = _RULE_REGEX.match(rule)
            if not m:
                msg = f"Invalid rule '{rule}' in cut '{cut_name}' (use '<var> <op> <number>')"
                raise ConfigError(msg)
            if m.group('var') not in _VARIABLES:
                msg = f"Unknown variable '{m.group('var')}' in cut '{cut_name}'"
                raise ConfigError(msg)


def load_config(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text())

    errs = sorted(
        Draft202012Validator(_JSON_SCHEMA).iter_errors(raw), key=lambda e: e.path
    )
    if errs:
        msg = '\n'.join(f'- {"/".join(map(str, e.path))}: {e.message}' for e in errs)
        msg = f'Schema validation failed:\n{msg}'
        raise ConfigError(msg)

    _check_semantics(raw)

    cuts = {
        k: Cut(rules=list(v.get('rules', [])), cache=bool(v.get('cache', False)))
        for k, v in raw['cuts'].items()
    }

    weights = {
        k: Weight(
            weight_type=v['weight_type'],
            sig=v.get('sig'),
            bkg=v.get('bkg'),
            cache=bool(v.get('cache', False)),
        )
        for k, v in raw['weights'].items()
    }

    datasets = {
        k: Dataset(source=v['source'], steps=list(v.get('steps', [])))
        for k, v in raw['datasets'].items()
    }
    if 'genmc' not in datasets:
        datasets['genmc'] = Dataset(source='genmc', steps=[])

    plots: dict[str, Plot1D | Plot2D] = {}
    for name, spec in raw['plots'].items():
        if 'variable' in spec:
            lo, hi = spec['limits']
            plots[name] = Plot1D(
                variable=spec['variable'],
                label=spec['label'],
                bins=int(spec['bins']),
                limits=(float(lo), float(hi)),
                units=spec.get('units', '') or '',
            )
        else:
            plots[name] = Plot2D(x=spec['x'], y=spec['y'])

    fits = {}
    for k, v in raw['fits'].items():
        lim = v.get('limits', [1.0, 2.0])
        fits[k] = Fit(
            waves=list(v['waves']),
            data=v['data'],
            accmc=v['accmc'],
            bins=int(v['bins']),
            limits=(float(lim[0]), float(lim[1])),
            n_iterations=int(v.get('n_iterations', 20)),
            n_bootstraps=int(v.get('n_bootstraps', 100)),
        )

    variations = {}
    for k, v in raw['variations'].items():
        plot_objs = [
            Plots(dataset=p['dataset'], items=list(p.get('items', [])))
            for p in v.get('plots', [])
        ]
        variations[k] = Variation(plots=plot_objs, fits=list(v.get('fits', [])))

    return Config(
        cuts=cuts,
        weights=weights,
        datasets=datasets,
        plots=plots,
        fits=fits,
        variations=variations,
    )
