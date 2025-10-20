from __future__ import annotations

import json
import operator
import pickle
import re
import tomllib
from dataclasses import dataclass, field
from tqdm.auto import trange
from functools import reduce
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import laddu as ld
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from jsonschema import Draft202012Validator
from numpy.typing import NDArray
from pint import Quantity, Unit, UnitRegistry
from loguru import logger

from gluex_ksks_paper_analysis.databases import CCDBData, get_all_polarized_run_numbers
from gluex_ksks_paper_analysis.environment import (
    BLACK,
    CMAP,
    DATASET_PATH,
    FITS_PATH,
    BLUE,
    RED,
    NORM,
    PLOTS_PATH,
)
from gluex_ksks_paper_analysis.fit import (
    AngularMomentum,
    Reflectivity,
    Wave,
    build_model,
)
from gluex_ksks_paper_analysis.splot import (
    SharedTauExponential2D,
    StepwisePDF2D,
    fit_mixture,
)
from gluex_ksks_paper_analysis.utilities import Histogram
from gluex_ksks_paper_analysis.variables import add_variable

if TYPE_CHECKING:
    from gluex_ksks_paper_analysis.splot import ComponentPDF


_JSON_SCHEMA = json.loads(
    (
        Path(resources.files(__package__)) / 'ksks_analysis_config.schema.json'
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
    'p4_0_E',
    'p4_0_Px',
    'p4_0_Py',
    'p4_0_Pz',
    'p4_1_E',
    'p4_1_Px',
    'p4_1_Py',
    'p4_1_Pz',
    'p4_2_E',
    'p4_2_Px',
    'p4_2_Py',
    'p4_2_Pz',
    'p4_3_E',
    'p4_3_Px',
    'p4_3_Py',
    'p4_3_Pz',
    'p4_4_E',
    'p4_4_Px',
    'p4_4_Py',
    'p4_4_Pz',
    'p4_5_E',
    'p4_5_Px',
    'p4_5_Py',
    'p4_5_Pz',
    'p4_6_E',
    'p4_6_Px',
    'p4_6_Py',
    'p4_6_Pz',
    'p4_7_E',
    'p4_7_Px',
    'p4_7_Py',
    'p4_7_Pz',
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
                ccdb = CCDBData()
                polarized_runs = get_all_polarized_run_numbers()
                return Weight.set_accidental_weights(
                    df, ccdb=ccdb, polarized_runs=polarized_runs, is_mc=is_mc
                )
            case 'splot':
                if self.sig == 'hist':
                    sigmc_data = df_sigmc.select('RFL1', 'RFL2', 'weight').collect()
                    sigmc_t1 = sigmc_data['RFL1'].to_numpy()
                    sigmc_t2 = sigmc_data['RFL2'].to_numpy()
                    sigmc_weights = sigmc_data['weight'].to_numpy()
                    signal = StepwisePDF2D.from_mc(
                        sigmc_t1, sigmc_t2, sigmc_weights, bins=2000
                    )
                elif self.sig == 'exp':
                    # TODO: get the right value
                    signal = SharedTauExponential2D(lda0=18.0)
                else:
                    msg = f'Unsupported sig {self.sig}'
                    raise ConfigError(msg)

                if self.bkg == 'hist':
                    bkgmc_data = df_bkgmc.select('RFL1', 'RFL2', 'weight').collect()
                    bkgmc_t1 = bkgmc_data['RFL1'].to_numpy()
                    bkgmc_t2 = bkgmc_data['RFL2'].to_numpy()
                    bkgmc_weights = bkgmc_data['weight'].to_numpy()
                    background = StepwisePDF2D.from_mc(
                        bkgmc_t1, bkgmc_t2, bkgmc_weights, bins=2000
                    )
                elif self.bkg == 'exp':
                    background = SharedTauExponential2D(lda0=110.0)
                else:
                    msg = f'Unsupported bkg {self.bkg}'
                    raise ConfigError(msg)
                return Weight.set_splot_weights(
                    df, signal=signal, background=background
                )
            case _:
                msg = f'Unsupported weight type {self.weight_type}'
                raise ConfigError(msg)

    @staticmethod
    def set_accidental_weights(
        df: pl.LazyFrame, *, ccdb: CCDBData, polarized_runs: set[int], is_mc: bool
    ) -> pl.LazyFrame:
        return (
            df.filter(pl.col('RunNumber').is_in(polarized_runs))
            .sort(['RunNumber', 'EventNumber', 'ChiSqDOF'])
            .group_by(['RunNumber', 'EventNumber'])
            .first()
            .with_columns(
                pl.struct('RunNumber', 'p4_0_E', 'RF', 'weight')
                .map_elements(
                    lambda s: s['weight']
                    * ccdb.get_accidental_weight(
                        s['RunNumber'], s['p4_0_E'], s['RF'], is_mc=is_mc
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
        # TODO: make plots and safe fit result
        # fit_result.plot_projection(rfls, weights)
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

    def write(self, df: pl.LazyFrame, output_path: Path) -> None:
        logger.info(f'Writing plot: {output_path}')
        plt.style.use('gluex_ksks_paper_analysis.style')
        _, ax = plt.subplots()
        df = add_variable(self.variable, df)
        df = df.select([self.variable, 'weight']).collect()
        variable = df[self.variable].to_numpy()
        weights = df['weight'].to_numpy()
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
        self, df: pl.LazyFrame, output_path: Path, plots1d: dict[str, Plot1D]
    ) -> None:
        logger.info(f'Writing plot: {output_path}')
        hist_plot_x = plots1d[self.x]
        hist_plot_y = plots1d[self.y]
        plt.style.use('gluex_ksks_paper_analysis.style')
        _, ax = plt.subplots()
        df = add_variable(hist_plot_x.variable, df)
        df = add_variable(hist_plot_y.variable, df)
        df = df.select([hist_plot_x.variable, hist_plot_y.variable, 'weight']).collect()
        data_x = df[self.hist_plot_x.variable].to_numpy()
        data_y = df[self.hist_plot_y.variable].to_numpy()
        weights = df['weight'].to_numpy()
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

    def df(self, cuts: dict[str, Cut], weights: dict[str, Weight]) -> pl.LazyFrame:
        logger.debug(f'Fetching dataset: {self.source} ({self.steps})')

        def get_from_steps(steps: list[str]) -> pl.LazyFrame:
            path = DATASET_PATH
            for step in steps:
                path /= step
            path /= self.source
            if path.exists():
                return pl.scan_parquet(path)
            df = get_from_steps(steps[:-1])
            last_step = steps[-1]
            if last_step in cuts and last_step in weights:
                msg = f'Ambiguous step {last_step} (both a cut and a weight)'
                raise ConfigError(msg)
            if cut := cuts.get(last_step):
                df = cut.apply(df)
                if cut.cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.select(_BASE_COLUMNS).sink_parquet(path)
                return df
            if weight := weights.get(last_step):
                ds_sigmc = Dataset('sigmc.parquet', steps[:-1])
                ds_bkgmc = Dataset('bkgmc.parquet', steps[:-1])
                is_mc = self.source != 'data.parquet'
                df = weight.apply(
                    df, ds_sigmc.df(cuts, weights), ds_bkgmc.df(cuts, weights), is_mc
                )
                if weight.cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.select(_BASE_COLUMNS).sink_parquet(path)
                return df
            else:
                msg = f'Failed to find {last_step} in either cuts or weights'
                raise ConfigError(msg)

        return get_from_steps(self.steps)


@dataclass
class FitResult:
    data: str
    accmc: str
    waves: list[Wave]
    fits: list[ld.MinimizationSummary]
    bootstraps: list[list[ld.MinimizationSummary]]
    bin_edges: NDArray
    projections: dict[str, Histogram]

    @property
    def bins(self) -> int:
        return len(self.bin_edges) - 1

    @property
    def limits(self) -> tuple[float, float]:
        return self.bin_edges[0], self.bin_edges[-1]


@dataclass(frozen=True)
class Fit:
    waves: list[str]
    data: str
    accmc: str
    bins: int
    limits: tuple[float, float] = (1.0, 2.0)
    n_iterations: int = 20
    n_bootstraps: int = 100
    required_columns: list[str] = field(
        default_factory=lambda: [
            'p4_0_E',
            'p4_0_Px',
            'p4_0_Py',
            'p4_0_Pz',
            'p4_1_E',
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_2_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_3_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'aux_0_x',
            'aux_0_y',
            'aux_0_z',
            'weight',
        ]
    )

    def fit(
        self,
        name: str,
        datasets: dict[str, Dataset],
        cuts: dict[str, Cut],
        weights: dict[str, Weight],
    ):
        fit_path = FITS_PATH
        for step in datasets[self.data].steps:
            fit_path /= step
        fit_path /= name + '.pkl'

        if not fit_path.exists():
            logger.info(f'Running fit: {name}')
            fit_result = self.fit_waves(datasets, cuts, weights)
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(fit_result, fit_path.open('wb'))
        else:
            logger.info(f'Skipping fit: {name}')
            fit_result = pickle.load(fit_path.open('rb'))

        plot_path = PLOTS_PATH
        for step in datasets[self.data].steps:
            plot_path /= step
        plot_path /= name + '.png'
        if not plot_path.exists():
            logger.info(f'Plotting fit: {name}')
            self.plot_fit(Path(plot_path), fit_result)
        else:
            logger.info(f'Skipping plot: {name}')

    def fit_waves(
        self,
        datasets: dict[str, Dataset],
        cuts: dict[str, Cut],
        weights: dict[str, Weight],
        genmc: str | None = None,
        seed: int = 0,
    ) -> FitResult:
        df_data = datasets[self.data].df(cuts, weights)
        df_data = add_variable('polarization', df_data)
        df_accmc = datasets[self.accmc].df(cuts, weights)
        df_accmc = add_variable('polarization', df_accmc)
        ds_data = ld.Dataset.from_polars(
            df_data.select(self.required_columns).collect(),
            rest_frame_indices=[1, 2, 3],
        )
        ds_accmc = ld.Dataset.from_polars(
            df_accmc.select(self.required_columns).collect(),
            rest_frame_indices=[1, 2, 3],
        )
        waves = [Wave(w) for w in self.waves]
        model = build_model(waves)
        mass = ld.Mass([2, 3])
        ds_data_binned = ds_data.bin_by(mass, self.bins, self.limits)
        ds_accmc_binned = ds_accmc.bin_by(mass, self.bins, self.limits)
        rng = np.random.default_rng(seed)
        best_fits: list[ld.MinimizationSummary] = []
        bootstraps: list[list[ld.MinimizationSummary]] = []
        for ibin in trange(self.bins, desc='Bins'):
            nll_ibin = ld.NLL(model, ds_data_binned[ibin], ds_accmc_binned[ibin])
            best_fit_ibin = None
            for iiter in trange(self.n_iterations, desc='Iterations'):
                p0 = rng.uniform(-100.0, 100.0, len(nll_ibin.parameters))
                res = nll_ibin.minimize(p0, settings={'skip_hessian': True})
                res = nll_ibin.minimize(res.x, method='nelder-mead')
                if best_fit_ibin is None or res.fx < best_fit_ibin.fx:
                    best_fit_ibin = res
            if best_fit_ibin is None:
                msg = f'Failed to fit bin {ibin}!'
                raise RuntimeError(msg)
            best_fits.append(best_fit_ibin)
            bootstraps_ibin: list[ld.MinimizationSummary] = []
            for iboot in trange(self.n_bootstraps, desc='Bootstraps'):
                nll_iboot = ld.NLL(
                    model, ds_data_binned[ibin].bootstrap(iboot), ds_accmc_binned[ibin]
                )
                bootstrap_res = nll_iboot.minimize(
                    best_fit_ibin.x, settings={'skip_hessian': True}
                )
                bootstrap_res = nll_iboot.minimize(
                    bootstrap_res.x, method='nelder-mead'
                )
                bootstraps_ibin.append(bootstrap_res)
            bootstraps.append(bootstraps_ibin)
        bin_edges = np.histogram_bin_edges([], self.bins, range=self.limits)
        logger.info('Projecting fit results...')
        if genmc is not None:
            df_genmc = datasets[genmc].df(cuts, weights)
            df_genmc = add_variable('polarization', df_genmc)
            ds_genmc = ld.Dataset.from_polars(
                df_genmc.select(self.required_columns).collect(),
                rest_frame_indices=[1, 2, 3],
            )
            ds_genmc_binned = ds_genmc.bin_by(mass, self.bins, self.limits)
        else:
            ds_genmc_binned = None
        projections: dict[str, Histogram] = {
            str(wave): Histogram.empty(self.bins, self.limits) for wave in waves
        }
        projections['total'] = Histogram.empty(self.bins, self.limits)
        projections['data'] = Histogram.empty(self.bins, self.limits)
        for ibin in trange(self.bins, desc='Bins'):
            nll_ibin = ld.NLL(model, ds_data_binned[ibin], ds_accmc_binned[ibin])
            ds_genmc_ibin = None if ds_genmc_binned is None else ds_genmc_binned[ibin]
            result_ibin = best_fits[ibin]
            bootstraps_ibin = bootstraps[ibin]
            genmc_evaluator = (
                None if ds_genmc_ibin is None else model.load(ds_genmc_ibin)
            )
            projections['total'].counts[ibin] = np.sum(
                nll_ibin.project(result_ibin.x, mc_evaluator=genmc_evaluator)
            )
            projections['total'].errors[ibin] = np.std(
                [
                    np.sum(nll_ibin.project(bootstrap.x, mc_evaluator=genmc_evaluator))
                    for bootstrap in bootstraps_ibin
                ],
                ddof=1,
            )
            projections['data'].counts[ibin] = ds_data_binned[ibin].n_events_weighted
            projections['data'].errors[ibin] = np.sqrt(
                np.sum(np.power(ds_data_binned[ibin].weights, 2))
            )
            for wave in waves:
                projections[str(wave)].counts[ibin] = np.sum(
                    nll_ibin.project_with(
                        result_ibin.x,
                        wave.amplitude_names,
                        mc_evaluator=genmc_evaluator,
                    )
                )
                projections[str(wave)].errors[ibin] = np.std(
                    [
                        np.sum(
                            nll_ibin.project_with(
                                bootstrap.x,
                                wave.amplitude_names,
                                mc_evaluator=genmc_evaluator,
                            )
                        )
                        for bootstrap in bootstraps_ibin
                    ],
                    ddof=1,
                )
        return FitResult(
            self.data, self.accmc, waves, best_fits, bootstraps, bin_edges, projections
        )

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
                fit_result.projections['data'].counts,
                fit_result.projections['data'].edges,
                color=BLACK,
                label='Data',
            )
            ax[i].errorbar(
                fit_result.projections['data'].centers,
                fit_result.projections['data'].counts,
                yerr=fit_result.projections['data'].errors,
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
                    fit_result.projections[str(wave)].centers,
                    fit_result.projections[str(wave)].counts,
                    yerr=fit_result.projections[str(wave)].errors,
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


@dataclass(frozen=True)
class Plots:
    dataset: str
    items: list[str]

    def make_plots(
        self,
        datasets: dict[str, Dataset],
        cuts: dict[str, Cut],
        weights: dict[str, Weight],
        plots: dict[str, Plot1D | Plot2D],
    ) -> None:
        ds = datasets[self.dataset]
        plots1d = {k: v for k, v in plots.items() if isinstance(v, Plot1D)}

        def build_path(steps: list[str], source: str, plot_name: str) -> Path:
            path = PLOTS_PATH
            for step in steps:
                path /= step
            base = source.replace('.parquet', '_')
            return path / f'{base}{plot_name}.png'

        def render_for(dataset: Dataset) -> None:
            dataframe = dataset.df(cuts, weights)
            for name, plot in plots.items():
                plot_path = build_path(dataset.steps, dataset.source, name)
                if plot_path.exists():
                    logger.info(f'Plot already exists: {plot_path}, skipping')
                    continue
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(plot, Plot1D):
                    plot.write(dataframe, plot_path)
                else:
                    plot.write(dataframe, plot_path, plots1d)

        for i in range(len(ds.steps) + 1):
            ds_i = Dataset(ds.source, ds.steps[:i])
            render_for(ds_i)


@dataclass(frozen=True)
class Variation:
    plots: list[Plots]
    fits: list[str]

    def process_variation(
        self,
        datasets: dict[str, Dataset],
        cuts: dict[str, Cut],
        weights: dict[str, Weight],
        plots: dict[str, Plot1D | Plot2D],
        fits: dict[str, Fit],
    ) -> None:
        for plot_set in self.plots:
            plot_set.make_plots(datasets, cuts, weights, plots)
        for fit in self.fits:
            fits[fit].fit(fit, datasets, cuts, weights)


@dataclass(frozen=True)
class Config:
    cuts: dict[str, Cut]
    weights: dict[str, Weight]
    datasets: dict[str, Dataset]
    plots: dict[str, Plot1D | Plot2D]
    fits: dict[str, Fit]
    variations: dict[str, Variation]

    def run(self, variation: str) -> None:
        self.variations[variation].process_variation(
            self.datasets, self.cuts, self.weights, self.plots, self.fits
        )


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
