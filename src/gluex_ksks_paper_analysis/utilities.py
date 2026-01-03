from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import uproot
from pint import Quantity, UnitRegistry


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray


class Histogram:
    def __init__(
        self, counts: ArrayLike, edges: ArrayLike, errors: ArrayLike | None = None
    ):
        self.counts: NDArray = np.asarray(counts)
        self.edges: NDArray = np.asarray(edges)
        self.errors: NDArray = (
            np.asarray(errors) if errors is not None else np.sqrt(np.abs(self.counts))
        )

    @property
    def bins(self) -> int:
        return len(self.edges) - 1

    @property
    def limits(self) -> tuple[float, float]:
        return self.edges[0], self.edges[-1]

    @property
    def bin_width(self) -> float:
        return float(np.diff(self.edges)[0])

    @property
    def centers(self) -> NDArray:
        return (self.edges[1:] + self.edges[:-1]) / 2.0

    def get_index(self, value: float) -> int | None:
        ibin = np.digitize(value, self.edges)
        if ibin == 0 or ibin == len(self.edges):
            return None
        return int(ibin) - 1

    @staticmethod
    def empty(bins: int, limits: tuple[float, float]) -> Histogram:
        edges = np.histogram_bin_edges([], bins=bins, range=limits)
        counts = np.zeros(bins)
        errors = np.zeros(bins)
        return Histogram(counts, edges, errors)

    @staticmethod
    def empty_like(h: Histogram) -> Histogram:
        edges = np.histogram_bin_edges([], bins=h.bins, range=h.limits)
        counts = np.zeros(h.bins)
        errors = np.zeros(h.bins)
        return Histogram(counts, edges, errors)

    def plot(self, *, variable_label: str, unit_label: str, **kwargs) -> Figure:
        plt.style.use('gluex_ksks_paper_analysis.style')
        fig, ax = plt.subplots()
        ax.stairs(self.counts, self.edges, **kwargs)
        ureg = UnitRegistry()
        unit = ureg.parse_units(unit_label)
        xlabel = f'{variable_label} (${unit:~L}$)'
        bin_width = Quantity(self.bin_width, unit).to_compact()
        bin_width = Quantity(round(bin_width.m, 2), bin_width.u)
        ylabel = f'Counts / ${bin_width:~L}$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

    def __add__(self, other: Histogram) -> Histogram:
        if not np.array_equal(self.edges, other.edges):
            msg = 'Histogram edges must match to add'
            raise ValueError(msg)
        out = Histogram.empty_like(self)
        out.counts = self.counts + other.counts
        out.errors = np.sqrt(np.power(self.errors, 2) + np.power(other.errors, 2))
        return out

    def __iadd__(self, other: Histogram):
        out = self + other
        self.counts = out.counts
        self.edges = out.edges
        self.errors = out.errors
        return self

    def __mul__(self, other: Histogram) -> Histogram:
        if not np.array_equal(self.edges, other.edges):
            msg = 'Histogram edges must match to multiply'
            raise ValueError(msg)
        out = Histogram.empty_like(self)
        out.counts = self.counts * other.counts
        out.errors = out.counts * np.sqrt(
            np.power(self.errors / self.counts, 2)
            + np.power(other.errors / other.counts, 2)
        )
        return out

    def __imul__(self, other: Histogram):
        out = self * other
        self.counts = out.counts
        self.edges = out.edges
        self.errors = out.errors
        return self

    def __truediv__(self, other: Histogram) -> Histogram:
        if not np.array_equal(self.edges, other.edges):
            msg = 'Histogram edges must match to divide'
            raise ValueError(msg)
        out = Histogram.empty_like(self)
        out.counts = self.counts / other.counts
        out.errors = out.counts * np.sqrt(
            np.power(self.errors / self.counts, 2)
            + np.power(other.errors / other.counts, 2)
        )
        return out

    def __itruediv__(self, other: Histogram):
        out = self / other
        self.counts = out.counts
        self.edges = out.edges
        self.errors = out.errors
        return self

    def scalar_add(self, value: float, error: float = 0.0) -> Histogram:
        out = Histogram.empty_like(self)
        out.counts = self.counts + value
        out.errors = np.sqrt(np.power(self.errors, 2) + np.power(error, 2))
        return out

    def scalar_mul(self, value: float, error: float = 0.0) -> Histogram:
        out = Histogram.empty_like(self)
        out.counts = self.counts * value
        out.errors = out.counts * np.sqrt(
            np.power(self.errors / self.counts, 2) + np.power(error / value, 2)
        )
        return out

    def scalar_div(self, value: float, error: float = 0.0) -> Histogram:
        out = Histogram.empty_like(self)
        out.counts = self.counts / value
        out.errors = out.counts * np.sqrt(
            np.power(self.errors / self.counts, 2) + np.power(error / value, 2)
        )
        return out

    @staticmethod
    def sum(histograms: list[Histogram]) -> Histogram:
        assert len(histograms) > 0
        out = Histogram.empty_like(histograms[0])
        for histogram in histograms:
            out += histogram
        return out


def merge_cli() -> None:
    parser = argparse.ArgumentParser(description='Merge multiple Parquet/ROOT files.')
    parser.add_argument(
        'inputs', nargs='+', type=Path, help='Input Parquet/ROOT files.'
    )
    parser.add_argument('output', type=Path, help='Output Parquet file.')
    parser.add_argument(
        '--compression-level',
        type=int,
        default=None,
        help='Optional compression level (min=1, max=22, default=3).',
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=None,
        help='Column names to exclude from the merged output.',
    )
    args = parser.parse_args()
    parquet_inputs: list[Path] = []
    for inp in args.inputs:
        if inp.suffix == '.root':
            parquet_input = inp.with_suffix('.parquet.tmp')
            root_to_dataframe(inp).write_parquet(parquet_input)
            parquet_inputs.append(parquet_input)
        elif inp.suffix == '.parquet':
            parquet_inputs.append(inp)
        else:
            print(f"Unsupported file type '{inp}' will be skipped")
    merge_parquet(
        parquet_inputs,
        args.output,
        compression_level=args.compression_level,
        exclude_columns=args.exclude,
    )
    for inp in parquet_inputs:
        if inp.suffix == '.tmp':
            inp.unlink()


def merge_parquet(
    inputs: list[Path],
    output: Path,
    *,
    compression_level: int | None = None,
    exclude_columns: list[str] | None = None,
) -> None:
    if exclude_columns is None:
        pl.scan_parquet(inputs, low_memory=True).sink_parquet(
            output, compression_level=compression_level, maintain_order=False
        )
    else:
        pl.scan_parquet(inputs, low_memory=True).drop(exclude_columns).sink_parquet(
            output, compression_level=compression_level, maintain_order=False
        )


def root_to_dataframe(input_path: Path, tree: str = 'kin') -> pl.DataFrame:
    tt = uproot.open(f'{input_path}:{tree}')
    root_data = tt.arrays(library='np')
    return pl.from_dict(root_data)
