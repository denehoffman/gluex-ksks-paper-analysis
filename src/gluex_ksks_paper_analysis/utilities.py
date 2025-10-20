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
        self.counts = counts
        self.edges = edges
        self.errors = errors if errors is not None else np.sqrt(np.abs(counts))

    @property
    def bins(self) -> int:
        return len(self.edges) - 1

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
