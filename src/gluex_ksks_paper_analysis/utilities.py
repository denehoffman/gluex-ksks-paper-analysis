from __future__ import annotations

import argparse
import getpass
import socket
from pathlib import Path, UnsupportedOperation
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import paramiko
import polars as pl
import uproot
from pint import Quantity, UnitRegistry

from gluex_ksks_paper_analysis.environment import DATASET_PATH

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


def connect_to_remote(
    username: str,
    hostname: str = 'ernest.phys.cmu.edu',
    port: int = 22,
) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname,
            port=port,
            username=username,
            allow_agent=True,
            look_for_keys=True,
            timeout=15,
        )
    except (
        paramiko.AuthenticationException,
        FileNotFoundError,
        paramiko.SSHException,
        OSError,
    ):
        password = getpass.getpass(
            f'Please enter the password for {username}@{hostname}'
        )
        client.connect(
            hostname,
            port=port,
            username=username,
            password=password,
            allow_agent=False,
            look_for_keys=False,
            timeout=15,
        )
    return client


def hardlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.hardlink_to(src)
    except (UnsupportedOperation, OSError):
        src.copy(dest)


def get_files(
    remote_paths: list[Path],
    local_dir: Path,
    user: str,
    hostname: str = 'ernest.phys.cmu.edu',
    port: int = 22,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    if socket.getfqdn().endswith('.phys.cmu.edu'):
        for src in remote_paths:
            dest = local_dir / src.name
            hardlink_or_copy(src, dest)
        return

    client = connect_to_remote(user, hostname, port)
    sftp = client.open_sftp()
    try:
        for src in remote_paths:
            dest = local_dir / src.name
            sftp.get(str(src), str(dest))
    finally:
        sftp.close()
        client.close()


def download_data() -> None:
    remote_dir = Path('/raid3/nhoffman/ksks-data')
    remote_paths = [
        remote_dir / f'{name}.parquet'
        for name in ['data', 'sigmc', 'bkgmc']
        if not (DATASET_PATH / f'{name}.parquet').exists()
    ]
    if remote_paths:
        print(
            'Some required data files were missing and need to be downloaded from ernest.phys.cmu.edu'
        )
        username = input('Please enter your username: ')
        get_files(remote_paths, DATASET_PATH, username)
