from __future__ import annotations

import getpass
import socket
from pathlib import Path, UnsupportedOperation
from typing import TYPE_CHECKING

import paramiko
from tqdm import tqdm

from gluex_ksks_paper_analysis.databases import build_caches
from gluex_ksks_paper_analysis.environment import (
    CCDB_CONNECTION,
    DATABASE_PATH,
    DATASET_PATH,
    POL_HIST_PATHS,
    POLARIZED_RUN_NUMBERS_PATH,
    PSFLUX_DATA_PATH,
    RCDB_CONNECTION,
    POLARIZATION_DATA_PATH,
    RUN_PERIODS,
)

if TYPE_CHECKING:
    pass


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


def _progress_callback(bar: tqdm):
    def callback(transferred: int, total: int):
        if total and bar.total != total:
            bar.reset(total=total)
        bar.update(transferred - bar.n)

    return callback


def get_files(
    download_map: list[tuple[list[Path] | Path, Path]],
    user: str,
    hostname: str = 'ernest.phys.cmu.edu',
    port: int = 22,
) -> None:
    for srcs, local_path in download_map:
        if isinstance(srcs, list):
            local_path.mkdir(parents=True, exist_ok=True)
    if socket.getfqdn().endswith('.phys.cmu.edu'):
        for srcs, local_path in download_map:
            if isinstance(srcs, list):
                for src in srcs:
                    dest = local_path / src.name
                    hardlink_or_copy(src, dest)
            else:
                hardlink_or_copy(srcs, local_path)
        return

    client = connect_to_remote(user, hostname, port)
    sftp = client.open_sftp()
    try:
        for srcs, local_path in download_map:
            if isinstance(srcs, list):
                for src in srcs:
                    dest = local_path / src.name
                    with tqdm(
                        unit='B', unit_scale=True, unit_divisor=1024, desc=src.name
                    ) as bar:
                        sftp.get(str(src), str(dest), callback=_progress_callback(bar))
            else:
                with tqdm(
                    unit='B', unit_scale=True, unit_divisor=1024, desc=srcs.name
                ) as bar:
                    sftp.get(
                        str(srcs), str(local_path), callback=_progress_callback(bar)
                    )

    finally:
        sftp.close()
        client.close()


def download_data() -> None:
    remote_dir = Path('/raid3/nhoffman/ksks-data')
    remote_data_paths = [
        remote_dir / f'{name}_{run_period}.parquet'
        for name in ['data', 'sigmc', 'bkgmc', 'genmc']
        for run_period in RUN_PERIODS
        if not (DATASET_PATH / f'{name}_{run_period}.parquet').exists()
    ]
    remote_pol_hist_paths = [
        remote_dir / Path('pol_hists') / f'{rp}.root'
        for rp in POL_HIST_PATHS
        if not POL_HIST_PATHS[rp].exists()
    ]
    skip_rcdb = (
        PSFLUX_DATA_PATH.exists()
        and POLARIZED_RUN_NUMBERS_PATH.exists()
        and POLARIZATION_DATA_PATH.exists()
    ) or RCDB_CONNECTION.exists()
    skip_ccdb = (PSFLUX_DATA_PATH.exists()) or CCDB_CONNECTION.exists()
    if remote_data_paths or remote_pol_hist_paths or (not skip_rcdb) or (not skip_ccdb):
        print(
            'Some required data files were missing and need to be downloaded from ernest.phys.cmu.edu'
        )
        username = input('Please enter your username: ')
        download_map = []
        if remote_pol_hist_paths:
            download_map.append((remote_pol_hist_paths, DATABASE_PATH))
        if not skip_rcdb:
            download_map.append((remote_dir / 'rcdb.sqlite', RCDB_CONNECTION))
        if not skip_ccdb:
            download_map.append((remote_dir / 'ccdb.sqlite', CCDB_CONNECTION))
        if remote_data_paths:
            download_map.append((remote_data_paths, DATASET_PATH))
        get_files(download_map, username)
        build_caches()
