from __future__ import annotations

import getpass
import socket
from pathlib import Path, UnsupportedOperation
from typing import TYPE_CHECKING

import paramiko
import requests
from tqdm import tqdm
from loguru import logger

from gluex_ksks_paper_analysis.databases import build_caches
from gluex_ksks_paper_analysis.environment import (
    CCDB_CONNECTION,
    CCDB_PATH,
    DATABASE_PATH,
    DATASET_PATH,
    POL_HIST_PATHS,
    POLARIZED_RUN_NUMBERS_PATH,
    PSFLUX_PATH,
    RCDB_CONNECTION,
    RCDB_PATH,
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
            with tqdm(
                unit='B', unit_scale=True, unit_divisor=1024, desc=src.name
            ) as bar:
                sftp.get(str(src), str(dest), callback=_progress_callback(bar))
    finally:
        sftp.close()
        client.close()


def download_file(url: str, path: Path) -> None:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    }
    logger.info(f'Downloading file from <{url}>')
    with requests.get(url, headers=headers, stream=True, timeout=10) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with (
            path.open('wb') as f,
            tqdm(
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=path.name,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    logger.info(f"File '{path}' downloaded successfully!")


def get_rcdb() -> None:
    if not RCDB_CONNECTION.exists():
        download_file('https://halldweb.jlab.org/dist/rcdb.sqlite', RCDB_CONNECTION)


def get_ccdb() -> None:
    if not CCDB_CONNECTION.exists():
        download_file('https://halldweb.jlab.org/dist/ccdb.sqlite', CCDB_CONNECTION)


def download_databases() -> None:
    skip_rcdb = (
        RCDB_PATH.exists()
        and PSFLUX_PATH.exists()
        and POLARIZED_RUN_NUMBERS_PATH.exists()
    )
    skip_ccdb = CCDB_PATH.exists() and PSFLUX_PATH.exists()
    if not skip_rcdb:
        get_rcdb()
    if not skip_ccdb:
        get_ccdb()


def download_data() -> None:
    remote_dir = Path('/raid3/nhoffman/ksks-data')
    remote_data_paths = [
        remote_dir / f'{name}.parquet'
        for name in ['data', 'sigmc', 'bkgmc']
        if not (DATASET_PATH / f'{name}.parquet').exists()
    ]
    remote_pol_hist_paths = [
        remote_dir / Path('pol_hists') / f'{rp}.root'
        for rp in POL_HIST_PATHS
        if not POL_HIST_PATHS[rp].exists()
    ]
    if remote_data_paths or remote_pol_hist_paths:
        print(
            'Some required data files were missing and need to be downloaded from ernest.phys.cmu.edu'
        )
        username = input('Please enter your username: ')
        if remote_pol_hist_paths:
            get_files(remote_pol_hist_paths, DATABASE_PATH, username)
        download_databases()
        build_caches()
        if remote_data_paths:
            get_files(remote_data_paths, DATASET_PATH, username)
