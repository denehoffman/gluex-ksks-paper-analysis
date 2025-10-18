import argparse
from pathlib import Path

from gluex_ksks_paper_analysis.config import load_config
from gluex_ksks_paper_analysis.databases import build_caches
from gluex_ksks_paper_analysis.utilities import download_data


def cli():
    parser = argparse.ArgumentParser(description='Run GlueX KsKs paper analysis.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.toml'),
        help='Path to configuration file (default: config.toml)',
    )
    parser.add_argument(
        '--variation',
        type=str,
        default='default',
        help='Variation name (default: default)',
    )
    args = parser.parse_args()
    download_data()
    build_caches()
    cfg = load_config(args.config)
    cfg.run(args.variation)
