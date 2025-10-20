import argparse
from pathlib import Path

from gluex_ksks_paper_analysis.config import load_config
from gluex_ksks_paper_analysis.environment import mkdirs, ANALYSIS_PATH
from gluex_ksks_paper_analysis.downloads import download_data
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format='<green>{time}</green> <level>{level}</level> {message}',
)
logger.add(
    ANALYSIS_PATH / 'analysis.log',
    colorize=False,
    format='{time} {level} {message}',
)


@logger.catch
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
    mkdirs()
    download_data()
    cfg = load_config(args.config)
    cfg.run(args.variation)
