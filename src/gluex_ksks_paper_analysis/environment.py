import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm, ListedColormap

ANALYSIS_PATH = Path(os.environ.get('ANALYSIS_PATH', 'analysis'))
DATASET_PATH = ANALYSIS_PATH / 'datasets'
PLOTS_PATH = ANALYSIS_PATH / 'plots'
FITS_PATH = ANALYSIS_PATH / 'fits'
REPORTS_PATH = ANALYSIS_PATH / 'reports'
DATABASE_PATH = ANALYSIS_PATH / 'databases'
RCDB_CONNECTION = DATABASE_PATH / 'rcdb.sqlite'
CCDB_CONNECTION = DATABASE_PATH / 'ccdb.sqlite'
POLARIZED_RUN_NUMBERS_PATH = DATABASE_PATH / 'polarized_run_numbers.pickle'
PSFLUX_DATA_PATH = DATABASE_PATH / 'psflux.pickle'
ACCIDENTAL_SCALING_FACTORS_PATH = DATABASE_PATH / 'accidental_scaling_factors.pickle'
POLARIZATION_DATA_PATH = DATABASE_PATH / 'polarization_data.pickle'

RUN_PERIODS = ['s17', 's18', 'f18', 's20']

POL_HIST_PATHS = {
    's17': DATABASE_PATH / 's17.root',
    's18': DATABASE_PATH / 's18.root',
    'f18': DATABASE_PATH / 'f18.root',
    's20': DATABASE_PATH / 's20.root',
}

CMAP = ListedColormap(
    np.vstack(
        [
            plt.get_cmap('GnBu_r', 256)(np.linspace(0.0, 1.0, 127)),  # TODO: check this
            # plt.get_cmap('GnBu', 256)(np.linspace(0.0, 1.0, 127))[::-1],
            np.ones((1, 4)),
            plt.get_cmap('afmhot_r', 256)(np.linspace(0.0, 1.0, 128)),
        ]
    )
)
NORM = CenteredNorm(vcenter=0.0)

BLUE = '#377eb8'
ORANGE = '#ff7f00'
GREEN = '#4daf4a'
RED = '#e41a1c'
PURPLE = '#984ea3'
BROWN = '#a65628'
PINK = '#f781bf'
GRAY = '#999999'
YELLOW = '#ffff33'
LIGHT_BLUE = '#97bfe0'
LIGHT_RED = '#f28c8d'
BLACK = '#000000'
WHITE = '#ffffff'


def mkdirs() -> None:
    ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    FITS_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.mkdir(parents=True, exist_ok=True)
