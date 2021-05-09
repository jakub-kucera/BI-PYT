"""This file contains default values and global constants."""

from typing import Final, Tuple, Optional

import numpy as np

#########################################################

DEFAULT_SHOW_PLOT = True
DEFAULT_SHOW_GUI = False

TRIALS_PER_PIXEL_COUNT: Final[int] = 2
POPULATION_SIZE: Final[int] = 200
MAX_GENERATIONS: Final[int] = 200
TOURNAMENT_SIZE_PERCENTAGE: Final[float] = 0.02
MUTATION_INCREASE_STEP: Final[float] = 0.02
MUTATE_PIXELS_MAX_PERCENTAGE: Final[float] = 0.4
CROSSOVER_PERCENTAGE: Final[float] = 0.9
SELECT_OLD_PERCENTAGE: Final[float] = 0.95

IMAGE_THRESHOLD_VALUE: Final[int] = 127
PLOT_IMG_DPI: Final[int] = 200
PLOT_THRESHOLD: Final[int] = -500

#########################################################

MAX_FITNESS: Final[int] = 0
NULL_FITNESS: Final[int] = 1
RANDOM_SEED: Final[Optional[int]] = None

DEFAULT_DATASET: Final[str] = "Datasets/dataset/"
DEFAULT_DATASET_SMALL_20: Final[str] = "Datasets/dataset_small_20/"
DEFAULT_DATASET_ADVANCED: Final[str] = "Datasets/written_hiragana_dataset/"

WHITE_COLOR: Final[Tuple[int, int, int]] = (255, 255, 255)
BLACK_COLOR: Final[Tuple[int, int, int]] = (0, 0, 0)
GREY_COLOR: Final[Tuple[int, int, int]] = (170, 170, 170)
PIXEL_OUTLINE_COLOR: Final[Tuple[int, int, int]] = (255, 0, 0)
PIXEL_OUTLINE_WIDTH: Final[int] = 1
COLORS: np.ndarray = np.array([WHITE_COLOR, BLACK_COLOR])
SIZE_MULTIPLIER: Final[int] = 10
DEFAULT_FPS: Final[int] = 30

OUTPUT_PLOTS_FILE: Final[str] = "output_plots/"
PLOTTER_COUNTER_FILE: Final[str] = OUTPUT_PLOTS_FILE + "plotter_counter.json"
OUTPUT_PLOT_IMG_TYPE: Final[str] = '.png'

DEBUG_PRINT: Final[bool] = False

RUN_DESCRIPTION: Final[str] = "Program which will find a combination of " \
                              "pixels of minimal lenght which are needed " \
                              "to differentiate between providated symbols."

SOLUTION_DATASET_6_PIXELS = [[14,  8], [3, 10], [8,  4], [14, 12], [6,  5], [1, 12]]

SOLUTION_DATASET_5_PIXELS = [[13,  6], [5,  5], [11, 12], [14, 11], [1,  3]]
# SOLUTION_DATASET_5_PIXELS = [[9, 12], [5,  5], [14, 11], [1,  3], [14,  6]]
