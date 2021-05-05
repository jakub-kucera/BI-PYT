from typing import Final, Tuple, Optional

import numpy as np

#########################################################

DEFAULT_SHOW_PLOT = False
DEFAULT_SHOW_GUI = False

TRIALS_PER_PIXEL_COUNT: Final[int] = 1
POPULATION_SIZE: Final[int] = 100
MAX_GENERATIONS: Final[int] = 1000
TOURNAMENT_SIZE_PERCENTAGE: Final[float] = 0.05
MUTATION_INCREASE_STEP: Final[float] = 0.01
MUTATE_PIXELS_MAX_PERCENTAGE: Final[float] = 0.8

IMAGE_THRESHOLD_VALUE: Final[int] = 127

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

"""Starting testing 8943 pixels.
Found best fitness
For 5 pixels, mutation swap count: 288
Elapsed time = 0.5646367073059082
Found solutions
Best solution: 
best_combination
[[13  6]
 [ 5  5]
 [11 12]
 [14 11]
 [ 1  3]]
Symbols:
[ True  True False False  True]
[False False False  True  True]
[ True False  True  True False]
[False  True  True False  True]
[False False  True  True  True]
[False  True False False  True]
[ True False  True  True  True]
[False  True False False False]
[ True  True False  True False]
[ True False False  True False]
[False False False False False]
[ True False  True False False]
[False False  True False False]
[ True  True False  True  True]
[ True False False False False]
[ True  True False False False]
[ True  True  True  True  True]
[False  True False  True False]
[False False False False  True]
[ True  True  True  True False]
[False  True False  True  True]
[ True False  True False  True]
[False  True  True  True  True]
[ True False False  True  True]
[False  True  True False False]
[False False False  True False]
/home/kucerj56/Documents/PYT/kucerj56/ocr_quick_and_easy/ocr/plotter.py:40: UserWarning: \
Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
"""
