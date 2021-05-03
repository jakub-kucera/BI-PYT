from typing import Final

RANDOM_SEED: Final[int] = 42
MAX_FITNESS: Final[int] = 0
NULL_FITNESS: Final[int] = 1

IMAGE_THRESHOLD_VALUE: Final[int] = 127
DEFAULT_DATASET: Final[str] = "Datasets/dataset/"
DEFAULT_DATASET_SMALL_20: Final[str] = "Datasets/dataset_small_20/"
DEFAULT_DATASET_ADVANCED: Final[str] = "Datasets/written_hiragana_dataset/"

POPULATION_SIZE: Final[int] = 100
MAX_GENERATIONS: Final[int] = 100
TOURNAMENT_SIZE: Final[int] = 2
MUTATION_INCREASE_STEP: Final[float] = 0.05

DEBUG_PRINT: Final[bool] = False

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
