import time
from typing import List, Tuple, Any, Callable, Type
import numpy as np

from config import *
from ocr.algorithm import OCRAlgorithm
from ocr.fitness import PixelFitnessCalculator
from ocr.image_loader import ImageLoader
from ocr.plotter import Plotter


class OCR:
    """Class that calculates a chooses the minimal amount of pixels needed to distinguish """

    def __init__(self,
                 plotter: Plotter = Plotter("None"),
                 dataset_directory: str = "Datasets/dataset/"):
        self.array_symbols = ImageLoader.load_symbols(dataset_directory=dataset_directory)
        self.fitness_calculator = PixelFitnessCalculator(self.array_symbols)

        self.plotter = plotter

        self.symbol_overlap = ImageLoader.create_overlap_distinct(symbols=self.array_symbols)
        self.overlapping_indexes = ImageLoader.get_filtered_matrix_indexes(overlap=self.symbol_overlap)
        self.total_overlap_pixel_count = len(self.overlapping_indexes)

    def get_shuffled_index_arrays(self):
        """Splits array of indexes into two arrays, each with only one of those indexes"""

        y_indexes, x_indexes = np.hsplit(self.overlapping_indexes, 2)
        np.random.default_rng(RANDOM_SEED).shuffle(y_indexes)
        np.random.default_rng(RANDOM_SEED).shuffle(x_indexes)

        return y_indexes, x_indexes

    def calculate(self, algorithm_type: Type[OCRAlgorithm]):
        """Runs calculation using provided method for increasing number of chosen pixels."""

        ocr_algorithm: OCRAlgorithm = algorithm_type(fitness_calculator=self.fitness_calculator,
                                                     plotter=self.plotter)

        y_indexes, x_indexes = self.get_shuffled_index_arrays()

        print(y_indexes) if DEBUG_PRINT else None
        print(x_indexes) if DEBUG_PRINT else None
        print(y_indexes.shape) if DEBUG_PRINT else None
        print(x_indexes.shape) if DEBUG_PRINT else None

        print(y_indexes[0]) if DEBUG_PRINT else None
        best_combination = None

        # for pixel_count in range(1, len(self.overlapping_indexes)):
        for pixel_count in range(1, 3):
            print(f"Starting testing {pixel_count} pixels.")
            start = time.time()
            found_solution, best_combination = ocr_algorithm.calculate_for_k_pixels(pixel_count=pixel_count,
                                                                                    y_index_array=y_indexes,
                                                                                    x_index_array=x_indexes)
            total = time.time() - start
            print(f"Elapsed time = {total}")

            if found_solution:
                print("Found solutions")
                break

        print("Best solution: ")
        print(best_combination)
