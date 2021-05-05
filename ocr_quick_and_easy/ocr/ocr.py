import math
import random
import time
from typing import Type, Optional

import numpy as np

from ocr.algorithms.algorithm import OCRAlgorithm
from ocr.gui.painter import Painter
from ocr.gui.sync_painter import SyncPainter
from ocr.utils.fitness import PixelFitnessCalculator
from ocr.utils.image_loader import ImageLoader
from ocr.utils.plotter import Plotter


class OCR:
    """Class that calculates a chooses the minimal amount of pixels needed to distinguish """

    def __init__(self,
                 plotter: Plotter = Plotter("None"),
                 dataset_directory: str = "Datasets/dataset/"):
        self.array_symbols = ImageLoader.load_symbols(dataset_directory=dataset_directory)
        self.symbol_overlap = ImageLoader.create_overlap_distinct(symbols=self.array_symbols)

        self.overlapping_indexes = ImageLoader.get_filtered_matrix_indexes(overlap=self.symbol_overlap)
        self.total_overlap_pixel_count = len(self.overlapping_indexes)

        self.plotter = plotter
        self.fitness_calculator = PixelFitnessCalculator(self.array_symbols)

    @staticmethod
    def calculate_pixel_count(symbol_count: int):
        """Calculates number of pixels that are needed to differentiate between given number of symbols."""
        return math.ceil(math.log2(symbol_count))

    def paint_only_combinations(self, chosen_pixels: np.ndarray):
        """Only show passed pixel combination using gui"""
        painter = SyncPainter(symbols=self.array_symbols)
        painter.init_painter()

        print(f"Fitness: {chosen_pixels}")
        print(self.fitness_calculator.calculate_fitness(chosen_pixels))

        while True:
            painter.change_chosen_pixels(chosen_pixels)

    def calculate(self, algorithm_type: Type[OCRAlgorithm],
                  painter_type: Type[Painter],
                  population_size: int,
                  generations_count: int,
                  trials_count: int,
                  seed: int
                  ):
        """Runs calculation using provided method for increasing number of chosen pixels."""

        # ocr_algorithm: OCRAlgorithm = algorithm_type(fitness_calculator=self.fitness_calculator,
        #                                              plotter=self.plotter)

        random.seed(seed)
        np.random.seed(seed)

        painter = painter_type(symbols=self.array_symbols)
        painter.init_painter()


        pixel_count = self.calculate_pixel_count(len(self.array_symbols))

        if pixel_count > len(self.overlapping_indexes):
            raise Exception("Provided symbols are too similar\
                                to differentiate between them")

        best_combination = None
        found_solution = False
        for current_pixel_count in range(pixel_count, len(self.overlapping_indexes)):
            for attempts_pixel_count in range(1, trials_count+1):
                print(f"Starting testing {current_pixel_count} pixels, trial #{attempts_pixel_count}")
                start = time.time()

                ocr_algorithm: OCRAlgorithm = algorithm_type(pixel_count=current_pixel_count,
                                                             indexes_array=self.overlapping_indexes.copy(),
                                                             fitness_calculator=self.fitness_calculator,
                                                             plotter=self.plotter, painter=painter,
                                                             seed=seed, population_size=population_size,
                                                             generations_count=generations_count)

                found_solution, best_combination = ocr_algorithm.calculate_for_k_pixels()
                total = time.time() - start
                print(f"Elapsed time = {total}")

                if best_combination is not None:
                    print("best_combination")
                    print(best_combination)

                if found_solution:
                    print("FOUND SOLUTION")
                    break

            if found_solution:
                break
