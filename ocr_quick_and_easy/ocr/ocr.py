import math
import time
from typing import Type

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
        painter = SyncPainter(symbols=self.array_symbols)
        painter.init_painter()

        print(f"Fitness: {chosen_pixels}")
        print(self.fitness_calculator.calculate_fitness(chosen_pixels))

        while True:
            painter.change_chosen_pixels(chosen_pixels)

    def calculate(self, algorithm_type: Type[OCRAlgorithm], painter_type: Type[Painter]):
        """Runs calculation using provided method for increasing number of chosen pixels."""

        # ocr_algorithm: OCRAlgorithm = algorithm_type(fitness_calculator=self.fitness_calculator,
        #                                              plotter=self.plotter)

        # random.seed(0)
        # np.random.seed(RANDOM_SEED)

        painter = painter_type(symbols=self.array_symbols)
        painter.init_painter()
        #
        best_combination = None

        # todo
        #  only run for this number
        #  genetic: increase generation size, number of generations inf?,
        #  smaller mutation. Only on pixels?
        #  make smaller changes using crossover

        pixel_count = self.calculate_pixel_count(len(self.array_symbols))

        if pixel_count > len(self.overlapping_indexes):
            raise Exception("Provided symbols are too similar\
                                to differentiate between them")

        # for pixel_count in range(6, 7):

        # for pixel_count in range(pixel_count, len(self.overlapping_indexes)):
        # for pixel_count in range(1, len(self.overlapping_indexes)):
        for attempts_pixel_count in range(1):
            pixel_count = 7
                # random.seed(RANDOM_SEED)
            print(f"Starting testing {pixel_count} pixels.")
            start = time.time()

            ocr_algorithm: OCRAlgorithm = algorithm_type(pixel_count=pixel_count,
                                                         indexes_array=self.overlapping_indexes.copy(),
                                                         fitness_calculator=self.fitness_calculator,
                                                         plotter=self.plotter, painter=painter)

            found_solution, best_combination = ocr_algorithm.calculate_for_k_pixels()
            total = time.time() - start
            print(f"Elapsed time = {total}")

            if found_solution:
                print("Found solutions")
                break

        if best_combination is not None:
            print("Best solution: ")
            print("best_combination")
            print(best_combination)
            print("Symbols:")
            # for symbol in self.array_symbols:
            #     print(symbol[best_combination.T[0], best_combination.T[1]])
