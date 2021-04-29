import numpy as np
from typing import List, Tuple, Any
from itertools import combinations

from config import *
from ocr.algorithm import OCRAlgorithm
from ocr.fitness import PixelFitnessCalculator
from ocr.plotter import Plotter


class OCRBruteForce(OCRAlgorithm):
    def __init__(self, fitness_calculator: PixelFitnessCalculator, plotter: Plotter):
        super().__init__(fitness_calculator, plotter)

    def calculate_for_k_pixels(self, pixel_count: int, indexes_array: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Tries all possible solutions for a given number of chosen pixels."""

        if pixel_count <= 0 or pixel_count >= indexes_array.size:
            raise Exception("Incorrect number of chosen pixels")

        if indexes_array[0].size != 2:
            raise Exception("Index arrays have different length")

        # creates iterators which generate all possible combinations of a given length
        index_combinations = combinations(indexes_array, pixel_count)

        best_fitness = NULL_FITNESS
        best_combination: np.ndarray = np.empty(shape=(0, 2), dtype=np.uint8)
        comb_count = 0

        # goes through all the elements
        for index_combination in (np.array(comb) for comb in index_combinations):
            comb_count += 1

            # calculates fitness for the current combinations of chosen pixels
            fitness = self.fitness_calculator.calculate_fitness(index_combination)

            # update current best fitness, if higher
            if best_fitness == NULL_FITNESS or fitness > best_fitness:
                best_fitness = fitness
                best_combination = index_combination

                # stop when desired final solution has been found
                if fitness == MAX_FITNESS:
                    break

        print(f"Comb count: {comb_count}")
        print(f"best_fitness: {best_fitness}")
        self.plotter.add_record(best_fitness)
        return best_fitness == MAX_FITNESS, best_combination
