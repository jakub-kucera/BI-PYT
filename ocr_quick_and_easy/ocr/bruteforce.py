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

    def calculate_for_k_pixels(self, pixel_count: int, y_index_array: List[int], x_index_array: List[int])\
            -> Tuple[bool, Tuple[Tuple[Any, ...], Tuple[Any, ...]]]:
        """Tries all possible solutions for a given number of chosen pixels."""

        # creates iterators which generate all possible combinations of a given length
        y_comb = combinations(y_index_array, pixel_count)
        x_comb = combinations(x_index_array, pixel_count)

        best_fitness = NULL_FITNESS
        best_combination: Tuple[Tuple[Any, ...], Tuple[Any, ...]] = ((), ())
        comb_count = 0

        # goes through all the elements
        for y, x in zip(y_comb, x_comb):
            comb_count += 1

            # calculates fitness for the current combinations of chosen pixels
            fitness = self.fitness_calculator.calculate_fitness(y, x)

            # update current best fitness, if higher
            if best_fitness == NULL_FITNESS or fitness > best_fitness:
                best_fitness = fitness
                best_combination = (y, x)

                # stop when desired final solution has been found
                if fitness == MAX_FITNESS:
                    break

        print(f"Comb count: {comb_count}") if DEBUG_PRINT else None
        print(f"best_fitness: {best_fitness}") if DEBUG_PRINT else None
        self.plotter.add_record(best_fitness)
        return best_fitness == MAX_FITNESS, best_combination
