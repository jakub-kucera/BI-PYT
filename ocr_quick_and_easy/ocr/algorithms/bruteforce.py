from itertools import combinations

from config import *
from ocr.algorithms.algorithm import OCRAlgorithm
from ocr.gui.painter import Painter
from ocr.utils.fitness import PixelFitnessCalculator
from ocr.gui.sync_painter import SyncPainter
from ocr.utils.plotter import Plotter


class OCRBruteForce(OCRAlgorithm):
    """Class for calculation pixel combination using bruteforce"""
    def __init__(self, pixel_count: int, indexes_array: np.ndarray, fitness_calculator: PixelFitnessCalculator,
                 plotter: Plotter, painter: Painter):
        super().__init__(pixel_count, indexes_array, fitness_calculator, plotter, painter)

    def calculate_for_k_pixels(self) -> Tuple[bool, np.ndarray]:
        """Tries all possible solutions for a given number of chosen pixels."""

        # creates iterator which generate all possible combinations of a given length
        index_combinations = combinations(self.indexes_array, self.pixel_count)

        best_fitness = NULL_FITNESS
        best_combination: np.ndarray = np.empty(shape=(0, 2), dtype=np.uint8)
        comb_count = 0

        # goes through all the elements
        for index_combination in (np.array(comb) for comb in index_combinations):
            comb_count += 1

            # calculates fitness for the current combinations of chosen pixels
            fitness = self.fitness_calculator.calculate_fitness(index_combination)
            self.painter.change_chosen_pixels(index_combination)

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
