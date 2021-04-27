import time
from typing import List, Tuple, Any
from xxhash import xxh3_64
import numpy as np
from itertools import combinations
from collections import Counter

from config import *
from ocr.image_loader import ImageLoader
from ocr.plotter import Plotter


class OCR:
    """Class that calculates a chooses the minimal amount of pixels needed to distinguish """

    def __init__(self, plotter: Plotter = Plotter(), dataset_directory: str = "Datasets/dataset/"):
        # self.img_loader = ImageLoader()
        # load_symbols()
        # array_symbols = load_symbols("Datasets/written_hiragana_dataset/")
        self.array_symbols = ImageLoader.load_symbols(dataset_directory=dataset_directory)
        self.symbol_count = len(self.array_symbols)
        # todo clean values
        self.symbol_overlap = ImageLoader.create_overlap(symbols=self.array_symbols)
        self.overlapping_indexes = ImageLoader.get_filtered_matrix_indexes(overlap=self.symbol_overlap)

        # self.total_pixel_count = self.overlapping_indexes.shape[0] * self.overlapping_indexes.shape[1]
        self.total_overlap_pixel_count = len(self.overlapping_indexes)
        self.plotter = plotter

    @staticmethod
    def invert_fitness(inverted_fitness: int) -> int:
        """"Invert fitness fot it to be in correct format."""
        return -(inverted_fitness - 1)

    def calculate_fitness(self, y_indexes: Tuple[Any, ...], x_indexes: Tuple[Any, ...]) -> int:
        """Calculates fitness for a given indexes of chosen pixels"""

        if len(y_indexes) != len(x_indexes):
            raise ValueError(f"y_indexes({y_indexes}) != x_indexes({x_indexes})")

        hashes = []
        for i in range(self.symbol_count):
            h = xxh3_64(self.array_symbols[i][y_indexes, x_indexes]).hexdigest()
            hashes += [h]

        print(Counter(hashes)) if DEBUG_PRINT else None
        inverted_fitness = 1
        total_occurrence_counter = 0
        for x in Counter(hashes).values():
            inverted_fitness *= x
            total_occurrence_counter += x

        if total_occurrence_counter != self.symbol_count:
            raise ValueError(f"Invalid number of symbols when calculating \
            fitness. Is {total_occurrence_counter}, but should be {self.symbol_count}")

        fitness = self.invert_fitness(inverted_fitness)
        print(fitness) if DEBUG_PRINT else None
        return fitness

    def bruteforce_k_pixels(self, pixel_count: int, y_index_array: np.ndarray, x_index_array: np.ndarray) -> Tuple[bool, Tuple[Tuple[Any, ...], Tuple[Any, ...]]]:
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
            fitness = self.calculate_fitness(y, x)
            # break
            # self.plotter.add_record(fitness)

            # update current best fitness, if higher
            if best_fitness == NULL_FITNESS or fitness > best_fitness:
                best_fitness = fitness
                best_combination = (y, x)

                # stop when wanted solution has been found
                if fitness == MAX_FITNESS:  # or highest possible
                    break
            # self.plotter.add_record(best_fitness)

        # self.plotter.add_record(best_fitness)

        print(f"Comb count: {comb_count}") if DEBUG_PRINT else None
        print(f"best_fitness: {best_fitness}") if DEBUG_PRINT else None
        # self.plotter.add_record(best_fitness)
        self.plotter.add_record(best_fitness)
        return best_fitness == MAX_FITNESS, best_combination

    def bruteforce(self):
        """Tries all possible solutions for increasing number of chosen pixels."""

        # split array of indexes into two arrays based on what dimension they indexed.
        y_indexes, x_indexes = np.hsplit(self.overlapping_indexes, 2)

        #  randomly shuffles elements in both arrays. (both are shuffled the same way)
        np.random.default_rng(RANDOM_SEED).shuffle(y_indexes)
        np.random.default_rng(RANDOM_SEED).shuffle(x_indexes)

        print(y_indexes) if DEBUG_PRINT else None
        print(x_indexes) if DEBUG_PRINT else None
        print(y_indexes.shape) if DEBUG_PRINT else None
        print(x_indexes.shape) if DEBUG_PRINT else None
        # print(shuffled_array)

        print(y_indexes[0]) if DEBUG_PRINT else None

        # self.bruteforce_k_pixels(pixel_count=2, y_index_array=y_indexes, x_index_array=x_indexes)
        # for p in range(1, len(self.overlapping_indexes)):
        for p in range(1, 4):
            print(f"Starting testing {p} pixels.")
            start = time.time()
            found_solution, best_combination = self.bruteforce_k_pixels(pixel_count=p, y_index_array=y_indexes,
                                                                        x_index_array=x_indexes)
            total = time.time() - start
            print(f"Elapsed time = {total}")
            if found_solution:
                print("Found solutions")
                break

