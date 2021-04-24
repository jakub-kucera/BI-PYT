from typing import List

import numpy as np
from itertools import combinations

from ocr.image_loader import ImageLoader
from ocr.plotter import Plotter

MAX_FITNESS = 10**10  # change

class OCR:
    def __init__(self, dataset_directory: str = "Datasets/dataset/"):
        # self.img_loader = ImageLoader()
        # load_symbols()
        # array_symbols = load_symbols("Datasets/written_hiragana_dataset/")
        self.array_symbols = ImageLoader.load_symbols(dataset_directory=dataset_directory)
        self.symbol_count = len(self.array_symbols)
        # todo clean values
        self.symbol_overlap = ImageLoader.create_overlap(symbols=self.array_symbols)
        self.overlapping_indexes = ImageLoader.get_filtered_matrix_indexes(overlap=self.symbol_overlap)

        # self.total_pixel_count = self.overlapping_indexes.shape[0] * self.overlapping_indexes.shape[1]
        self.total_pixel_count = len(self.overlapping_indexes)
        self.plotter = Plotter()

    def calculate_fitness(self, symbols_chosen_pixes: List[np.ndarray]) -> int:
        # todo
        return 1

    def bruteforce_k_pixels(self, pixel_count: int, y_index_array: np.ndarray, x_index_array: np.ndarray) -> bool:
        # creates iterators which generate all possible combinations of a given length
        y_comb = combinations(y_index_array, pixel_count)
        x_comb = combinations(x_index_array, pixel_count)

        best_fitness = 0
        comb_count = 0

        # goes through all the elements
        for y, x in zip(y_comb, x_comb):
            comb_count += 1
            symbols_chosen_pixels = []
            # gets chosen pixels from all the symbols
            for i in range(self.symbol_count):
                print(f"i: {i}")
                symbols_chosen_pixels += [self.array_symbols[i][y, x]]
            print(symbols_chosen_pixels)
            # calculates fitness for the current combinations of pixels
            fitness = self.calculate_fitness(symbols_chosen_pixels)
            if fitness > best_fitness:
                best_fitness = fitness
            if fitness == MAX_FITNESS:  # or highest possible
                break

        print(f"Comb count: {comb_count}")
        self.plotter.add_record(best_fitness)
        return best_fitness == MAX_FITNESS

    def bruteforce(self):

        # split array of indexes into two arrays based on what dimension they indexed.
        y_indexes, x_indexes = np.hsplit(self.overlapping_indexes, 2)

        #  randomly shuffles elements in both arrays. (both are shuffled the same way)
        np.random.default_rng(42).shuffle(y_indexes)
        np.random.default_rng(42).shuffle(x_indexes)

        print(y_indexes)
        print(x_indexes)
        print(y_indexes.shape)
        print(x_indexes.shape)
        # print(shuffled_array)

        print(y_indexes[0])

        self.bruteforce_k_pixels(pixel_count=2, y_index_array=y_indexes, x_index_array=x_indexes)
        # for p in range(1, self.overlapping_indexes):
        #     if self.bruteforce_k_pixels(p):
        #         pass

