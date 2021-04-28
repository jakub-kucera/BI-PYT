import numpy as np
from collections import Counter
from xxhash import xxh3_64
from typing import Tuple, Any, List

from config import *


class PixelFitnessCalculator:
    def __init__(self, symbols: List[np.ndarray]):
        self.symbols = symbols
        self.symbol_count = len(self.symbols)

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
            hashed_array = xxh3_64(self.symbols[i][y_indexes, x_indexes]).hexdigest()
            hashes += [hashed_array]

        print(Counter(hashes)) if DEBUG_PRINT else None
        inverted_fitness = 1
        total_occurrence_counter = 0
        for count in Counter(hashes).values():
            inverted_fitness *= count
            total_occurrence_counter += count

        if total_occurrence_counter != self.symbol_count:
            raise ValueError(f"Invalid number of symbols when calculating \
            fitness. Is {total_occurrence_counter}, but should be {self.symbol_count}")

        fitness = self.invert_fitness(inverted_fitness)
        print(fitness) if DEBUG_PRINT else None
        return fitness
