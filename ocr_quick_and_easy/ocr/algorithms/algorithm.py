from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from ocr.utils.fitness import PixelFitnessCalculator
from ocr.gui.sync_painter import SymbolPainter
from ocr.utils.plotter import Plotter


class OCRAlgorithm(ABC):
    """Abstract class which defines interfaces for
    algorithms witch can finds the correct combination
    of pixels needed to differentiate between all symbols"""

    def __init__(self, pixel_count: int,
                 indexes_array: np.ndarray,
                 fitness_calculator: PixelFitnessCalculator,
                 plotter: Plotter, painter: SymbolPainter):
        self.pixel_count = pixel_count
        self.indexes_array = indexes_array
        self.fitness_calculator = fitness_calculator
        self.plotter = plotter
        self.painter = painter

        if self.pixel_count <= 0 or self.pixel_count >= self.indexes_array.size:
            raise Exception("Incorrect number of chosen pixels")

        if self.indexes_array[0].size != 2:
            raise Exception("Index arrays have different length")

    @abstractmethod
    def calculate_for_k_pixels(self) -> Tuple[bool, np.ndarray]:
        """Calculates pixel combination"""
        pass

    @staticmethod
    def shuffle_index_array(indexes_array, shuffle_seed: int = None):
        """Shuffles array of indexes"""
        np.random.default_rng(shuffle_seed).shuffle(indexes_array)
