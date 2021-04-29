from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any, List

from config import RANDOM_SEED
from ocr.fitness import PixelFitnessCalculator
from ocr.plotter import Plotter


class OCRAlgorithm(ABC):
    def __init__(self, fitness_calculator: PixelFitnessCalculator, plotter: Plotter):
        self.fitness_calculator = fitness_calculator
        self.plotter = plotter

    @abstractmethod
    def calculate_for_k_pixels(self, pixel_count: int, indexes_array: np.ndarray) -> Tuple[bool, np.ndarray]:
        pass

    @staticmethod
    def shuffle_index_array(indexes_array, shuffle_seed: int = RANDOM_SEED):
        np.random.default_rng(shuffle_seed).shuffle(indexes_array)
