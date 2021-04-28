from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any, List

from ocr.fitness import PixelFitnessCalculator
from ocr.plotter import Plotter


class OCRAlgorithm(ABC):
    def __init__(self, fitness_calculator: PixelFitnessCalculator, plotter: Plotter):
        self.fitness_calculator = fitness_calculator
        self.plotter = plotter

    @abstractmethod
    def calculate_for_k_pixels(self, pixel_count: int, y_index_array: List[int], x_index_array: List[int])\
            -> Tuple[bool, Tuple[Tuple[Any, ...], Tuple[Any, ...]]]:
        pass
