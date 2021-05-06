import pytest
import numpy as np

from config import SOLUTION_DATASET_5_PIXELS, SOLUTION_DATASET_6_PIXELS
from ocr.utils.fitness import PixelFitnessCalculator
from tests.test_datasets.symbols_dataset import ENGLISH_SYMBOLS_NP, get_all_indexes

TEST_DEFAULT_DATASET = "tests/test_datasets/dataset/"
TEST_DEFAULT_DATASET_ADVANCED = "tests/test_datasets/written_hiragana_dataset/"


@pytest.mark.parametrize('symbols', [
    ENGLISH_SYMBOLS_NP,
    # TEST_DEFAULT_DATASET_ADVANCED,
])
def test_fitness_create(symbols):
    fitness_calc = PixelFitnessCalculator(symbols)
    assert fitness_calc.symbol_count == len(symbols)


@pytest.mark.parametrize('in_number, out_fitness', [
    (0, 1),
    (1, 0),
    (20, -19),
    (345, -344),
])
def test_fitness_invert_fitness(in_number, out_fitness):
    assert out_fitness == PixelFitnessCalculator.invert_fitness(in_number)


@pytest.mark.parametrize('symbols, indexes_array, fitness', [
    (ENGLISH_SYMBOLS_NP, np.array([[2, 2], [5, 5]]), -1259),
    (ENGLISH_SYMBOLS_NP, np.array(SOLUTION_DATASET_5_PIXELS), 0),
    (ENGLISH_SYMBOLS_NP, np.array(SOLUTION_DATASET_6_PIXELS), 0),
    (ENGLISH_SYMBOLS_NP, np.array(get_all_indexes(16, 16)), 0),
    (ENGLISH_SYMBOLS_NP, np.array([[1, 8], [9, 2], [1, 1], [13, 12]]), -1295),
])
def test_fitness_calculate_fitness(symbols, indexes_array, fitness):
    fitness_calc = PixelFitnessCalculator(symbols)
    assert fitness_calc.calculate_fitness(indexes_array) == fitness
