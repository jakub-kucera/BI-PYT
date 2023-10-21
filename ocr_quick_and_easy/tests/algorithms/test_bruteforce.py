import pytest
import numpy as np

from ocr.algorithms.bruteforce import OCRBruteForce
from ocr.gui.dummy_painter import DummyPainter
from ocr.gui.sync_painter import SyncPainter
from ocr.utils.fitness import PixelFitnessCalculator
from ocr.utils.plotter import Plotter
from tests.test_datasets.symbols_dataset import ENGLISH_SYMBOLS_NP, get_all_indexes

TEST_DEFAULT_DATASET = "tests/test_datasets/dataset/"
TEST_DEFAULT_DATASET_ADVANCED = "tests/test_datasets/written_hiragana_dataset/"

###########################################################################################


@pytest.mark.parametrize('pixel_count,\
                            indexes_array, \
                            fitness_calculator, \
                            plotter, \
                            painter, \
                            seed, \
                            population_size, \
                            generations_count,', [
    (
        5,
        np.array(get_all_indexes(16, 16)),
        PixelFitnessCalculator,
        Plotter,
        DummyPainter,
        1234567890,
        1000,
        1000,
    ),
    (
        10,
        np.array(get_all_indexes(10, 10)),
        PixelFitnessCalculator,
        Plotter,
        SyncPainter,
        None,
        5,
        8,
    ),
    # TEST_DEFAULT_DATASET_ADVANCED,
])
def test_bruteforce_create(pixel_count,
                           indexes_array,
                           fitness_calculator,
                           plotter,
                           painter,
                           seed,
                           population_size,
                           generations_count,
                           mocker):
    OCRBruteForce(pixel_count=pixel_count,
                  indexes_array=indexes_array,
                  fitness_calculator=fitness_calculator,
                  plotter=plotter,
                  painter=painter,
                  seed=seed,
                  population_size=population_size,
                  generations_count=generations_count)

###########################################################################################


@pytest.mark.parametrize('pixel_count, \
                                indexes_array, \
                                fitness_calculator, \
                                plotter, \
                                painter, \
                                seed, \
                                population_size, \
                                generations_count,\
                                found_combination, \
                                combination,', [
    (
        2,
        np.array(get_all_indexes(16, 16)),
        PixelFitnessCalculator,
        Plotter,
        DummyPainter,
        1234567890,
        1000,
        1000,
        False,
        np.array([[2, 1], [8, 1]]),
    ),
    (
        2,
        np.array(get_all_indexes(10, 10)),
        PixelFitnessCalculator,
        Plotter,
        SyncPainter,
        None,
        5,
        8,
        False,
        np.array([[2, 1], [8, 1]]),
    ),
    # TEST_DEFAULT_DATASET_ADVANCED,
])
def test_bruteforce_calculate_for_k_pixels(pixel_count,
                                           indexes_array,
                                           fitness_calculator,
                                           plotter,
                                           painter,
                                           seed,
                                           population_size,
                                           generations_count,
                                           found_combination,
                                           combination,
                                           mocker):

    # mocker.patch('ocr.utils.plotter.Plotter.write_counter')
    # mocker.patch('itertools.combinations')

    # combinations = itertools.combinations(indexes_array, pixel_count)
    #
    # # with mock.patch('itertools.combinations', )
    # #
    # with mock.patch('itertools.combinations', autospec=True, spec_set=True) as mocked_combinations:
    #     mocked_combinations.return_value = combinations
    #     mocked_combinations.return_value = (SOLUTION_DATASET_5_PIXELS, SOLUTION_DATASET_6_PIXELS)
    # with mock.patch('ocr.utils.fitness.PixelFitnessCalculator.calculate_fitness', autospec=True, spec_set=True) as mocked_calculator:

    # itertools.combinations = MagicMock(return_value=itertools.combinations(indexes_array, pixel_count))

    algorithm = OCRBruteForce(pixel_count=pixel_count,
                              indexes_array=indexes_array,
                              fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                              plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                              painter=painter(ENGLISH_SYMBOLS_NP),
                              seed=seed,
                              population_size=population_size,
                              generations_count=generations_count)

    # with mock.patch('ocr.algorithms.bruteforce.fitness_calculator.calculate_fitness', autospec=True, spec_set=True) as mocked_calculator:
    # with mock.patch('ocr.utils.fitness.PixelFitnessCalculator', autospec=True, spec_set=True) as mocked_calculator:
    #     mocked_calculator.return_value.calculate_fitness.return_value = 0
    f, c = algorithm.calculate_for_k_pixels()
    assert f == found_combination
    assert np.array_equal(c, combination)

# (
#     pixel_count,
#     indexes_array,
#     fitness_calculator,
#     plotter,
#     painter,
#     seed,
#     population_size,
#     generations_count,
# ),
