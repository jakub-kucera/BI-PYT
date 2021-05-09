import pytest
import numpy as np

from config import NULL_FITNESS, SELECT_OLD_PERCENTAGE
from ocr.algorithms.genetic import OCRGenetic
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
])
def test_genetic_create(pixel_count,
                        indexes_array,
                        fitness_calculator,
                        plotter,
                        painter,
                        seed,
                        population_size,
                        generations_count,
                        mocker):
    OCRGenetic(pixel_count=pixel_count,
                  indexes_array=indexes_array,
                  fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                  plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                  painter=painter(ENGLISH_SYMBOLS_NP),
                  seed=seed,
                  population_size=population_size,
                  generations_count=generations_count)

############################################################################################


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
        40,
        8,
    ),
])
def test_genetic_create_first_generation(pixel_count,
                        indexes_array,
                        fitness_calculator,
                        plotter,
                        painter,
                        seed,
                        population_size,
                        generations_count,
                        mocker):
    algortihm = OCRGenetic(pixel_count=pixel_count,
                  indexes_array=indexes_array,
                  fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                  plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                  painter=painter(ENGLISH_SYMBOLS_NP),
                  seed=seed,
                  population_size=population_size,
                  generations_count=generations_count)

    algortihm.create_first_generation()

    assert len(algortihm.population) == population_size
    assert algortihm.population[0].fitness == NULL_FITNESS
    assert len(algortihm.population[0].indexes) == pixel_count


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
])
def test_genetic_recalculate_fitness(pixel_count,
                        indexes_array,
                        fitness_calculator,
                        plotter,
                        painter,
                        seed,
                        population_size,
                        generations_count,
                        mocker):
    algortihm = OCRGenetic(pixel_count=pixel_count,
                  indexes_array=indexes_array,
                  fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                  plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                  painter=painter(ENGLISH_SYMBOLS_NP),
                  seed=seed,
                  population_size=population_size,
                  generations_count=generations_count)

    algortihm.create_first_generation()
    algortihm.recalculate_fitness()

    assert len(algortihm.population) == population_size
    for individual in algortihm.population:
        assert individual.fitness != NULL_FITNESS
        assert individual.fitness <= 0
        assert len(individual.indexes) == pixel_count

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
        100,
        8,
    ),
])
def test_genetic_select(pixel_count,
                        indexes_array,
                        fitness_calculator,
                        plotter,
                        painter,
                        seed,
                        population_size,
                        generations_count,
                        mocker):

    algortihm = OCRGenetic(pixel_count=pixel_count,
                           indexes_array=indexes_array,
                           fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                           plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                           painter=painter(ENGLISH_SYMBOLS_NP),
                           seed=seed,
                           population_size=population_size,
                           generations_count=generations_count)

    algortihm.create_first_generation()
    algortihm.recalculate_fitness()

    fitness_sum_orig = sum(inv.fitness for inv in algortihm.population)

    algortihm.select()
    assert len(algortihm.population) == population_size

    fitness_sum_post = sum(inv.fitness for inv in algortihm.population)
    assert fitness_sum_post > fitness_sum_orig

    for individual in algortihm.population:
        assert individual.fitness <= NULL_FITNESS or individual.fitness <= 0
        assert len(individual.indexes) == pixel_count

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
        40,
        8,
    ),
])
def test_genetic_crossover(pixel_count,
                        indexes_array,
                        fitness_calculator,
                        plotter,
                        painter,
                        seed,
                        population_size,
                        generations_count,
                        mocker):

    algortihm = OCRGenetic(pixel_count=pixel_count,
                           indexes_array=indexes_array,
                           fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                           plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                           painter=painter(ENGLISH_SYMBOLS_NP),
                           seed=seed,
                           population_size=population_size,
                           generations_count=generations_count)

    algortihm.create_first_generation()
    algortihm.recalculate_fitness()

    algortihm.crossover()
    algortihm.recalculate_fitness()

    assert len(algortihm.population) == population_size


    for individual in algortihm.population:
        assert individual.fitness != NULL_FITNESS
        assert individual.fitness <= 0
        assert len(individual.indexes) == pixel_count

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
        40,
        8,
    ),
])
def test_genetic_mutation(pixel_count,
                          indexes_array,
                          fitness_calculator,
                          plotter,
                          painter,
                          seed,
                          population_size,
                          generations_count,
                          mocker):

    algortihm = OCRGenetic(pixel_count=pixel_count,
                           indexes_array=indexes_array,
                           fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                           plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                           painter=painter(ENGLISH_SYMBOLS_NP),
                           seed=seed,
                           population_size=population_size,
                           generations_count=generations_count)

    algortihm.create_first_generation()
    algortihm.recalculate_fitness()
    algortihm.mutation_probability = 1.0

    algortihm.mutate()
    algortihm.recalculate_fitness()

    assert len(algortihm.population) == population_size

    for individual in algortihm.population:
        assert individual.fitness != NULL_FITNESS
        assert individual.fitness <= 0
        assert len(individual.indexes) == pixel_count

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
        10,
    ),
    (
        10,
        np.array(get_all_indexes(10, 10)),
        PixelFitnessCalculator,
        Plotter,
        SyncPainter,
        None,
        100,
        8,
    ),
])
def test_genetic_calculate_for_k_pixels(pixel_count,
                                        indexes_array,
                                        fitness_calculator,
                                        plotter,
                                        painter,
                                        seed,
                                        population_size,
                                        generations_count,
                                        mocker):

    algorthm = OCRGenetic(pixel_count=pixel_count,
                          indexes_array=indexes_array,
                          fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
                          plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
                          painter=painter(ENGLISH_SYMBOLS_NP),
                          seed=seed,
                          population_size=population_size,
                          generations_count=generations_count)

    # algorithm.calculate_for_k_pixels()

    found, comb = algorthm.calculate_for_k_pixels()
    assert type(found) == bool
    assert len(comb) == pixel_count

    assert len(algorthm.population) == population_size

    for individual in algorthm.population:
        assert individual.fitness != NULL_FITNESS
        assert individual.fitness <= 0
        assert len(individual.indexes) == pixel_count


###########################################################################################

# @pytest.mark.parametrize('pixel_count, \
#                                 indexes_array, \
#                                 fitness_calculator, \
#                                 plotter, \
#                                 painter, \
#                                 seed, \
#                                 population_size, \
#                                 generations_count,\
#                                 found_combination, \
#                                 combination,', [
#     (
#         2,
#         np.array(get_all_indexes(16, 16)),
#         PixelFitnessCalculator,
#         Plotter,
#         DummyPainter,
#         1234567890,
#         1000,
#         1000,
#         False,
#         np.array([[2, 1], [8, 1]]),
#     ),
#     (
#         2,
#         np.array(get_all_indexes(10, 10)),
#         PixelFitnessCalculator,
#         Plotter,
#         SyncPainter,
#         None,
#         5,
#         8,
#         False,
#         np.array([[2, 1], [8, 1]]),
#     ),
#     # TEST_DEFAULT_DATASET_ADVANCED,
# ])
# def test_genetic_calculate_for_k_pixels(pixel_count,
#                                         indexes_array,
#                                         fitness_calculator,
#                                         plotter,
#                                         painter,
#                                         seed,
#                                         population_size,
#                                         generations_count,
#                                         found_combination,
#                                         combination,
#                                         mocker):
#
#     # mocker.patch('ocr.utils.plotter.Plotter.write_counter')
#     # mocker.patch('itertools.combinations')
#
#     # combinations = itertools.combinations(indexes_array, pixel_count)
#     #
#     # # with mock.patch('itertools.combinations', )
#     # #
#     # with mock.patch('itertools.combinations', autospec=True, spec_set=True) as mocked_combinations:
#     #     mocked_combinations.return_value = combinations
#     #     mocked_combinations.return_value = (SOLUTION_DATASET_5_PIXELS, SOLUTION_DATASET_6_PIXELS)
#     # with mock.patch('ocr.utils.fitness.PixelFitnessCalculator.calculate_fitness', autospec=True, spec_set=True) as mocked_calculator:
#
#     # itertools.combinations = MagicMock(return_value=itertools.combinations(indexes_array, pixel_count))
#
#     algorithm = OCRBruteForce(pixel_count=pixel_count,
#                               indexes_array=indexes_array,
#                               fitness_calculator=PixelFitnessCalculator(ENGLISH_SYMBOLS_NP),
#                               plotter=plotter(dataset_directory=TEST_DEFAULT_DATASET),
#                               painter=painter(ENGLISH_SYMBOLS_NP),
#                               seed=seed,
#                               population_size=population_size,
#                               generations_count=generations_count)
#
#     # with mock.patch('ocr.algorithms.bruteforce.fitness_calculator.calculate_fitness', autospec=True, spec_set=True) as mocked_calculator:
#     # with mock.patch('ocr.utils.fitness.PixelFitnessCalculator', autospec=True, spec_set=True) as mocked_calculator:
#     #     mocked_calculator.return_value.calculate_fitness.return_value = 0
#     f, c = algorithm.calculate_for_k_pixels()
#     assert f == found_combination
#     assert np.array_equal(c, combination)
#
# # (
# #     pixel_count,
# #     indexes_array,
# #     fitness_calculator,
# #     plotter,
# #     painter,
# #     seed,
# #     population_size,
# #     generations_count,
# # ),
