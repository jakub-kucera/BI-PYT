import random
from typing import Mapping, Any, Optional, List, Type

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image
import ocr
from ocr.algorithms.algorithm import OCRAlgorithm
from ocr.algorithms.bruteforce import OCRBruteForce
from ocr.algorithms.genetic import OCRGenetic
from ocr.gui.dummy_painter import DummyPainter
from ocr.gui.painter import Painter
from ocr.ocr import OCR
from ocr.utils.fitness import PixelFitnessCalculator
from ocr.utils.plotter import Plotter

TEST_DEFAULT_DATASET = "tests/test_datasets/dataset/"
TEST_DEFAULT_DATASET_ADVANCED = "tests/test_datasets/written_hiragana_dataset/"

TEST_ALGORITHM_CLASS_TO_PACKAGE = {OCRGenetic: "genetic.OCRGenetic", OCRAlgorithm: "algorithm.OCRAlgorithm", OCRBruteForce: "bruteforce.OCRBruteForce"}

@pytest.mark.parametrize("dataset_directory, overlapping_indexes_count", [
    (TEST_DEFAULT_DATASET, 214),
    (TEST_DEFAULT_DATASET_ADVANCED, 174),
])
def test_ocr_create(dataset_directory, overlapping_indexes_count, mocker):
    test_ocr = OCR(plotter=Plotter(dataset_directory), dataset_directory=dataset_directory)
    assert overlapping_indexes_count == test_ocr.total_overlap_pixel_count
    assert type(test_ocr.fitness_calculator) == PixelFitnessCalculator


@pytest.mark.parametrize('symbol_count, pixel_count', [
    (26, 5),
    (50, 6),
    (32, 5),
    (2, 1),
    (150, 8)

])
def test_ocr_calculate_pixel_count(symbol_count, pixel_count):
    assert pixel_count == OCR.calculate_pixel_count(symbol_count)


@pytest.mark.parametrize("dataset_directory, algorithm_type, painter_type, \
                        population_size, generations_count, trials_count, seed", [
    (
        TEST_DEFAULT_DATASET,
        OCRGenetic,
        DummyPainter,
        100,
        100,
        1,
        None
    ),
    (
        TEST_DEFAULT_DATASET_ADVANCED,
        OCRBruteForce,
        DummyPainter,
        1000,
        1000,
        2,
        123
    ),
])
def test_ocr_calculate(dataset_directory: str,
                       algorithm_type: Type[OCRAlgorithm],
                       painter_type: Type[Painter],
                       population_size: int,
                       generations_count: int,
                       trials_count: int,
                       seed: int,
                       mocker
):
    mocker.patch('random.seed')
    mocker.patch('numpy.random.seed')
    mocker.patch('ocr.gui.painter.Painter.__init__', autospec=True)
    mocker.patch('ocr.algorithms.' + TEST_ALGORITHM_CLASS_TO_PACKAGE[algorithm_type] + '.calculate_for_k_pixels',
                 autospec=True, return_value=(True, np.ndarray([1, 2])))

    test_ocr = OCR(dataset_directory=dataset_directory)

    test_ocr.calculate(
        algorithm_type=algorithm_type,
        painter_type=painter_type,
        population_size=population_size,
        generations_count=generations_count,
        trials_count=trials_count,
        seed=seed)

    random.seed.assert_called_once_with(seed)
    np.random.seed.assert_called_once_with(seed)
    ocr.gui.painter.Painter.__init__.assert_called_once()

    # ocr.algorithms.genetic.OCRGenetic.calculate_for_k_pixels.assert_called_once()
    algorithm_type.calculate_for_k_pixels.assert_called_once()
