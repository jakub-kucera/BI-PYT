import pytest
import numpy as np
from typing import List

from ocr.image_loader import ImageLoader

TEST_DATASETS = "tests/test_datasets/"

TEST_DEFAULT_DATASET = "dataset/"
TEST_DEFAULT_DATASET_ADVANCED = "written_hiragana_dataset/"


@pytest.mark.parametrize('database_directory', [
    TEST_DEFAULT_DATASET,
    TEST_DEFAULT_DATASET_ADVANCED,
])
def test_load_symbols_success(database_directory: str):
    ImageLoader.load_symbols(TEST_DATASETS + database_directory)


@pytest.mark.parametrize('database_directory', [
    "dummy/"
    "empty/"
    "incorrect_size/"
])
def test_load_symbols_create_error(database_directory: str):
    with pytest.raises(Exception):
        ImageLoader.load_symbols(TEST_DATASETS + database_directory)


@pytest.mark.parametrize('database_directory, count', [
    (TEST_DEFAULT_DATASET, 26),
    (TEST_DEFAULT_DATASET_ADVANCED, 50),
])
def test_load_symbols_load_count(database_directory: str, count: int):
    symbols_loaded = ImageLoader.load_symbols(TEST_DATASETS + database_directory)
    assert len(symbols_loaded) == count


@pytest.mark.parametrize('symbols, overlap', [
    (
            [
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
            ],
            np.array([[0, 0], [0, 0]], dtype=np.uint8)
    ),
    (
            [
                np.array([[1, 1], [1, 1]], dtype=np.uint8),
                np.array([[1, 1], [1, 1]], dtype=np.uint8),
            ],
            np.array([[0, 0], [0, 0]], dtype=np.uint8)
    ),
    (
            [
                np.array([[1, 1], [1, 1]], dtype=np.uint8),
                np.array([[1, 1], [1, 255]], dtype=np.uint8),
            ],
            np.array([[0, 0], [0, 254]], dtype=np.uint8)
    ),
    (
            [
                np.array([[1, 1], [1, 1]], dtype=np.uint8),
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
            ],
            np.array([[1, 1], [1, 1]], dtype=np.uint8)
    ),
    (
            [
                np.array([[0, 1], [1, 1]], dtype=np.uint8),
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
            ],
            np.array([[1, 1], [1, 0]], dtype=np.uint8)
    ),
    (
            [
                np.array([[0, 1], [1, 1]], dtype=np.uint8),
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
                np.array([[1, 1], [1, 1]], dtype=np.uint8),
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
            ],
            np.array([[1, 1], [1, 0]], dtype=np.uint8)
    ),
    (
            [
                np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 0, 0]], dtype=np.uint8),
                np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.uint8),
            ],
            np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 0, 1]], dtype=np.uint8)
    ),
])
def test_create_overlap_distinct(symbols: List[np.ndarray], overlap: np.ndarray):
    created_overlap = ImageLoader.create_overlap_distinct(symbols)
    assert (created_overlap == overlap).all()
    assert np.array_equal(created_overlap, overlap)


@pytest.mark.parametrize('symbols', [
    (
            [
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0, 1], [0, 0, 1]], dtype=np.uint8),
            ]
    ),
    (
            [
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0], [0, 0], [0, 0]], dtype=np.uint8),
            ]
    ),
    (
            [
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8),
            ]
    ),
    (
            [
                np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8),
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
            ]
    ),
    (
            [
                np.array([[0, 0], [0, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0], [1, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 7], [0, 0], [1, 0]], dtype=np.uint8),
                np.array([[0, 0], [0, 0]], dtype=np.uint8),
                np.array([[0, 0], [0, 8], [0, 0]], dtype=np.uint8),
            ]
    ),
])
def test_create_overlap_distinct_error(symbols: List[np.ndarray]):
    with pytest.raises(Exception):
        ImageLoader.create_overlap_distinct(symbols)


@pytest.mark.parametrize('overlap, threshold, indexes', [
    (
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
            127,
            ([], [])
    ),
    (
            np.array([[255, 255], [255, 255]], dtype=np.uint8),
            127,
            ([0, 0, 1, 1], [0, 1, 0, 1])
    ),
    (
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
            127,
            ([0, 1], [1, 0])
    ),
    (
            np.array([[0, 255], [128, 126]], dtype=np.uint8),
            127,
            ([0, 1], [1, 0])
    ),

])
def test_get_filtered_matrix_indexes(overlap: np.ndarray, threshold: int, indexes: np.ndarray):
    overlapping_indexes = ImageLoader.get_filtered_matrix_indexes(overlap=overlap,
                                                                  threshold=threshold)
    assert overlapping_indexes == indexes
    # assert np.array_equal(overlapping_indexes, indexes)
