import os
import pathlib
from typing import List, Tuple
import numpy as np
from PIL import Image
import filetype

from config import IMAGE_THRESHOLD_VALUE, DEFAULT_DATASET, DEBUG_PRINT


class ImageLoader:
    """Class that takes cares of loading symbols from files and their processing"""

    @staticmethod
    def covert_symbols_bool(symbols: List[np.ndarray], threshold: int) -> List[np.ndarray]:
        """Converts a 2D array of integers into a new 2D array of boolean values"""
        converted_symbols = []
        for symbol in symbols:
            converted_symbols += [np.vectorize(lambda x: x > threshold)(symbol)]
        return converted_symbols

    @staticmethod
    def load_symbols(dataset_directory: str = DEFAULT_DATASET,
                     threshold: int = IMAGE_THRESHOLD_VALUE) -> List[np.ndarray]:
        """Converts all images from a given directory into arrays"""

        # various checks for dataset directory
        if not os.path.exists(dataset_directory):
            raise Exception(f"{dataset_directory} does not exist")

        if not os.path.isdir(dataset_directory):
            raise Exception(f"{dataset_directory} is not a directory")

        if not os.listdir(dataset_directory):
            raise Exception(f"Directory {dataset_directory} is empty")

        dataset_file = pathlib.Path(dataset_directory)
        symbols = []
        img_size = None

        # goes through all files in directory
        for symbol_path in sorted(dataset_file.iterdir()):
            print(symbol_path) if DEBUG_PRINT else None
            if symbol_path.is_file() and filetype.is_image(symbol_path.__str__()):
                img = Image.open(symbol_path).convert('L')

                if img_size is None:
                    img_size = img.size
                elif img_size != img.size:
                    raise ValueError(f"Image {symbol_path} has a different dimensions. ")

                array_flattened = np.array(img.getdata(), dtype=np.uint8)
                array = np.resize(array_flattened, (img.size[0], img.size[1]))
                array_inverted = np.invert(array)

                symbols += [array_inverted]

        # return symbols
        return ImageLoader.covert_symbols_bool(symbols, threshold)

    @staticmethod
    def create_overlap_distinct(symbols: List[np.ndarray]) -> np.ndarray:
        """Creates a new array indicating value overlap of distinct pixels of all given arrays"""
        image_shape = symbols[0].shape

        overlap_max = np.zeros(symbols[0].shape, dtype=np.uint8)
        overlap_min = np.full(symbols[0].shape, fill_value=255, dtype=np.uint8)

        for symbol in symbols:
            if symbol.shape != image_shape:
                raise Exception("symbols in array have variable shape")

            overlap_max = np.maximum(overlap_max, symbol)
            overlap_min = np.minimum(overlap_min, symbol)

        overlap = np.subtract(overlap_max, overlap_min)

        return overlap

    @staticmethod
    def get_filtered_matrix_indexes(overlap: np.ndarray) -> np.ndarray:
        """Return indexes of pixels with larger than passed value"""
        # threshold: int = IMAGE_THRESHOLD_VALUE) -> Tuple[List[int], List[int]]:

        indexes = []
        # y_indexes = []
        # x_indexes = []

        for i in np.ndindex(overlap.shape):
            if overlap[i]:
                # y_indexes += [i[0]]
                # x_indexes += [i[1]]
                indexes += [i]

        if DEBUG_PRINT:
            print(f"Symbols have {len(overlap) * len(overlap[0])} total pixels")
            print(f"Symbols have {len(indexes)} overlapping pixels")

        # indexes_np = np.array(indexes, dtype=np.dtype('uint8, uint8'))
        indexes_np = np.array(indexes, dtype=np.uint8)

        return indexes_np
