import pathlib
from typing import List
import numpy as np
from PIL import Image
import filetype

from config import IMAGE_THRESHOLD_VALUE, DEFAULT_DATASET, DEBUG_PRINT


class ImageLoader:
    """Class that takes cares of loading symbols from files"""

    @staticmethod
    def load_symbols(dataset_directory: str = DEFAULT_DATASET) -> List[np.ndarray]:
        """Converts all images from a given directory into arrays"""

        dataset_file = pathlib.Path(dataset_directory)
        symbols = []

        img_size = None

        # goes through all files in directory
        for symbol_path in dataset_file.iterdir():
            print(symbol_path) if DEBUG_PRINT else None
            if symbol_path.is_file() and filetype.is_image(symbol_path.__str__()):
                img = Image.open(symbol_path).convert('L')

                if img_size is None:
                    img_size = img.size
                elif img_size != img.size:
                    print(f"Image {symbol_path} has a different dimensions. ")
                    continue

                array_flattened = np.array(img.getdata(), dtype=np.uint8)
                array = np.resize(array_flattened, (img.size[0], img.size[1]))
                array_inverted = np.invert(array)

                symbols += [array_inverted]

        return symbols

    @staticmethod
    def create_overlap(symbols: List[np.ndarray]) -> np.ndarray:
        """Creates a new array indicating value overlap of all given arrays"""

        overlap = np.zeros(symbols[0].shape)

        for symbol in symbols:
            overlap = np.maximum(overlap, symbol)

        return overlap

    @staticmethod
    def get_filtered_matrix_indexes(overlap: np.ndarray, threshold: int = IMAGE_THRESHOLD_VALUE) -> np.ndarray:
        """Return indexes of pixels with larger than passed value"""

        indexes = []
        # y_indexes = []
        # x_indexes = []

        for i in np.ndindex(overlap.shape):
            if overlap[i] > threshold:
                # y_indexes += [i[0]]
                # x_indexes += [i[1]]
                indexes += [i]

        # indexes_np = np.array(indexes, dtype=np.dtype('uint8, uint8'))
        indexes_np_idk = np.array(indexes, dtype=np.dtype('uint8'))

        # print("indexes_np")
        # print(indexes_np)
        # print("indexes_np_idk")
        # print(indexes_np_idk)

        return indexes_np_idk
