import pathlib
from typing import List, Tuple

import numpy as np
from PIL import Image
import filetype

IMAGE_THRESHOLD_VALUE = 127


class ImageLoader:
    # def __init__(self, dataset_directory: str = "Datasets/dataset/"):
    #     self.dataset_directory = dataset_directory

    @staticmethod
    def load_symbols(dataset_directory: str = "Datasets/dataset/") -> List[np.ndarray]:
        """Converts all images from a given directory into an array"""
        dataset_file = pathlib.Path(dataset_directory)
        symbols = []

        img_size = None

        # goes through all files in directory
        for symbol_path in dataset_file.iterdir():
            print(symbol_path)
            if symbol_path.is_file() and filetype.is_image(symbol_path.__str__()):
                img = Image.open(symbol_path).convert('L')

                if img_size is None:
                    img_size = img.size
                elif img_size != img.size:
                    print(f"Image {symbol_path} has a different dimensions. ")
                    continue

                array_flattened = np.array(img.getdata(), dtype=np.uint8)

                # todo maybe use bool
                # array_1d = np.array(img.getdata(), dtype=np.bool8)
                array = np.resize(array_flattened, (img.size[0], img.size[1]))

                array = np.invert(array)
                symbols += [array]

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
        # def get_filtered_matrix_indexes(overlap: np.array, threshold: int = IMAGE_THRESHOLD_VALUE) -> List[Tuple[int, int]]:

        indexes = []
        # y_indexes = []
        # x_indexes = []
        # indexes_np_s = np.array(0, dtype=('uint8', 'uint8'))

        for i in np.ndindex(overlap.shape):
            if overlap[i] > threshold:
                # y_indexes += [i[0]]
                # x_indexes += [i[1]]
                indexes += [i]
                # indexes_np_s = np.append(indexes_np_s, i)

        indexes_np = np.array(indexes, dtype=np.dtype('uint8, uint8'))
        indexes_np_idk = np.array(indexes, dtype=np.dtype('uint8'))

        # print("indexes_np")
        # print(indexes_np)
        # print("indexes_np_idk")
        # print(indexes_np_idk)

        return indexes_np_idk
