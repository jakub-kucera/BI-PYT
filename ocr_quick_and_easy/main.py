import pathlib
from typing import List, Tuple

import numpy as np
from PIL import Image
import filetype

IMAGE_THRESHOLD_VALUE = 127

# 1. source venv/bin/activate

# , img_width=16, img_height=16


def load_symbols(directory: str = "Datasets/dataset/") -> List[np.array]:
    """Converts all images from a given directory into an array"""
    dataset_file = pathlib.Path(directory)
    symbols = []

    # goes through all files in directory
    for symbol_path in dataset_file.iterdir():
        print(symbol_path)
        if symbol_path.is_file() and filetype.is_image(symbol_path.__str__()):
            img = Image.open(symbol_path).convert('L')

            # todo add size check

            array_1d = np.array(img.getdata(), dtype=np.uint8)
            # todo maybe use bool
            # array_1d = np.array(img.getdata(), dtype=np.bool8)
            array = np.resize(array_1d, (img.size[0], img.size[1]))
            array = np.invert(array)
            print(array)

            symbols += [array]

    return symbols


def create_overlap(symbols: List[np.array]) -> np.array:
    """Creates a new array indicating value overlap of all given arrays"""
    overlap = np.zeros(symbols[0].shape)

    for symbol in symbols:
        overlap = np.maximum(overlap, symbol)

    return overlap


def get_filtered_matrix_indexes(overlap: np.array, threshold: int = IMAGE_THRESHOLD_VALUE) -> List[Tuple[int, int]]:

    indexes = []
    indexes_np_s = np.array(0, dtype=('uint8', 'uint8'))
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            if overlap[i][j] > threshold:
                indexes += [(i, j)]
                indexes_np_s = np.append(indexes_np_s, (i, j))

    indexes_np = np.array(indexes, dtype=np.dtype('uint8, uint8'))
    print("numpy convert")
    print(indexes_np)
    print("numpy direct")
    print(indexes_np_s)
    return indexes


if __name__ == '__main__':
    print("Start")
    # load_symbols()
    # array_symbols = load_symbols("Datasets/written_hiragana_dataset/")
    array_symbols = load_symbols()
    # todo clean values
    symbol_overlap = create_overlap(array_symbols)
    print("OVERLAP:")
    print(symbol_overlap)
    overlapping_indexes = get_filtered_matrix_indexes(symbol_overlap)

    for pixel_count in range(1, len(overlapping_indexes)):
        pass
        # todo check numpy.random.permutation


# for loop num_pixels++ until solution found
# all permutations of indexes for given number of pixels
# for all symbols take elements on indexes (np.take(symbol, chosen_indexes))
# hill climbing, genetic algo

# input arguments
# for hill-climbing/genetic track fitness -> matplotlib
# pygame drawing
