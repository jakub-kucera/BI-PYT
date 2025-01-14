import numpy as np

import filtering.helpers

PADDING_VALUE = 0

from filtering.helpers import *


# def get_convolution_element_value(r: int, c: int, d: int, in_img: np.array, kernel: np.array):
#     return PADDING_PIXEL_VALUE

# out_img[r, c, d] = get_convolution_element_value(r, c, d, in_img, kernel)

def apply_filter_rgb(in_img: np.array, flipped_kernel: np.array) -> np.array:
    padding_size = int(flipped_kernel.shape[0] / 2 + 1)
    kernel_sides = int(flipped_kernel.shape[0] / 2)
    padded_img = np.pad(in_img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant',
                        constant_values=((PADDING_VALUE, PADDING_VALUE), (PADDING_VALUE, PADDING_VALUE), (0, 0)))

    # padded_img = padded_img.astype(flipped_kernel.dtype)
    # padded_img = padded_img.astype(in_img.dtype)

    # out_img = np.zeros(in_img.shape, dtype=flipped_kernel.dtype)
    out_img = np.zeros(padded_img.shape, dtype=in_img.dtype)

    img_rows = in_img.shape[0]
    img_columns = in_img.shape[1]
    dimensions = in_img.shape[2]

    for dim in range(dimensions):
        for img_row in range(padding_size, img_rows + padding_size + 1):
            for img_col in range(padding_size, img_columns + padding_size + 1):
                arr_window = padded_img[(img_row - kernel_sides):(img_row + kernel_sides + 1),
                             (img_col - kernel_sides):(img_col + kernel_sides + 1), dim]
                # flipped_arr = np.flip(arr_window, axis=(0, 1))
                pixel_value = np.sum(arr_window * flipped_kernel)
                # pixel_value = np.sum(flipped_arr * kernel)

                if pixel_value < 0:
                    pixel_value = 0
                elif pixel_value > 255:
                    pixel_value = 255

                out_img[img_row, img_col, dim] = pixel_value

    # return out_img
    return out_img[padding_size:img_rows + padding_size, padding_size:img_columns + padding_size, ]


def apply_filter_gray(in_img: np.array, flipped_kernel: np.array) -> np.array:
    padding_size = int(flipped_kernel.shape[0] / 2 + 1)
    kernel_sides = int(flipped_kernel.shape[0] / 2)
    padded_img = np.pad(in_img, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                        constant_values=((PADDING_VALUE, PADDING_VALUE), (PADDING_VALUE, PADDING_VALUE)))

    padded_img = padded_img.astype(flipped_kernel.dtype)
    # padded_img = padded_img.astype(in_img.dtype)

    out_img = np.zeros(padded_img.shape, dtype=flipped_kernel.dtype)
    # out_img = np.zeros(padded_img.shape, dtype=in_img.dtype)

    img_rows = in_img.shape[0]
    img_columns = in_img.shape[1]

    for img_row in range(padding_size, img_rows + padding_size + 1):
        for img_col in range(padding_size, img_columns + padding_size + 1):

            arr_window = padded_img[(img_row - kernel_sides):(img_row + kernel_sides + 1),
                         (img_col - kernel_sides):(img_col + kernel_sides + 1)]
            # flipped_arr = np.flip(arr_window, axis=(0, 1))
            pixel_value = np.sum(arr_window * flipped_kernel)
            # pixel_value = np.sum(flipped_arr * kernel)

            if pixel_value < 0:
                pixel_value = 0
            elif pixel_value > 255:
                pixel_value = 255

            out_img[img_row, img_col] = pixel_value

    return out_img[padding_size:img_rows + padding_size, padding_size:img_columns + padding_size].astype(
        flipped_kernel.dtype)
    # return out_img[padding_size:img_rows + padding_size, padding_size:img_columns + padding_size].astype(in_img.dtype)
    # return out_img[padding_size:img_rows + padding_size, padding_size:img_columns + padding_size]


def apply_filter(in_img: np.array, kernel: np.array) -> np.array:
    # A given image has to have either 2 (grayscale) or 3 (RGB) dimensions
    assert in_img.ndim in [2, 3]
    # # A given filter has to be 2 dimensional and square
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]

    # if kernel == identity_kernel:
    #     return in_img.copy()
    #
    # if np.array_equal(kernel, filtering.helpers.roberts_cross_1_kernel):
    #     kernel = np.pad(kernel, ((0, 1), (0, 1)), 'constant', constant_values=((0, PADDING_VALUE), (0, PADDING_VALUE)))
    #
    # if np.array_equal(kernel, filtering.helpers.roberts_cross_2_kernel):
    #     kernel = np.pad(kernel, ((0, 1), (1, 0)), 'constant', constant_values=((0, PADDING_VALUE), (0, PADDING_VALUE)))
    flipped_kernel = kernel
    # flipped_kernel = np.flip(flipped_kernel, axis=(0, 1))


    if flipped_kernel.shape[0] % 2 == 0:
        flipped_kernel = np.pad(flipped_kernel, ((0, 1), (0, 1)), 'constant',
                                constant_values=((PADDING_VALUE, PADDING_VALUE), (PADDING_VALUE, PADDING_VALUE)))
    else:
        flipped_kernel = np.flip(flipped_kernel, axis=(0, 1))

    print(kernel)
    print(flipped_kernel)

    # flipped_kernel = np.flip(flipped_kernel, axis=(0, 1))
    #

    # flipped_kernel = np.insert(flipped_kernel, 1, 0, axis=1)
    # flipped_kernel = np.insert(flipped_kernel, 1, 0, axis=0)

    # flipped_kernel = np.flip(flipped_kernel, axis=(0, 1))

    if in_img.ndim == 2:
        return apply_filter_gray(in_img, flipped_kernel)

    if in_img.ndim == 3:
        return apply_filter_rgb(in_img, flipped_kernel)
