from filtering.filtering import apply_filter
from filtering.helpers import *
from time import time


if __name__ == "__main__":
    image = read_image('tests/lenna.png')
    start_total = time()
    for filter_name, kernel in filters.items():
        print(f"start {filter_name}")
        start_local = time()
        display_image(apply_filter(image, kernel))
        end_local = time() - start_local
        print(f"{filter_name} time: {end_local.real}")
    end_total = time() - start_total
    print(f"Total run time: {end_total.real}")
