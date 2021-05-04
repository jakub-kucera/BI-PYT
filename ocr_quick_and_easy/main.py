from config import DEFAULT_DATASET, MAX_GENERATIONS, DEFAULT_DATASET_ADVANCED, DEFAULT_DATASET_SMALL_20, \
    PLOTTER_COUNTER_FILE
from ocr.algorithms.bruteforce import OCRBruteForce
from ocr.algorithms.genetic import OCRGenetic
from ocr.gui.dummy_painter import DummyPainter
import json
from ocr.gui.sync_painter import SyncPainter
from ocr.ocr import OCR

# def main(dataset_directory: str = DEFAULT_DATASET_ADVANCED):
from ocr.utils.plotter import Plotter


# def main(dataset_directory: str = DEFAULT_DATASET_SMALL_20):
# def main(dataset_directory: str = DEFAULT_DATASET):
def main(dataset_directory: str = DEFAULT_DATASET_ADVANCED):
    print("Start")
    plotter = Plotter(dataset_directory)
    ocrko = OCR(plotter=plotter, dataset_directory=dataset_directory)

    # ocrko.calculate(algorithm_type=OCRGenetic, painter_type=SyncPainter)
    ocrko.calculate(algorithm_type=OCRGenetic, painter_type=DummyPainter)
    # ocrko.calculate(algorithm_type=OCRBruteForce, painter_type=DummyPainter)
    # ocrko.calculate(algorithm_type=OCRBruteForce, painter_type=SyncPainter)

    # plotter.show(section_width=MAX_GENERATIONS)
    plotter.show()


if __name__ == '__main__':
    main()

    # symbols = ImageLoader.load_symbols()
    # comb_distinct_overlap_pixels = []
    # comb_count = 0
    # best_comb = tuple()
    # best_small_count = 255
    # for comb in combinations(symbols, 20):
    #     overlap = ImageLoader.create_overlap_distinct(list(comb))
    #     distinct_pixels = ImageLoader.get_filtered_matrix_indexes(overlap)
    #     distinct_pixels_count = len(distinct_pixels)
    #     if distinct_pixels_count < best_small_count:
    #         best_small_count = distinct_pixels_count
    #         best_comb = comb
    #
    #     comb_distinct_overlap_pixels += [[comb, len(distinct_pixels)]]
    #     # comb_count += 1
    #     # print(comb_count)
    #
    # for comb, pixels in comb_distinct_overlap_pixels:
    #     print(pixels)
    #
    # for cmb in best_comb:
    #     print("================================================================================")
    #     print(cmb)


# for loop num_pixels++ until solution found
# all combinations of indexes for given number of pixels
# for all symbols take elements on indexes (np.take(symbol, chosen_indexes))
# hill climbing, genetic algo

# input arguments
# for hill-climbing/genetic track fitness -> matplotlib
# pygame drawing
