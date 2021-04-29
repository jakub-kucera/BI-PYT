# , img_width=16, img_height=16
from config import DEFAULT_DATASET_ADVANCED, DEFAULT_DATASET
from ocr.algorithm import OCRAlgorithm
from ocr.bruteforce import OCRBruteForce
from ocr.genetic import OCRGenetic
from ocr.ocr import OCR
from ocr.plotter import Plotter


def main(dataset_directory: str = DEFAULT_DATASET_ADVANCED):
# def main(dataset_directory: str = DEFAULT_DATASET):
    print("Start")
    plotter = Plotter(dataset_directory)
    ocrko = OCR(plotter=plotter, dataset_directory=dataset_directory)

    ocrko.calculate(algorithm_type=OCRGenetic)
    # ocrko.calculate(algorithm_type=OCRBruteForce)
    plotter.show()


if __name__ == '__main__':
    main()

# for loop num_pixels++ until solution found
# all combinations of indexes for given number of pixels
# for all symbols take elements on indexes (np.take(symbol, chosen_indexes))
# hill climbing, genetic algo

# input arguments
# for hill-climbing/genetic track fitness -> matplotlib
# pygame drawing
