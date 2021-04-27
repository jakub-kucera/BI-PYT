# , img_width=16, img_height=16
from ocr.ocr import OCR
from ocr.plotter import Plotter

if __name__ == '__main__':
    print("Start")
    plotter = Plotter()
    ocrko = OCR(plotter=plotter)

    ocrko.bruteforce()
    plotter.show()


# for loop num_pixels++ until solution found
# all permutations of indexes for given number of pixels
# for all symbols take elements on indexes (np.take(symbol, chosen_indexes))
# hill climbing, genetic algo

# input arguments
# for hill-climbing/genetic track fitness -> matplotlib
# pygame drawing
