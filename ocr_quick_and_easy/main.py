# , img_width=16, img_height=16
from ocr.ocr import OCR

if __name__ == '__main__':
    print("Start")
    ocrko = OCR()

    ocrko.bruteforce()


# for loop num_pixels++ until solution found
# all permutations of indexes for given number of pixels
# for all symbols take elements on indexes (np.take(symbol, chosen_indexes))
# hill climbing, genetic algo

# input arguments
# for hill-climbing/genetic track fitness -> matplotlib
# pygame drawing
