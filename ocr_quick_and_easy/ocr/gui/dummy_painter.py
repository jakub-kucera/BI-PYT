from typing import List

import numpy as np

from ocr.gui.painter import Painter


class DummyPainter(Painter):
    """Class that functions as a dummy Painter object. """

    def __init__(self, symbols: List[np.ndarray]):
        super().__init__(symbols)

    def init_painter(self):
        return

    def change_chosen_pixels(self, chosen_pixels: np.ndarray):
        return
