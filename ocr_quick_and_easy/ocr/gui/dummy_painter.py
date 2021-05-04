from typing import List

import numpy as np

from ocr.gui.painter import Painter


class DummyPainter(Painter):
    def __init__(self, symbols: List[np.ndarray]):
        super().__init__(symbols)

    def init_painter(self):
        pass

    def change_chosen_pixels(self, chosen_pixels: np.ndarray):
        pass
