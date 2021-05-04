from typing import List

import numpy as np
import pygame as pg
from pygame.constants import KEYDOWN, K_ESCAPE

from config import BLACK_COLOR, DEFAULT_FPS
from ocr.gui.painter import Painter


class SymbolPainter(Painter):
    def __init__(self, symbols: List[np.ndarray]):
        super().__init__(symbols)
        self.create_clean_symbol_surfaces()

    def init_painter(self):
        self.screen = pg.display.set_mode((self.symbols_horizontal * self.symbol_width,
                                           self.symbols_vertical * self.symbol_height))
        pg.init()
        self.clock = pg.time.Clock()

    def change_chosen_pixels(self, chosen_pixels: np.ndarray):
        if self.clock is None or self.screen is None:
            return

        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False

        self.screen.fill(BLACK_COLOR)
        self.create_clean_symbol_surfaces()
        self.mark_pixel_combination(chosen_pixels)
        self.add_symbols_on_screen()
        pg.display.flip()
        # self.clock.tick(self.fps)

    def __start_painter(self, fps: int = DEFAULT_FPS):
        pg.init()
        clock = pg.time.Clock()

        self.screen = pg.display.set_mode((self.symbols_horizontal * self.symbol_width,
                                           self.symbols_vertical * self.symbol_height))

        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            self.screen.fill(BLACK_COLOR)
            # self.mark_pixel_combination()
            self.add_symbols_on_screen()
            pg.display.flip()
            clock.tick(fps)
