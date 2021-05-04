import itertools
import math
from abc import ABC, abstractmethod
from typing import List, Union, Optional

import numpy as np
import pygame as pg
from pygame.surface import SurfaceType, Surface
from pygame.time import Clock

from config import SIZE_MULTIPLIER, COLORS, BLACK_COLOR, PIXEL_OUTLINE_WIDTH, PIXEL_OUTLINE_COLOR, DEFAULT_FPS


class Painter(ABC):
    def __init__(self, symbols: List[np.ndarray]):
        self.symbols = symbols
        self.symbol_count = len(self.symbols)

        self.symbol_orig_height, self.symbol_orig_width = symbols[0].shape
        self.symbol_height = self.symbol_orig_height * SIZE_MULTIPLIER
        self.symbol_width = self.symbol_orig_width * SIZE_MULTIPLIER

        self.symbols_vertical = math.floor(math.sqrt(self.symbol_count))
        self.symbols_horizontal = math.ceil(math.sqrt(self.symbol_count))

        self.symbol_surfaces: List[Union[pg.Surface, SurfaceType]] = []
        self.screen: Optional[Union[Surface, SurfaceType]] = None
        self.clock: Optional[Clock] = None
        self.fps = DEFAULT_FPS

        self.chosen_pixels: List[np.ndarray] = []

    @abstractmethod
    def init_painter(self):
        pass

    @abstractmethod
    def change_chosen_pixels(self, chosen_pixels: np.ndarray):
        pass

    def create_clean_symbol_surfaces(self):
        self.symbol_surfaces = []
        for symbol in self.symbols:
            surface = pg.surfarray.make_surface(COLORS[symbol.astype(dtype=np.uint8).T])
            surface_scaled = pg.transform.scale(surface, (self.symbol_width, self.symbol_height))
            self.symbol_surfaces += [surface_scaled]

    def paint_symbol_border(self):
        for symbol in self.symbol_surfaces:
            pg.draw.rect(symbol, BLACK_COLOR, ((0, 0), (1, self.symbol_height)))
            pg.draw.rect(symbol, BLACK_COLOR, ((0, 0), (self.symbol_width, 1)))

    def paint_chosen_pixels(self, chosen_pixels: np.ndarray):
        for symbol, (y_pixel, x_pixel) in itertools.product(self.symbol_surfaces, chosen_pixels):
            position_0_0 = (x_pixel * SIZE_MULTIPLIER, y_pixel*SIZE_MULTIPLIER)
            position_1_0 = ((x_pixel + 1) * SIZE_MULTIPLIER, y_pixel*SIZE_MULTIPLIER)
            position_0_1 = (x_pixel * SIZE_MULTIPLIER, (y_pixel + 1)*SIZE_MULTIPLIER)
            position_1_width = (PIXEL_OUTLINE_WIDTH, SIZE_MULTIPLIER)
            position_1_height = (SIZE_MULTIPLIER, PIXEL_OUTLINE_WIDTH)

            pg.draw.rect(symbol, PIXEL_OUTLINE_COLOR,
                         (position_0_0, position_1_width))
            pg.draw.rect(symbol, PIXEL_OUTLINE_COLOR,
                         (position_0_0, position_1_height))
            pg.draw.rect(symbol, PIXEL_OUTLINE_COLOR,
                         (position_1_0, position_1_width))
            pg.draw.rect(symbol, PIXEL_OUTLINE_COLOR,
                         (position_0_1, position_1_height))

    def mark_pixel_combination(self, chosen_pixels: np.ndarray):
        self.paint_symbol_border()
        self.paint_chosen_pixels(chosen_pixels)

    def add_symbols_on_screen(self):
        location_iterator = itertools.product(range(0, self.symbols_vertical * self.symbol_height, self.symbol_height),
                                              range(0, self.symbols_horizontal * self.symbol_width, self.symbol_width))
        for surface, (y, x) in zip(self.symbol_surfaces, location_iterator):
            self.screen.blit(surface, (x, y))
