
# def __start_painter(self, fps: int = DEFAULT_FPS):
#     pg.init()
#     clock = pg.time.Clock()
#
#     self.screen = pg.display.set_mode((self.symbols_horizontal * self.symbol_width,
#                                        self.symbols_vertical * self.symbol_height))
#
#     running = True
#     while running:
#         for event in pg.event.get():
#             if event.type == pg.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
#                 running = False
#
#         self.screen.fill(BLACK_COLOR)
#         # self.mark_pixel_combination()
#         self.add_symbols_on_screen()
#         pg.display.flip()
#         clock.tick(fps)
