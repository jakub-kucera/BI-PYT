import argparse
from typing import Type, Dict

from config import RUN_DESCRIPTION, DEFAULT_DATASET, POPULATION_SIZE, MAX_GENERATIONS, \
    TRIALS_PER_PIXEL_COUNT, DEFAULT_SHOW_GUI, DEFAULT_SHOW_PLOT
from ocr.algorithms.algorithm import OCRAlgorithm
from ocr.algorithms.bruteforce import OCRBruteForce

from ocr.algorithms.genetic import OCRGenetic
from ocr.gui.dummy_painter import DummyPainter
from ocr.gui.painter import Painter
from ocr.gui.sync_painter import SyncPainter
from ocr.ocr import OCR

from ocr.utils.plotter import Plotter

ALGORITHM_TYPE_TO_CLASS: Dict[str, Type[OCRAlgorithm]] = {'Genetic': OCRGenetic, 'BruteForce': OCRBruteForce}
# PAINTER_TYPE_TO_CLASS: Dict[str, Type[Painter]] = {'Dummy': DummyPainter, 'Sync': SyncPainter}
PAINTER_TYPE_TO_CLASS: Dict[bool, Type[Painter]] = {False: DummyPainter, True: SyncPainter}

# todo add tests
# todo write func/class descriptions
# todo write report


def main(dataset_directory: str,
         algorithm_type: Type[OCRAlgorithm],
         painter_type: Type[Painter],
         show_plot: bool,
         population_size: int,
         generations_count: int,
         trials_count: int,
         seed: int
         ):
    print("Start")
    plotter = Plotter(dataset_directory)
    ocrko = OCR(plotter=plotter, dataset_directory=dataset_directory)

    ocrko.calculate(algorithm_type=algorithm_type, painter_type=painter_type,
                    population_size=population_size, generations_count=generations_count,
                    trials_count=trials_count, seed=seed)

    # plotter.show(section_width=MAX_GENERATIONS)
    plotter.show(show_plot=show_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ocr.py', description=RUN_DESCRIPTION)
    parser.add_argument('-dr', '--directory', action='store', default=DEFAULT_DATASET, metavar='PATH',
                        type=str, help='Path to directory with symbol images.')
    parser.add_argument('-al', '--algorithm', action='store', default='Genetic', metavar='TYPE',
                        type=str, help='Type of algorithm that is supposed to be used.',
                        choices=ALGORITHM_TYPE_TO_CLASS.keys())
    parser.add_argument('-ui', '--show_gui', action='store_true', default=DEFAULT_SHOW_GUI,
                        help='Show GUI with all the symbols and currently chosen pixels')
    parser.add_argument('-pl', '--plot_fitness', action='store_true', default=DEFAULT_SHOW_PLOT,
                        help='Show plot representing fitness value over time')
    parser.add_argument('-ps', '--population_size', action='store', default=POPULATION_SIZE, metavar='SIZE',
                        type=int, help='Size of each generation when using genetic algorithm.')
    parser.add_argument('-gc', '--generations_count', action='store', default=MAX_GENERATIONS, metavar='COUNT',
                        type=int, help='Maximum number of generations in one run.')
    parser.add_argument('-tc', '--trials_count', action='store', default=TRIALS_PER_PIXEL_COUNT, metavar='COUNT',
                        type=int, help='Maximum number of attempted runs of chosen algorithm for each pixel count.')
    parser.add_argument('-sd', '--seed', action='store', default=None, metavar='SEED',
                        type=int, help="Seed which is used to initialize random functions.")

    args = parser.parse_args()

    main(
        dataset_directory=args.directory,
        algorithm_type=ALGORITHM_TYPE_TO_CLASS[args.algorithm],
        painter_type=PAINTER_TYPE_TO_CLASS[args.show_gui],
        show_plot=args.plot_fitness,
        population_size=args.population_size,
        generations_count=args.generations_count,
        trials_count=args.trials_count,
        seed=args.seed
    )
