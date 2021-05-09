import json
import os.path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from config import PLOTTER_COUNTER_FILE, OUTPUT_PLOT_IMG_TYPE, OUTPUT_PLOTS_FILE, POPULATION_SIZE, MAX_GENERATIONS, \
    PLOT_IMG_DPI, PLOT_THRESHOLD


class Plotter:
    """Class that makes a plot of values that are generated overtime"""

    def __init__(self, dataset_directory: str, records: List[int] = None,
                 cutoff: int = -200, section_width: int = None):
        self.title = dataset_directory
        self.records: List[int] = []
        self.cutoff = cutoff

        self.__set_section_width(section_width=section_width)

        records = records or []
        for record in records:
            self.add_record(record)

    def __set_section_width(self, section_width: Optional[int] = None):
        """Sets with for sections which are displayed on final plot"""
        if section_width is not None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xticks(np.arange(0, len(self.records), section_width))
            plt.grid(color='0.55', linestyle='-', linewidth=1, axis='x')

    def add_record(self, value: int):
        """Adds a new record by integer value"""
        if value > self.cutoff:
            self.records += [value]
        else:
            self.records += [self.cutoff]

    @staticmethod
    def get_counter() -> int:
        """Reads counter from json file"""
        counter = 0
        if os.path.isfile(PLOTTER_COUNTER_FILE):
            with open(PLOTTER_COUNTER_FILE, 'r') as infile:
                counter = json.load(infile)['count']
        return counter

    @staticmethod
    def write_counter(counter: int):
        """Writes counter into json file"""
        with open(PLOTTER_COUNTER_FILE, 'w') as outfile:
            json.dump({"count": counter}, outfile)

    def show(self, section_width: int = None, show_plot: bool = True):
        """Shows the plot"""

        counter = self.get_counter()

        self.__set_section_width(section_width)
        plt.suptitle(self.title)
        plt.title(f"Result number: {counter}")
        plt.xlabel("Records")
        plt.ylabel("Fitness")
        plt.plot(self.records)

        counter += 1
        output_img = f"{OUTPUT_PLOTS_FILE}/{counter}_plot{OUTPUT_PLOT_IMG_TYPE}"
        plt.savefig(output_img, dpi=PLOT_IMG_DPI)
        self.write_counter(counter)

        if show_plot:
            img = Image.open(output_img)
            img.show()
        # plt.show()


class PlotterExperimental3D:
    """Class that makes a plot of values that are generated overtime"""

    def __init__(self, dataset_directory: str, records: List[int] = None,
                 cutoff: int = -200, section_width: int = None):
        self.title = dataset_directory
        self.records: List[List[int]] = []
        self.cutoff = cutoff

    def add_record(self, value: List[int]):
        """Adds a new record by integer value"""
        self.records += [value]

    @staticmethod
    def get_counter() -> int:
        """Reads counter from json file"""
        counter = 0
        if os.path.isfile(PLOTTER_COUNTER_FILE):
            with open(PLOTTER_COUNTER_FILE, 'r') as infile:
                counter = json.load(infile)['count']
        return counter

    @staticmethod
    def write_counter(counter: int):
        """Writes counter into json file"""
        with open(PLOTTER_COUNTER_FILE, 'w') as outfile:
            json.dump({"count": counter}, outfile)

    def show(self, section_width: int = None, show_plot: bool = True):
        """Shows the plot"""

        # self.records = self.records[::-1]


        counter = self.get_counter()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, POPULATION_SIZE, 1).tolist()
        Y = np.arange(0, POPULATION_SIZE, 1).tolist()
        X, Y = np.meshgrid(X, Y)

        plt.suptitle(self.title)
        plt.title(f"Population size: {POPULATION_SIZE}, Generations count: {MAX_GENERATIONS}")

        Z = np.zeros(X.shape)
        for i in range(0, len(X)):
        # for i in range(0, len(X)):
            # for j in range(len(Y), 0):
            for j in range(0, len(Y)):
                value = self.records[X[i, j]][Y[i, j]]
                # print(value)
                # if value > 200:
                #     Z[99 - i, 99 - j] = value
                #     print(value)
                # else:\
                #     Z[99 - i, 99 - j] = 200
                # Z[(POPULATION_SIZE - 1) - i, (POPULATION_SIZE - 1) - j] = value
                Z[(POPULATION_SIZE - 1)-i, (POPULATION_SIZE - 1)-j] = value if (value > PLOT_THRESHOLD) else PLOT_THRESHOLD

        # plt.plot(self.records)

        ax.plot_surface(X, Y, Z)
        # ax.invert_xaxis()
        # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, zdir='z', c='red')

        counter += 1
        output_img = f"{OUTPUT_PLOTS_FILE}/{counter}_plot{OUTPUT_PLOT_IMG_TYPE}"

        plt.xlabel("Generations")
        # plt.zlabel("Fitness")
        plt.ylabel("individuals")

        plt.savefig(output_img, dpi=PLOT_IMG_DPI)
        self.write_counter(counter)

        # plt.figure(dpi=1200)
        # plt.show()
        img = Image.open(output_img)
        img.show()
