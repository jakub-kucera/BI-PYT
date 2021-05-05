import json
import os.path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import PLOTTER_COUNTER_FILE, OUTPUT_PLOT_IMG_TYPE, OUTPUT_PLOTS_FILE


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
        plt.savefig(output_img)
        self.write_counter(counter)

        if show_plot:
            img = Image.open(output_img)
            img.show()
        # plt.show()
