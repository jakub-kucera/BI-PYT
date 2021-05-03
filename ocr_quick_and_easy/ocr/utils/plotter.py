from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """Class that makes a plot of values that are generated overtime"""

    def __init__(self, dataset_directory: str, records: List[int] = None,
                 cutoff: int = -200, section_width: int = None):
        self.title = dataset_directory
        self.records: List[int] = []
        self.cutoff = cutoff

        print("=========================Constructor=========================")

        self.__set_section_width(section_width=section_width)

        records = records or []
        for record in records:
            self.add_record(record)

    def __set_section_width(self, section_width: Optional[int] = None):
        """Sets with for sections which are displayed on final plot"""
        # print("=========================Before IF=========================")
        if section_width is not None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xticks(np.arange(0, len(self.records), section_width))
            # print("=========================called plot=========================")
            plt.grid(color='0.55', linestyle='-', linewidth=1, axis='x')
            # print("=========================called plot=========================")

    def add_record(self, value: int):
        """Adds a new record by integer value"""

        if value > self.cutoff:
            self.records += [value]
        else:
            self.records += [self.cutoff]

    def show(self, section_width: int = None):
        """Shows the plot"""

        self.__set_section_width(section_width)
        plt.title(self.title)
        plt.xlabel("Records")
        plt.ylabel("Fitness")

        plt.plot(self.records)
        plt.show()
