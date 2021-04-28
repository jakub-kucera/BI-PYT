from typing import List
import matplotlib.pyplot as plt


class Plotter:
    """Class that makes a plot of values that are generated overtime"""
    def __init__(self, dataset_directory: str, records: List[int] = None):
        plt.title(dataset_directory)
        plt.xlabel("Records")
        plt.ylabel("Fitness")

        self.records: List[int] = []
        if records is not None:
            self.records += records

    def add_record(self, value: int):
        """Adds a new record by integer value"""
        self.records += [value]

    def show(self):
        """Shows the plot"""
        plt.plot(self.records)
        plt.show()
