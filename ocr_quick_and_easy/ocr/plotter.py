from typing import List

import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, dataset_directory: str):
        self.records: List[int] = []
        plt.title(dataset_directory)
        # self.records = np.array()

    def add_record(self, value: int):
        self.records += [value]

    def show(self):
        plt.plot(self.records)
        plt.show()
