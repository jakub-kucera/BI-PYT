import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.records = []
        # self.records = np.array()

    def add_record(self, value: int):
        self.records += [value]

    def show(self):
        plt.plot(self.records)
        plt.show()
