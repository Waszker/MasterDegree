import os
import numpy as np
from numpy import genfromtxt


class DatasetReader:
    """
    Takes care of reading data sets for training and testing.
    """

    def __init__(self, dataset_path):
        self.path = dataset_path

    def read_digits(self, filename='digits.csv'):
        """
        Reads data from digits file in .csv format.
        :param filename: optional parameter if filename is different than 'digits.csv'
        :return: numpy row-oriented data array
        """
        data = genfromtxt(self.path + os.sep + filename, delimiter=',', skip_header=0)
        return np.asarray(data)

    def read_letters(self, filename='letters.csv'):
        """
        Reads data from letters file in .csv format.
        :param filename: optional parameter if filename is different than 'letters.csv'
        :return: numpy row-oriented data array
        """
        data = genfromtxt(self.path + os.sep + filename, delimiter=',', skip_header=0)
        data = np.asarray(data)
        return np.delete(data, 0, 1)
