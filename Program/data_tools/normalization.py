import numpy as np


class Normalizer:
    """
    Class responsible for normalizing data sets and storing values for future normalizations.
    """

    def __init__(self, data):
        self.min_vector, self.max_vector = Normalizer._get_min_max_vectors(data)
        self.difference_vector = self.max_vector - self.min_vector

    def get_normalized_data_matrix(self, data):
        """
        Normalizes data to [0, 1] values. Deletes columns that are always zero.
        :param data: numpy array, row-oriented data
        :return: normalized data without 'zero columns'
        """
        new_data = np.asarray(data)

        for row in new_data:
            for i in range(0, len(self.difference_vector)):
                if self.difference_vector[i] == 0:
                    row[i] = 0
                else:
                    row[i] = float(row[i] - self.min_vector[i]) / self.difference_vector[i]

        return self._delete_zero_columns(np.asarray(new_data))

    @staticmethod
    def _get_min_max_vectors(data):
        min_vector = np.amin(data, axis=0)
        max_vector = np.amax(data, axis=0)

        return min_vector, max_vector

    def _delete_zero_columns(self, data):
        difference = np.tile(np.asarray(self.difference_vector), (2, 1))
        zero_columns = np.nonzero(difference.sum(axis=0) == 0)
        return np.delete(data, zero_columns, axis=1)
