import numpy as np


class Dataset:
    """
    Class storing training and test data.
    """

    def __init__(self, data, division_ratio=0.7):
        """
        Created training and test data sets from provided data.
        :param data: numpy array containing row-oriented data with first column being class labels
        :param division_ratio: training to test data division ratio
        """
        self.training_labels = []
        self.test_labels = []

        row_count = data.shape[0]
        training_data = data[:int(row_count * division_ratio), :]
        test_data = data[int(row_count * division_ratio):, :]

        for row in training_data:
            self.training_labels.append(row[0])
        for row in test_data:
            self.test_labels.append(row[0])

        self.training_data = np.delete(training_data, 0, 1)
        self.test_data = np.delete(test_data, 0, 1)

    def get_classes_distribution(self):
        """
        Calculates classes distribution for each data set. Evenly distributed classes help in training and testing
        classifiers.
        :return: array for training data and array for test data
        """
        training_classes_count = np.amax(self.training_labels) - np.amin(self.training_labels) + 1
        test_classes_count = np.amax(self.test_labels) - np.amin(self.test_labels) + 1
        training_classes_counter = np.array([0.] * training_classes_count)
        test_classes_counter = np.array([0.] * test_classes_count)

        for label in self.training_labels:
            training_classes_counter[label] += 1
        for label in self.test_labels:
            test_classes_counter[label] += 1

        training_classes_distribution = training_classes_counter / len(self.training_labels)
        test_classes_distribution = test_classes_counter / len(self.test_labels)

        return training_classes_distribution, test_classes_distribution
