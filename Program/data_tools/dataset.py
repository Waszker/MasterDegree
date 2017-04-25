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
        row_count = data.shape[0]
        training_data = data[:int(row_count * division_ratio), :]
        test_data = data[int(row_count * division_ratio):, :]
        self.training_labels = [row[0] for row in training_data]
        self.test_labels = [row[0] for row in test_data]
        self.training_data = np.delete(training_data, 0, 1)
        self.test_data = np.delete(test_data, 0, 1)

    def get_patterns_by_class(self):
        """
        Returns two dictionaries with keys being the class labels and the values being list of patterns from this class.
        :return: two dictionaries for training and test set
        """
        native_classes = set(self.training_labels)
        training = {k: [p for i, p in enumerate(self.training_data) if self.training_labels[i] == k] for k in
                    native_classes}
        test = {k: [p for i, p in enumerate(self.test_data) if self.test_labels[i] == k] for k in native_classes}

        return training, test

    def get_classes_distribution(self):
        """
        Calculates classes distribution for each data set. Evenly distributed classes help in training and testing
        models.
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

    @staticmethod
    def calculate_central_points(patterns_by_class):
        """
        Calculates central points for each class in provided data.
        Central points are the average of all points in certain class.
        :param patterns_by_class: dictionary containing list of patterns for each class number key
        :return: dictionary containing central points for each class
        """

        def get_division_vector(c): return [len(patterns_by_class[c])] * len(patterns_by_class[c][0])

        def get_central_point(c): return np.asarray(patterns_by_class[c]).sum(axis=0) / get_division_vector(c)

        return {i: get_central_point(i) for i in patterns_by_class.keys()}
