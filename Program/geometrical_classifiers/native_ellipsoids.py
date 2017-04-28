import numpy as np
from models.ellipsoid import MVEE


class NativeEllipsoids:
    """
    Class constructing set of ellipsoids for each native class in the provided training dataset.
    """

    def __init__(self, dataset):
        """
        Creates set of ellipsoids based on training data from data parameter.
        :param dataset: instance of Dataset class with training and test sets
        """
        self.dataset = dataset
        self.ellipsoids = self._create_ellipsoids_for_classes()

    def get_results(self, foreign_elements=None):
        """
        Returns matrix with one row for each ellipsoid and columns corresponding to percent of patterns from some class
        being rejected by this ellipsoid.
        :return: list of lists, which can be easily converted to two dimensional numpy array
        """
        training, test = self.dataset.get_patterns_by_class()
        result_matrix = []
        for ellipsoid in self.ellipsoids:
            training_results = [ellipsoid.calculate_error(training[native_class]) for native_class in training.keys()]
            test_results = [ellipsoid.calculate_error(test[native_class]) for native_class in test.keys()]
            ellipsoid_results = [(float(e1 + e2) / 2.) for e1, e2 in zip(training_results, test_results)]
            if foreign_elements is not None:
                ellipsoid_results.append(ellipsoid.calculate_error(foreign_elements))
            result_matrix.append(ellipsoid_results)
        return result_matrix

    def get_confusion_matrix(self, foreign_elements=None, tolerance=0.001):
        """
        Returns confusion matrix for native and (optionally) foreign elements by traversing array of ellipsoids.
        :param foreign_elements: optional list of elements that should be rejected during classification process
        :param tolerance: float value that impacts ellipsoid tolerance for outliers identification
        :return: confusion matrix in form of list of lists of int values
        """

        def classify_element(pattern, result_array, tol):
            (minimum_distance, best_ellipsoid) = min((ellipsoid.calculate_distance(pattern), i)
                                                     for i, ellipsoid in enumerate(self.ellipsoids))
            result_array[best_ellipsoid if (minimum_distance <= (1. + tol)) else -1] += 1

        confusion_matrix = []
        training, test = self.dataset.get_patterns_by_class()
        for native_class in training.keys():
            native_results = [0] * (len(training.keys()) + 1)
            [classify_element(element, native_results, tolerance) for element in training[native_class]]
            [classify_element(element, native_results, tolerance) for element in test[native_class]]
            confusion_matrix.append(native_results)
        if foreign_elements is not None:
            foreign_results = [0] * (len(training.keys()) + 1)
            [classify_element(element, foreign_results, tolerance) for element in foreign_elements]
            confusion_matrix.append(foreign_results)

        return np.asarray(confusion_matrix, dtype=float)

    def _create_ellipsoids_for_classes(self):
        patterns_by_class, _ = self.dataset.get_patterns_by_class()
        ellipsoids = [MVEE(patterns_by_class[native_class]) for native_class in patterns_by_class.keys()]
        return ellipsoids
