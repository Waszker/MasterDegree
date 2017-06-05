import numpy as np
from models.minimum_volume_figures.minimum_volume_figure import MinimumVolumeFigure
from models.minimum_volume_figures.ellipsoid import MVEE


class NativeFigures:
    """
    Class constructing set of minimum volume figures for each native class in the provided training dataset.
    """

    def __init__(self, dataset, minimum_volume_figure_class=MVEE):
        """
        Creates set of figures based on training data from data parameter.
        :param dataset: instance of Dataset class with training and test sets
        """
        self.dataset = dataset
        self.figure_class = minimum_volume_figure_class
        self.figures = self._create_figures_for_classes()

    def get_results(self, foreign_elements=None):
        """
        Returns matrix with one row for each figure and columns corresponding to percent of patterns from some class
        being rejected by this figure.
        :return: list of lists, which can be easily converted to two dimensional numpy array
        """
        training, test = self.dataset.get_patterns_by_class()
        result_matrix = []
        for figure in self.figures:
            training_results = [figure.calculate_error(training[native_class]) for native_class in training.keys()]
            test_results = [figure.calculate_error(test[native_class]) for native_class in test.keys()]
            figures_results = [(float(e1 + e2) / 2.) for e1, e2 in zip(training_results, test_results)]
            if foreign_elements is not None:
                figures_results.append(figure.calculate_error(foreign_elements))
            result_matrix.append(figures_results)
        return result_matrix

    def get_confusion_matrix(self, foreign_elements=None, tolerance=0.001):
        """
        Returns confusion matrix for native and (optionally) foreign elements by traversing array of figures.
        :param foreign_elements: optional list of elements that should be rejected during classification process
        :param tolerance: float value that impacts figure tolerance for outliers identification
        :return: confusion matrix in form of list of lists of int values
        """

        def classify_element(pattern, result_array, tol):
            (minimum_distance, best_figure) = min((figure.calculate_distance(pattern), i)
                                                  for i, figure in enumerate(self.figures))
            result_array[best_figure if (minimum_distance <= (1. + tol)) else -1] += 1

        confusion_matrix = []
        confusion_matrix2 = []
        training, test = self.dataset.get_patterns_by_class()
        for native_class in training.keys():
            native_results = [0] * (len(training.keys()) + 1)
            native_results2 = [0] * (len(test.keys()) + 1)
            [classify_element(element, native_results, tolerance) for element in training[native_class]]
            [classify_element(element, native_results2, tolerance) for element in test[native_class]]
            confusion_matrix.append(native_results)
            confusion_matrix2.append(native_results2)
        confusion_matrix.extend(confusion_matrix2)
        if foreign_elements is not None:
            foreign_results = [0] * (len(training.keys()) + 1)
            [classify_element(element, foreign_results, tolerance) for element in foreign_elements]
            confusion_matrix.append(foreign_results)

        return np.asarray(confusion_matrix, dtype=float)

    def _create_figures_for_classes(self):
        patterns_by_class, _ = self.dataset.get_patterns_by_class()
        figure_class = self.figure_class
        figures = [figure_class(patterns_by_class[native_class]) for native_class in patterns_by_class.keys()]
        return figures
