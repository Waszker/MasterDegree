import enum
import numpy as np
from native_figures import NativeFigures
from models.minimum_volume_figures.minimum_volume_figure import MinimumVolumeFigure
from models.minimum_volume_figures.ellipsoid import MVEE


class ShrinkingOption(enum.Enum):
    TOLERANCE_MANIPULATION = 1
    ELEMENTS_REJECTION = 2


class ShrinkingFigures(NativeFigures):
    """
    Minimum volume figure used in testing purposes that shrinks its size in order to increase
    rejection option rate by sacrificing classification one.
    """

    def __init__(self, dataset, foreign_elements, minimum_volume_figure_class=MVEE):
        """
        Creates initial figures for each class.
        :param dataset: instance of Dataset class with training and test sets
        """
        NativeFigures.__init__(self, dataset, minimum_volume_figure_class)
        self.foreign_elements = foreign_elements
        self.figure_tolerance = 0.001

    def perform_tests(self, steps=10, shrinking_option=ShrinkingOption.TOLERANCE_MANIPULATION, filename="partial.csv"):
        """
        Performs rejection and classification ratios tests on decreasing figure.
        :param steps: number of shrinking steps to make
        :param shrinking_option: ShrinkingOption enumerator defining shrinking algorithm
        :param filename: name of the file to which partial results will be appended
        """
        original_figure_tolerance = self.figure_tolerance
        if shrinking_option is ShrinkingOption.TOLERANCE_MANIPULATION: self.figure_tolerance = 1.
        original_figures = self.figures
        training = self.dataset.get_patterns_by_class()[0]
        filename = "../Results/%s" % filename

        for step in xrange(1, steps + 1):
            print "Performing step %i of %i" % (step, steps)
            matrix = self.get_confusion_matrix(self.foreign_elements, self.figure_tolerance)
            self.figures = [self._shrink_figure(training[i], figure, shrinking_option, step=step)
                            for i, figure in enumerate(self.figures)]
            if shrinking_option is ShrinkingOption.TOLERANCE_MANIPULATION:
                self.figure_tolerance -= 0.02
            with open(filename, 'a') as f:
                [f.write("%s\n" % ','.join(map(str, row))) for row in matrix]
                f.write("\n")

        self.figure_tolerance = original_figure_tolerance
        self.figures = original_figures

    def _get_ratios(self, native_class, native_elements, figure):
        tolerance = self.figure_tolerance
        others = [patterns for i, class_elements in enumerate(native_elements) if i != native_class
                  for patterns in class_elements]
        classification_rate = 1.0 - figure.calculate_error(np.asmatrix(native_elements[native_class]), tolerance)
        native_sensitivity = 1.0 - figure.calculate_error(np.asmatrix(others), tolerance)
        rejection_rate = figure.calculate_error(np.asmatrix(self.foreign_elements), self.figure_tolerance)

        return classification_rate, native_sensitivity, rejection_rate

    def _get_ratios_from_matrix(self, confusion_matrix):
        matrix = confusion_matrix
        first_test_row = (len(matrix) - 1) / 2
        correctly_classified_training = sum([matrix[i][i] for i in xrange(first_test_row)])
        correctly_classified_test = sum([matrix[first_test_row + i][i] for i in xrange(first_test_row)])
        correctly_native_training = sum(
            [matrix[i][j] for i in xrange(first_test_row) for j in xrange(len(matrix[i]) - 1)])
        correctly_native_test = sum(
            [matrix[first_test_row + i][j] for i in xrange(first_test_row) for j in xrange(len(matrix[i]) - 1)])
        correctly_rejected = matrix[-1][-1]
        all_native_training = len(self.dataset.training_data)
        all_native_test = len(self.dataset.test_data)
        all_foreign = len(self.foreign_elements)

        classification_training = float(correctly_classified_training) / all_native_training
        classification_test = float(correctly_classified_test) / all_native_test
        identification_training = float(correctly_native_training) / all_native_training
        identification_test = float(correctly_native_test) / all_native_test
        rejection = float(correctly_rejected) / all_foreign

        return classification_training, identification_training, rejection, classification_test, identification_test, rejection

    def _shrink_figure(self, training, figure, shrinking_option, step=1):
        new_figure = figure
        if shrinking_option is ShrinkingOption.TOLERANCE_MANIPULATION:
            pass
        elif shrinking_option is ShrinkingOption.ELEMENTS_REJECTION:
            elements_removed = 5
            sorted_patterns = sorted(training, key=figure.calculate_distances)
            new_figure = self.figure_class(sorted_patterns[:-1 * step * elements_removed])
        else:
            raise TypeError("Unsupported shrinking_option value")

        return new_figure
