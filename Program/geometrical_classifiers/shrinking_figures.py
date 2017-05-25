import enum
import numpy as np
from native_figures import NativeFigures
from models.minimum_volume_figures.minimum_volume_figure import MinimumVolumeFigure
from models.minimum_volume_figures.ellipsoid import MVEE


class ShrinkingFigures(NativeFigures):
    """
    Minimum volume figure used in testing purposes that shrinks its size in order to increase
    rejection option rate by sacrificing classification one.
    """

    class ShrinkingOption(enum.Enum):
        TOLERANCE_MANIPULATION = 1
        ELEMENTS_REJECTION = 2

    def __init__(self, dataset, foreign_elements, minimum_volume_figure_class=MVEE):
        """
        Creates initial figures for each class.
        :param dataset: instance of Dataset class with training and test sets
        """
        NativeFigures.__init__(self, dataset, minimum_volume_figure_class)
        self.foreign_elements = foreign_elements
        self.figure_tolerance = 1.

    def perform_tests(self, steps=10, shrinking_option=ShrinkingOption.TOLERANCE_MANIPULATION):
        """
        Performs rejection and classification ratios tests on decreasing figure.
        :param steps: number of shrinking steps to make
        :param shrinking_option: ShrinkingOption enumerator defining shrinking algorithm
        :return: list of tuples containing strict native and foreign accuracies
        """
        original_figure_tolerance = self.figure_tolerance
        original_figures = self.figures
        results = []
        training = self.dataset.get_patterns_by_class()[0]
        filename = "../Results/%s_shrinking_ellipsoids.csv" % "rejection"
        for step in xrange(1, steps + 1):
            print "Performed step %i of %i" % (step, steps)
            matrix = self.get_confusion_matrix(self.foreign_elements, self.figure_tolerance)
            results.append(self._get_ratios_from_matrix(matrix))
            self.figures = [self._shrink_figure(training[i], figure, shrinking_option, step=step)
                            for i, figure in enumerate(self.figures)]
            if shrinking_option is self.ShrinkingOption.TOLERANCE_MANIPULATION:
                self.figure_tolerance -= 0.02
            with open(filename, 'a') as f:
                f.write("%f, %f, %f\n" % tuple(results[-1]))

        self.figure_tolerance = original_figure_tolerance
        self.figures = original_figures

        return results

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
        correctly_classified = sum([matrix[i][i] for i in xrange(len(matrix) - 1)])
        correctly_native = sum([matrix[i][j] for i in xrange(len(matrix) - 1) for j in xrange(len(matrix[i]) - 1)])
        correctly_rejected = matrix[-1][-1]
        all_native = len(self.dataset.training_data) + len(self.dataset.test_data)
        all_foreign = len(self.foreign_elements)

        classification = float(correctly_classified) / all_native
        identification = float(correctly_native) / all_native
        rejection = float(correctly_rejected) / all_foreign

        return classification, identification, rejection

    def _shrink_figure(self, training, figure, shrinking_option, step=1):
        new_figure = figure
        if shrinking_option is self.ShrinkingOption.TOLERANCE_MANIPULATION:
            pass
        elif shrinking_option is self.ShrinkingOption.ELEMENTS_REJECTION:
            elements_removed = 5
            distances = [figure.calculate_distance(pattern) for pattern in training]
            too_big_distance = sorted(distances)[-1 * (elements_removed * step)]
            new_figure = MVEE([pattern for i, pattern in enumerate(training) if distances[i] < too_big_distance])
        else:
            raise TypeError("Unsupported shrinking_option value")

        return new_figure
