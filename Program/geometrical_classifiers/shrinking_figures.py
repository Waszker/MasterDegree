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
        figures = self.figures
        results = [[] for _ in xrange(len(figures))]
        training, test = self._create_native_lists()
        natives = list(training)
        [natives[i].extend(test[i]) for i in xrange(len(training))]

        for step in xrange(1, steps+1):
            [results[i].append(self._get_ratios(i, natives, figure)) for i, figure in enumerate(figures)]
            figures = [self._shrink_figure(training[i], figure, shrinking_option, step=step)
                       for i, figure in enumerate(figures)]
            if shrinking_option is self.ShrinkingOption.TOLERANCE_MANIPULATION:
                self.figure_tolerance -= 0.02

        self.figure_tolerance = original_figure_tolerance

        return results

    def _create_native_lists(self):
        training, test = self.dataset.get_patterns_by_class()
        natives_training, natives_test = [training[k] for k in training.keys()], [test[k] for k in test.keys()]
        return natives_training, natives_test

    def _get_ratios(self, native_class, native_elements, figure):
        tolerance = self.figure_tolerance
        others = [patterns for i, class_elements in enumerate(native_elements) if i != native_class
                  for patterns in class_elements]
        classification_rate = 1.0 - figure.calculate_error(np.asmatrix(native_elements[native_class]), tolerance)
        native_sensitivity = 1.0 - figure.calculate_error(np.asmatrix(others), tolerance)
        rejection_rate = figure.calculate_error(np.asmatrix(self.foreign_elements), self.figure_tolerance)

        return classification_rate, native_sensitivity, rejection_rate

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
