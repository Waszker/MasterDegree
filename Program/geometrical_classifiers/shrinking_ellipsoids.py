import enum
import numpy as np
from models.ellipsoid import MVEE
from native_ellipsoids import NativeEllipsoids


class ShrinkingEllipsoid(NativeEllipsoids):
    """
    Ellipsoid used in testing purposes that shrinks its size in order to increase
    rejection option rate by sacrificing classification one.
    """

    class ShrinkingOption(enum.Enum):
        TOLERANCE_MANIPULATION = 1
        ELEMENTS_REJECTION = 2

    def __init__(self, dataset, foreign_elements):
        """
        Creates initial ellipsoids for each class.
        :param dataset: instance of Dataset class with training and test sets
        """
        NativeEllipsoids.__init__(self, dataset)
        self.foreign_elements = foreign_elements
        self.ellipsoid_tolerance = 1e-3

    def perform_tests(self, steps=10, shrinking_option=ShrinkingOption.TOLERANCE_MANIPULATION):
        """
        Performs rejection and classification ratios tests on decreasing ellipsoid.
        :param steps: number of shrinking steps to make
        :param shrinking_option: ShrinkingOption enumerator defining shrinking algorithm
        :return: list of tuples containing strict native and foreign accuracies
        """
        original_ellipsoid_tolerance = self.ellipsoid_tolerance
        ellipsoids = self.ellipsoids
        results = [[] for _ in xrange(len(ellipsoids))]
        training, test = self._create_native_lists()
        natives = list(training)
        [natives[i].extend(test[i]) for i in xrange(len(training))]

        for step in xrange(steps):
            [results[i].append(self._get_ratios(natives[i], ellipsoid)) for i, ellipsoid in enumerate(ellipsoids)]
            ellipsoids = [self._shrink_ellipsoid(training[i], ellipsoid, shrinking_option, step=step)
                          for i, ellipsoid in enumerate(ellipsoids)]

        self.ellipsoid_tolerance = original_ellipsoid_tolerance

        return results

    def _create_native_lists(self):
        training, test = self.dataset.get_patterns_by_class()
        natives_training, natives_test = [training[k] for k in training.keys()], [test[k] for k in test.keys()]
        return natives_training, natives_test

    def _get_ratios(self, native_elements, ellipsoid):
        classification_rate = 1.0 - ellipsoid.calculate_error(np.asmatrix(native_elements), self.ellipsoid_tolerance)
        rejection_rate = ellipsoid.calculate_error(np.asmatrix(self.foreign_elements), self.ellipsoid_tolerance)

        return classification_rate, rejection_rate

    def _shrink_ellipsoid(self, training, ellipsoid, shrinking_option, step=1):
        new_ellipsoid = ellipsoid
        if shrinking_option is self.ShrinkingOption.TOLERANCE_MANIPULATION:
            self.ellipsoid_tolerance /= 2.
        elif shrinking_option is self.ShrinkingOption.ELEMENTS_REJECTION:
            elements_removed = 5
            distances = [ellipsoid.calculate_distance(pattern) for pattern in training]
            too_big_distance = sorted(list(distances))[-1 * (elements_removed * step)]
            new_ellipsoid = MVEE([pattern for i, pattern in enumerate(training) if distances[i] < too_big_distance])
        else:
            raise TypeError("Unsupported shrinking_option value")

        return new_ellipsoid
