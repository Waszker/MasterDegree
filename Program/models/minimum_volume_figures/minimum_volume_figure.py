from abc import abstractmethod
import numpy as np


class MinimumVolumeFigure:
    """
    Abstract class for minimum volume figure implementation.
    """

    def __init__(self):
        pass

    def calculate_error(self, points, tolerance=0.001):
        """
        Calculates collective error (outliers rate) for provided points set.
        :param points: row-ordered points matrix
        :param tolerance: tolerance for point distance value
        :return: float number with error rate
        """
        points = np.asarray(points)
        counter = [(1 if self.calculate_distance(point) > (1. + tolerance) else 0) for point in points]
        return float(sum(counter)) / len(points)

    @abstractmethod
    def calculate_distance(self, point):
        """
        Calculates point distance in regards to figure center.
        :param point: vector of point coordinates in figure's n-space
        :return: float value denoting distance to figure center (1.0 if point lies on figure surface)
        """
        pass
