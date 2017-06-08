import numpy as np
from math import sqrt
from minimum_volume_figure import MinimumVolumeFigure


class HyperRectangle(MinimumVolumeFigure):
    def __init__(self, points):
        """
        Constructs hyper rectangle using provided points.
        :param points: list of lists (denoting points in n-dimensional space)
        """
        MinimumVolumeFigure.__init__(self)
        self._calculate_bounds(points)

    def calculate_distance(self, point):
        """
        Calculates point distance in regards to hyper rectangle center.
        :param point: vector of point coordinates in hyper rectangle n-space
        :return: float value denoting distance to hyper rectangle center (1.0 if point lies on hyper rectangle surface)
        """
        distances = [(abs(coord - center) / width) if width > 0 else None
                     for coord, center, width in zip(point, self.centres, self.widths)]
        return max(distances)

    def calculate_distances(self, point):
        """
        Calculates point distance in regards to hyper rectangle center.
        :param point: vector of point coordinates in hyper rectangle n-space
        :return: float value denoting distance to hyper rectangle center (1.0 if point lies on hyper rectangle surface)
        """
        return self.calculate_distance(point)

    def _calculate_bounds(self, points):
        self.left_bounds = np.amin(points, axis=0).tolist()
        self.right_bounds = np.amax(points, axis=0).tolist()
        self.widths = [(float(right - left) / 2.) for left, right in zip(self.left_bounds, self.right_bounds)]
        self.centres = [(left + float(right - left) / 2.) for left, right in zip(self.left_bounds, self.right_bounds)]
