import numpy as np
import numpy.linalg as la


class Ellipsoid:
    """
    Class representing minimum volume enclosing ellipsoid classifier.
    """

    def __init__(self, points, tolerance=0.001):
        """
        Constructs ellipsoid using provided points.
        :param points: list of lists (denoting points in n-dimensional space)
        :param tolerance: stop parameter for ellipsoid construction algorithm
        """
        self.a, self.c = self._mvee(points, tolerance)

    def calculate_error(self, points, tolerance=0.001):
        """
        Calculates collective error (outliers rate) for provided points set.
        :param points: rwo-ordered points matrix
        :param tolerance: tolerance for point distance value
        :return: float number with error rate
        """
        counter = [(1 if self.calculate_distance(point) > (1. + tolerance) else 0) for point in points]
        return float(sum(counter)) / len(points)

    def calculate_distance(self, point):
        """
        Calculates point distance in regards to ellipsoid center.
        :param point: vector of point coordinates in ellipsoid n-space
        :return: float value denoting distance to ellipsoid center (1.0 if point lies on ellipsoid surface)
        """
        point = np.asmatrix(point) - self.c
        return point * self.a * np.transpose(point)

    def _mvee(self, points, tolerance):
        # Taken from: http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
        points = np.asmatrix(points)
        n, d = points.shape
        q = np.column_stack((points, np.ones(n))).T
        err = tolerance + 1.0
        u = np.ones(n) / n
        while err > tolerance:
            # assert u.sum() == 1 # invariant
            x = q * np.diag(u) * q.T
            m = np.diag(q.T * la.inv(x) * q)
            jdx = np.argmax(m)
            step_size = (m[jdx] - d - 1.0) / ((d + 1) * (m[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
        c = u * points
        a = la.inv(points.T * np.diag(u) * points - c.T * c) / d
        return np.asarray(a), np.squeeze(np.asarray(c))


class MVEE(Ellipsoid):
    """
    The same class as Ellipsoid. This is just another name.
    """
