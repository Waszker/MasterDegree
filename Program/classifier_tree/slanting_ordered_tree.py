from slanting_tree import SlantingTree
from models.ellipsoid import MVEE
from data_tools.dataset import Dataset


class SlantingOrderedTree(SlantingTree):
    """
    Classifier tree with ordered classes path. It uses ellipsoids for class ordering.
    Instead of using arbitrary ordering of classes as in SlantingTree class, this time MVEE is used for determining
    class order. Creation of minimum volume enclosing ellipsoid is done for "class representatives points".
    Next, the point lying farthest from ellipsoid center gets removed and the whole procedure is repeated.
    This is done until the last class remains. Rejected classes are stored in the same order they were rejected.
    """

    def build(self, labels, patterns):
        """
        Builds tree and trains classifiers in each node.
        :param labels: class labels for each provided pattern
        :param patterns: patterns used for classifier training
        """
        classes = set(int(l) for l in labels)
        patterns_by_class = {k: [p for i, p in enumerate(patterns) if labels[i] == k] for k in classes}
        ordered_classes = SlantingOrderedTree._get_sorted_classes(classes, patterns_by_class)
        self._root = self._create_tree_node(ordered_classes, patterns_by_class)

    def get_name(self):
        return "SlantingOrderedTree"

    @staticmethod
    def _get_sorted_classes(classes, patterns_by_class):
        labels = [label for label in classes]
        central_points = Dataset.calculate_central_points(patterns_by_class)
        central_points = [central_points[i] for i in classes]
        ordered_classes = []

        # Order classes using ellipsoid identifier
        while len(central_points) > 1:
            mvee = MVEE(central_points)
            errors = [mvee.calculate_distance(point) for point in central_points]
            worst_index = errors.index(max(errors))
            ordered_classes.append(labels[worst_index])
            del central_points[worst_index]
            del labels[worst_index]
        ordered_classes.extend(labels)

        return ordered_classes
