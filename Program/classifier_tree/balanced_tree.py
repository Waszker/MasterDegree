import numpy as np
import copy
from Program.models import classifiers
from tree import BinaryTree, Node


class BalancedTree(BinaryTree):
    """
    Tree for classifying and rejecting patterns in the shape of almost-balanced tree.
    """

    def __init__(self, clustering_method=("spectral", None), classification_method=("svm", None)):
        """
        Creates BalancedTree using provided clustering and classification methods.
        Raises a ValueError exception if provided parameters are incorrect.
        :param clustering_method: tuple consisting of string containing name of clustering algorithm
        and its parameters (None for default ones)
        :param classification_method: tuple consisting of string containing name of classification algorithm
        and its parameters (None for default ones)
        """
        BinaryTree.__init__(self)
        self.clustering = classifiers.clustering_object(clustering_method)
        self.classifier = classifiers.classifier_object(classification_method)

    def build(self, labels, patterns):
        # TODO: Finish method description
        """
        Builds tree from provided patterns. Patterns are divided into two potential classes...
        :param labels: list of correct data classes for each pattern array row
        :param patterns: row-ordered numpy array data
        :return:
        """
        class_representation_points = BalancedTree._get_class_representation_points(labels, patterns)
        central_points = {c: p for c, p in enumerate(class_representation_points)}
        self._root = self._create_tree_node(central_points, labels, patterns)

    def show(self):
        """
        Displays tree nodes and their corresponding classes.
        """
        BalancedTree._show_tree_node(self._root, 0)

    def classify_patterns(self, patterns):
        """
        Classifies provided patterns.
        :param patterns: numpy array of row-oriented patters
        :return: row-oriented numpy vector of strings of correct classes ("none" if rejected)
        """
        if self._root is None:
            raise ValueError('Tree has not yet been constructed. Root does not exist.')

        correct_classes = []

        for pattern in patterns:
            correct_classes.append(self._classify_pattern(pattern))

        return np.asarray(correct_classes)

    def _classify_pattern(self, pattern):
        return BalancedTree._classify_traverse(self._root, pattern)

    @staticmethod
    def _classify_traverse(node, pattern):
        classified_as = node.classifier.predict(pattern.reshape(1, -1))
        is_leaf = len(node.classes) == 1
        if is_leaf and classified_as == 1:
            return -1
        elif is_leaf:
            return node.classes[0]
        elif classified_as == 0:
            return BalancedTree._classify_traverse(node.left, pattern)
        else:
            return BalancedTree._classify_traverse(node.right, pattern)

    def _create_tree_node(self, central_points, labels, patterns):
        """
        Creates BalancedTree node.
        :param central_points: dictionary containing class:central_point data
        :param labels: list of correct data classes for each pattern array row
        :param patterns: row-oriented numpy array containing data for classifier training
        :return: created node
        """
        if len(central_points) == 1:
            return self._create_tree_leaf(central_points.keys()[0], labels, patterns)

        node = Node()
        node.classes = central_points.keys()
        node.classifier = copy.deepcopy(self.classifier)
        clusters = self._get_clusters(central_points)
        BalancedTree._fit_node_classifier(node, central_points.keys(), clusters, labels, patterns)
        node.left = self._create_tree_node({k: v for k, v in central_points.iteritems() if k in node.classes_left},
                                           labels, patterns)
        node.right = self._create_tree_node({k: v for k, v in central_points.iteritems() if k in node.classes_right},
                                            labels, patterns)

        return node

    def _create_tree_leaf(self, leaf_class, labels, patterns):
        """
        Creates tree leaf node with classifier trained on leaf_class vs. rest data.
        :param leaf_class: class represented by leaf
        :param labels: list of correct data classes for each pattern array row
        :param patterns: row-oriented numpy array containing data for classifier training
        :return: created node
        """
        node = Node()
        node.classes = [leaf_class]
        node.classifier = copy.deepcopy(self.classifier)
        native_patterns = np.asarray([row for index, row in enumerate(patterns) if labels[index] == leaf_class])
        foreign_patterns = np.asarray([row for index, row in enumerate(patterns) if labels[index] != leaf_class])

        native_count = native_patterns.shape[0]
        foreign_patterns = foreign_patterns[0:native_count, :]
        labels = [0] * native_count
        labels.extend([1] * foreign_patterns.shape[0])

        node.classifier.fit(np.concatenate((native_patterns, foreign_patterns), axis=0), labels)
        return node

    def _get_clusters(self, central_points):
        """
        Returns clusters for provided central class points.
        :param central_points: dictionary containing class:central_point data
        :return: list containing cluster numbers for each data class
        """
        class_count = len(central_points.keys())
        class_representation_points = np.array([row for (_, row) in central_points.iteritems()])
        clusters = [0, 1]

        if class_count > 2:
            self.clustering.n_neighbors = class_count
            clusters = self.clustering.fit_predict(class_representation_points, [0, 1])

        return clusters

    @staticmethod
    def _fit_node_classifier(node, class_numbers, clusters, labels, patterns):
        """
        Trains classifier for provided node, along with filling information about classes in left and right children.
        :param node: node that should have classifier trained
        :param class_numbers: original class labels (as int()) for each cluster result
        :param clusters: result vector for clustering central class points
        :param labels: list of correct data classes for each pattern array row
        :param patterns: row-oriented numpy array containing data for classifier training
        """
        node.classes_left = [class_numbers[c] for c, value in enumerate(clusters) if value == 0]
        node.classes_right = [class_numbers[c] for c, value in enumerate(clusters) if value == 1]
        class_numbers_clusters = {k: 0 for k in node.classes_left}
        class_numbers_clusters.update({k: 1 for k in node.classes_right})
        training_labels, training_data = BalancedTree._get_training_data_with_labels(class_numbers_clusters, labels,
                                                                                     patterns)
        node.classifier.fit(training_data, training_labels)

    @staticmethod
    def _get_training_data_with_labels(class_numbers_clusters, labels, patterns):
        """
        Prepares training data array and labels list for classifier training.
        :param class_numbers_clusters: dictionary with keys being class numbers and values being corresponding
        cluster class
        :param labels: list of correct data classes for each pattern array row
        :param patterns: row-oriented numpy array containing data for classifier training
        :return: training_labels and training_data ready to feed them to classifier fit method
        """
        training_labels = []
        training_data = []
        for index, row in enumerate(patterns):
            pattern_class = labels[index]
            if pattern_class not in class_numbers_clusters.keys():
                continue

            cluster_class = class_numbers_clusters[pattern_class]
            training_labels.append(cluster_class)
            training_data.append(row)

        return training_labels, np.asarray(training_data)

    @staticmethod
    def _get_class_representation_points(labels, patterns):
        """
        Calculates "central class points" from provided patterns. It helps in clustering to divide certain classes.
        :param labels: list of correct data classes for each pattern array row
        :param patterns: row-oriented numpy array containing data for classifier training
        :return: row-oriented numpy array containing "central class point" for each class
        """
        classes = set(labels)
        patterns_by_class = {key: value for key, value in zip(classes, [[] for _ in classes])}
        representation_points = []

        for i, pattern in enumerate(patterns):
            patterns_by_class[labels[i]].append(pattern)

        for key, patterns_in_class in patterns_by_class.iteritems():
            patterns_array = np.asarray(patterns_in_class)
            elements_in_class_count = float(patterns_array.shape[0])
            columns_count = patterns_array.shape[1]
            central_class_point = patterns_array.sum(axis=0) / [elements_in_class_count] * columns_count
            representation_points.append(central_class_point)

        return np.asarray(representation_points)

    @staticmethod
    def _show_tree_node(node, level):
        if node is not None:
            print str(level) + ": (" + str(node.classes) + ")"
            BalancedTree._show_tree_node(node.left, level + 1)
            BalancedTree._show_tree_node(node.right, level + 1)
