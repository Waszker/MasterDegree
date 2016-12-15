import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
        self.clustering = BalancedTree._get_clustering_object(clustering_method)
        self.classifier = BalancedTree._get_classifier_object(classification_method)

    def build_tree(self, patterns):
        """
        TODO: Finish method description
        Builds tree from provided patterns. Patterns are divided into two potential classes...
        :param patterns:
        :return:
        """
        # TODO: Implement that!
        self._root = Node()
        return self.clustering.fit_predict(patterns, ['0', '1'])

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
        # TODO: Implement that!
        return 'none'

    @staticmethod
    def _get_clustering_object((clustering_method, parameters)):
        try:
            return {
                'spectral': BalancedTree._get_spectral(parameters),
            }[clustering_method]
        except KeyError:
            raise ValueError('Provided clustering method name \'' + str(clustering_method) + '\' not recognized.')

    @staticmethod
    def _get_classifier_object((classification_method, parameters)):
        try:
            return {
                'svm': BalancedTree._get_svm(parameters),
                'rf': BalancedTree._get_rf(parameters),
                'knn': BalancedTree._get_knn(parameters),
            }[classification_method]
        except KeyError:
            raise ValueError('Provided classifier name \'' + str(classification_method) + '\' not recognized.')

    @staticmethod
    def _get_spectral(parameters):
        if parameters is None:
            parameters = {
                'n_clusters': 2
            }
        return spectral_clustering(**parameters)

    @staticmethod
    def _get_svm(parameters):
        if parameters is None:
            parameters = {
                'C': 8,
                'kernel': 'rbf',
                'gamma': 0.5
            }
        return svm.SVC(**parameters)

    @staticmethod
    def _get_rf(parameters):
        if parameters is None:
            parameters = {
                'n_estimators': 100,
            }
        return RandomForestClassifier(**parameters)

    @staticmethod
    def _get_knn(parameters):
        if parameters is None:
            parameters = {
                'n_neighbors': 5,
            }
        return KNeighborsClassifier(**parameters)
