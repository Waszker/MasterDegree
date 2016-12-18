import copy
import numpy as np
from balanced_tree import BinaryTree, Node
from Program.models import classifiers


class SlantingTree(BinaryTree):
    """
    Tree for classifying and rejecting patterns in the shape of a slanted tree.
    """

    def __init__(self, classification_method=("svm", None)):
        """
        Creates SlantingTree with provided classifier.
        Raises a ValueError exception if provided parameters are incorrect.
        :param classification_method: tuple consisting of string containing name of classification algorithm
        and its parameters (None for default ones)
        """
        BinaryTree.__init__(self)
        self.classifier = classifiers.classifier_object(classification_method)

    def build(self, labels, patterns):
        # TODO: Add method description
        classes = set(int(l) for l in labels)
        patterns_by_class = {k: [p for i, p in enumerate(patterns) if labels[i] == k] for k in classes}
        self._root = self._create_tree_node(list(classes), patterns_by_class)

    def show(self):
        """
        Displays tree nodes and their corresponding classes.
        """
        if self._root is None:
            raise ValueError('Tree has not yet been constructed. Root does not exist.')

        node = self._root
        while node.right is not None:
            print '[' + str(node.right.classes) + '] -> ' + str(node.right.left.classes)
            node = node.left
        print '[' + str(node.classes) + ']'

    def classify_patterns(self, patterns):
        """
        Classifies provided patterns.
        :param patterns: numpy array of row-oriented patters
        :return: row-oriented numpy vector of strings of correct classes ("none" if rejected)
        """
        if self._root is None:
            raise ValueError('Tree has not yet been constructed. Root does not exist.')

        return [SlantingTree._classify_pattern(self._root, p) for p in patterns]

    def _create_tree_node(self, classes, patterns_by_class):
        if len(classes) == 1:
            return self._create_tree_leaf(classes[0], None, patterns_by_class)
        node = Node()
        node.classes = classes
        node.classes_left = classes[1:]
        node.classes_right = classes[0]
        node.left = self._create_tree_node(node.classes_left, patterns_by_class)
        node.right = self._create_tree_leaf(node.classes_right, node.left, patterns_by_class)
        self._train_classifier(node, node.classes_right, patterns_by_class)
        return node

    def _create_tree_leaf(self, class_number, failure_node, patterns_by_class):
        node = Node()
        node.classes = class_number
        node.left = failure_node
        node.right = None
        self._train_classifier(node, node.classes, patterns_by_class)

        return node

    def _train_classifier(self, node, class_number, patterns_by_class):
        node.classifier = copy.deepcopy(self.classifier)
        native_class_samples_count = len(patterns_by_class[class_number])
        foreign_class_samples_count = native_class_samples_count / (len(patterns_by_class) - 1)
        native_samples = np.asarray(patterns_by_class[class_number])
        foreign_samples = np.asarray(sum(
            [patterns_by_class[i][:foreign_class_samples_count] for i in patterns_by_class.keys() if i != class_number],
            []))
        labels = sum([[1] * native_class_samples_count, [0] * foreign_samples.shape[0]], [])
        node.classifier.fit(np.concatenate((native_samples, foreign_samples), axis=0), labels)

    @staticmethod
    def _classify_pattern(node, pattern):
        classified_as = node.classifier.predict(pattern.reshape(1, -1))
        if classified_as == 1 and node.right is None:
            return node.classes
        elif classified_as == 0 and node.left is None:
            return -1

        return SlantingTree._classify_pattern(node.left if classified_as == 0 else node.right, pattern)
