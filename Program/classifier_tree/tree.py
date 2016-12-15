from abc import abstractmethod


class BinaryTree:
    """
    Abstract binary tree class.
    """

    def __init__(self):
        self._root = None

    @abstractmethod
    def build_tree(self, patterns):
        pass

    @abstractmethod
    def classify_patterns(self, patterns):
        pass

    @abstractmethod
    def _classify_pattern(self, pattern):
        pass


class Node:
    """
    Just a tree node.
    It holds information about other nodes (children) beneath it (left and right fields), classifier used for
    incoming patterns and list of class labels for each child node.
    """

    def __init__(self):
        self.left = self.right = None
        self.classifier = None
        self.classes_left = self.classes_right = []
