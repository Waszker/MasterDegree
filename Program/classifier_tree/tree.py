from abc import abstractmethod


class BinaryTree:
    """
    Abstract binary tree class.
    """

    def __init__(self):
        self._root = None

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def build(self, labels, patterns):
        pass

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def classify_patterns(self, patterns):
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
        self.classes = self.classes_left = self.classes_right = []
