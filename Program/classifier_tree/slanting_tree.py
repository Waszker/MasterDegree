import copy
import numpy as np
from random import randint
from tree import BinaryTree, Node
from models import classifiers


class SlantingTree(BinaryTree):
    """
    Tree for classifying and rejecting patterns in the shape of a slanted tree.
    """

    def __init__(self, classification_method=("svm", None), point_generation_method=1):
        """
        Creates SlantingTree with provided classifier.
        Raises a ValueError exception if provided parameters are incorrect.
        :param classification_method: tuple consisting of string containing name of classification algorithm
        and its parameters (None for default ones)
        """
        BinaryTree.__init__(self)
        self.classifier = classifiers.classifier_object(classification_method)
        self._point_generation_method = point_generation_method

    def get_name(self):
        return "SlantingTree"

    def build(self, labels, patterns):
        # TODO: Finish method description
        """
        Builds tree and trains classifiers in each node.
        :param labels:
        :param patterns:
        :return:
        """
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
        while node.right is not None and node.right.left is not None:
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
        node = Node()
        node.classes = classes
        node.classes_right = classes[0]
        node.classes_left = classes[1:]
        self._train_classifier(node, node.classes_right, patterns_by_class)
        node.left = self._create_tree_node(node.classes_left, patterns_by_class) if len(classes) > 1 else None
        node.right = self._create_tree_leaf(node.classes_right, node, patterns_by_class)

        return node

    def _create_tree_leaf(self, class_number, parent_node, patterns_by_class):
        node = Node()
        node.classes = class_number
        node.left = parent_node.left
        self._train_classifier_for_leaf(node, node.classes, parent_node, patterns_by_class)

        return node

    def _train_classifier(self, node, class_number, patterns_by_class):
        node.classifier = copy.deepcopy(self.classifier)
        native_samples, foreign_samples, labels = SlantingTree._create_training_datasets_and_labels(class_number,
                                                                                                    patterns_by_class)
        node.classifier.fit(np.concatenate((native_samples, foreign_samples), axis=0), labels)

    def _train_classifier_for_leaf(self, node, class_number, parent_node, patterns_by_class):
        parent_classifier = parent_node.classifier
        node.classifier = copy.deepcopy(self.classifier)
        native_samples, foreign_samples, labels = SlantingTree._create_training_datasets_and_labels(class_number,
                                                                                                    patterns_by_class)
        dataset = np.concatenate((native_samples, foreign_samples), axis=0)
        results = parent_classifier.predict(dataset)
        native_samples, foreign_samples, labels = self._create_additional_training_points(dataset, labels, results)
        node.classifier.fit(np.concatenate((native_samples, foreign_samples), axis=0), labels)

    def _create_additional_training_points(self, dataset, labels, results):
        # TODO: Check which method should be used when creating new points
        native_samples, foreign_samples = SlantingTree._get_seed_samples(dataset, labels, results)
        native_samples, foreign_samples = SlantingTree._generate_new_samples(native_samples, foreign_samples,
                                                                             sum([1 for l in labels if l == 1]),
                                                                             self._point_generation_method)
        native_samples_count, foreign_samples_count = len(native_samples), len(foreign_samples)
        native_samples, foreign_samples = np.asarray(native_samples), np.asarray(foreign_samples)
        labels = sum([[1] * native_samples_count, [0] * foreign_samples_count], [])

        return native_samples, foreign_samples, labels

    @staticmethod
    def _create_training_datasets_and_labels(class_number, patterns_by_class):
        """
        Creates equal training native and foreign data sets and their corresponding labels.
        :param class_number: native class number
        :param patterns_by_class: dictionary of all patterns with class number as a key
        :return: native and foreign patterns classes with corresponding labels list
        """
        native_class_samples_count = len(patterns_by_class[class_number])
        foreign_class_samples_count = native_class_samples_count / (len(patterns_by_class) - 1)
        native_samples = np.asarray(patterns_by_class[class_number])
        foreign_samples = np.asarray(sum(
            [patterns_by_class[i][:foreign_class_samples_count] for i in patterns_by_class.keys() if i != class_number],
            []))
        labels = sum([[1] * native_class_samples_count, [0] * foreign_samples.shape[0]], [])

        return native_samples, foreign_samples, labels

    @staticmethod
    def _get_seed_samples(dataset, labels, results):
        """
        Chooses those patterns that can be used in artificial patterns generation.
        :param dataset: set of data used for previous classifier prediction
        :param labels: proper classes for each pattern
        :param results: classes recognized by classifier
        :return: native and foreign patterns that can be used for generation new samples
        """
        native_samples, foreign_samples = [], []
        for i, p in enumerate(dataset):
            if results[i] == 0:
                continue
            elif labels[i] == 1:
                native_samples.append(p)
            else:
                foreign_samples.append(p)

        if len(native_samples) <= 10:
            native_samples = [p for i, p in enumerate(dataset) if labels[i] == 1]
            native_samples = native_samples[:len(native_samples) / 2]
        if len(foreign_samples) <= 10:
            foreign_samples = [p for i, p in enumerate(dataset) if labels[i] == 0]
            foreign_samples = foreign_samples[:len(foreign_samples) / 2]

        return native_samples, foreign_samples

    @staticmethod
    def _generate_new_samples(native_samples, foreign_samples, proper_class_count, method_number):
        """
        Creates new, artificial patterns from correctly classified native ones and wrongly classified foreign ones.
        Both samples lists must not be empty!
        There are few approaches towards this problem:
            1) For randomly selected pattern apply gaussian distortion vector
        :param native_samples: patterns from native class that were correctly classified
        :param foreign_samples: patterns from foreign class that were wrongly classified
        :param proper_class_count: number of patterns in each class needed for classifier training
        :param method_number: point generation method identifier
        :return: newly created classes
        """
        native_count, foreign_count = len(native_samples), len(foreign_samples)

        for i in range(native_count, proper_class_count):
            native_samples.append(SlantingTree._apply_gaussian_distortion(native_samples[randint(0, native_count - 1)]))
        for i in range(foreign_count, proper_class_count):
            foreign_samples.append(
                SlantingTree._apply_gaussian_distortion(foreign_samples[randint(0, foreign_count - 1)]))

        return native_samples, foreign_samples

    @staticmethod
    def _apply_gaussian_distortion(pattern):
        distortion = np.random.normal(loc=0.0, scale=1.0, size=pattern.shape[0])
        return pattern + distortion

    @staticmethod
    def _classify_pattern(node, pattern):
        classified_as = node.classifier.predict(pattern.reshape(1, -1))
        if classified_as == 1 and node.right is None:
            return node.classes
        elif classified_as == 0 and node.left is None:
            return -1

        return SlantingTree._classify_pattern(node.left if classified_as == 0 else node.right, pattern)
