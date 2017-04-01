import numpy as np
import copy
from models import classifiers
from slanting_tree import SlantingTree


class SlantingDualTree(SlantingTree):
    """
    Tree for classifying and rejecting patterns with similar structure to SlantingTree but using two different
    classifiers on different tree levels. The first-order classifier is used in tree nodes whereas second-order one
    is kept in tree leafs.
    """

    def __init__(self, node_classifier=("svm", None), leaf_classifier=("rf", None)):
        SlantingTree.__init__(self, classification_method=node_classifier)
        self._node_classifier = self.classifier
        self._leaf_classifier = classifiers.classifier_object(leaf_classifier)

    def get_name(self):
        return "SlantingDualTree"

    def _train_classifier_for_leaf(self, node, class_number, parent_node, patterns_by_class):
        node.classifier = copy.deepcopy(self._leaf_classifier)
        native_samples, foreign_samples, labels = SlantingTree._create_training_datasets_and_labels(class_number,
                                                                                                    patterns_by_class)
        dataset = np.concatenate((native_samples, foreign_samples), axis=0)
        node.classifier.fit(dataset, labels)
