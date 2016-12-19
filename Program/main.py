import numpy as np
from data_tools.dataset_reader import DatasetReader
from data_tools.dataset import Dataset
from data_tools.normalization import Normalizer
from classifier_tree.balanced_tree import BalancedTree
from classifier_tree.slanting_tree import SlantingTree


def get_result_matrix(tree, patterns, labels, matrix_size):
    result_list = tree.classify_patterns(patterns)
    result_matrix = np.asarray([[0] * (matrix_size[0] + 1)] * (matrix_size[1] + 1))
    for i, r in enumerate(result_list):
        if r == -1:
            r = matrix_size[1]
        result_matrix[labels[i]][r] += 1

    return result_matrix


def run_balanced_tree_test(digits_data, letters):
    t = BalancedTree(classification_method=("svm", None), clustering_method=("kmeans", None))
    t.build(digits_data.training_labels, digits_data.training_data)
    t.show()

    print str(get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))) + str("\n\n")
    print str(get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))) + str("\n\n")
    print get_result_matrix(t, letters, [10] * letters.shape[0], (10, 10))


def run_slanting_tree_test(digits_data, letters):
    t = SlantingTree(classification_method=("svm", None))
    t.build(digits_data.training_labels, digits_data.training_data)
    t.show()

    print str(get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))) + str("\n\n")
    print str(get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))) + str("\n\n")
    print get_result_matrix(t, letters, [10] * letters.shape[0], (10, 10))

if __name__ == "__main__":
    """
    Main program entry function.
    """
    reader = DatasetReader("../Datasets")
    raw_data = reader.read_digits(filename='digits.csv')
    data = Dataset(raw_data, division_ratio=0.70)
    normalizer = Normalizer(data.training_data)
    data.training_data = normalizer.get_normalized_data_matrix(data.training_data)
    data.test_data = normalizer.get_normalized_data_matrix(data.test_data)
    raw_data = reader.read_letters()
    raw_data = normalizer.get_normalized_data_matrix(raw_data)

    run_balanced_tree_test(data, raw_data)
    run_slanting_tree_test(data, raw_data)

