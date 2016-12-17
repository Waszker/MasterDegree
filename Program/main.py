import numpy as np
from data_tools.dataset_reader import DatasetReader
from data_tools.dataset import Dataset
from classifier_tree.balanced_tree import BalancedTree


def get_result_matrix(tree, patterns, labels, matrix_size):
    result_list = tree.classify_patterns(patterns)
    result_matrix = np.asarray([[0] * (matrix_size[0] + 1)] * (matrix_size[1] + 1))
    for i, r in enumerate(result_list):
        if r == -1:
            r = matrix_size[1]
        result_matrix[labels[i]][r] += 1

    return result_matrix


if __name__ == "__main__":
    """
    Main program entry function.
    """
    reader = DatasetReader("../Datasets")
    raw_data = reader.read_digits(filename='digits.csv')
    data = Dataset(raw_data, division_ratio=0.70)
    raw_data = reader.read_letters()

    t = BalancedTree(classification_method=("svm", None), clustering_method=("kmeans", None))
    t.build(data.training_labels, data.training_data)
    t.show()

    print str(get_result_matrix(t, data.training_data, data.training_labels, (10, 10))) + str("\n\n")
    print str(get_result_matrix(t, data.test_data, data.test_labels, (10, 10))) + str("\n\n")
    print get_result_matrix(t, raw_data, [10] * raw_data.shape[0], (10, 10))
