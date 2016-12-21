import numpy as np
import multiprocessing as mp
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


def run_balanced_tree_test(digits_data, letters_data, classifier, classifier_parameters):
    t = BalancedTree(classification_method=(classifier, classifier_parameters), clustering_method=("kmeans", None))
    t.build(digits_data.training_labels, digits_data.training_data)

    training_matrix = get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))
    test_matrix = get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))
    foreign_matrix = get_result_matrix(t, letters_data, [10] * letters_data.shape[0], (10, 10))

    filename = "../Results/balanced_tree " + str(classifier) + "_" + str(classifier_parameters) + ".csv"
    np.savetxt(filename, np.concatenate((training_matrix, test_matrix, foreign_matrix), axis=0), delimiter=',')


def run_slanting_tree_test(digits_data, letters_data, classifier, classifier_parameters):
    t = SlantingTree(classification_method=(classifier, classifier_parameters))
    t.build(digits_data.training_labels, digits_data.training_data)

    training_matrix = get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))
    test_matrix = get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))
    foreign_matrix = get_result_matrix(t, letters_data, [10] * letters_data.shape[0], (10, 10))

    filename = "../Results/slanting_tree " + str(classifier) + "_" + str(classifier_parameters) + ".csv"
    np.savetxt(filename, np.concatenate((training_matrix, test_matrix, foreign_matrix), axis=0), delimiter=',')


def run_tests(digits_data, letters_data):
    classifiers = {
        "svm": {
            'C': [8, 16],
            'kernel': ['rbf', 'poly'],
            'gamma': [pow(2, -1), pow(2, -2), pow(2, -3)]
        },
        "knn": {
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        "rf": {
            'n_estimators': [30, 50, 100, 150]
        }
    }
    pool = mp.Pool()

    for classifier, parameters_dict in classifiers.iteritems():
        traverse_parameters_run_test(classifier, parameters_dict, {}, digits_data, letters_data)
        pool.apply_async(traverse_parameters_run_test,
                         args=(classifier, parameters_dict, {}, digits_data, letters_data))
    pool.close()
    pool.join()


def traverse_parameters_run_test(classifier, parameters_dict, parameters, digits_data, letters_data):
    current_level = len(parameters)

    if current_level == len(parameters_dict):
        print "Running tests for classifier " + str(classifier) + " with parameters " + str(parameters)
        run_balanced_tree_test(digits_data, letters_data, classifier, parameters)
        run_slanting_tree_test(digits_data, letters_data, classifier, parameters)
        return

    parameter_name = parameters_dict.keys()[current_level]
    parameter_values = parameters_dict[parameter_name]

    for value in parameter_values:
        parameters[parameter_name] = value
        traverse_parameters_run_test(classifier, parameters_dict, parameters, digits_data, letters_data)
        del parameters[parameter_name]


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

    run_tests(data, raw_data)
