#!/usr/bin/python2.7
import getopt
import sys
import numpy as np
import multiprocessing as mp
from data_tools.dataset_reader import DatasetReader
from data_tools.dataset import Dataset
from data_tools.normalization import Normalizer
from classifier_tree.balanced_tree import BalancedTree
from classifier_tree.slanting_tree import SlantingTree
from classifier_tree.slanting_dual_tree import SlantingDualTree


def get_result_matrix(tree, patterns, labels, matrix_size):
    result_list = tree.classify_patterns(patterns)
    result_matrix = np.asarray([[0] * (matrix_size[0] + 1)] * (matrix_size[1] + 1))
    for i, r in enumerate(result_list):
        if r == -1:
            r = matrix_size[1]
        result_matrix[int(labels[i])][r] += 1

    return result_matrix


def run_balanced_tree_test(digits_data, letters_data, classifier, classifier_parameters):
    print "BalancedTree tests for classifier " + str(classifier) + " with parameters " + str(classifier_parameters)
    t = BalancedTree(classification_method=(classifier, classifier_parameters), clustering_method=("kmeans", None))
    t.build(digits_data.training_labels, digits_data.training_data)

    training_matrix = get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))
    test_matrix = get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))
    foreign_matrix = get_result_matrix(t, letters_data, [10] * letters_data.shape[0], (10, 10))

    filename = "../Results/balanced_tree " + str(classifier) + "_" + str(classifier_parameters) + ".csv"
    np.savetxt(filename, np.concatenate((training_matrix, test_matrix, foreign_matrix), axis=0), delimiter=',',
               fmt='%i')


def run_slanting_tree_test(digits_data, letters_data, classifier, classifier_parameters):
    print "SlantingTree tests for classifier " + str(classifier) + " with parameters " + str(classifier_parameters)
    t = SlantingTree(classification_method=(classifier, classifier_parameters))
    t.build(digits_data.training_labels, digits_data.training_data)

    training_matrix = get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))
    test_matrix = get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))
    foreign_matrix = get_result_matrix(t, letters_data, [10] * letters_data.shape[0], (10, 10))

    filename = "../Results/slanting_tree " + str(classifier) + "_" + str(classifier_parameters) + ".csv"
    np.savetxt(filename, np.concatenate((training_matrix, test_matrix, foreign_matrix), axis=0), delimiter=',',
               fmt='%i')


def run_slanting_dual_tree_test(digits_data, letters_data, node_classifier, node_classifier_parameters, leaf_classifier,
                                leaf_classifier_parameters):
    # TODO: Add leaf classifier parameter
    print "SlantingDualTree tests for node classifier " + str(node_classifier) + " with parameters " + str(
        node_classifier_parameters) + " and leaf classifier " + str(leaf_classifier) + " with parameters " + str(
        leaf_classifier_parameters)
    t = SlantingDualTree(node_classifier=(node_classifier, node_classifier_parameters),
                         leaf_classifier=(leaf_classifier, leaf_classifier_parameters))
    t.build(digits_data.training_labels, digits_data.training_data)

    training_matrix = get_result_matrix(t, digits_data.training_data, digits_data.training_labels, (10, 10))
    test_matrix = get_result_matrix(t, digits_data.test_data, digits_data.test_labels, (10, 10))
    foreign_matrix = get_result_matrix(t, letters_data, [10] * letters_data.shape[0], (10, 10))

    filename = "../Results/slanting_dual_tree " + str(node_classifier) + "_" + str(
        node_classifier_parameters) + "&" + str(leaf_classifier) + "_" + str(leaf_classifier_parameters) + ".csv"
    np.savetxt(filename, np.concatenate((training_matrix, test_matrix, foreign_matrix), axis=0), delimiter=',',
               fmt='%i')


def _run_balanced_tree_calculations(pool, digits_data, letters_data):
    classifier_parameters_combinations = _get_all_parameters_combinations()
    for classifier, parameters_combinations in classifier_parameters_combinations.iteritems():
        for combination in parameters_combinations:
            pool.apply_async(run_balanced_tree_test, args=(digits_data, letters_data, classifier, dict(combination)))


def _run_slanting_tree_calculations(pool, digits_data, letters_data):
    classifier_parameters_combinations = _get_all_parameters_combinations()
    for classifier, parameters_combinations in classifier_parameters_combinations.iteritems():
        for combination in parameters_combinations:
            pool.apply_async(run_slanting_tree_test, args=(digits_data, letters_data, classifier, dict(combination)))


def _run_slanting_dual_tree_calculations(pool, digits_data, letters_data):
    classifier_parameters_combinations = _get_all_parameters_combinations()
    for classifier, parameters_combinations in classifier_parameters_combinations.iteritems():
        for combination in parameters_combinations:
            for second_classifier, parameters_combinations2 in classifier_parameters_combinations.iteritems():
                for combination2 in parameters_combinations2:
                    pool.apply_async(run_slanting_dual_tree_test,
                                     args=(digits_data, letters_data, classifier, dict(combination),
                                           second_classifier, dict(combination2)))


def _get_all_parameters_combinations():
    classifiers = {
        "svm": {
            'C': [8, 16],
            'kernel': ['rbf', 'poly'],
            'gamma': [pow(2, -1), pow(2, -2), pow(2, -3)]
        },
        "knn": {
            'n_neighbors': [3, 5, 7, 10],
        },
        "rf": {
            'n_estimators': [30, 50, 100, 150]
        }
    }
    combinations = {}

    for classifier in classifiers:
        combinations[classifier] = _get_combinations_list(classifiers[classifier])

    return combinations


def _get_combinations_list(parameters_dictionary, combination=None, all_combinations=None):
    if combination is None: combination = {}
    if all_combinations is None: all_combinations = []

    recursion_level = len(combination)
    if recursion_level == len(parameters_dictionary):
        all_combinations.append(combination.copy())
    else:
        parameter_name = parameters_dictionary.keys()[recursion_level]
        parameter_values = parameters_dictionary[parameter_name]

        for value in parameter_values:
            combination[parameter_name] = value
            all_combinations = _get_combinations_list(parameters_dictionary, combination, all_combinations)
            del combination[parameter_name]

    return all_combinations


if __name__ == "__main__":
    """
    Main program entry function.
    """
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "123", [])
        if len(opts) == 0: raise getopt.GetoptError("No options specified")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)

    reader = DatasetReader("../Datasets")
    raw_data = reader.read_digits(filename='digits.csv')
    data = Dataset(raw_data, division_ratio=0.70)
    normalizer = Normalizer(data.training_data)
    data.training_data = normalizer.get_normalized_data_matrix(data.training_data)
    data.test_data = normalizer.get_normalized_data_matrix(data.test_data)
    raw_data = reader.read_letters()
    raw_data = normalizer.get_normalized_data_matrix(raw_data)

    pool = mp.Pool()
    for o, a in opts:
        if o == "-1":
            _run_balanced_tree_calculations(pool, data, raw_data)
        elif o == "-2":
            _run_slanting_tree_calculations(pool, data, raw_data)
        elif o == "-3":
            _run_slanting_dual_tree_calculations(pool, data, raw_data)

    pool.close()
    pool.join()
