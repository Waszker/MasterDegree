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
from classifier_tree.slanting_ordered_tree import SlantingOrderedTree
from geometrical_classifiers.native_figures import NativeFigures
from geometrical_classifiers.shrinking_figures import ShrinkingFigures, ShrinkingOption
from models.minimum_volume_figures.hyper_rectangle import HyperRectangle
from models.minimum_volume_figures.ellipsoid import MVEE


def get_result_matrix(tree, patterns, labels, matrix_size):
    result_list = tree.classify_patterns(patterns)
    result_matrix = np.asarray([[0] * (matrix_size[0] + 1)] * (matrix_size[1] + 1))
    for i, r in enumerate(result_list):
        if r == -1:
            r = matrix_size[1]
        result_matrix[int(labels[i])][r] += 1

    return result_matrix


def _get_all_parameters_combinations():
    classifiers = {
        "svm": {
            'C': [8, 16],
            'kernel': ['rbf', 'poly'],
            'gamma': ['auto', pow(2, -4), pow(2, -1), pow(2, -2), pow(2, -3)],
            'tol': [1e-3, 1e-5]
        },
        "knn": {
            'n_neighbors': [3, 5, 7, 10],
        },
        "rf": {
            'n_estimators': [30, 50, 100, 150]
        }
    }
    combinations = {classifier: _get_combinations_list(classifiers[classifier]) for classifier in classifiers}
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


def _calculate_matrix(tree, digits_data, letters_data):
    tree.build(digits_data.training_labels, digits_data.training_data)
    training_matrix = get_result_matrix(tree, digits_data.training_data, digits_data.training_labels, (10, 10))
    test_matrix = get_result_matrix(tree, digits_data.test_data, digits_data.test_labels, (10, 10))
    foreign_matrix = get_result_matrix(tree, letters_data, [10] * letters_data.shape[0], (10, 10))
    return np.concatenate((training_matrix, test_matrix, foreign_matrix), axis=0)


def _run_test(tree_builder, datasets, arguments):
    (digits_data, letters_data) = datasets
    classifier_tree = tree_builder(*arguments)
    print "Starting test for %s with arguments %s" % (str(classifier_tree.get_name()), str(arguments))
    result = _calculate_matrix(classifier_tree, digits_data, letters_data)
    filename = "../Results/%s_%s.csv" % (str(classifier_tree.get_name()), str(arguments))
    np.savetxt(filename, result, delimiter=',', fmt='%i')


def _run_parallel_calculations(tree_builder, digits_data, letters_data):
    datasets = (digits_data, letters_data)
    classifier_parameters_combinations = _get_all_parameters_combinations()
    classifiers = [(classifier, parameters)
                   for classifier, combinations in classifier_parameters_combinations.iteritems()
                   for parameters in combinations]
    [pool.apply_async(_run_test, args=(tree_builder, datasets, classifier)) for classifier in classifiers]


def _run_parallel_calculations2(tree_builder, digits_data, letters_data):
    datasets = (digits_data, letters_data)
    classifier_parameters_combinations = _get_all_parameters_combinations()
    classifiers = [(classifier, parameters)
                   for classifier, combinations in classifier_parameters_combinations.iteritems()
                   for parameters in combinations]
    [pool.apply_async(_run_test, args=(tree_builder, datasets, (classifier1, classifier2)))
     for classifier1 in classifiers for classifier2 in classifiers]


def _run_minimum_figure_calculations(figure_class, shrinking_option, datasets, datasets_name):
    print "Running calculations for " + str(figure_class)
    figures = ShrinkingFigures(*datasets, minimum_volume_figure_class=figure_class)
    fname = "%s_shrinking_%s_%s" % (str(shrinking_option),
                                    'rectangles' if figure_class is HyperRectangle else 'ellipsoids', datasets_name)
    figures.perform_tests(steps=100, shrinking_option=shrinking_option, filename=fname)


if __name__ == "__main__":
    """
    Main program entry function.
    """
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hi:n12345678", ["help", "input="])
        if len(opts) == 0: raise getopt.GetoptError("No options specified")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(1)

    if ('-h', '') in opts:
        print "Pattern recognition program help:"
        print "-1: BalancedTree\n-2: SlantingTree\n-3: SlantingDualTree\n-4: SlantingOrderedTree\n-5:NativeEllipsoids" \
              "\n-6:NativeEllipsoids (confusion matrix)\n-7:ShrinkingEllipsoids (tolerance manipulation)" \
              "\n-8:ShrinkingEllipsoids (element rejection)"
        sys.exit(0)

    turn_normalization_off = ('-n', '') in opts
    data_file = "digits"

    for o, a in opts:
        if o in ('-i', '--input'):
            data_file = a

    print "Reading data from %s.csv" % data_file
    reader = DatasetReader("../Datasets")
    raw_data = reader.read_digits(filename="%s_native.csv" % data_file)
    digits = Dataset(raw_data, division_ratio=0.70)
    letters = reader.read_letters(filename="%s_foreign.csv" % data_file)

    if not turn_normalization_off:
        print "Normalizing data before usage"
        normalizer = Normalizer(digits.training_data)
        digits.training_data = normalizer.get_normalized_data_matrix(digits.training_data)
        digits.test_data = normalizer.get_normalized_data_matrix(digits.test_data)
        letters = normalizer.get_normalized_data_matrix(letters)


    # @formatter:off
    def balanced_builder(x, y): return BalancedTree(classification_method=(x, y), clustering_method=("kmeans", None))
    def slanting_builder(x, y): return SlantingTree(classification_method=(x, y))
    def slanting2_builder(x, y): return SlantingDualTree(node_classifier=x, leaf_classifier=y)
    def slanting3_builder(x, y): return SlantingOrderedTree(classification_method=(x, y))
    # @formatter:on

    pool = mp.Pool()
    for o, a in opts:
        if o == "-1":
            _run_parallel_calculations(balanced_builder, digits, letters)
        elif o == "-2":
            _run_parallel_calculations(slanting_builder, digits, letters)
        elif o == "-3":
            _run_parallel_calculations2(slanting2_builder, digits, letters)
        elif o == "-4":
            _run_parallel_calculations(slanting3_builder, digits, letters)
        elif o == "-5":
            ellipsoids = NativeFigures(digits)
            results = np.asarray(ellipsoids.get_results(letters), dtype=float)
            filename = "../Results/native_ellipsoids.csv"
            np.savetxt(filename, results, delimiter=',', fmt='%f')
        elif o == "-6":
            ellipsoids = NativeFigures(digits, minimum_volume_figure_class=MVEE)
            matrix = ellipsoids.get_confusion_matrix(letters, tolerance=0.001)
            filename = "../Results/native_ellipsoid2.csv"
            np.savetxt(filename, matrix, delimiter=',', fmt='%i')
        elif o == "-7" or o == "-8":
            figure_classes = [MVEE, HyperRectangle]
            shrinking_options = [ShrinkingOption.TOLERANCE_MANIPULATION, ShrinkingOption.ELEMENTS_REJECTION]
            [pool.apply_async(_run_minimum_figure_calculations,
                              args=(figure_class, shrinking_option, (digits, letters), data_file))
             for figure_class in figure_classes for shrinking_option in shrinking_options]

    pool.close()
    pool.join()
