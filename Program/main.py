from data_tools.dataset_reader import DatasetReader
from data_tools.dataset import Dataset
from classifier_tree.balanced_tree import BalancedTree

if __name__ == "__main__":
    """
    Main program entry function.
    """
    reader = DatasetReader("../Datasets")
    raw_data = reader.read_digits(filename='digits.csv')
    data = Dataset(raw_data)
    t = BalancedTree()
    t.build(data.training_labels, data.training_data)
    t.show()

    results = t.classify_patterns(data.training_data)
    correct = sum([1 for index, correct in enumerate(data.training_labels) if correct == results[index]])
    print 'Results for training data: ' + str(float(correct) / len(data.training_labels))

    results = t.classify_patterns(data.test_data)
    correct = sum([1 for index, correct in enumerate(data.test_labels) if correct == results[index]])
    print 'Results for test data: ' + str(float(correct) / len(data.test_labels))
