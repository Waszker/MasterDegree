import numpy as np
from classifier_tree.balanced_tree import BalancedTree

if __name__ == "__main__":
    """
    Main program entry function.
    """
    t = BalancedTree()
    t.build_tree(None)
    t.classify_patterns(np.asarray([[3, 4, 5], [3, 4, 4]]))