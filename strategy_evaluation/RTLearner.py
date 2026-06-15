
import numpy as np
from AbstractTreeLearner import AbstractTreeLearner


class RTLearner(AbstractTreeLearner):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def get_i_and_split_value(self, xs, y):
        """
        @summary: Pick a random i and split value.

        Make sure that not all X are the same for i and also pick
        different values to average the split_value from.
        """
        i = np.random.randint(0, xs.shape[1])
        while np.all(xs[0,i] == xs[:,i]):
            i = np.random.randint(0, xs.shape[1])

        # I don't know about the performance of this, but at least it
        # terminates reliably. If the two elements are the same something is
        # wrong.
        a = np.array(list(set(xs[:, i])))
        r1, r2 = np.random.choice(a, size = 2, replace = False)
        assert(r1 != r2)
        split_value = (r1 + r2) / 2.0
        return i, split_value