import numpy as np


class AbstractTreeLearner:
    LEAF = -1
    NA = -1

    def author(self):
        return 'felixm' # replace tb34 with your Georgia Tech username

    def create_node(self, factor, split_value, left, right):
        return np.array([(factor, split_value, left, right), ],
                        dtype='|i4, f4, i4,  i4')

    def query_point(self, point):
        node_index = 0
        while self.rel_tree[node_index][0] != self.LEAF:
            node = self.rel_tree[node_index]
            split_factor = node[0]
            split_value = node[1]
            if point[split_factor] <= split_value:
                # Recurse into left sub-tree.
                node_index += node[2]
            else:
                node_index += node[3]
        v = self.rel_tree[node_index][1]
        return v

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        query_point = lambda p: self.query_point(p)
        r = np.apply_along_axis(query_point, 1, points)
        return r

    def build_tree(self, xs, y):
        """
        @summary: Build a decision tree from the training data.
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        assert(xs.shape[0] == y.shape[0])
        assert(xs.shape[0] > 0) # If this is 0 something went wrong.

        if xs.shape[0] <= self.leaf_size:
            value = np.mean(y)
            if value < -0.2:
                value = -1
            elif value > 0.2:
                value = 1
            else:
                value = 0
            return self.create_node(self.LEAF, value, self.NA, self.NA)

        if np.all(y[0] == y):
            return self.create_node(self.LEAF, y[0], self.NA, self.NA)

        i, split_value = self.get_i_and_split_value(xs, y)
        select_l = xs[:, i] <= split_value
        select_r = xs[:, i] > split_value
        lt = self.build_tree(xs[select_l], y[select_l])
        rt = self.build_tree(xs[select_r], y[select_r])
        root = self.create_node(i, split_value, 1, lt.shape[0] + 1)
        root = np.concatenate([root, lt, rt])
        return root

    def addEvidence(self, data_x, data_y):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.rel_tree = self.build_tree(data_x, data_y)

