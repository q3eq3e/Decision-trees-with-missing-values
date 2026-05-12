import numpy as np
from .node import Node
from .splits import best_threshold, best_split_discrete


class DecisionTreeID3:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def majority(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=self.max_depth)

    def _build_tree(self, X, y, depth):
        node = Node()
        node.majority_class = self.majority(y)

        if len(np.unique(y)) == 1 or depth == 0:
            node.is_leaf = True
            node.prediction = self.majority(y)
            return node

        best_gain = -np.inf
        best_attr = None
        best_type = None
        best_val = None

        for col in X.columns:
            if np.issubdtype(X[col].dropna().dtype, np.number):
                gain, t = best_threshold(X, y, col)
                if gain > best_gain:
                    best_gain, best_attr, best_type, best_val = gain, col, "cont", t
            else:
                gain, split = best_split_discrete(X, y, col)
                if gain > best_gain:
                    best_gain, best_attr, best_type, best_val = gain, col, "disc", split

        if best_gain == -np.inf:
            node.is_leaf = True
            node.prediction = self.majority(y)
            return node

        node.attribute = best_attr

        if best_type == "cont":
            node.threshold = best_val
            left_idx = X[best_attr] <= best_val
        else:
            node.split = best_val
            left_idx = X[best_attr].isin(best_val)

        right_idx = ~left_idx

        node.left = self._build_tree(X[left_idx], y[left_idx], depth - 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth - 1)

        node.default_route = "left" if left_idx.sum() >= right_idx.sum() else "right"
        return node
