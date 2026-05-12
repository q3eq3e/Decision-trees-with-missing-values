import numpy as np
from .entropy import entropy


def information_gain(parent_y, left_y, right_y):
    if len(left_y) == 0 or len(right_y) == 0:
        return -np.inf
    H = entropy(parent_y)
    return (
        H
        - (len(left_y) / len(parent_y)) * entropy(left_y)
        - (len(right_y) / len(parent_y)) * entropy(right_y)
    )


def best_threshold(X, y, col):
    data = X[[col]].copy()
    data["y"] = y
    data = data.dropna()
    data = data.sort_values(col)

    best_gain = -np.inf
    best_t = None

    values = data[col].values
    classes = data["y"].values

    for i in range(len(values) - 1):
        if classes[i] != classes[i + 1]:
            t = (values[i] + values[i + 1]) / 2
            left = y[X[col] <= t]
            right = y[X[col] > t]
            gain = information_gain(y, left, right)

            if gain > best_gain:
                best_gain, best_t = gain, t

    return best_gain, best_t


def best_split_discrete(X, y, col):
    values = X[col].dropna().unique()
    best_gain = -np.inf
    best_subset = None

    from itertools import combinations

    for i in range(1, len(values)):
        for subset in combinations(values, i):
            subset = set(subset)
            left = y[X[col].isin(subset)]
            right = y[~X[col].isin(subset)]
            gain = information_gain(y, left, right)
            if gain > best_gain:
                best_gain, best_subset = gain, subset

    return best_gain, best_subset
