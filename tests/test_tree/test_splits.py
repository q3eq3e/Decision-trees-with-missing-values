import numpy as np
import pandas as pd
import pytest
from src.tree.splits import best_split_discrete, best_threshold, information_gain


def test_information_gain_returns_minus_inf_for_empty_split():
    parent = np.array([0, 1, 1])

    gain = information_gain(parent, np.array([]), np.array([0, 1, 1]))

    assert np.isneginf(gain)


def test_information_gain_finds_perfect_split():
    parent = np.array([0, 0, 1, 1])
    left = np.array([0, 0])
    right = np.array([1, 1])

    gain = information_gain(parent, left, right)

    assert gain == pytest.approx(1.0)


def test_best_threshold_returns_best_midpoint():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 0, 1, 1])

    gain, threshold = best_threshold(X, y, "feature")

    assert gain == pytest.approx(1.0)
    assert threshold == pytest.approx(2.5)


def test_best_split_discrete_prefers_pure_subset():
    X = pd.DataFrame({"color": ["red", "red", "blue", "green"]})
    y = pd.Series([0, 0, 1, 1])

    gain, subset = best_split_discrete(X, y, "color")

    assert gain == pytest.approx(1.0)
    assert subset == {"red"}
