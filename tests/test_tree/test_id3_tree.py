import numpy as np
import pandas as pd
import pytest

from src.tree.id3_tree import (
    DecisionTree,
    MissingStrategy,
    best_split,
    best_threshold,
)


def test_best_threshold_returns_midpoint_and_left_default():
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    dataset = list(zip(X.to_dict(orient="records"), y.to_numpy()))

    gain, threshold, default_route = best_threshold(
        "feature",
        dataset,
        MissingStrategy.MAJORITY,
    )

    assert gain == pytest.approx(1.0)
    assert threshold == pytest.approx(1.5)
    assert default_route == "left"


def test_best_split_returns_pure_partition():
    X = pd.DataFrame({"color": ["red", "red", "blue", "blue"]})
    y = pd.Series([0, 0, 1, 1])
    dataset = list(zip(X.to_dict(orient="records"), y.to_numpy()))

    gain, split = best_split("color", dataset, MissingStrategy.MAJORITY)

    assert gain == pytest.approx(1.0)
    assert split is not None

    left_labels = {label for (row, label) in dataset if row["color"] in split}
    right_labels = {label for (row, label) in dataset if row["color"] not in split}

    assert len(left_labels) == 1
    assert len(right_labels) == 1


def test_decision_tree_uses_surrogate_for_missing_values():
    X = pd.DataFrame(
        {
            "color": ["red", "red", "blue", "blue"],
            "feature": [0.0, 0.1, 2.0, 2.1],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    tree = DecisionTree(
        discrete_attrs=["color"],
        continuous_attrs=["feature"],
        strategy=MissingStrategy.SURROGATE,
        max_surrogate_splits=5,
    )
    tree.fit(X, y)

    assert tree.root.condition_attr == "color"
    assert tree.root.surrogate_splits
    assert {entry.attribute for entry in tree.root.surrogate_splits} == {"feature"}

    prediction = tree.predict_one(pd.Series({"color": np.nan, "feature": 0.0}))

    assert prediction == 0
