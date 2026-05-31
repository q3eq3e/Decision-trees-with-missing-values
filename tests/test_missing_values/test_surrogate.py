from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from src.missing_values.surrogate import (
    SurrogateEntry,
    agreement_measure,
    best_surrogate_split,
    best_surrogate_threshold,
    surrogate_split,
    surrogate_split_predict,
)


def test_agreement_measure_returns_expected_overlap():
    left = pd.Index([0, 1])
    left_surrogate = pd.Index([0, 2])
    right = pd.Index([2, 3])
    right_surrogate = pd.Index([1, 3])

    agreement = agreement_measure(left, left_surrogate, right, right_surrogate)

    assert agreement == pytest.approx(0.5)


def test_agreement_measure_returns_zero_when_no_overlap():
    left = pd.Index([0])
    left_surrogate = pd.Index([2])
    right = pd.Index([1])
    right_surrogate = pd.Index([3])

    agreement = agreement_measure(left, left_surrogate, right, right_surrogate)

    assert agreement == pytest.approx(0.0)


def test_best_surrogate_threshold_finds_perfect_continuous_split():
    U = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}, index=[0, 1, 2, 3])

    agreement, threshold, direction = best_surrogate_threshold(
        "feature",
        U,
        pd.Index([0, 1]),
        pd.Index([2, 3]),
    )

    assert agreement == pytest.approx(1.0)
    assert threshold == pytest.approx(1.5)
    assert direction == "left"


def test_best_surrogate_threshold_uses_reverse_direction_when_needed():
    U = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}, index=[0, 1, 2, 3])

    agreement, threshold, direction = best_surrogate_threshold(
        "feature",
        U,
        pd.Index([2, 3]),
        pd.Index([0, 1]),
    )

    assert agreement == pytest.approx(1.0)
    assert threshold == pytest.approx(1.5)
    assert direction == "right"


def test_best_surrogate_split_finds_exact_discrete_subset():
    U = pd.DataFrame(
        {"color": ["red", "red", "blue", "green"]},
        index=[0, 1, 2, 3],
    )
    values = sorted(U["color"].dropna().unique().tolist())

    agreement, split = best_surrogate_split(
        "color",
        U,
        pd.Index([0, 1]),
        pd.Index([2, 3]),
        frozenset(),
        0,
        values,
    )

    assert agreement == pytest.approx(1.0)
    assert split == frozenset({"red"})


def test_surrogate_split_returns_continuous_and_discrete_surrogates():
    U = pd.DataFrame(
        {
            "color": ["red", "red", "blue", "blue"],
            "feature": [10.0, 11.0, 20.0, 21.0],
            "primary": [0.1, 0.2, 5.0, 6.0],
        },
        index=[0, 1, 2, 3],
    )

    entries = surrogate_split(
        U,
        D=["color"],
        C=["feature"],
        best_attr="primary",
        U_left_idx=pd.Index([0, 1]),
        U_right_idx=pd.Index([2, 3]),
        size=2,
    )

    assert len(entries) == 2
    assert {entry.attribute for entry in entries} == {"color", "feature"}
    assert any(entry.split_values is not None for entry in entries)
    assert any(entry.threshold is not None for entry in entries)
    assert all(entry.agreement == pytest.approx(1.0) for entry in entries)


def test_surrogate_split_predict_uses_surrogate_when_primary_is_missing():
    node = SimpleNamespace(
        left="left_leaf",
        right="right_leaf",
        surrogate_splits=[
            SurrogateEntry(agreement=1.0, attribute="color", split_values={"red"})
        ],
        default_route="left",
        prediction=0,
        is_leaf=False,
    )

    result = surrogate_split_predict(node, pd.Series({"color": "red"}))

    assert result == "left_leaf"


def test_surrogate_split_predict_falls_back_to_default_route_when_missing():
    node = SimpleNamespace(
        left="left_leaf",
        right="right_leaf",
        surrogate_splits=[
            SurrogateEntry(agreement=1.0, attribute="color", split_values={"red"})
        ],
        default_route="right",
        prediction=0,
        is_leaf=False,
    )

    result = surrogate_split_predict(node, pd.Series({"color": np.nan}))

    assert result == "right_leaf"
