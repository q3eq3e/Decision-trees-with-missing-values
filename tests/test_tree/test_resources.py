## authors: Jakub Bagiński, Maciej Borkowski

import math
import pandas as pd
import pytest

from src.tree.resources import is_missing, entropy, majority


def test_is_missing_for_none():
    assert is_missing(None) is True


def test_is_missing_for_nan():
    assert is_missing(float("nan")) is True


def test_is_missing_for_regular_values():
    assert is_missing(0) is False
    assert is_missing("") is False
    assert is_missing("abc") is False
    assert is_missing(False) is False


def test_entropy_empty_dataset():
    assert entropy([]) == 0.0


def test_entropy_single_class_is_zero():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "A"),
    ]
    assert entropy(dataset) == pytest.approx(0.0)


def test_entropy_balanced_binary_classes():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
    ]

    # H = 1 bit
    assert entropy(dataset) == pytest.approx(1.0)


def test_entropy_three_classes():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "C"),
    ]

    expected = -(0.5 * math.log2(0.5) + 0.25 * math.log2(0.25) + 0.25 * math.log2(0.25))

    assert entropy(dataset) == pytest.approx(expected)


def test_majority_returns_most_frequent_class():
    y = pd.Series(["A", "B", "A", "A", "C"])
    assert majority(y) == "A"


def test_majority_single_class():
    y = pd.Series(["YES"] * 5)
    assert majority(y) == "YES"


def test_continuous_split_produces_correct_partition():
    attr = "age"
    t = 30

    sorted_u = [
        ({"age": 20}, "N"),
        ({"age": 30}, "Y"),
        ({"age": 31}, "Y"),
        ({"age": 50}, "N"),
    ]

    left = [(x, y) for x, y in sorted_u if x[attr] <= t]
    right = [(x, y) for x, y in sorted_u if x[attr] > t]

    assert len(left) == 2
    assert len(right) == 2

    assert all(x["age"] <= t for x, _ in left)
    assert all(x["age"] > t for x, _ in right)


def test_information_gain_perfect_split():
    attr = "x"
    t = 2.5

    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
    ]

    n = len(dataset)
    base_ent = entropy(dataset)

    left = [(x, y) for x, y in dataset if x[attr] <= t]
    right = [(x, y) for x, y in dataset if x[attr] > t]

    gain = (
        base_ent - (len(left) / n) * entropy(left) - (len(right) / n) * entropy(right)
    )

    assert base_ent == pytest.approx(1.0)
    assert entropy(left) == pytest.approx(0.0)
    assert entropy(right) == pytest.approx(0.0)
    assert gain == pytest.approx(1.0)


def test_missing_detection_used_by_tree_builder():
    attr = "age"

    dataset = [
        ({"age": None}, "A"),
        ({"age": float("nan")}, "B"),
        ({"age": "?"}, "C"),
        ({"age": 20}, "D"),
    ]

    u_missing = [
        (x, y) for x, y in dataset if is_missing(x.get(attr)) or x.get(attr) == "?"
    ]

    assert len(u_missing) == 3
    assert [y for _, y in u_missing] == ["A", "B", "C"]


def test_all_same_true_when_every_attribute_constant():
    X = pd.DataFrame(
        {
            "disc": ["A", "A", "A"],
            "cont": [5, 5, 5],
        }
    )

    D = ["disc"]
    C = ["cont"]

    all_same = all(
        len({x.get(a) for x in X.to_dict(orient="records")}) <= 1 for a in D + C
    )

    assert all_same is True


def test_all_same_false_when_any_attribute_varies():
    X = pd.DataFrame(
        {
            "disc": ["A", "B", "A"],
            "cont": [5, 5, 5],
        }
    )

    D = ["disc"]
    C = ["cont"]

    all_same = all(
        len({x.get(a) for x in X.to_dict(orient="records")}) <= 1 for a in D + C
    )

    assert all_same is False


def test_discrete_binary_subset_split_example():
    subset = {"red", "blue"}
    attr = "color"

    dataset = [
        ({"color": "red"}, "A"),
        ({"color": "green"}, "B"),
        ({"color": "blue"}, "A"),
        ({"color": "yellow"}, "B"),
    ]

    left = [(x, y) for x, y in dataset if x[attr] in subset]
    right = [(x, y) for x, y in dataset if x[attr] not in subset]

    assert {x["color"] for x, _ in left} == {"red", "blue"}
    assert {x["color"] for x, _ in right} == {"green", "yellow"}
