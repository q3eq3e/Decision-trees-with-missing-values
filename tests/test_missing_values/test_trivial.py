import math
import pytest

from src.missing_values.trival import best_threshold_trival


def test_returns_minus_inf_when_no_numeric_values():
    dataset = [
        ({"x": None}, "A"),
        ({"x": "?"}, "B"),
        ({"x": float("nan")}, "A"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert gain == -math.inf
    assert t is None
    assert default == "left"


def test_returns_minus_inf_when_only_one_numeric_value():
    dataset = [
        ({"x": 10}, "A"),
        ({"x": None}, "B"),
        ({"x": "?"}, "A"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert gain == -math.inf
    assert t is None
    assert default == "left"


def test_finds_perfect_threshold_without_missing_values():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert gain == pytest.approx(1.0)
    assert t == pytest.approx(2.5)
    assert default in {"left", "right"}


def test_ignores_missing_values_when_computing_threshold():
    dataset = [
        ({"x": None}, "A"),
        ({"x": "?"}, "B"),
        ({"x": float("nan")}, "A"),
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert t == pytest.approx(2.5)
    assert gain > 0


def test_selects_right_route_when_missing_class_matches_right_branch():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
        ({"x": None}, "B"),
        ({"x": "?"}, "B"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert t == pytest.approx(2.5)
    assert default == "right"


def test_selects_left_route_when_missing_class_matches_left_branch():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
        ({"x": None}, "A"),
        ({"x": "?"}, "A"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert t == pytest.approx(2.5)
    assert default == "left"


def test_skips_candidate_when_adjacent_examples_have_same_class():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "A"),
        ({"x": 4}, "B"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    # jedyny sensowny podział między 3 i 4
    assert t == pytest.approx(3.5)
    assert gain > 0


def test_skips_candidate_when_adjacent_values_are_equal():
    dataset = [
        ({"x": 1}, "A"),
        ({"x": 1}, "B"),
        ({"x": 3}, "B"),
        ({"x": 4}, "A"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert t is not None
    assert math.isfinite(gain)


def test_handles_nan_missing_values():
    dataset = [
        ({"x": float("nan")}, "A"),
        ({"x": 1}, "A"),
        ({"x": 2}, "A"),
        ({"x": 3}, "B"),
        ({"x": 4}, "B"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert t == pytest.approx(2.5)
    assert gain > 0


def test_all_numeric_values_identical_produces_no_valid_split():
    dataset = [
        ({"x": 5}, "A"),
        ({"x": 5}, "B"),
        ({"x": 5}, "A"),
        ({"x": 5}, "B"),
    ]

    gain, t, default = best_threshold_trival("x", dataset)

    assert gain == -math.inf
    assert t is None
    assert default == "left"
