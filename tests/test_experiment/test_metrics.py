import numpy as np
import pytest
import warnings
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.experiment.metrics import compute_metrics


def test_perfect_classification():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)

    np.testing.assert_array_equal(metrics["confusion"], np.array([[2, 0], [0, 2]]))


def test_completely_wrong_binary_classification():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["B", "B", "A", "A"]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)

    np.testing.assert_array_equal(metrics["confusion"], np.array([[0, 2], [2, 0]]))


def test_metrics_match_sklearn_reference():
    y_true = ["A", "A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "B", "A"]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))

    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted"))

    np.testing.assert_array_equal(
        metrics["confusion"], confusion_matrix(y_true, y_pred)
    )


def test_multiclass_classification():
    y_true = ["A", "B", "C", "A", "B", "C"]
    y_pred = ["A", "B", "A", "A", "C", "C"]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))

    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted"))

    np.testing.assert_array_equal(
        metrics["confusion"], confusion_matrix(y_true, y_pred)
    )


def test_single_class_prediction():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        y_true = ["A", "A", "A", "A"]
        y_pred = ["A", "A", "A", "A"]

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

        np.testing.assert_array_equal(metrics["confusion"], np.array([[4]]))


def test_returned_dictionary_has_expected_keys():
    metrics = compute_metrics(["A", "B"], ["A", "B"])

    assert set(metrics.keys()) == {
        "accuracy",
        "f1",
        "confusion",
    }


def test_confusion_matrix_shape_for_three_classes():
    y_true = ["A", "B", "C", "A", "B", "C"]
    y_pred = ["A", "A", "A", "B", "B", "C"]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["confusion"].shape == (3, 3)
