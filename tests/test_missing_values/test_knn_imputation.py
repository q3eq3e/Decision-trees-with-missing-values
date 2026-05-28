import numpy as np
import pandas as pd
import pytest

from src.missing_values.knn_imputation import CustomKNNImputer


def test_fit_detects_nominal_and_numeric_columns():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "blue"],
            "age": [20, 30, 40, 50],
        }
    )

    imputer = CustomKNNImputer(
        n_neighbors=2,
        discrete_columns=["color"],
        continuous_columns=["age"],
    )

    imputer.fit(X)

    assert imputer.nominal_cols == ["color"]
    assert imputer.ordinal_cols == []
    assert imputer.numeric_cols == ["age"]
    assert imputer.min_["age"] == 20
    assert imputer.max_["age"] == 50


def test_transform_imputes_missing_nominal_and_numeric_values():
    X_train = pd.DataFrame(
        {
            "color": ["red", "red", "blue", "blue"],
            "age": [20, 30, 40, 50],
        }
    )
    X_missing = pd.DataFrame({"color": [np.nan], "age": [np.nan]}, index=[99])

    imputer = CustomKNNImputer(
        n_neighbors=2,
        discrete_columns=["color"],
        continuous_columns=["age"],
    )
    imputer.fit(X_train)

    transformed = imputer.transform(X_missing)

    assert transformed.loc[99, "color"] == "red"
    assert transformed.loc[99, "age"] == pytest.approx(25.0)
