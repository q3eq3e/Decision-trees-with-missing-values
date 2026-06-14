## authors: Jakub Bagiński, Maciej Borkowski

import pytest
import pandas as pd
from src.data.preprocessing import DataPreprocessor, masking


def test_prepare_adult_maps_target_and_missing(monkeypatch):
    df = pd.DataFrame(
        {
            "workclass": [" Private", "?"],
            "martial-status": [" Married", " Single"],
            "relationship": [" Husband", " Own-child"],
            "race": [" White", " Black"],
            "sex": [" Male", " Female"],
            "age": [30, 40],
            "fnlwgt": [1, 2],
            "education-num": [10, 12],
            "capital-gain": [0, 0],
            "capital-loss": [0, 0],
            "hours-per-week": [40, 50],
            "income": ["<=50K", ">50K"],
        }
    )

    monkeypatch.setattr("src.data.preprocessing.load_adult", lambda: df)

    X, y = DataPreprocessor()._prepare_adult()

    assert y.tolist() == [0, 1]
    assert pd.isna(X["workclass"].iloc[1])

    assert "workclass" in X.attrs["discrete_columns"]
    assert "age" in X.attrs["continuous_columns"]


def test_prepare_titanic_rounds_age(monkeypatch):
    df = pd.DataFrame(
        {
            "Age": [20.2, 30.8],
            "Pclass": [1, 2],
            "Sex": ["male", "female"],
            "Embarked": ["S", "C"],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Fare": [10.5, 20.5],
            "Survived": [1, 0],
        }
    )

    monkeypatch.setattr("src.data.preprocessing.load_titanic", lambda: df)

    X, y = DataPreprocessor()._prepare_titanic()

    assert X["Age"].tolist() == [20.0, 31.0]


def test_prepare_car_creates_price_classes(monkeypatch):
    df = pd.DataFrame(
        {
            "Make": ["A", "B", "C"],
            "Colour": ["Red", "Blue", "Black"],
            "Doors": [4, 4, 2],
            "Odometer (KM)": [1000, 2000, 3000],
            "Price": [
                "Rs500,000",
                "Rs700,000",
                "Rs1,500,000",
            ],
        }
    )

    monkeypatch.setattr("src.data.preprocessing.load_car", lambda: df)

    X, y = DataPreprocessor()._prepare_car()

    assert y.tolist() == [0, 1, 2]

    assert "Price" not in X.columns
    assert "Price_numeric" not in X.columns


def test_split_preserves_column_metadata():
    X = pd.DataFrame(
        {
            "a": [1] * 20,
            "b": list(range(20)),
        }
    )

    X.attrs["discrete_columns"] = ["a"]
    X.attrs["continuous_columns"] = ["b"]

    y = pd.Series([0, 1] * 10)

    X_train, X_test, y_train, y_test = DataPreprocessor().split(X, y)

    assert X_train.attrs["discrete_columns"] == ["a"]
    assert X_train.attrs["continuous_columns"] == ["b"]

    assert X_test.attrs["discrete_columns"] == ["a"]
    assert X_test.attrs["continuous_columns"] == ["b"]


def test_masking_zero_rate_does_nothing():
    df = pd.DataFrame({"x": [1, 2, 3]})

    result = masking(df.copy(), 0.0, "x")

    assert result["x"].isna().sum() == 0


def test_masking_unknown_column_does_nothing():
    df = pd.DataFrame({"x": [1, 2, 3]})

    result = masking(df.copy(), 1.0, "y")

    pd.testing.assert_frame_equal(df, result)


def test_masking_full_rate_masks_everything():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})

    result = masking(df.copy(), 1.0, "x")

    assert result["x"].isna().all()


def test_masking_is_reproducible():
    df1 = pd.DataFrame({"x": range(100)})
    df2 = pd.DataFrame({"x": range(100)})

    masking(df1, 0.3, "x", seed=123)
    masking(df2, 0.3, "x", seed=123)

    pd.testing.assert_frame_equal(df1, df2)


def test_load_dataset_unknown_name():
    with pytest.raises(ValueError):
        DataPreprocessor().load_dataset("abc")
