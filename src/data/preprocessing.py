## authors: Jakub Bagiński, Maciej Borkowski

import pandas as pd
import numpy as np
from typing import Tuple
from enum import Enum
from sklearn.model_selection import train_test_split

from src.data.loaders import load_titanic, load_adult, load_car


class Dataset(Enum):
    TITANIC = 0
    ADULT = 1
    CARSALES = 2


class DataPreprocessor:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def _prepare_titanic(self):
        df = load_titanic()

        df = df.replace(["", "?", "NA", "N/A", "na", "null"], np.nan)

        if "Age" in df.columns:
            df["Age"] = df["Age"].round()

        target = "Survived"
        X = df.drop(columns=[target])
        y = df[target]

        X.attrs["discrete_columns"] = ["Pclass", "Sex", "Embarked"]
        X.attrs["continuous_columns"] = ["Age", "SibSp", "Parch", "Fare"]

        return X, y

    def _prepare_adult(self):
        df = load_adult()

        df = df.apply(
            lambda x: (
                x.str.replace(" ", "", n=1, regex=False) if x.dtype == "object" else x
            )
        )

        df = df.replace(["", "?", "NA", "N/A", "na", "null"], np.nan)

        # target mapping
        df["income"] = df["income"].map(
            {
                "<=50K": 0,
                ">50K": 1,
            }
        )

        target = "income"
        X = df.drop(columns=[target])
        y = df[target]

        X.attrs["discrete_columns"] = [
            "workclass",
            "martial-status",
            "relationship",
            "race",
            "sex",
        ]
        X.attrs["continuous_columns"] = [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]

        return X, y

    def _prepare_car(self):
        df = load_car()

        df = df.replace(["", "?", "NA", "N/A", "na", "null"], np.nan)

        # drop rows with missing price, since it's our target
        df = df.dropna(subset=["Price"])

        df["Price_numeric"] = (
            df["Price"]
            .str.replace("Rs", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # mapping price to classes
        def price_to_class(price):
            if price < 600000:
                return 0  # tanie
            elif price >= 1_000_000:
                return 2  # drogie
            else:
                return 1  # średnie

        df["Price_class"] = df["Price_numeric"].apply(price_to_class)

        # drop original price columns
        df = df.drop(columns=["Price", "Price_numeric"])

        target = "Price_class"
        X = df.drop(columns=[target])
        y = df[target]

        X.attrs["discrete_columns"] = ["Make", "Colour"]
        X.attrs["continuous_columns"] = ["Odometer (KM)", "Doors"]

        return X, y

    def load_dataset(self, name: Dataset) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main dispatcher
        """
        if name == Dataset.TITANIC:
            return self._prepare_titanic()

        elif name == Dataset.ADULT:
            return self._prepare_adult()

        elif name == Dataset.CARSALES:
            return self._prepare_car()

        else:
            raise ValueError(f"Unknown dataset: {name}")

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Test-train split with stratification for classification tasks and metadata preservation.
        """
        stratify = y if len(y.unique()) < 20 else None
        disc_cols = X.attrs.get("discrete_columns", [])
        cont_cols = X.attrs.get("continuous_columns", [])
        result_list = train_test_split(
            X,
            y,
            test_size=self.test_size,  # 80/20 zgodnie z wymaganiem
            random_state=self.random_state,
            stratify=stratify,
        )
        result_list[0].attrs["discrete_columns"] = disc_cols
        result_list[0].attrs["continuous_columns"] = cont_cols
        result_list[1].attrs["discrete_columns"] = disc_cols
        result_list[1].attrs["continuous_columns"] = cont_cols
        return result_list

    def prepare(
        self, dataset_name: Dataset
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Full preparation pipeline
        """
        X, y = self.load_dataset(dataset_name)
        return self.split(X, y)


def masking(dataset: pd.DataFrame, missing_rate: float, column: str, seed: int = 42):
    """
    Randomly masks values in the specified column with NaN according to the given missing rate.
    """
    if missing_rate <= 0 or column not in dataset.columns:
        return dataset

    np.random.seed(seed)

    for idx in dataset.index:
        if np.random.rand() < missing_rate:
            dataset.loc[idx, column] = np.nan

    return dataset
