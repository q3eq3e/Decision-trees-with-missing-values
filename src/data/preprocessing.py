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
    """
    Dataset-aware preprocessing.
    Nie imputuje braków.
    Nie koduje na siłę kategorii (drzewo obsługuje mixed types).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    # ======================================================
    # TITANIC
    # ======================================================
    def _prepare_titanic(self):
        df = load_titanic()

        # standaryzacja braków
        df = df.replace(["", "?", "NA", "N/A", "na", "null"], np.nan)

        # Age -> zaokrąglenie
        if "Age" in df.columns:
            df["Age"] = df["Age"].round()

        target = "Survived"
        X = df.drop(columns=[target])
        y = df[target]

        X.discrete_columns = ["Pclass", "Sex", "Embarked"]
        X.continuous_columns = ["Age", "SibSp", "Parch", "Fare"]

        return X, y

    # ======================================================
    # ADULT
    # ======================================================
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

        X.discrete_columns = [
            "workclass",
            "marital-status",
            "relationship",
            "race",
            "sex",
        ]
        X.continuous_columns = [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]

        return X, y

    # ======================================================
    # CAR SALES
    # ======================================================
    def _prepare_car(self):
        df = load_car()

        df = df.replace(["", "?", "NA", "N/A", "na", "null"], np.nan)

        # usuń brak targetu
        df = df.dropna(subset=["Price"])

        # usunięcie "Rs" i konwersja na int
        df["Price_numeric"] = (
            df["Price"]
            .str.replace("Rs", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # klasy cenowe
        def price_to_class(price):
            if price < 600000:
                return 0  # tanie
            elif price >= 1_000_000:
                return 2  # drogie
            else:
                return 1  # średnie

        df["Price_class"] = df["Price_numeric"].apply(price_to_class)

        # usuwamy oryginalną cenę
        df = df.drop(columns=["Price", "Price_numeric"])

        target = "Price_class"
        X = df.drop(columns=[target])
        y = df[target]

        X.discrete_columns = ["Make", "Colour"]
        X.continuous_columns = ["Odometer (KM)", "Doors"]

        return X, y

    # ======================================================
    # MAIN DISPATCHER
    # ======================================================
    def load_dataset(self, name: Dataset) -> Tuple[pd.DataFrame, pd.Series]:

        if name == Dataset.TITANIC:
            return self._prepare_titanic()

        elif name == Dataset.ADULT:
            return self._prepare_adult()

        elif name == Dataset.CARSALES:
            return self._prepare_car()

        else:
            raise ValueError(f"Unknown dataset: {name}")

    # ======================================================
    # TRAIN / TEST SPLIT
    # ======================================================
    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        stratify = y if len(y.unique()) < 20 else None
        disc_cols = getattr(X, "discrete_columns", [])
        cont_cols = getattr(X, "continuous_columns", [])
        result_list = train_test_split(
            X,
            y,
            test_size=self.test_size,  # 80/20 zgodnie z wymaganiem
            random_state=self.random_state,
            stratify=stratify,
        )
        result_list[0].discrete_columns = disc_cols
        result_list[0].continuous_columns = cont_cols
        result_list[1].discrete_columns = disc_cols
        result_list[1].continuous_columns = cont_cols
        return result_list

    # ======================================================
    # FULL PIPELINE
    # ======================================================
    def prepare(
        self, dataset_name: Dataset
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X, y = self.load_dataset(dataset_name)
        return self.split(X, y)
