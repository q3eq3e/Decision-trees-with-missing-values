import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """
    Uniwersalny preprocessing dla wszystkich datasetów.
    Nie imputuje braków — robią to strategie missing values.
    """

    def __init__(
        self, target_column: str, test_size: float = 0.2, random_state: int = 42
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.encoders: Dict[str, LabelEncoder] = {}

    # --------------------------------------------------
    # LOADING
    # --------------------------------------------------
    def load_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # standardyzacja braków
        df = df.replace(["?", "NA", "N/A", "na", "null"], np.nan)

        return df

    # --------------------------------------------------
    # TARGET SPLIT
    # --------------------------------------------------
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y

    # --------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------
    def _encode_column(self, series: pd.Series, col: str) -> pd.Series:
        """
        Label encoding zachowujący NaN.
        """
        le = LabelEncoder()

        mask = series.notna()
        encoded = series.copy()

        if mask.sum() > 0:
            le.fit(series[mask])
            encoded.loc[mask] = le.transform(series[mask])
            self.encoders[col] = le

        return encoded.astype(float)

    def encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Zamienia wszystkie kolumny kategoryczne na numeryczne.
        Drzewo implementujemy tylko dla cech numerycznych.
        """
        X = X.copy()

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = self._encode_column(X[col], col)

        return X

    # --------------------------------------------------
    # TARGET ENCODING (dla klasyfikacji)
    # --------------------------------------------------
    def encode_target(self, y: pd.Series) -> pd.Series:
        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=y.name)
        return y

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------
    def train_test_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        stratify = y if len(y.unique()) < 20 else None

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    # --------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------
    def prepare(self, path: str):
        """
        Pełny preprocessing:
        load → split → encode → split train/test
        """

        df = self.load_dataset(path)
        X, y = self.split_features_target(df)

        X = self.encode_categorical(X)
        y = self.encode_target(y)

        return self.train_test_split(X, y)
