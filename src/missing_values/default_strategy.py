import pandas as pd
import numpy as np


class DefaultMissingStrategy:
    """
    Numeryczne → średnia
    Kategoryczne → moda
    """

    def __init__(self):
        self.fill_values = {}

    def fit(self, X: pd.DataFrame):
        for col in X.columns:
            if X[col].dtype == "object":
                self.fill_values[col] = X[col].mode()[0]
            else:
                self.fill_values[col] = X[col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col, value in self.fill_values.items():
            X_copy[col] = X_copy[col].fillna(value)
        return X_copy

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
