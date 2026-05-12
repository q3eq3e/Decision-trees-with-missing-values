import pandas as pd


class SurrogatePipeline:
    """
    Pipeline dla surrogate splits.
    Nie imputuje braków — pozostawia NaN.
    """

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()
