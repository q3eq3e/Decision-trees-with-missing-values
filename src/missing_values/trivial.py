import pandas as pd


class TrivialMissingStrategy:
    """
    Usuwa wszystkie rekordy zawierające brakujące wartości.
    """

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.dropna().reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transform(X)
