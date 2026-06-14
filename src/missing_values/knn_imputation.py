## authors: Jakub Bagiński, Maciej Borkowski

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class CustomKNNImputer:
    def __init__(
        self,
        n_neighbors: int = 5,
        discrete_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ) -> None:
        self.k: int = n_neighbors
        self.discrete_cols_input: List[str] = discrete_columns or []
        self.continuous_cols: List[str] = continuous_columns or []

        self.nominal_cols: List[str] = []
        self.ordinal_cols: List[str] = []
        self.numeric_cols: List[str] = []

        self.min_: Dict[str, float] = {}
        self.max_: Dict[str, float] = {}

        self.X_train: Optional[pd.DataFrame] = None
        self.train_index_to_pos: Dict[int, int] = {}

    def fit(self, X: pd.DataFrame) -> "CustomKNNImputer":
        X = X.copy()
        self.X_train = X

        for col in self.discrete_cols_input:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.ordinal_cols.append(col)
            else:
                self.nominal_cols.append(col)

        self.numeric_cols = self.ordinal_cols + self.continuous_cols

        for col in self.numeric_cols:
            self.min_[col] = float(X[col].min())
            self.max_[col] = float(X[col].max())

        self.train_index_to_pos = {
            idx: pos for pos, idx in enumerate(self.X_train.index)
        }
        return self

    def _normalize_numeric(
        self,
        col: str,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Min-max normalization to [0, 1].
        """
        min_val = self.min_[col]
        max_val = self.max_[col]

        if max_val == min_val:
            return np.zeros_like(values, dtype=float)

        return (values - min_val) / (max_val - min_val)

    def _distance_to_all(self, row: pd.Series) -> np.ndarray:
        """
        Computes distance between a single sample and all training samples.
        """
        X = self.X_train
        assert X is not None

        distances = np.zeros(len(X), dtype=float)

        for col in self.nominal_cols:
            a = row[col]
            b = X[col].values

            missing_mask = pd.isna(a) | pd.isna(b)
            diff = (b != a).astype(float)
            diff[missing_mask] = 1.0
            distances += diff

        for col in self.numeric_cols:
            a = row[col]
            b = X[col].values

            missing_mask = pd.isna(a) | pd.isna(b)

            a_norm = self._normalize_numeric(col, np.array([a]))[0]
            b_norm = self._normalize_numeric(col, b.astype(float))

            diff = np.abs(b_norm - a_norm)
            diff[missing_mask] = 1.0
            distances += diff

        return distances

    def _impute_from_neighbors(
        self,
        neighbors: pd.DataFrame,
        column: str,
    ) -> float | str | None:
        """
        Imputes a single feature value using nearest neighbors.

        - nominal → mode
        - numeric → mean
        """
        col_values = neighbors[column].dropna()

        if len(col_values) == 0:
            return np.nan

        if column in self.nominal_cols:
            return col_values.mode().iloc[0]
        return col_values.mean()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using KNN-based strategy.
        """
        X = X.copy()

        for col in self.nominal_cols + self.ordinal_cols:
            if col in X.columns:
                X[col] = X[col].astype("object")

        for idx, row in X.iterrows():
            if not row.isna().any():
                continue

            distances = self._distance_to_all(row)

            if idx in self.train_index_to_pos:
                pos = self.train_index_to_pos[idx]
                distances[pos] = np.inf

            missing_cols = row.index[row.isna()]
            valid_mask = ~self.X_train[missing_cols].isna().any(axis=1)

            distances[~valid_mask.to_numpy()] = np.inf

            knn_idx = np.argsort(distances)[: self.k]
            neighbors = self.X_train.iloc[knn_idx]

            for col in X.columns:
                if pd.isna(row[col]):
                    X.at[idx, col] = self._impute_from_neighbors(
                        neighbors,
                        col,
                    )

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        self.X_train = X.copy()
        return self.transform(X)
