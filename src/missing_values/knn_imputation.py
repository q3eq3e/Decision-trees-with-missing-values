import numpy as np
import pandas as pd


class CustomKNNImputer:
    """
    KNN imputer zgodny z pseudokodem z pracy.

    Typy kolumn:
        nominal_columns     -> dystans 0/1
        discrete_columns    -> minmax + L1
        continuous_columns  -> minmax + L1
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        discrete_columns: list[str] | None = None,
        continuous_columns: list[str] | None = None,
    ):
        self.k = n_neighbors
        self.discrete_cols_input = discrete_columns or []
        self.continuous_cols = continuous_columns or []

        # zostaną wykryte w fit()
        self.nominal_cols = []
        self.ordinal_cols = []
        self.numeric_cols = []

        self.min_ = {}
        self.max_ = {}
        self.X_train = None
        self.train_index_to_pos = None

    # ======================================================
    # FIT
    # ======================================================
    def fit(self, X: pd.DataFrame):
        X = X.copy()
        self.X_train = X

        # 🔹 rozbij discrete -> nominal vs ordinal
        for col in self.discrete_cols_input:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.ordinal_cols.append(col)
            else:
                self.nominal_cols.append(col)

        # numeric = ordinal + continuous
        self.numeric_cols = self.ordinal_cols + self.continuous_cols

        # zapamiętaj min/max dla skalowania
        for col in self.numeric_cols:
            self.min_[col] = X[col].min()
            self.max_[col] = X[col].max()

        self.train_index_to_pos = {
            idx: pos for pos, idx in enumerate(self.X_train.index)
        }

        return self

    # ======================================================
    # MINMAX NORMALIZATION
    # ======================================================
    def _normalize_numeric(self, col, values):
        min_val = self.min_[col]
        max_val = self.max_[col]

        if max_val == min_val:
            return np.zeros_like(values, dtype=float)

        return (values - min_val) / (max_val - min_val)

    # ======================================================
    # DISTANCE RECORD vs MATRIX (VECTORISED)
    # ======================================================
    def _distance_to_all(self, row: pd.Series) -> np.ndarray:
        X = self.X_train
        distances = np.zeros(len(X))

        # -------- NOMINAL --------
        for col in self.nominal_cols:
            a = row[col]
            b = X[col].values

            missing_mask = pd.isna(a) | pd.isna(b)
            diff = (b != a).astype(float)
            diff[missing_mask] = 1
            distances += diff

        # -------- NUMERIC (discrete + continuous) --------
        for col in self.numeric_cols:
            a = row[col]
            b = X[col].values

            missing_mask = pd.isna(a) | pd.isna(b)

            a_norm = self._normalize_numeric(col, np.array([a]))[0]
            b_norm = self._normalize_numeric(col, b.astype(float))

            diff = np.abs(b_norm - a_norm)
            diff[missing_mask] = 1
            distances += diff

        return distances

    # ======================================================
    # IMPUTE SINGLE COLUMN FROM NEIGHBOURS
    # ======================================================
    def _impute_from_neighbors(self, neighbors: pd.DataFrame, column: str):
        col_values = neighbors[column].dropna()

        if len(col_values) == 0:
            return np.nan

        if column in self.nominal_cols:
            return col_values.mode().iloc[0]
        else:
            return col_values.mean()

    # ======================================================
    # TRANSFORM
    # ======================================================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for idx, row in X.iterrows():
            if not row.isna().any():
                continue

            distances = self._distance_to_all(row)

            if idx in self.train_index_to_pos:
                pos = self.train_index_to_pos[idx]
                distances[pos] = np.inf

            knn_idx = np.argsort(distances)[: self.k]
            neighbors = self.X_train.iloc[knn_idx]

            for col in X.columns:
                if pd.isna(row[col]):
                    X.at[idx, col] = self._impute_from_neighbors(neighbors, col)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        self.X_train = X.copy()
        return self.transform(X)
