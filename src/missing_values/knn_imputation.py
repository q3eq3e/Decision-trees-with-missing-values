import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


class KNNImputationStrategy:
    """
    Imputacja braków przy użyciu sklearn KNNImputer.

    Pipeline:
        1. Rozpoznanie kolumn numerycznych i kategorycznych
        2. Tymczasowe kodowanie kategorii (OrdinalEncoder)
        3. Imputacja KNN
        4. Przywrócenie kategorii
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.num_cols = None
        self.cat_cols = None

        self.encoder = None
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights="uniform",
            metric="nan_euclidean",
        )

    # ======================================================
    # FIT
    # ======================================================
    def fit(self, X: pd.DataFrame):
        X = X.copy()

        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # encoder dla kategorii (obsługa NaN!)
        if len(self.cat_cols) > 0:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            )
            self.encoder.fit(X[self.cat_cols])

        # przygotuj dane numeryczne do fit imputera
        X_encoded = self._encode(X)
        self.imputer.fit(X_encoded)

        return self

    # ======================================================
    # ENCODE / DECODE
    # ======================================================
    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.cat_cols:
            X[self.cat_cols] = self.encoder.transform(X[self.cat_cols])

        return X.astype(float)

    def _decode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.cat_cols:
            # zaokrąglamy bo imputacja daje floaty
            X[self.cat_cols] = np.round(X[self.cat_cols])
            X[self.cat_cols] = self.encoder.inverse_transform(X[self.cat_cols])

        return X

    # ======================================================
    # TRANSFORM
    # ======================================================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_original = X.copy()

        X_encoded = self._encode(X_original)

        X_imputed = self.imputer.transform(X_encoded)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        X_decoded = self._decode(X_imputed)

        return X_decoded

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


# Usage:
# from missing_values.knn_imputation import KNNImputationStrategy

# imputer = KNNImputationStrategy(n_neighbors=5)
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)
