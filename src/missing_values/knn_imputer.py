import numpy as np


class KNNImputerCustom:
    def __init__(self, k=3):
        self.k = k

    def fit_transform(self, X):
        X = X.copy()

        for idx, row in X.iterrows():
            for col in X.columns:
                if pd.isna(row[col]):
                    neighbors = self._find_neighbors(X, idx, col)
                    X.loc[idx, col] = neighbors.mode()[0]

        return X

    def _distance(self, a, b):
        d = 0
        for col in a.index:
            if pd.isna(a[col]) or pd.isna(b[col]):
                d += 1
            elif isinstance(a[col], str):
                d += 0 if a[col] == b[col] else 1
            else:
                d += abs(a[col] - b[col])
        return d

    def _find_neighbors(self, X, idx, col):
        distances = []
        for i in X.index:
            if not pd.isna(X.loc[i, col]) and i != idx:
                distances.append((self._distance(X.loc[idx], X.loc[i]), i))
        distances.sort()
        return X.loc[[i for _, i in distances[: self.k]], col]
