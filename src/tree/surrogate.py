import numpy as np
import pandas as pd
from typing import List, Tuple


class SurrogateSplitFinder:
    """
    Szuka surrogate splitów maksymalizujących zgodność z głównym splitem.
    """

    def __init__(self, max_surrogates: int = 3):
        self.max_surrogates = max_surrogates

    def _agreement_score(self, primary_left, surrogate_left):
        """
        Oblicza zgodność dwóch podziałów.
        """
        mask = ~np.isnan(primary_left) & ~np.isnan(surrogate_left)
        if mask.sum() == 0:
            return 0
        return np.mean(primary_left[mask] == surrogate_left[mask])

    def _numeric_split(self, feature: pd.Series, threshold: float):
        return feature <= threshold

    def find_surrogates(
        self,
        X: pd.DataFrame,
        primary_feature: str,
        primary_threshold: float,
    ) -> List[Tuple[str, float]]:
        """
        Zwraca listę surrogate splitów:
        [(feature, threshold), ...]
        """

        surrogates = []

        primary_split = self._numeric_split(
            X[primary_feature], primary_threshold
        ).astype(float)

        for col in X.columns:
            if col == primary_feature:
                continue

            if not np.issubdtype(X[col].dtype, np.number):
                continue

            values = X[col].dropna().unique()
            if len(values) < 5:
                continue

            thresholds = np.percentile(values, [20, 40, 60, 80])

            best_score = -1
            best_thr = None

            for thr in thresholds:
                surrogate_split = self._numeric_split(X[col], thr).astype(float)
                score = self._agreement_score(primary_split, surrogate_split)

                if score > best_score:
                    best_score = score
                    best_thr = thr

            if best_thr is not None:
                surrogates.append((col, best_thr, best_score))

        surrogates.sort(key=lambda x: x[2], reverse=True)
        return surrogates[: self.max_surrogates]


def apply_split_with_surrogates(
    X: pd.DataFrame,
    feature: str,
    threshold: float,
    surrogates: List[Tuple[str, float]],
):
    """
    Zwraca maskę lewego/prawego dziecka uwzględniając surrogate splits.
    """

    left_mask = X[feature] <= threshold
    missing_mask = X[feature].isna()

    for s_feature, s_thr, _ in surrogates:
        surrogate_left = X[s_feature] <= s_thr
        left_mask = left_mask | (missing_mask & surrogate_left)
        missing_mask = missing_mask & X[s_feature].isna()

    # pozostałe NaN → większość
    majority_left = left_mask.mean() >= 0.5
    left_mask = left_mask | missing_mask if majority_left else left_mask

    right_mask = ~left_mask
    return left_mask, right_mask
