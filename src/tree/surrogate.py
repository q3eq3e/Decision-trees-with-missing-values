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
import numpy as np
import pandas as pd
import heapq
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple


def _is_discrete(series: pd.Series) -> bool:
    """
    Returns True if the column should be treated as a discrete (categorical)
    attribute: object, string, boolean, or pandas Categorical dtype.
    Numeric dtypes are treated as continuous.
    """
    return pd.api.types.is_object_dtype(series) \
        or pd.api.types.is_bool_dtype(series) \
        or isinstance(series.dtype, pd.CategoricalDtype) \
        or pd.api.types.is_string_dtype(series)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SurrogateEntry:
    """
    Represents a single surrogate split (continuous or discrete).

    Continuous:  threshold + direction ("left" / "right")
    Discrete:    split_values (set of values that go left)
    """
    agreement: float
    attribute: str
    # continuous surrogate fields
    threshold: Optional[float] = None
    direction: Optional[str] = None   # "left" | "right"
    # discrete surrogate fields
    split_values: Optional[Set[Any]] = None

    # heapq is a min-heap; we want a max-heap by agreement, so invert sign
    def __lt__(self, other: "SurrogateEntry") -> bool:
        return self.agreement > other.agreement   # descending


# ---------------------------------------------------------------------------
# 2.2.3.4  Agreement measure
# ---------------------------------------------------------------------------

def agreement_measure(
    U_left: pd.Index,
    U_lsur: pd.Index,
    U_right: pd.Index,
    U_rsur: pd.Index,
) -> float:
    """
    AgreementMeasure(U_left, U_lsur, U_right, U_rsur) =
        (|U_left ∩ U_lsur| + |U_right ∩ U_rsur|)
        / |(U_left ∪ U_right) ∩ (U_lsur ∪ U_rsur)|

    All arguments are pandas Index objects (row labels of the respective sets).
    Returns 0 if the denominator is zero.
    """
    primary_union   = U_left.union(U_right)
    surrogate_union = U_lsur.union(U_rsur)
    denom = len(primary_union.intersection(surrogate_union))
    if denom == 0:
        return 0.0
    numer = (
        len(U_left.intersection(U_lsur))
        + len(U_right.intersection(U_rsur))
    )
    return numer / denom


# ---------------------------------------------------------------------------
# 2.2.3.2  BestSurrogateThreshold  (continuous attribute)
# ---------------------------------------------------------------------------

def best_surrogate_threshold(
    c: str,
    U: pd.DataFrame,
    U_left_idx: pd.Index,
    U_right_idx: pd.Index,
) -> Tuple[float, Optional[float], str]:
    """
    Finds the threshold t for continuous attribute c that best mimics the
    primary split (U_left, U_right).

    Returns:
        (best_agreement, best_t, best_direction)
        best_direction in {"left", "right"} – "left" means U[c] <= t goes left.
    """
    # Work only on rows that have a value for c
    col = U[c].dropna()
    if col.empty:
        return 0.0, None, "left"

    sorted_idx = col.sort_values().index
    sorted_vals = col.loc[sorted_idx]

    best_agreement = -np.inf
    best_t: Optional[float] = None
    best_direction = "left"

    for i in range(len(sorted_idx) - 1):
        # candidate threshold at every consecutive-pair boundary
        v_i   = sorted_vals.iloc[i]
        v_ip1 = sorted_vals.iloc[i + 1]
        if v_i == v_ip1:
            continue

        t = (v_i + v_ip1) / 2.0

        # U_below: rows in U with c <= t,  U_above: rows with c > t
        col_all = U[c].dropna()
        U_below_idx = col_all[col_all <= t].index
        U_above_idx = col_all[col_all >  t].index

        agreement = agreement_measure(U_left_idx, U_below_idx,
                                      U_right_idx, U_above_idx)

        if agreement > best_agreement:
            best_agreement = agreement
            best_t = t
            best_direction = "left"

        # "right" direction: U[c] <= t → right  (i.e. swap surrogate sides)
        inv_agreement = agreement_measure(U_left_idx, U_above_idx,
                                          U_right_idx, U_below_idx)
        if inv_agreement > best_agreement:
            best_agreement = inv_agreement
            best_t = t
            best_direction = "right"

    return best_agreement, best_t, best_direction


# ---------------------------------------------------------------------------
# 2.2.3.3  BestSurrogateSplit  (discrete attribute)
# ---------------------------------------------------------------------------

def best_surrogate_split(
    d: str,
    U: pd.DataFrame,
    U_left_idx: pd.Index,
    U_right_idx: pd.Index,
    split: frozenset,
    i: int,
    values: List[Any],
) -> Tuple[float, frozenset]:
    """
    Recursively enumerates all subsets of `values` to find the subset
    assignment that maximises agreement with the primary split.

    Parameters
    ----------
    d         : discrete column name
    U         : full dataset (only rows without missing d are considered)
    U_left_idx, U_right_idx : row-label indices of primary left/right sets
    split     : current candidate set of values that go left
    i         : index into `values` currently being decided
    values    : sorted list of unique values of d (non-missing)

    Returns
    -------
    (best_agreement, best_split_frozenset)
    """
    if i == len(values):
        col = U[d].dropna()
        U_l_idx = col[col.isin(split)].index
        U_r_idx = col[~col.isin(split)].index

        a1 = agreement_measure(U_left_idx, U_l_idx, U_right_idx, U_r_idx)
        a2 = agreement_measure(U_left_idx, U_r_idx, U_right_idx, U_l_idx)

        if a1 >= a2:
            return a1, split
        else:
            all_values = frozenset(values)
            return a2, all_values - split

    # Branch 1: do NOT add values[i] to the split
    a1, s1 = best_surrogate_split(d, U, U_left_idx, U_right_idx,
                                   split, i + 1, values)
    # Branch 2: add values[i] to the split
    a2, s2 = best_surrogate_split(d, U, U_left_idx, U_right_idx,
                                   split | frozenset([values[i]]), i + 1, values)

    if a1 >= a2:
        return a1, s1
    return a2, s2


# ---------------------------------------------------------------------------
# 2.2.3.1  SurrogateSplit  (main finder)
# ---------------------------------------------------------------------------

def surrogate_split(
    U: pd.DataFrame,
    best_attr: str,
    U_left_idx: pd.Index,
    U_right_idx: pd.Index,
    size: int,
) -> List[SurrogateEntry]:
    """
    Finds the top-`size` surrogate splits for a node whose primary split is
    on `best_attr`.

    Column types are inferred directly from the DataFrame:
      - object / bool / Categorical / string  →  discrete  (BestSurrogateSplit)
      - numeric                                →  continuous (BestSurrogateThreshold)

    Parameters
    ----------
    U            : training rows at this node
    best_attr    : the primary split attribute (excluded from candidates)
    U_left_idx   : row labels that went left in the primary split
    U_right_idx  : row labels that went right in the primary split
    size         : maximum number of surrogates to keep

    Returns
    -------
    List of SurrogateEntry objects sorted by agreement (descending).
    """
    heap: List[Tuple[float, SurrogateEntry]] = []

    def _push(entry: SurrogateEntry):
        heapq.heappush(heap, (-entry.agreement, entry))
        if len(heap) > size:
            heapq.heappop(heap)

    for attr in U.columns:
        if attr == best_attr:
            continue

        if _is_discrete(U[attr]):
            col = U[attr].dropna()
            values = sorted(col.unique().tolist())
            if not values:
                continue
            a, s = best_surrogate_split(
                attr, U, U_left_idx, U_right_idx,
                frozenset(), 0, values
            )
            _push(SurrogateEntry(agreement=a, attribute=attr, split_values=set(s)))

        elif pd.api.types.is_numeric_dtype(U[attr]):
            a, t, direction = best_surrogate_threshold(
                attr, U, U_left_idx, U_right_idx
            )
            if t is None:
                continue
            _push(SurrogateEntry(agreement=a, attribute=attr,
                                 threshold=t, direction=direction))

        # other dtypes (datetime, etc.) are skipped

    return [entry for _, entry in sorted(heap, key=lambda x: -x[1].agreement)]


# ---------------------------------------------------------------------------
# 2.2.3.5  SurrogateSplitPredict
# ---------------------------------------------------------------------------

def surrogate_split_predict(node: Any, x: pd.Series) -> Any:
    """
    Traverses the decision tree from `node` to a leaf, using surrogate splits
    whenever the primary attribute is missing ("?" or NaN).

    Assumes each tree node has:
        node.attribute      – primary split attribute name
        node.condition(x)   – callable; True → go left
        node.left           – left child node
        node.right          – right child node
        node.surrogates     – list of SurrogateEntry
        node.default_route  – "left" | "right" (majority class direction)
        node.prediction     – leaf prediction value
        node.is_leaf        – bool
    """
    def _is_missing(value) -> bool:
        if value == "?":
            return True
        try:
            return pd.isna(value)
        except (TypeError, ValueError):
            return False

    def _surrogate_condition(entry: SurrogateEntry, x: pd.Series) -> bool:
        """Returns True if surrogate sends x to the LEFT child."""
        val = x.get(entry.attribute, np.nan)
        if entry.threshold is not None:
            # continuous surrogate
            goes_left_by_threshold = val <= entry.threshold
            if entry.direction == "left":
                return goes_left_by_threshold
            else:
                return not goes_left_by_threshold
        else:
            # discrete surrogate
            return val in entry.split_values

    while not node.is_leaf:
        primary_val = x.get(node.attribute, np.nan)

        if not _is_missing(primary_val):
            # Primary attribute is available
            if node.condition(x):
                node = node.left
            else:
                node = node.right
        else:
            # Primary attribute is missing – try surrogates in order
            found_surrogate = False
            for surrogate in node.surrogates:
                s_val = x.get(surrogate.attribute, np.nan)
                if not _is_missing(s_val):
                    if _surrogate_condition(surrogate, x):
                        node = node.left
                    else:
                        node = node.right
                    found_surrogate = True
                    break

            if not found_surrogate:
                # No surrogate available → take the majority route
                return node.default_route

    return node.prediction


# ---------------------------------------------------------------------------
# Legacy helper kept for backward compatibility
# ---------------------------------------------------------------------------

def apply_split_with_surrogates(
    X: pd.DataFrame,
    feature: str,
    threshold: float,
    surrogates: List[SurrogateEntry],
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (left_mask, right_mask) for a node, routing missing primary
    values through surrogate splits and ultimately the majority direction.
    """
    left_mask    = X[feature] <= threshold
    missing_mask = X[feature].isna()

    for entry in surrogates:
        if missing_mask.sum() == 0:
            break
        if entry.threshold is not None:
            if entry.direction == "left":
                surrogate_left = X[entry.attribute] <= entry.threshold
            else:
                surrogate_left = X[entry.attribute] > entry.threshold
        else:
            surrogate_left = X[entry.attribute].isin(entry.split_values)

        left_mask    = left_mask | (missing_mask & surrogate_left)
        missing_mask = missing_mask & X[entry.attribute].isna()

    # Remaining NaN → majority direction
    majority_left = left_mask.mean() >= 0.5
    if majority_left:
        left_mask = left_mask | missing_mask

    right_mask = ~left_mask
    return left_mask, right_mask

# ----------------------------
