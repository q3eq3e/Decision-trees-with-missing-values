## authors: Jakub Bagiński, Maciej Borkowski

import numpy as np
import pandas as pd
import heapq
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple
from ..tree.resources import is_missing

@dataclass
class SurrogateEntry:
    """
    Represents a single surrogate split (continuous or discrete).
    """
    agreement: float
    attribute: str
    # not None for continuous surrogate fields
    threshold: Optional[float] = None
    direction: Optional[str] = None   # "left" | "right"
    # not None for discrete surrogate fields
    split_values: Optional[Set[Any]] = None

    def __lt__(self, other: "SurrogateEntry") -> bool:
        return self.agreement > other.agreement


def agreement_measure(
    U_left: pd.Index,
    U_lsur: pd.Index,
    U_right: pd.Index,
    U_rsur: pd.Index,
) -> float:
    """
    AgreementMeasure(U_left, U_lsur, U_right, U_rsur) =
        (|U_left ∩ U_lsur| + |U_right ∩ U_rsur|)
        / |(U_left U U_right) ∩ (U_lsur U U_rsur)|
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


def best_surrogate_threshold(
    c: str,
    U: pd.DataFrame,
    U_left_idx: pd.Index,
    U_right_idx: pd.Index,
) -> Tuple[float, Optional[float], str]:
    """
    Finds the threshold t for continuous attribute c that best mimics the
    primary split (U_left, U_right).
    """
    col = U[c].dropna()
    if col.empty:
        return 0.0, None, "left"

    sorted_idx = col.sort_values().index
    sorted_vals = col.loc[sorted_idx]

    best_agreement = -np.inf
    best_t: Optional[float] = None
    best_direction = "left"

    for i in range(len(sorted_idx) - 1):
        v_i   = sorted_vals.iloc[i]
        v_ip1 = sorted_vals.iloc[i + 1]
        if v_i == v_ip1:
            continue

        t = (v_i + v_ip1) / 2.0

        col_all = U[c].dropna()
        U_below_idx = col_all[col_all <= t].index
        U_above_idx = col_all[col_all >  t].index

        agreement = agreement_measure(U_left_idx, U_below_idx,
                                      U_right_idx, U_above_idx)

        if agreement > best_agreement:
            best_agreement = agreement
            best_t = t
            best_direction = "left"

        inv_agreement = agreement_measure(U_left_idx, U_above_idx,
                                          U_right_idx, U_below_idx)
        if inv_agreement > best_agreement:
            best_agreement = inv_agreement
            best_t = t
            best_direction = "right"

    return best_agreement, best_t, best_direction


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
    """
    if i == len(values)-1:
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


def surrogate_split(
    U: pd.DataFrame,
    D: List[str],
    C: List[str],
    best_attr: str,
    U_left_idx: pd.Index,
    U_right_idx: pd.Index,
    size: int,
) -> List[SurrogateEntry]:
    """
    Finds the top-`size` surrogate splits for a node whose primary split is
    on `best_attr`.
    """
    heap: List[Tuple[float, SurrogateEntry]] = []

    def _push(entry: SurrogateEntry):
        heapq.heappush(heap, (entry.agreement, entry))
        if len(heap) > size:
            heapq.heappop(heap)

    for attr in D:
        if attr == best_attr:
            continue
        col = U[attr].dropna()
        values = sorted(col.unique().tolist())
        if not values:
            continue
        a, s = best_surrogate_split(
            attr, U, U_left_idx, U_right_idx,
            frozenset(), 0, values
        )
        _push(SurrogateEntry(agreement=a, attribute=attr, split_values=set(s)))
    for attr in C:
        if attr == best_attr:
            continue
        a, t, direction = best_surrogate_threshold(
            attr, U, U_left_idx, U_right_idx
        )
        if t is None:
            continue
        _push(SurrogateEntry(agreement=a, attribute=attr,
                                threshold=t, direction=direction))

    return [entry for _, entry in sorted(heap, key=lambda x: -x[1].agreement)]


def surrogate_split_predict(node: Any, x: pd.Series) -> Any:
    """
    Makes decision for a single instance `x` if the value is missing at a node with surrogate splits.
    """
    def _surrogate_condition(entry: SurrogateEntry, x: pd.Series) -> bool:
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
    for surrogate in node.surrogate_splits:
        s_val = x.get(surrogate.attribute, np.nan)
        if not is_missing(s_val):
            if _surrogate_condition(surrogate, x):
                return node.left
            else:
                return node.right
    if node.default_route == "left":
        return node.left
    else:
        return node.right
