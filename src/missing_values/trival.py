import math
from typing import Any, Dict, List, Optional, Tuple

from src.tree.resources import is_missing, entropy


def best_threshold_trival(
    attr: str,
    dataset: List[Tuple[Dict[str, Any], Any]],
) -> Tuple[float, Optional[float], str]:
    """
    Finds the best threshold for a continuous attribute using a
    trivial missing-value handling strategy.

    Missing values are assigned to either the left or right branch
    and the split quality is evaluated accordingly.

    Parameters
    ----------
    attr : str
        Name of the attribute for which the threshold is computed.

    dataset : list of tuples (x, y)
        Dataset represented as (feature_dict, label).

    Returns
    -------
    best_gain : float
        Maximum information gain achieved by the split.

    best_threshold : float | None
        Optimal threshold value. None if no valid split exists.

    best_default_branch : str
        Direction for missing values:
        - "left"  → assign missing values to left branch
        - "right" → assign missing values to right branch
    """

    U_missing: List[Tuple[Dict[str, Any], Any]] = [
        (x, y) for x, y in dataset if is_missing(x.get(attr)) or x.get(attr) == "?"
    ]

    U_num: List[Tuple[Dict[str, Any], Any]] = [
        (x, y) for x, y in dataset if not is_missing(x.get(attr)) and x.get(attr) != "?"
    ]

    if len(U_num) < 2:
        return -math.inf, None, "left"

    sorted_u = sorted(U_num, key=lambda t: t[0][attr])

    best_gain: float = -math.inf
    best_t: Optional[float] = None
    best_default: str = "left"

    n_total: int = len(dataset)
    base_ent: float = entropy(dataset)

    for i in range(len(sorted_u) - 1):

        y1 = sorted_u[i][1]
        y2 = sorted_u[i + 1][1]

        # skip if same class (no useful split)
        if y1 == y2:
            continue

        v1 = sorted_u[i][0][attr]
        v2 = sorted_u[i + 1][0][attr]

        # skip identical feature values
        if v1 == v2:
            continue

        t: float = (v1 + v2) / 2

        U_left = [(x, y) for x, y in sorted_u if x[attr] <= t]
        U_right = [(x, y) for x, y in sorted_u if x[attr] > t]

        left_with_missing = U_left + U_missing

        gain_left: float = (
            base_ent
            - (len(left_with_missing) / n_total) * entropy(left_with_missing)
            - (len(U_right) / n_total) * entropy(U_right)
        )

        if gain_left > best_gain:
            best_gain = gain_left
            best_t = t
            best_default = "left"

        right_with_missing = U_right + U_missing

        gain_right: float = (
            base_ent
            - (len(U_left) / n_total) * entropy(U_left)
            - (len(right_with_missing) / n_total) * entropy(right_with_missing)
        )

        if gain_right > best_gain:
            best_gain = gain_right
            best_t = t
            best_default = "right"

    return best_gain, best_t, best_default
