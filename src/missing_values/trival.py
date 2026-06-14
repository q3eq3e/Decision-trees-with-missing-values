## authors: Jakub Bagiński, Maciej Borkowski

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
    """
    U_missing = [
        (x, y) for x, y in dataset if is_missing(x.get(attr)) or x.get(attr) == "?"
    ]
    U_num = [
        (x, y) for x, y in dataset if not is_missing(x.get(attr)) and x.get(attr) != "?"
    ]
    if len(U_num) < 2:
        return -math.inf, None, "left"

    sorted_u = sorted(U_num, key=lambda t : t[0][attr])

    best_gain = -math.inf
    best_threshold  = None
    best_default = "left"   # default branch for missing values

    n_total = len(dataset)
    base_ent = entropy(dataset)

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

        threshold  = (v1 + v2) / 2

        U_left = [(x, y) for x, y in sorted_u if x[attr] <= threshold ]
        U_right = [(x, y) for x, y in sorted_u if x[attr] > threshold ]

        left_with_missing = U_left + U_missing

        gain_left = (
            base_ent
            - (len(left_with_missing) / n_total) * entropy(left_with_missing)
            - (len(U_right) / n_total) * entropy(U_right)
        )

        if gain_left > best_gain:
            best_gain = gain_left
            best_threshold  = threshold 
            best_default = "left"

        right_with_missing = U_right + U_missing

        gain_right = (
            base_ent
            - (len(U_left) / n_total) * entropy(U_left)
            - (len(right_with_missing) / n_total) * entropy(right_with_missing)
        )

        if gain_right > best_gain:
            best_gain = gain_right
            best_threshold  = threshold 
            best_default = "right"

    return best_gain, best_threshold , best_default
