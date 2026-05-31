import math
from collections import Counter
from typing import Any, Tuple, Optional
import pandas as pd

from ..tree.resources import is_missing, entropy


def best_threshold_trival(
    attr: str,
    dataset: list,
) -> Tuple[float, Optional[float], str]:
        U_missing = [
            (x, y) for x, y in dataset if is_missing(x.get(attr)) or x.get(attr) == "?"
        ]

        U_num = [
            (x, y)
            for x, y in dataset
            if not is_missing(x.get(attr)) and x.get(attr) != "?"
        ]

        if len(U_num) < 2:
            return -math.inf, None, "left"

        sorted_u = sorted(U_num, key=lambda t: t[0][attr])

        best_gain = -math.inf
        best_t = None
        best_default = "left"

        n_total = len(dataset)
        base_ent = entropy(dataset)

        for i in range(len(sorted_u) - 1):

            if sorted_u[i][1] == sorted_u[i + 1][1]:
                continue

            v1 = sorted_u[i][0][attr]
            v2 = sorted_u[i + 1][0][attr]

            if v1 == v2:
                continue

            t = (v1 + v2) / 2

            U_left = [(x, y) for x, y in sorted_u if x[attr] <= t]
            U_right = [(x, y) for x, y in sorted_u if x[attr] > t]

            # missing -> left
            left_with_missing = U_left + U_missing

            gain_left = (
                base_ent
                - (len(left_with_missing) / n_total) * entropy(left_with_missing)
                - (len(U_right) / n_total) * entropy(U_right)
            )

            if gain_left > best_gain:
                best_gain = gain_left
                best_t = t
                best_default = "left"

            # missing -> right
            right_with_missing = U_right + U_missing

            gain_right = (
                base_ent
                - (len(U_left) / n_total) * entropy(U_left)
                - (len(right_with_missing) / n_total) * entropy(right_with_missing)
            )

            if gain_right > best_gain:
                best_gain = gain_right
                best_t = t
                best_default = "right"

        return best_gain, best_t, best_default
