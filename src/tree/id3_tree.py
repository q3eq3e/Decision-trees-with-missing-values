import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..missing_values.surrogate import surrogate_split, surrogate_split_predict
from ..missing_values.trival import best_threshold_trival
from .node import Leaf, Node
from .resources import entropy, is_missing, majority
import pandas as pd


class MissingStrategy(Enum):
    MAJORITY = 0
    TRIVIAL = 1
    IMPUTATION = 2
    SURROGATE = 3


# ---------------------------------------------------------------------------
# best_threshold – podział atrybutów ciągłych
# ---------------------------------------------------------------------------


def best_threshold(attr: str, dataset: list) -> Tuple[float, Optional[float]]:

    prepared = [
        dataset[i] for i in range(len(dataset)) if not is_missing(dataset[i][0][attr])
    ]

    if len(prepared) < 2:
        return -math.inf, None

    sorted_u = sorted(prepared, key=lambda t: t[0][attr])

    base_ent = entropy(prepared)
    n = len(sorted_u)

    best_gain = -math.inf
    best_t = None

    for i in range(n - 1):

        if sorted_u[i][0][attr] == sorted_u[i + 1][0][attr]:
            continue

        t = (sorted_u[i][0][attr] + sorted_u[i + 1][0][attr]) / 2

        left = [(x, y) for x, y in sorted_u if x[attr] <= t]
        right = [(x, y) for x, y in sorted_u if x[attr] > t]

        gain = (
            base_ent
            - (len(left) / n) * entropy(left)
            - (len(right) / n) * entropy(right)
        )

        if gain > best_gain:
            best_gain = gain
            best_t = t

    return best_gain, best_t


# ---------------------------------------------------------------------------
# best_split – podział atrybutów dyskretnych
# ---------------------------------------------------------------------------


def best_split(attr: str, dataset: List) -> Tuple[float, Optional[frozenset]]:

    if not dataset:
        return -math.inf, None

    values = list({x[attr] for x, _ in dataset})
    if len(values) < 2:
        return -math.inf, None

    def recurse(split: frozenset, idx: int) -> Tuple[float, frozenset]:
        if idx == len(values):
            if not split or split == frozenset(values):
                return -math.inf, split
            left = [(x, y) for x, y in dataset if x[attr] in split]
            right = [(x, y) for x, y in dataset if x[attr] not in split]
            if not left or not right:
                return -math.inf, split
            n = len(dataset)
            gain = (
                entropy(dataset)
                - (len(left) / n) * entropy(left)
                - (len(right) / n) * entropy(right)
            )
            return gain, split

        g1, s1 = recurse(split, idx + 1)
        g2, s2 = recurse(split | {values[idx]}, idx + 1)
        return (g1, s1) if g1 >= g2 else (g2, s2)

    gain, s = recurse(frozenset(), 0)
    return gain, s


class DecisionTree:
    """
    Drzewo decyzyjne ID3 z parametryzowaną strategią obsługi braków.

    Parameters
    ----------
    discrete_attrs : lista nazw atrybutów dyskretnych
    continuous_attrs : lista nazw atrybutów ciągłych
    max_depth : maksymalna głębokość drzewa (g)
    strategy : MissingStrategy – strategia obsługi brakujących wartości
    max_surrogate_splits : maksymalna liczba surrogate splits
    """

    def __init__(
        self,
        discrete_attrs: List[str],
        continuous_attrs: List[str],
        max_depth: int = 10,
        strategy: MissingStrategy = MissingStrategy.MAJORITY,
        max_surrogate_splits: int = 5,
    ):
        self.discrete_attrs = list(discrete_attrs)
        self.continuous_attrs = list(continuous_attrs)
        self.max_depth = max_depth
        self.strategy = strategy
        self.root: Any = None
        self.max_surrogate_splits = max_surrogate_splits

    # ------------------------------------------------------------------
    # Budowanie
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTree":
        classes = sorted(set(y))
        self.root = self._build(
            classes, self.discrete_attrs, self.continuous_attrs, X, y, self.max_depth
        )
        return self

    def _build(self, Y, D, C, X: pd.DataFrame, y: pd.Series, g: int):
        classes = [row for row in y.values]
        if len(set(classes)) == 1:
            return Leaf(classes[0])

        all_same = all(
            len({x.get(a) for x in X.to_dict(orient="records")}) <= 1 for a in D + C
        )
        if all_same or g == 0:
            return Leaf(majority(y))

        # --- szukamy najlepszego podziału ---
        best_gain = -math.inf
        best_attr = None
        best_type = None  # "discrete" | "continuous"
        best_split_set = None
        best_t = None
        U = list(zip(X.to_dict(orient="records"), y.values))
        for d in D:
            if self.strategy == MissingStrategy.TRIVIAL:
                prepared_dataset = []
                for i in range(len(U)):
                    prepared_dataset.append(U[i])
                    if is_missing(U[i][0][d]):
                        prepared_dataset[-1][0][d] = "?"
            else:
                prepared_dataset = [
                    U[i] for i in range(len(U)) if not is_missing(U[i][0][d])
                ]
            gain, s = best_split(d, prepared_dataset)
            if gain > best_gain:
                best_gain = gain
                best_attr = d
                best_split_set = s
                best_type = "discrete"

        best_default_route = default_route = "left"
        for c in C:
            vals = [x[c] for x, _ in U if not is_missing(x.get(c))]
            if not vals or min(vals) == max(vals):
                continue
            if self.strategy == MissingStrategy.TRIVIAL:
                gain, t, default_route = best_threshold_trival(c, U)
            else:
                gain, t = best_threshold(c, U)
            if gain > best_gain:
                best_gain = gain
                best_attr = c
                best_t = t
                best_type = "continuous"
                best_default_route = default_route

        if best_attr is None or best_gain <= 0:
            return Leaf(majority(y))

        # --- budujemy węzeł ---
        maj = majority(y)

        if best_type == "discrete":
            if self.strategy == MissingStrategy.SURROGATE:

                U_left = []
                U_right = []
                id_left = []
                id_right = []
                for i, (x, y) in enumerate(U):
                    if is_missing(x.get(best_attr)) or x.get(best_attr) == "?":
                        continue
                    elif x.get(best_attr) in best_split_set:
                        U_left.append((x, y))
                        id_left.append(i)
                    else:
                        U_right.append((x, y))
                        id_right.append(i)
            else:
                U_left = [
                    (x, y)
                    for x, y in U
                    if not is_missing(val := x.get(best_attr)) and val in best_split_set
                ]
                U_right = [
                    (x, y)
                    for x, y in U
                    if not is_missing(val := x.get(best_attr))
                    and val not in best_split_set
                ]
            X_left, y_left = zip(*U_left) if U_left else ([], [])
            X_right, y_right = zip(*U_right) if U_right else ([], [])

            node = Node(
                majority_class=maj,
                condition_attr=best_attr,
                is_continuous=False,
                split_set=best_split_set,
            )
            node.default_route = "right" if len(U_right) > len(U_left) else "left"

        else:  # continuous
            U_left = []
            U_right = []
            if self.strategy == MissingStrategy.SURROGATE:
                id_left = []
                id_right = []
                for i, (x_row, y_row) in enumerate(U):
                    val = x_row.get(best_attr)
                    # brak wartości
                    if is_missing(val) or val == "?":
                        continue

                    if val <= best_t:
                        U_left.append((x_row, y_row))
                        id_left.append(i)
                    else:
                        U_right.append((x_row, y_row))
                        id_right.append(i)

            else:
                for x_row, y_row in U:

                    val = x_row.get(best_attr)

                    # brak wartości
                    if is_missing(val) or val == "?":

                        if self.strategy == MissingStrategy.TRIVIAL:

                            if best_default_route == "left":
                                U_left.append((x_row, y_row))
                            else:
                                U_right.append((x_row, y_row))

                        continue

                    if val <= best_t:
                        U_left.append((x_row, y_row))
                    else:
                        U_right.append((x_row, y_row))

            X_left, y_left = zip(*U_left) if U_left else ([], [])
            X_right, y_right = zip(*U_right) if U_right else ([], [])

            node = Node(
                majority_class=maj,
                condition_attr=best_attr,
                is_continuous=True,
                threshold=best_t,
            )

            if self.strategy == MissingStrategy.TRIVIAL:
                node.default_route = best_default_route
            else:
                node.default_route = "right" if len(y_right) > len(y_left) else "left"
        if not y_left:
            return Leaf(maj)
        if not y_right:
            return Leaf(maj)

        node.left = self._build(Y, D, C, pd.DataFrame(X_left), pd.Series(y_left), g - 1)
        node.right = self._build(
            Y, D, C, pd.DataFrame(X_right), pd.Series(y_right), g - 1
        )
        if self.strategy == MissingStrategy.SURROGATE:
            node.surrogate_splits = surrogate_split(
                X,
                D,
                C,
                best_attr,
                pd.Index(id_left),
                pd.Index(id_right),
                self.max_surrogate_splits,
            )
        return node

    # ------------------------------------------------------------------
    # Predykcja
    # ------------------------------------------------------------------

    def predict_one(self, x: pd.Series) -> Any:
        node = self.root
        while not node.is_leaf():
            result = node.condition(x)
            if result is None:
                if self.strategy == MissingStrategy.TRIVIAL:

                    if node.default_route == "left":
                        node = node.left
                    else:
                        node = node.right

                    continue
                elif (
                    self.strategy == MissingStrategy.SURROGATE and node.surrogate_splits
                ):
                    node = surrogate_split_predict(node, x)
                    continue
                return node.majority_class
            elif result:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X: pd.DataFrame) -> List[Any]:
        return [self.predict_one(x[1]) for x in X.iterrows()]

    # ------------------------------------------------------------------
    # Wizualizacja tekstowa
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines: List[str] = []
        self._print_node(self.root, lines, prefix="", is_left=True, is_root=True)
        return "\n".join(lines)

    def _print_node(self, node, lines, prefix, is_left, is_root):
        connector = "" if is_root else ("├── L: " if is_left else "└── R: ")
        if node.is_leaf():
            lines.append(f"{prefix}{connector}[LEAF] class={node.prediction}")
            return
        if node.is_continuous:
            cond = f"{node.condition_attr} ≤ {node.threshold:.4f}"
        else:
            cond = f"{node.condition_attr} ∈ {set(node.split_set)}"
        lines.append(
            f"{prefix}{connector}[NODE] {cond}  (default→{node.default_route})"
        )
        child_prefix = prefix + ("    " if is_root else ("│   " if is_left else "    "))
        self._print_node(node.left, lines, child_prefix, is_left=True, is_root=False)
        self._print_node(node.right, lines, child_prefix, is_left=False, is_root=False)
