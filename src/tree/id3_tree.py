


import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from .node import Leaf, Node
import pandas as pd
# ---------------------------------------------------------------------------
# Typy pomocnicze
# ---------------------------------------------------------------------------

 
class MissingStrategy(Enum):
    MAJORITY = 0
    TRIVAL = 1
    IMPUTATION = 2
    SURROGATE = 3

def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False

def _entropy(dataset: pd.DataFrame) -> float:
    if not dataset:
        return 0.0
    counts = Counter(y for _, y in dataset)
    n = len(dataset)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)
 
 
def _majority(Y: pd.DataFrame) -> Any:
    return Counter(Y).most_common(1)[0][0]


def best_threshold(attr: str, dataset: list,
                   strategy: MissingStrategy) -> Tuple[float, Optional[float]]:
    # prepared = _prepare_dataset(dataset, attr, strategy, is_continuous=True)
    prepared = [dataset[i] for i in range(len(dataset)) if not _is_missing(dataset[i][0][attr])]
    if len(prepared) < 2:
        return -math.inf, None
 
    sorted_u = sorted(prepared, key=lambda t: t[0][attr])
    base_ent = _entropy(prepared)
    n = len(sorted_u)
 
    best_gain = -math.inf
    best_t: Optional[float] = None
 
    for i in range(n - 1):
        if sorted_u[i][0][attr] == sorted_u[i + 1][0][attr]:
            continue
        t = (sorted_u[i][0][attr] + sorted_u[i + 1][0][attr]) / 2
        left  = [(x, y) for x, y in sorted_u if x[attr] <= t]
        right = [(x, y) for x, y in sorted_u if x[attr] >  t]
        if strategy == MissingStrategy.SURROGATE:
            pass
        else:
            gain = base_ent - (len(left) / n) * _entropy(left) \
                             - (len(right) / n) * _entropy(right)
 
        if gain > best_gain:
            best_gain = gain
            best_t = t
 
    return best_gain, best_t
 
 
# ---------------------------------------------------------------------------
# BestSplit – atrybuty dyskretne
# ---------------------------------------------------------------------------
 
def best_split(attr: str, dataset: pd.DataFrame,
               strategy: MissingStrategy) -> Tuple[float, Optional[frozenset]]:
    # prepared = _prepare_dataset(dataset, attr, strategy, is_continuous=False)
    prepared = dataset
    if not prepared:
        return -math.inf, None
 
    values = list({x[attr] for x, _ in prepared})
    if len(values) < 2:
        return -math.inf, None
 
    best_gain = -math.inf
    best_s: Optional[frozenset] = None
 
    def recurse(split: frozenset, idx: int) -> Tuple[float, frozenset]:
        nonlocal best_gain, best_s
        if idx == len(values):
            if not split or split == frozenset(values):
                return -math.inf, split
            left  = [(x, y) for x, y in prepared if x[attr]     in split]
            right = [(x, y) for x, y in prepared if x[attr] not in split]
            if not left or not right:
                return -math.inf, split
            n = len(prepared)
            if strategy == MissingStrategy.SURROGATE:
                pass
            else:
                gain = _entropy(prepared) \
                       - (len(left)  / n) * _entropy(left) \
                       - (len(right) / n) * _entropy(right)
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
    """
 
    def __init__(
        self,
        discrete_attrs:   List[str],
        continuous_attrs: List[str],
        max_depth:        int = 10,
        strategy:         MissingStrategy = MissingStrategy.MAJORITY,
    ):
        self.discrete_attrs   = list(discrete_attrs)
        self.continuous_attrs = list(continuous_attrs)
        self.max_depth        = max_depth
        self.strategy         = strategy
        self.root: Any        = None
 
    # ------------------------------------------------------------------
    # Budowanie
    # ------------------------------------------------------------------
 
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTree":
        dataset: pd.DataFrame = pd.concat([X, y], axis=1)
        classes = sorted(set(y))
        self.root = self._build(classes, self.discrete_attrs,
                                self.continuous_attrs, X, y, self.max_depth)
        return self
 
    def _build(self, Y, D, C, X: pd.DataFrame, y: pd.Series, g: int):
        classes = [row for row in y.values]
        if len(set(classes)) == 1:
            return Leaf(classes[0])
 
        all_same = all(
            len({x.get(a) for x in X.to_dict(orient="records")}) <= 1 for a in D + C
        )
        if all_same or g == 0:
            return Leaf(_majority(y))

        # --- szukamy najlepszego podziału ---
        best_gain = -math.inf
        best_attr = None
        best_type = None   # "discrete" | "continuous"
        best_split_set = None
        best_t = None
        U = list(zip(X.to_dict(orient="records"), y.values))
        for d in D:
            gain, s = best_split(d, U, self.strategy)
            if gain > best_gain:
                best_gain = gain
                best_attr = d
                best_split_set = s
                best_type = "discrete"
 
        for c in C:
            vals = [x[c] for x, _ in U if not _is_missing(x.get(c))]
            if not vals or min(vals) == max(vals):
                continue
            gain, t = best_threshold(c, U, self.strategy)
            if gain > best_gain:
                best_gain = gain
                best_attr = c
                best_t = t
                best_type = "continuous"
 
        if best_attr is None or best_gain <= 0:
            return Leaf(_majority(y))
 
        # --- budujemy węzeł ---
        maj = _majority(y)
 
        if best_type == "discrete":
            U_left = [(x, y) for x, y in U 
                        if not _is_missing(val := x.get(best_attr)) and val in best_split_set]
            X_left, y_left = zip(*U_left) if U_left else ([], [])
            U_right = [(x, y) for x, y in U 
                         if not _is_missing(val := x.get(best_attr)) and val not in best_split_set]
            X_right, y_right = zip(*U_right) if U_right else ([], [])
            if self.strategy == MissingStrategy.SURROGATE:
                pass

            node = Node(
                majority_class=maj,
                condition_attr=best_attr,
                is_continuous=False,
                split_set=best_split_set,
            )
            node.default_route = "right" if len(U_right) > len(U_left) else "left"
 
        else:  # continuous
            U_left = [(x, y) for x, y in U 
                        if not _is_missing(val := x.get(best_attr)) and val <= best_t]
            X_left, y_left = zip(*U_left) if U_left else ([], [])
            U_right = [(x, y) for x, y in U 
                                                 if not _is_missing(val := x.get(best_attr)) and val > best_t]
            X_right, y_right = zip(*U_right) if U_right else ([], [])

            if self.strategy == MissingStrategy.SURROGATE:
                pass
 
            node = Node(
                majority_class=maj,
                condition_attr=best_attr,
                is_continuous=True,
                threshold=best_t,
            )
            node.default_route = "right" if len(y_right) > len(y_left) else "left"
 
        if not y_left:
            return Leaf(maj)
        if not y_right:
            return Leaf(maj)
 
        node.left  = self._build(Y, D, C, pd.DataFrame(X_left), pd.Series(y_left),  g - 1)
        node.right = self._build(Y, D, C, pd.DataFrame(X_right), pd.Series(y_right), g - 1)
        return node
 
    # ------------------------------------------------------------------
    # Predykcja
    # ------------------------------------------------------------------
 
    def predict_one(self, x: pd.Series) -> Any:
        node = self.root
        while not node.is_leaf():
            result = node.condition(x)
            if result is None:

                # brak wartości – idź domyślną gałęzią
                node = node.left if node.default_route == "left" else node.right
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
        lines.append(f"{prefix}{connector}[NODE] {cond}  (default→{node.default_route})")
        child_prefix = prefix + ("    " if is_root else ("│   " if is_left else "    "))
        self._print_node(node.left,  lines, child_prefix, is_left=True,  is_root=False)
        self._print_node(node.right, lines, child_prefix, is_left=False, is_root=False)
 