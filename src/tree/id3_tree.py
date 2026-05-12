


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
 
# Sample = Dict[str, Any]          # słownik cecha -> wartość (None / NaN = brak)
Dataset = pd.DataFrame
 
class MissingStrategy(Enum):
    TRIVAL = auto()
    MAJORITY = auto()
    SURROGATE = auto()
    IMPUTATION = auto()

def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False

def _entropy(dataset: Dataset) -> float:
    if not dataset:
        return 0.0
    counts = Counter(y for _, y in dataset)
    n = len(dataset)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)
 
 
def _majority(dataset: Dataset) -> Any:
    return Counter(y for _, y in dataset).most_common(1)[0][0]


def best_threshold(attr: str, dataset: Dataset,
                   strategy: MissingStrategy) -> Tuple[float, Optional[float]]:
    # prepared = _prepare_dataset(dataset, attr, strategy, is_continuous=True)
    prepared = dataset
    if len(prepared) < 2:
        return -math.inf, None
 
    sorted_u = sorted(prepared, key=lambda t: t[0][attr])
    base_ent = _entropy(prepared)
    n = len(sorted_u)
 
    best_gain = -math.inf
    best_t: Optional[float] = None
 
    for i in range(n - 1):
        if sorted_u[i][1] == sorted_u[i + 1][1]:
            continue
        t = (sorted_u[i][0][attr] + sorted_u[i + 1][0][attr]) / 2
        left  = [(x, y) for x, y in sorted_u if x[attr] <= t]
        right = [(x, y) for x, y in sorted_u if x[attr] >  t]
        if strategy == MissingStrategy.SURROGATE:
            pass
        # if strategy == MissingStrategy.FRACTION:
        #     gain = _gain_fraction(
        #         dataset, attr,
        #         lambda x, _t=t: float(x[attr]) <= _t,
        #         lambda x, _t=t: float(x[attr]) >  _t,
        #     )
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
 
def best_split(attr: str, dataset: Dataset,
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
        strategy:         MissingStrategy = MissingStrategy.IGNORE,
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
        dataset: Dataset = pd.concat([X, y], axis=1)
        classes = sorted(set(y))
        self.root = self._build(classes, self.discrete_attrs,
                                self.continuous_attrs, dataset, self.max_depth)
        return self
 
    def _build(self, Y, D, C, U: Dataset, g: int):
        # --- warunki stopu ---
        classes = [row.iloc[-1] for row in U.iterrows()]
        if len(set(classes)) == 1:
            return Leaf(classes[0])
 
        all_same = all(
            len({x.get(a) for x, _ in U}) <= 1 for a in D + C
        )
        if all_same or g == 0:
            return Leaf(_majority(U))
 
        # --- szukamy najlepszego podziału ---
        best_gain = -math.inf
        best_attr = None
        best_type = None   # "discrete" | "continuous"
        best_split_set = None
        best_t = None
 
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
            return Leaf(_majority(U))
 
        # --- budujemy węzeł ---
        maj = _majority(U)
 
        if best_type == "discrete":
            U_left  = [(x, y) for x, y in U if not _is_missing(x.get(best_attr)) and     x[best_attr]  in best_split_set]
            U_right = [(x, y) for x, y in U if not _is_missing(x.get(best_attr)) and     x[best_attr] not in best_split_set]
 
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
            U_left  = [(x, y) for x, y in U if not _is_missing(x.get(best_attr)) and float(x[best_attr]) <= best_t]
            U_right = [(x, y) for x, y in U if not _is_missing(x.get(best_attr)) and float(x[best_attr]) >  best_t]
 
            if self.strategy == MissingStrategy.SURROGATE:
                pass
 
            node = Node(
                majority_class=maj,
                condition_attr=best_attr,
                is_continuous=True,
                threshold=best_t,
            )
            node.default_route = "right" if len(U_right) > len(U_left) else "left"
 
        if not U_left:
            return Leaf(maj)
        if not U_right:
            return Leaf(maj)
 
        node.left  = self._build(Y, D, C, U_left,  g - 1)
        node.right = self._build(Y, D, C, U_right, g - 1)
        return node
 
    # ------------------------------------------------------------------
    # Predykcja
    # ------------------------------------------------------------------
 
    def predict_one(self, x: pd.DataFrame) -> Any:
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
        return [self.predict_one(x) for x in X]
 