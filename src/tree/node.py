import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

@dataclass
class Leaf:
    prediction: Any
 
    def is_leaf(self) -> bool:
        return True
 
 
@dataclass
class Node:
    majority_class: Any
    condition_attr: str
    is_continuous: bool
    # dyskretny: zbiór wartości trafiających w lewo
    split_set: Optional[frozenset] = None
    # ciągły: próg
    threshold: Optional[float] = None
    # domyślna gałąź gdy brak wartości atrybutu
    default_route: str = "left"
    surrogate_splits = None 
 
    left:  Any = field(default=None, repr=False)   # Node | Leaf
    right: Any = field(default=None, repr=False)   # Node | Leaf
 
    def is_leaf(self) -> bool:
        return False
 
    def condition(self, x: pd.DataFrame) -> Optional[bool]:
        """Zwraca True -> lewo, False -> prawo, None -> brak wartości."""
        val = x.get(self.condition_attr)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        if self.is_continuous:
            return float(val) <= self.threshold
        return val in self.split_set
 