## authors: Jakub Bagiński, Maciej Borkowski

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional
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
    # set of attribute values for discrete attributes
    split_set: Optional[frozenset] = None
    # threshold for continuous attributes
    threshold: Optional[float] = None
    # default branch when attribute value is missing (used in default mode)
    default_route: str = "left"
    surrogate_splits: Optional[List] = None
 
    left:  Any = field(default=None, repr=False)   # Node | Leaf
    right: Any = field(default=None, repr=False)   # Node | Leaf
 
    def is_leaf(self) -> bool:
        return False
 
    def condition(self, x: pd.Series) -> Optional[bool]:
        """
        True -> go left
        False -> go right
        """
        val = x.get(self.condition_attr)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        if self.is_continuous:
            return float(val) <= self.threshold
        return val in self.split_set
