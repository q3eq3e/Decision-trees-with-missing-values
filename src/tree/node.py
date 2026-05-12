from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SurrogateRule:
    attribute: str
    threshold: Any = None
    split: set = None
    direction: str = "left"  # for continuous surrogate


@dataclass
class Node:
    is_leaf: bool = False
    prediction: Any = None
    majority_class: Any = None

    attribute: Optional[str] = None
    threshold: Optional[float] = None
    split: Optional[set] = None

    default_route: str = "left"
    left: Any = None
    right: Any = None

    surrogates: List[SurrogateRule] = field(default_factory=list)
