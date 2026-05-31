import math
from collections import Counter
from typing import Any
import pandas as pd


def is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def entropy(dataset: pd.DataFrame) -> float:
    if not dataset:
        return 0.0
    counts = Counter(y for _, y in dataset)
    n = len(dataset)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def majority(Y: pd.DataFrame) -> Any:
    return Counter(Y).most_common(1)[0][0]

