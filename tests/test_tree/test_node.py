
from src.tree.node import Node
import pandas as pd
import numpy as np
    
def test_condition_continuous():
    node = Node(
        majority_class=0,
        condition_attr="feature",
        is_continuous=True,
        threshold=2.5,
    )

    # Test continuous condition
    assert node.condition(pd.Series({"feature": 2.0})) is True
    assert node.condition(pd.Series({"feature": 3.0})) is False

    # Test missing value
    assert node.condition(pd.Series({"feature": np.nan})) is None

def test_condition_discrete():
    node = Node(
        majority_class=0,
        condition_attr="color",
        is_continuous=False,
        split_set=frozenset({"red", "blue"}),
    )

    # Test discrete condition
    assert node.condition(pd.Series({"color": "red"})) is True
    assert node.condition(pd.Series({"color": "green"})) is False

    # Test missing value
    assert node.condition(pd.Series({"color": None})) is None