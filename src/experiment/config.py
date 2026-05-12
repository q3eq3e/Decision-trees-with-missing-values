from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DatasetConfig:
    name: str
    path: str
    target_column: str
    task: str  # "classification" or "regression"


@dataclass
class ExperimentConfig:
    max_depth: int = 8
    min_samples_split: int = 10
    n_folds: int = 5
    random_state: int = 42


DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        name="Titanic",
        path="data/Titanic.csv",
        target_column="Survived",
        task="classification",
    ),
    DatasetConfig(
        name="CarSales",
        path="data/CarSales.csv",
        target_column="Price",
        task="regression",
    ),
    DatasetConfig(
        name="Adult",
        path="data/Adult.csv",
        target_column="income",
        task="classification",
    ),
]

EXPERIMENT = ExperimentConfig()

MISSING_STRATEGIES = [
    "trivial",
    "default",
    "surrogate",
]
