## authors: Jakub Bagiński, Maciej Borkowski

from dataclasses import dataclass
from typing import List


@dataclass
class DatasetConfig:
    name: str
    path: str
    target_column: str


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
    ),
    DatasetConfig(
        name="CarSales",
        path="data/CarSales.csv",
        target_column="Price",
    ),
    DatasetConfig(
        name="Adult",
        path="data/Adult.csv",
        target_column="income",
    ),
]

EXPERIMENT = ExperimentConfig()

MISSING_STRATEGIES = [
    "trivial",
    "default",
    "surrogate",
]
