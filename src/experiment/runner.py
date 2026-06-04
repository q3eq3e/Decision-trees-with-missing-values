import numpy as np
from typing import Dict, List, Any

from src.data.preprocessing import DataPreprocessor, Dataset, masking
from src.experiment.metrics import compute_metrics
from src.tree.id3_tree import DecisionTree, MissingStrategy


def run_single(
    seed: int,
    dataset_name: Dataset,
    mode: str,
    masking_column: str | None = None,
    masking_rate: float = 0.0,
    visualize=False,
) -> Dict[str, Any]:
    """
    Executes a single experiment run for a given dataset,
    missing-value handling strategy, and random seed.

    Parameters
    ----------
    seed : int
        Random seed controlling dataset split reproducibility.

    dataset_name : Dataset
        Dataset identifier.

    mode : str
        Missing value handling strategy name:
        - "default"
        - "trivial"
        - "surrogate"
        - "impute"

    visualize : bool
        Prints built tree.

    Returns
    -------
    dict
        Dictionary containing train/test metrics and overfitting gap.
    """

    X, X_test, y, y_test = DataPreprocessor(random_state=seed).prepare(dataset_name)

    if masking_column is not None:
        X = masking(X, masking_rate, masking_column, seed=seed)

    strategy_mapping: Dict[str, MissingStrategy] = {
        "default": MissingStrategy.MAJORITY,
        "trivial": MissingStrategy.TRIVIAL,
        "surrogate": MissingStrategy.SURROGATE,
        "impute": MissingStrategy.IMPUTATION,
    }

    strategy = strategy_mapping[mode]

    tree = DecisionTree(
        discrete_attrs=X.attrs["discrete_columns"],
        continuous_attrs=X.attrs["continuous_columns"],
        max_depth=10,
        strategy=strategy,
        max_surrogate_splits=5,
    )

    tree.fit(X, y)

    train_pred = tree.predict(X)
    test_pred = tree.predict(X_test)

    if visualize:
        print(tree)

    train_metrics = compute_metrics(y, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    return {
        "train": train_metrics,
        "test": test_metrics,
        "gap_accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],
        "gap_f1": train_metrics["f1"] - test_metrics["f1"],
    }


def run_experiment_25(
    dataset_name: Dataset,
    mode: str,
    masking_column: str | None = None,
    masking_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Runs 25 independent experiments with different random seeds
    and aggregates the results.

    Parameters
    ----------
    dataset_name : Dataset
        Dataset identifier.

    mode : str
        Missing value handling strategy.

    Returns
    -------
    dict
        Aggregated statistics over 25 runs.
    """

    seeds: List[int] = list(range(25))

    results: List[Dict[str, Any]] = [
        run_single(
            seed, dataset_name, mode, masking_column, masking_rate, visualize=False
        )
        for seed in seeds
    ]

    return aggregate_results(results)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates experiment results across multiple runs.

    Computes mean, standard deviation, minimum and maximum
    for each evaluation metric.

    Parameters
    ----------
    results : list of dict
        Output of multiple `run_single` executions.

    Returns
    -------
    dict
        Summary statistics for each metric.
    """

    def extract(metric: str) -> np.ndarray:
        return np.array([r["test"][metric] for r in results])

    summary: Dict[str, Dict[str, float]] = {}

    for metric in ["accuracy", "f1"]:
        values = extract(metric)

        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    gap_acc = np.array([r["gap_accuracy"] for r in results])
    gap_f1 = np.array([r["gap_f1"] for r in results])

    summary["gap_accuracy"] = {
        "mean": float(gap_acc.mean()),
        "std": float(gap_acc.std()),
        "min": float(gap_acc.min()),
        "max": float(gap_acc.max()),
    }

    summary["gap_f1"] = {
        "mean": float(gap_f1.mean()),
        "std": float(gap_f1.std()),
        "min": float(gap_f1.min()),
        "max": float(gap_f1.max()),
    }

    return summary
