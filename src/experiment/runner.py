## authors: Jakub Bagiński, Maciej Borkowski

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
    X, X_test, y, y_test = DataPreprocessor(random_state=seed).prepare(dataset_name)

    if masking_column is not None and masking_rate is not None and masking_rate > 0:
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
    """
    seeds = list(range(25))

    results = [
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
    for each evaluation metric on training and test sets.
    """
    metrics = [
        "accuracy",
        "f1",
    ]

    summary = {
        "train": {},
        "test": {},
    }

    for split in ["train", "test"]:
        for metric in metrics:
            values = np.array([r[split][metric] for r in results])

            summary[split][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        cms = [r[split]["confusion_matrix"] for r in results]

        summary[split]["confusion_matrix"] = np.sum(
            cms,
            axis=0,
        )

    for gap_metric in ["gap_accuracy", "gap_f1"]:
        values = np.array([r[gap_metric] for r in results])
        summary[gap_metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return summary
