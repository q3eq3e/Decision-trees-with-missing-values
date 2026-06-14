import numpy as np
from typing import Dict, List, Any

from src.missing_values.knn_imputation import CustomKNNImputer
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
        Dictionary containing train/test metrics.
    """

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

    if strategy == MissingStrategy.IMPUTATION:
        imputer = CustomKNNImputer(
            n_neighbors=2,
            discrete_columns=X.attrs["discrete_columns"],
            continuous_columns=X.attrs["continuous_columns"],
        )
        X = imputer.fit_transform(X)
        X_test = imputer.transform(X_test)

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
    for each evaluation metric on training and test sets.

    Parameters
    ----------
    results : list of dict
        Output of multiple `run_single` executions.

    Returns
    -------
    dict
        Summary statistics for training metrics, test metrics
        and train-test performance gaps.
    """

    metrics = [
        "accuracy",
        "f1",
    ]

    summary: Dict[str, Any] = {
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
