import csv
import json
import numpy as np
from typing import Dict, Any, List, Tuple

from src.data.preprocessing import Dataset
from src.experiment.runner import run_experiment_25

DATASETS: List[Dataset] = [
    Dataset.TITANIC,
    Dataset.ADULT,
    Dataset.CARSALES,
]

STRATEGIES: List[str] = [
    "default",
    "trivial",
    "surrogate",
    "impute",
]


def flatten_result(
    dataset: Dataset,
    strategy: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts nested experiment results into a flat structure
    suitable for CSV export.

    Parameters
    ----------
    dataset : Dataset
        Dataset identifier.

    strategy : str
        Missing value handling strategy.

    result : dict
        Aggregated experiment results containing summary
        statistics for training and test metrics, as well as
        gap metrics between train and test performance.

        Expected structure:

        {
            "train": {
                "accuracy": {...},
                "f1": {...},
            },
            "test": {
                "accuracy": {...},
                "f1": {...},
            },
            "gap_accuracy": {...},
            "gap_f1": {...},
        }

        Additionally, each split contains a confusion matrix:

        {
            "train": {
                "accuracy": {...},
                "f1": {...},
                "confusion_matrix": ndarray (summed over runs)
            },
            "test": {
                "accuracy": {...},
                "f1": {...},
                "confusion_matrix": ndarray (summed over runs)
            }
        }

    Returns
    -------
    dict
        Flattened dictionary containing dataset metadata and
        aggregated scalar statistics for training and test
        metrics.

        Keys follow the convention:

        <split>_<metric>_<stat>

        Examples:
        - train_accuracy_mean
        - test_f1_std

        Gap metrics:
        - gap_accuracy_mean
        - gap_accuracy_std
        - gap_f1_mean
        - gap_f1_std

        Confusion matrices are not expanded into scalar statistics.
        They are stored separately (e.g. as JSON or flattened columns
        depending on implementation) and are not included in the
        statistical aggregation step.
    """

    row = {
        "dataset": dataset.name,
        "strategy": strategy,
    }

    metrics = [
        "accuracy",
        "f1",
    ]

    for split in ["train", "test"]:
        for metric in metrics:
            for stat in ["mean", "std", "min", "max"]:
                row[f"{split}_{metric}_{stat}"] = result[split][metric][stat]

    for metric in ["gap_accuracy", "gap_f1"]:
        row[f"{metric}_mean"] = result[metric]["mean"]
        row[f"{metric}_std"] = result[metric]["std"]

    row["train_confusion_matrix"] = json.dumps(
        result["train"]["confusion_matrix"].tolist()
    )
    row["test_confusion_matrix"] = json.dumps(
        result["test"]["confusion_matrix"].tolist()
    )

    return row


def run_all() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs full experimental grid:
    - multiple datasets
    - multiple missing-value strategies
    - 25 random seeds per configuration

    Returns
    -------
    tuple
        - flattened results for CSV export
        - hierarchical results for JSON export
    """

    all_results: List[Dict[str, Any]] = []
    detailed_results: Dict[str, Dict[str, Any]] = {}

    for dataset in DATASETS:
        detailed_results[dataset.name] = {}

        for strategy in STRATEGIES:
            print(f"Running experiment: {dataset.name} | {strategy}")

            result = run_experiment_25(
                dataset_name=dataset,
                mode=strategy,
            )

            detailed_results[dataset.name][strategy] = result

            all_results.append(flatten_result(dataset, strategy, result))

    return all_results, detailed_results


def save_csv(rows: List[Dict[str, Any]], path: str = "results/results.csv") -> None:
    """
    Saves flattened experiment results to CSV file.
    """
    if not rows:
        return

    keys = rows[0].keys()

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: Dict[str, Any], path: str = "results/results.json") -> None:
    """
    Saves full hierarchical experiment results to JSON file.
    """

    data = make_json_serializable(data)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    csv_rows, full_results = run_all()

    save_csv(csv_rows, "results/results.csv")
    save_json(full_results, "results/results.json")

    print("Finished experiments")
