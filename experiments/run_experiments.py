import csv
import json
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
        Aggregated experiment results.

    Returns
    -------
    dict
        Flattened dictionary for tabular storage.
    """

    row: Dict[str, Any] = {
        "dataset": dataset.name,
        "strategy": strategy,
    }

    for metric in ["accuracy", "f1"]:
        for stat in ["mean", "std", "min", "max"]:
            row[f"{metric}_{stat}"] = result[metric][stat]

    if "gap_accuracy" in result:
        row["gap_accuracy_mean"] = result["gap_accuracy"]["mean"]
        row["gap_accuracy_std"] = result["gap_accuracy"]["std"]

    if "gap_f1" in result:
        row["gap_f1_mean"] = result["gap_f1"]["mean"]
        row["gap_f1_std"] = result["gap_f1"]["std"]

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


def save_csv(rows: List[Dict[str, Any]], path: str = "results.csv") -> None:
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


def save_json(data: Dict[str, Any], path: str = "results.txt") -> None:
    """
    Saves full hierarchical experiment results to JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    csv_rows, full_results = run_all()

    save_csv(csv_rows, "results.csv")
    save_json(full_results, "results.txt")

    print("Finished experiments ✔")
