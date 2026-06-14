## authors: Jakub Bagiński, Maciej Borkowski

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
    masking_rate: float,
    strategy: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:

    row = {
        "dataset": dataset.name,
        "masking_rate": masking_rate,
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

    all_results = []
    detailed_results = {}

    for dataset in DATASETS:
        detailed_results[dataset.name] = {}
        for synteticly_missing_rate in [0, 0.25, 0.5]:
            detailed_results[dataset.name][synteticly_missing_rate] = {}
            # Choosing first data column
            if dataset == Dataset.CARSALES:
                synteticly_missing_column = "Make"
            elif dataset == Dataset.ADULT:
                synteticly_missing_column = "age"
            else:
                synteticly_missing_column = "Pclass"
            for strategy in STRATEGIES:
                print(f"Running experiment: {dataset.name} | {synteticly_missing_rate} | {strategy}")

                result = run_experiment_25(
                    dataset_name=dataset,
                    mode=strategy,
                    masking_column=synteticly_missing_column,
                    masking_rate=synteticly_missing_rate if synteticly_missing_rate else 0.
                )

                detailed_results[dataset.name][synteticly_missing_rate][strategy] = result

                all_results.append(flatten_result(dataset, synteticly_missing_rate, strategy, result))

    return all_results, detailed_results


def save_csv(rows: List[Dict[str, Any]], path: str = "results/results.csv") -> None:
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

    data = make_json_serializable(data)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    csv_rows, full_results = run_all()

    save_csv(csv_rows, "results/results.csv")
    save_json(full_results, "results/results.json")

    print("Finished experiments")
