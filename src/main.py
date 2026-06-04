import warnings
import argparse
from src.experiment.runner import run_single
from src.data.preprocessing import Dataset

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # add argparser to determine dataset from CLI and whether to impute or not
    parser = argparse.ArgumentParser(description="Run experiment on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["titanic", "adult", "carsales"],
        default="carsales",
        help="Dataset to run the experiment on.",
    )

    parser.add_argument(
        "--masking_column",
        type=str,
        required=False,
        help="Column to apply masking to.",
    )
    parser.add_argument(
        "--masking_rate",
        type=float,
        default=0.3,
        help="Rate of masking for the specified column (between 0 and 1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["impute", "default", "trivial", "surrogate"],
        default="default",
        help="Mode of operation for the experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2137,
        help="Run a single experiment with given seed.",
    )
    args = parser.parse_args()

    dataset = Dataset[args.dataset.upper()]

    result = run_single(
        seed=args.seed,
        dataset_name=dataset,
        mode=args.mode,
        masking_column=args.masking_column,
        masking_rate=args.masking_rate,
        visualize=True,
    )
    print(result)
