import argparse

from src.data.loaders import load_titanic, load_adult, load_car
from src.experiment.runner import run_experiment
from src.data.preprocessing import DataPreprocessor, Dataset
import warnings
from src.missing_values.knn_imputation import CustomKNNImputer
import argparse

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
        "--mode",
        type=str,
        choices=["impute", "default", "trivial", "surrogate"],
        default="default",
        help="Mode of operation for the experiment.",
    )
    args = parser.parse_args()

    knn_impute = args.mode == "impute"
    dataset_mapping = {
        "titanic": Dataset.TITANIC,
        "adult": Dataset.ADULT,
        "carsales": Dataset.CARSALES,
    }
    dataset = dataset_mapping[args.dataset]
    X, test_x, y, test_y = DataPreprocessor().prepare(dataset)
    discrete_columns = X.discrete_columns
    continuous_columns = X.continuous_columns
    if knn_impute:
        imputer = CustomKNNImputer(
            n_neighbors=1,
            discrete_columns=discrete_columns,
            continuous_columns=continuous_columns,
        )
        X = imputer.fit_transform(X)
        test_x = imputer.transform(test_x)
        X.discrete_columns = discrete_columns
        X.continuous_columns = continuous_columns
        test_x.discrete_columns = discrete_columns
        test_x.continuous_columns = continuous_columns
    # print(len(X), len(y), len(test_x), len(test_y))
    # # print("Car:", X, y)
    print(X.isna().sum(), test_x.isna().sum(), y.isna().sum(), test_y.isna().sum())
    print(run_experiment(X, y, test_x, test_y, args.mode))
