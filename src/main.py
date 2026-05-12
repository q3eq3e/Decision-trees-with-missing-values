from src.data.loaders import load_titanic, load_adult, load_car
from src.experiment.runner import run_experiment
from src.data.preprocessing import DataPreprocessor, Dataset


def run_dataset(loader):
    df = loader()
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return run_experiment(X, y)


if __name__ == "__main__":
    # print("Titanic:", run_dataset(load_titanic))
    # print("Adult:", run_dataset(load_adult))
    # print("Car:", run_dataset(load_car))
    X, test_x, y, test_y = DataPreprocessor().prepare(Dataset.CARSALES)
    print(len(X), len(y), len(test_x), len(test_y))
    # print("Titanic:", X, y, X.discrete_columns, X.continuous_columns)
    # print("Adult:", DataPreprocessor().prepare(Dataset.ADULT))
    print("Car:", DataPreprocessor().prepare(Dataset.CARSALES))
