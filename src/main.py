from src.data.loaders import load_titanic, load_adult, load_car
from src.experiments.runner import run_experiment


def run_dataset(loader):
    df = loader()
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return run_experiment(X, y)


if __name__ == "__main__":
    print("Titanic:", run_dataset(load_titanic))
    print("Adult:", run_dataset(load_adult))
    print("Car:", run_dataset(load_car))
