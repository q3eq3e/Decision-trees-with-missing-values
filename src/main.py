from src.data.loaders import load_titanic, load_adult, load_car
from src.experiment.runner import run_experiment
from src.data.preprocessing import DataPreprocessor, Dataset
import warnings
from src.missing_values.knn_imputation import KNNImputationStrategy

warnings.filterwarnings("ignore")


def run_dataset(loader):
    df = loader()
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return run_experiment(X, y)


if __name__ == "__main__":
    X, test_x, y, test_y = DataPreprocessor().prepare(Dataset.ADULT)
    imputer = KNNImputationStrategy(n_neighbors=5)
    X = imputer.fit_transform(X)
    test_x = imputer.transform(test_x)
    print(len(X), len(y), len(test_x), len(test_y))
    # print("Car:", X, y)
    print(X.isna().sum(), test_x.isna().sum(), y.isna().sum(), test_y.isna().sum())
