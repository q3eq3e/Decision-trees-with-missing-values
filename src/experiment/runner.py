import numpy as np
from sklearn.model_selection import train_test_split
from ..tree.id3_tree import DecisionTree
from ..tree.predict import predict
from .metrics import compute_metrics


def run_experiment(X, y, runs=25):
    results = []

    for seed in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        tree = DecisionTree(
            discrete_attrs=X_train.discrete_columns,
            continuous_attrs=X_train.continuous_columns,
            max_depth=5
        )
        tree.fit(X_train, y_train)

        train_pred = predict(tree, X_train)
        test_pred = predict(tree, X_test)

        results.append(
            {
                "train": compute_metrics(y_train, train_pred),
                "test": compute_metrics(y_test, test_pred),
            }
        )

    return results
