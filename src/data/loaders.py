import pandas as pd


def load_titanic():
    df = pd.read_csv("data/Titanic.csv")
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    return df


def load_adult():
    df = pd.read_csv("data/Adult1000.csv")
    df = df.drop(columns=["education", "occupation", "native-country"])
    return df


def load_car():
    df = pd.read_csv("data/CarSales.csv")
    return df
