import pandas as pd


def load_portfolio_data():
    df = pd.read_csv("data/raw/borrower_portfolio.csv")
    return df


def preprocess_portfolio(df):
    df = df.copy()

    df["industry"] = df["industry"].astype(str)

    df = pd.get_dummies(df, columns=["industry"], drop_first=True)

    return df