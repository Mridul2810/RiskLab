from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def train_default_model(df):
    df = df.copy()

    X = df.drop(columns=["default", "default_probability_true"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_prob, auc, report


def add_predicted_pd(df, model):
    df = df.copy()

    X = df.drop(columns=["default", "default_probability_true"])
    df["predicted_pd"] = model.predict_proba(X)[:, 1]

    return df