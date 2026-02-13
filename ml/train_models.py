from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X, y):
    """
    Train Logistic Regression model
    using class_weight to handle imbalance
    """

    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced"
    )

    model.fit(X, y)
    return model


def train_decision_tree(X, y):
    """
    Train Decision Tree model
    """

    model = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)
    return model
