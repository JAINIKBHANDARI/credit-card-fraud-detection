from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_logistic_regression(X, y):
    model = LogisticRegression(
    max_iter=5000,
    solver="lbfgs",
   
)

    model.fit(X, y)
    return model

def train_decision_tree(X, y):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model
