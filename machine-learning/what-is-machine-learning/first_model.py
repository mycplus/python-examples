"""A first machine learning model, end to end, in 39 lines."""
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_split(seed=0):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=seed, stratify=iris.target
    )
    return iris, X_train, X_test, y_train, y_test


def fit(X_train, y_train):
    # Scaling inside the pipeline keeps the test rows out of the fit.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(X_train, y_train)


def main():
    iris, X_train, X_test, y_train, y_test = load_split()
    model = fit(X_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    predicted, expected = model.predict(X_test), baseline.predict(X_test)
    print(f"data     {len(iris.data)} rows, {iris.data.shape[1]} features")
    print(f"split    {len(X_train)} training rows, {len(X_test)} test rows")
    print(f"baseline accuracy {accuracy_score(y_test, expected):.4f}")
    print(f"model accuracy    {accuracy_score(y_test, predicted):.4f}")
    print("confusion matrix, rows are true classes")
    print(confusion_matrix(y_test, predicted))


if __name__ == "__main__":
    main()
