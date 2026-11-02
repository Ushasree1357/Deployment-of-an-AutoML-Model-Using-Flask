from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


def main():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compare model families with cross-validation using training data only.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    parameter_grid = [
        {"classifier": [LogisticRegression(max_iter=2000, random_state=42)],
         "classifier__C": [0.1, 1.0, 10.0]},
        {"classifier": [SVC(probability=True, random_state=42)],
         "classifier__C": [0.1, 1.0, 10.0],
         "classifier__kernel": ["linear", "rbf"]},
        {"classifier": [RandomForestClassifier(random_state=42)],
         "classifier__n_estimators": [100, 200],
         "classifier__max_depth": [None, 5]},
        {"classifier": [ExtraTreesClassifier(random_state=42)],
         "classifier__n_estimators": [100, 200],
         "classifier__max_depth": [None, 5]},
        {"classifier": [DecisionTreeClassifier(random_state=42)],
         "classifier__max_depth": [None, 3, 5]},
    ]
    search = GridSearchCV(
        pipeline, parameter_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=iris.target_names,
        output_dict=True,
        zero_division=0,
    )
    bundle = {
        "model": model,
        "accuracy": float(acc),
        "classification_report": report,
        "best_model": type(model.named_steps["classifier"]).__name__,
        "best_params": search.best_params_,
        "feature_names": list(X.columns),
        "target_names": list(iris.target_names),
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"Selected model: {bundle['best_model']}")
    print(f"Best cross-validation accuracy: {search.best_score_:.2%}")
    print(f"Held-out test accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(f"Saved model bundle to {MODEL_PATH}")

if __name__ == '__main__':
    main()
