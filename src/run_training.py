import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV

from src.data_prep import DATA_OUT, TARGET_COL
from src.fe import engineer_features, NUMERIC_FEATS, CATEGORICAL_FEATS
from src.model_builder import make_pipeline
from src.schema import run_validation

MODEL_DIR = Path("models")


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_OUT, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_OUT, "test.csv"))
    # валидация
    run_validation(train_df)
    run_validation(test_df)
    return train_df, test_df


def prepare_xy(train_df, test_df):
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    feats = NUMERIC_FEATS + CATEGORICAL_FEATS
    X_tr, y_tr = train_df[feats], train_df[TARGET_COL]
    X_te, y_te = test_df[feats], test_df[TARGET_COL]
    return X_tr, y_tr, X_te, y_te


def evaluate(pipe, X_te, y_te):
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_hat = pipe.predict(X_te)
    return {
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        "precision": round(precision_score(y_te, y_hat), 4),
        "recall": round(recall_score(y_te, y_hat), 4),
        "f1": round(f1_score(y_te, y_hat), 4),
    }, y_prob


def save_roc_plot(y_te, y_prob, path="models/roc.png"):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax, name="best model")
    ax.set_title("ROC curve — test set")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train_df, test_df = load_data()
    X_tr, y_tr, X_te, y_te = prepare_xy(train_df, test_df)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit_default")

    experiments = [
        (
            "gbc_baseline",
            "gbc",
            {
                "model__n_estimators": 100,
                "model__max_depth": 3,
                "model__learning_rate": 0.1,
            },
        ),
        (
            "gbc_deeper",
            "gbc",
            {
                "model__n_estimators": 200,
                "model__max_depth": 5,
                "model__learning_rate": 0.05,
            },
        ),
        (
            "gbc_more_trees",
            "gbc",
            {
                "model__n_estimators": 400,
                "model__max_depth": 4,
                "model__learning_rate": 0.05,
            },
        ),
        ("logreg_l2", "logreg", {"model__C": 1.0, "model__penalty": "l2"}),
        (
            "logreg_l1",
            "logreg",
            {"model__C": 0.5, "model__penalty": "l1", "model__solver": "saga"},
        ),
    ]

    results = []
    best_auc = 0

    for name, mtype, params in experiments:
        with mlflow.start_run(run_name=name):
            pipe = make_pipeline(NUMERIC_FEATS, CATEGORICAL_FEATS, model_type=mtype)
            pipe.set_params(**params)
            pipe.fit(X_tr, y_tr)

            mets, _ = evaluate(pipe, X_te, y_te)
            mlflow.log_params(params)
            mlflow.log_params({"model_type": mtype})
            mlflow.log_metrics(mets)
            mlflow.sklearn.log_model(
                pipe,
                f"model_{name}",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )

            results.append((name, mets))
            print(f"  [{name}] AUC={mets['roc_auc']}  F1={mets['f1']}")

            if mets["roc_auc"] > best_auc:
                best_auc = mets["roc_auc"]

    with mlflow.start_run(run_name="grid_search_gbc"):
        base_pipe = make_pipeline(NUMERIC_FEATS, CATEGORICAL_FEATS, model_type="gbc")
        grid = {
            "model__n_estimators": [150, 300],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.05, 0.1],
        }
        gs = GridSearchCV(base_pipe, grid, cv=4, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)

        final_pipe = gs.best_estimator_
        mets, y_prob = evaluate(final_pipe, X_te, y_te)

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("cv_auc", round(gs.best_score_, 4))
        mlflow.log_metrics(mets)

        save_roc_plot(y_te, y_prob)
        mlflow.log_artifact("models/roc.png")
        mlflow.sklearn.log_model(
            final_pipe,
            "final_model",
            registered_model_name="CreditDefault_v2",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

        print(f"GridSearch best params: {gs.best_params_}")
        print(f"Test AUC={mets['roc_auc']}, F1={mets['f1']}")

    # сохраняем на диск
    joblib.dump(final_pipe, MODEL_DIR / "final_model.pkl")

    with open("metrics.json", "w") as f:
        json.dump(mets, f, indent=2)

    print("model saved to models/final_model.pkl")


if __name__ == "__main__":
    main()
