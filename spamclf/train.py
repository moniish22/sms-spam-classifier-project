"""Train, compare and save the best model.

Usage (from project root):
    python -m spamclf.train                  # baseline comparison
    python -m spamclf.train --tune           # + grid-search each model
    python -m spamclf.train --embeddings     # + Word2Vec models (needs gensim)
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from . import config
from .data import load_dataset, split_data
from .evaluate import (compute_metrics, get_scores, misclassified, plot_comparison,
                       plot_confusion_matrix, top_features)
from .models import PARAM_GRIDS, build_pipelines


def main(argv=None):
    p = argparse.ArgumentParser(description="Train SMS spam classifiers")
    p.add_argument("--data", default=str(config.DATA_PATH))
    p.add_argument("--tune", action="store_true", help="grid-search hyper-parameters")
    p.add_argument("--embeddings", action="store_true", help="also try Word2Vec models")
    args = p.parse_args(argv)

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    df = load_dataset(args.data)
    print(f"Loaded {len(df)} unique messages | spam = {df.label.mean():.1%}")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train = {len(X_train)} | Test = {len(X_test)}\n")

    pipelines = build_pipelines()
    if args.embeddings:
        from .embeddings import build_embedding_pipelines
        pipelines.update(build_embedding_pipelines())

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}

    for name, pipe in pipelines.items():
        print(f"=== {name} ===")
        if args.tune and name in PARAM_GRIDS:
            search = GridSearchCV(pipe, PARAM_GRIDS[name], scoring="f1", cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            cv_f1 = search.best_score_
            print(f"best params: {search.best_params_}")
        else:
            cv_f1 = cross_val_score(pipe, X_train, y_train, scoring="f1", cv=cv, n_jobs=-1).mean()
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, get_scores(pipe, X_test))
        metrics.update({"model": name, "cv_f1_train": cv_f1})
        rows.append(metrics)
        fitted[name] = (pipe, y_pred)
        print(f"CV F1 (train) = {cv_f1:.4f} | Test F1 = {metrics['f1']:.4f} | "
              f"P = {metrics['precision']:.4f} | R = {metrics['recall']:.4f}\n")
        plot_confusion_matrix(y_test, y_pred, name, config.REPORTS_DIR / f"cm_{name}.png")

    results = pd.DataFrame(rows)
    first = ["model", "cv_f1_train", "accuracy", "precision", "recall", "f1", "roc_auc"]
    results = results[[c for c in first if c in results.columns]].sort_values(
        "cv_f1_train", ascending=False)
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    plot_comparison(results, config.REPORTS_DIR / "model_comparison.png")
    print(results.round(4).to_string(index=False))

    # Choose the winner using CROSS-VALIDATION on training data (not the test set),
    # so the test set stays an honest, untouched final exam.
    best_name = results.iloc[0]["model"]
    best_pipe, best_pred = fitted[best_name]
    print(f"\nBest model (by CV F1): {best_name}\n")
    print(classification_report(y_test, best_pred, target_names=["ham", "spam"], digits=4))

    joblib.dump(best_pipe, config.MODELS_DIR / "best_model.joblib")
    with open(config.MODELS_DIR / "best_model_info.json", "w") as f:
        json.dump({"model": best_name,
                   "metrics": results.iloc[0].to_dict()}, f, indent=2, default=float)

    misclassified(X_test, y_test, best_pred).to_csv(
        config.REPORTS_DIR / "misclassified.csv", index=False)
    tf = top_features(best_pipe)
    if tf is not None:
        tf.to_csv(config.REPORTS_DIR / "top_features.csv", index=False)
    print(f"Saved model to {config.MODELS_DIR / 'best_model.joblib'}")
    print(f"Reports in {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
