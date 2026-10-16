"""Metrics, plots and error analysis helpers."""
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def get_scores(model, X):
    """Continuous spam score: probability if available, else squashed margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    margin = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-margin))


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(set(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
    return m


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(results: pd.DataFrame, path):
    cols = [c for c in ["precision", "recall", "f1"] if c in results.columns]
    ax = results.set_index("model")[cols].plot(kind="bar", figsize=(8, 4.5), ylim=(0.8, 1.0))
    ax.set_title("Test-set performance by model")
    ax.set_ylabel("score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def top_features(pipeline, n=20):
    """Most spam-indicating / ham-indicating words (linear models only)."""
    if "tfidf" not in pipeline.named_steps:
        return None
    clf = pipeline.named_steps["clf"]
    names = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    if hasattr(clf, "coef_"):
        weights = clf.coef_.ravel()
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return None
    order = np.argsort(weights)
    n = min(n, len(names))
    return pd.DataFrame({
        "top_spam_words": names[order[::-1][:n]],
        "top_ham_words": names[order[:n]],
    })


def misclassified(X_test, y_test, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": y_pred})
    df["error_type"] = np.where(df["actual"] == 1, "false_negative (spam missed)",
                                "false_positive (ham flagged)")
    return df[df["actual"] != df["predicted"]]
