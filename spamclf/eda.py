"""Exploratory data analysis. Run: python -m spamclf.eda  (plots saved to reports/eda)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from . import config
from .data import load_dataset
from .preprocess import clean_text


def top_words(texts, n=15):
    cv = CountVectorizer(preprocessor=clean_text, stop_words="english")
    counts = cv.fit_transform(texts).sum(axis=0).A1
    return pd.Series(counts, index=cv.get_feature_names_out()).nlargest(n)


def main():
    out = config.REPORTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(drop_duplicates=False)
    df = load_dataset()
    print(f"Rows (raw): {len(raw)} | after removing duplicates: {len(df)}")
    print("\nClass balance:\n", df.label.map({0: "ham", 1: "spam"}).value_counts())

    df["char_len"] = df.text.str.len()
    df["word_len"] = df.text.str.split().str.len()
    df["digits"] = df.text.str.count(r"\d")
    df["has_url"] = df.text.str.contains(r"http|www\.", case=False).astype(int)
    print("\nMean stats per class (0=ham, 1=spam):\n",
          df.groupby("label")[["char_len", "word_len", "digits", "has_url"]].mean().round(2))

    plt.figure(figsize=(5, 4))
    sns.countplot(x=df.label.map({0: "ham", 1: "spam"}))
    plt.title("Class balance"); plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=df.assign(cls=df.label.map({0: "ham", 1: "spam"})), x="char_len",
                 hue="cls", bins=40, stat="density", common_norm=False)
    plt.title("Message length by class"); plt.tight_layout()
    plt.savefig(out / "length_by_class.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (lab, name) in zip(axes, [(0, "ham"), (1, "spam")]):
        tw = top_words(df[df.label == lab].text)
        sns.barplot(x=tw.values, y=tw.index, ax=ax); ax.set_title(f"Top words: {name}")
    plt.tight_layout(); plt.savefig(out / "top_words.png", dpi=150); plt.close()
    print(f"\nPlots saved in {out}")


if __name__ == "__main__":
    main()
