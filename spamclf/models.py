"""Model definitions: TF-IDF + {Naive Bayes, Logistic Regression, SVM}."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE
from .preprocess import clean_text


def _word_tfidf():
    return TfidfVectorizer(
        preprocessor=clean_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )


def _char_tfidf():
    return TfidfVectorizer(
        preprocessor=clean_text, analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, sublinear_tf=True,
    )


def build_pipelines() -> dict:
    """Name -> sklearn Pipeline. The vectorizer lives INSIDE the pipeline so it
    is fitted only on training folds (no leakage)."""
    return {
        "tfidf_naive_bayes": Pipeline([("tfidf", _word_tfidf()),
                                       ("clf", MultinomialNB(alpha=0.1))]),
        "tfidf_logreg": Pipeline([("tfidf", _word_tfidf()),
                                  ("clf", LogisticRegression(C=10, max_iter=2000,
                                                             class_weight="balanced",
                                                             random_state=RANDOM_STATE))]),
        "tfidf_linear_svm": Pipeline([("tfidf", _word_tfidf()),
                                      ("clf", LinearSVC(C=1.0, class_weight="balanced",
                                                        random_state=RANDOM_STATE))]),
        "char_tfidf_linear_svm": Pipeline([("tfidf", _char_tfidf()),
                                           ("clf", LinearSVC(C=1.0, class_weight="balanced",
                                                             random_state=RANDOM_STATE))]),
    }


# Hyper-parameter grids used with --tune (keys are "<pipeline step>__<parameter>")
PARAM_GRIDS = {
    "tfidf_naive_bayes": {"tfidf__ngram_range": [(1, 1), (1, 2)],
                          "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "tfidf_logreg": {"tfidf__ngram_range": [(1, 1), (1, 2)],
                     "clf__C": [1, 5, 10, 50]},
    "tfidf_linear_svm": {"tfidf__ngram_range": [(1, 1), (1, 2)],
                         "clf__C": [0.1, 0.5, 1, 5]},
    "char_tfidf_linear_svm": {"clf__C": [0.1, 0.5, 1, 5]},
}
