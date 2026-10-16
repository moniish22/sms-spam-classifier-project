"""Optional: word-embedding features (average Word2Vec) instead of TF-IDF.
Requires `pip install gensim`. Trained on the training texts only."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import RANDOM_STATE
from .preprocess import clean_text


class AvgWord2Vec(BaseEstimator, TransformerMixin):
    """Turns each SMS into the average of its word vectors."""

    def __init__(self, vector_size=100, window=5, min_count=2, epochs=30):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

    def _tokens(self, texts):
        return [clean_text(t).split() for t in texts]

    def fit(self, X, y=None):
        try:
            from gensim.models import Word2Vec
        except ImportError as err:
            raise ImportError("Install gensim: pip install gensim") from err
        self.w2v_ = Word2Vec(
            sentences=self._tokens(X), vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, epochs=self.epochs, seed=RANDOM_STATE, workers=1,
        )
        return self

    def transform(self, X):
        wv = self.w2v_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, tokens in enumerate(self._tokens(X)):
            vecs = [wv[t] for t in tokens if t in wv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


def build_embedding_pipelines() -> dict:
    return {
        "w2v_logreg": Pipeline([("w2v", AvgWord2Vec()),
                                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                           random_state=RANDOM_STATE))]),
        "w2v_rbf_svm": Pipeline([("w2v", AvgWord2Vec()),
                                 ("clf", SVC(kernel="rbf", C=5, class_weight="balanced",
                                             random_state=RANDOM_STATE))]),
    }
