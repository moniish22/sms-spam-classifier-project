import pandas as pd

from spamclf.data import load_dataset
from spamclf.models import build_pipelines
from spamclf.preprocess import clean_text


def test_clean_text_replaces_entities():
    out = clean_text("WIN £1000 now!! Call 08001234567 or visit www.prize.com")
    assert "currencytoken" in out and "phonetoken" in out and "urltoken" in out
    assert out == out.lower()


def test_loader_handles_kaggle_format(tmp_path):
    f = tmp_path / "spam.csv"
    pd.DataFrame({"v1": ["ham", "spam", "spam"], "v2": ["hi", "win cash", "win cash"],
                  "Unnamed: 2": [None] * 3}).to_csv(f, index=False, encoding="latin-1")
    df = load_dataset(f)
    assert list(df.columns) == ["label", "text"]
    assert len(df) == 2  # duplicate removed
    assert set(df.label) == {0, 1}


def test_pipelines_fit_and_predict():
    X = pd.Series(["win free cash now", "claim prize free", "free entry win",
                   "see you at lunch", "call me later", "ok cool thanks"] * 4)
    y = pd.Series([1, 1, 1, 0, 0, 0] * 4)
    for name, pipe in build_pipelines().items():
        pipe.fit(X, y)
        assert pipe.predict(["free cash prize"])[0] == 1, name
