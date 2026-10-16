# SMS Spam Classifier

Classify SMS messages as **spam** or **legitimate (ham)** using TF-IDF features
(and optionally Word2Vec embeddings) with Naive Bayes, Logistic Regression and
Support Vector Machines.

> **Start here:** open `docs/INTERN_GUIDE.md` and follow it step by step.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put your Kaggle file at data/spam.csv, then:
python -m spamclf.eda                # explore the data
python -m spamclf.train              # train + compare models
python -m spamclf.train --tune       # (optional) hyper-parameter search
python -m spamclf.predict "Congratulations! You won a free £500 voucher, call now"
python -m pytest -q                  # run tests
```

Optional: `pip install -r requirements-optional.txt` then
`python -m spamclf.train --embeddings` and `streamlit run app.py`.

## Project layout

```
sms-spam-classifier/
├── data/                 <- put spam.csv here (not included)
├── spamclf/
│   ├── config.py         paths + constants
│   ├── data.py           load, clean labels, de-duplicate, split
│   ├── preprocess.py     text cleaning (URL/phone/number tokens)
│   ├── models.py         TF-IDF + NB / LogReg / SVM pipelines + tuning grids
│   ├── embeddings.py     optional average-Word2Vec features
│   ├── evaluate.py       metrics, plots, error analysis, top words
│   ├── eda.py            exploratory analysis
│   ├── train.py          train / compare / save best model
│   └── predict.py        classify new messages (CLI)
├── tests/test_basic.py   unit tests
├── app.py                optional Streamlit demo
├── models/               saved model appears here
├── reports/              charts + CSV results appear here
└── docs/
    ├── INTERN_GUIDE.md   detailed step-by-step guide
    └── REPORT_TEMPLATE.md  final write-up template
```
