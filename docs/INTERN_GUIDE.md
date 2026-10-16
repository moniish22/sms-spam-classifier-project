# Intern Guide: Build an SMS Spam Classifier (Step by Step)

**Time needed:** about 6-8 working days (can be compressed to 3-4 if you are experienced)
**Skills practised:** Python, pandas, text preprocessing, TF-IDF, Naive Bayes, Logistic Regression, SVM, model evaluation, error analysis, packaging a model.
**You will deliver:** a trained model, charts, a results table, an error analysis, and a short report.

---

## 0. The big picture

Phone users get two kinds of text messages:

- **ham** = legitimate (a friend, a bank OTP, a delivery update)
- **spam** = unwanted/fraudulent ("WINNER!! Claim your £1000 prize now")

Your job: teach a program to look at the text of a message and say *spam* or *ham*.

```
raw SMS  ->  clean text  ->  numbers (TF-IDF)  ->  classifier  ->  spam / ham
```

Computers cannot read words, so we convert each message into a list of numbers (**features**). A classifier then learns which numbers usually mean spam.

---

## 1. Concepts you must understand first (read before coding)

### 1.1 TF-IDF
- **TF (term frequency):** how often a word appears in *this* message.
- **IDF (inverse document frequency):** how *rare* the word is across all messages. Common words ("the", "u") get low weight; distinctive words ("prize", "claim") get high weight.
- **TF-IDF = TF x IDF.** Each message becomes a long vector, one number per vocabulary word.
- **n-grams:** word pairs like "call now" or "free entry" (`ngram_range=(1,2)`) capture phrases that single words miss.
- **Character n-grams:** pieces like "wi", "win", "winn" help with misspellings ("W1NNER", "fr33").

### 1.2 The three classifiers
| Model | Idea in one line | Strengths | Watch out |
|---|---|---|---|
| **Multinomial Naive Bayes** | Uses word-count probabilities, assuming words are independent | Very fast, strong text baseline | Probabilities are over-confident |
| **Logistic Regression** | Learns a weight per word, then squashes the sum into a 0-1 probability | Interpretable, gives probabilities | Needs enough iterations/regularisation |
| **Linear SVM (LinearSVC)** | Finds the widest-margin line separating spam from ham | Often best on TF-IDF text | No native probability (we use margin score) |

### 1.3 Word embeddings (optional experiment)
Word2Vec gives each word a dense vector (say 100 numbers) where similar words are close. We average the word vectors of a message. With only ~5,000 short SMS, TF-IDF usually *beats* embeddings; you will test this yourself and explain why.

### 1.4 Metrics (very important: accuracy alone is misleading)
Only ~13% of the data is spam. A silly model that says "ham" for everything gets ~87% accuracy and catches **zero** spam.

- **Precision** = of messages predicted spam, how many really are spam? (Low precision = real messages get blocked.)
- **Recall** = of all real spam, how many did we catch? (Low recall = spam gets through.)
- **F1** = balance of precision and recall.
- **Confusion matrix:** counts of TP, TN, FP, FN.
  - **False positive (FP)** = legit message marked spam. *Usually the worse mistake* (a user misses a bank OTP).
  - **False negative (FN)** = spam let through.
- **ROC-AUC** = how well the model *ranks* spam above ham, independent of threshold.

### 1.5 Data leakage (the #1 beginner mistake)
Information from the test set must never influence training. Two common leaks:
1. Duplicate messages in both train and test (this dataset has ~400 duplicates; we remove them).
2. Fitting TF-IDF on the *whole* dataset before splitting. We avoid this by putting TF-IDF **inside a `Pipeline`**, so it only learns vocabulary from training data.

---

## 2. Setup (Day 1, ~1 hour)

### Step 2.1: Check Python
```bash
python --version        # need 3.9 or newer
```

### Step 2.2: Create a virtual environment
```bash
cd sms-spam-classifier
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
```
You should see `(.venv)` in your terminal. A virtual environment keeps this project's libraries separate from everything else on your computer.

### Step 2.3: Install libraries
```bash
pip install -r requirements.txt
```

### Step 2.4: Add your dataset
Copy the Kaggle file to **`data/spam.csv`** (exact name). Quick check:
```bash
python -c "from spamclf.data import load_dataset; d=load_dataset(); print(d.shape); print(d.head())"
```
Expected: roughly `(5169, 2)` with columns `label` and `text`. (Raw file has 5,572 rows; the loader removes duplicates.)

### Step 2.5: Run the tests
```bash
python -m pytest -q
```
Expected: `3 passed`. If not, see Troubleshooting (section 9).

### Step 2.6: Git (recommended)
```bash
git init && git add . && git commit -m "Initial project setup"
```
Commit after every day of work.

---

## 3. Understand the data: EDA (Day 1-2)

```bash
python -m spamclf.eda
```
Outputs go to `reports/eda/`:
- `class_balance.png`
- `length_by_class.png`
- `top_words.png`

**What to look for and write down:**
1. How many spam vs ham? (Expect about 12-13% spam: an *imbalanced* dataset.)
2. Is spam longer than ham? (Usually yes.)
3. Do spam messages contain more digits and URLs? (Yes: phone numbers, prices, links.)
4. Which words dominate spam? ("free", "call", "txt", "claim", "prize", "urgent"...)

**Exercise A (do it yourself in a notebook or script):**
- Print 10 random spam and 10 random ham messages: `df[df.label==1].sample(10, random_state=1)`.
- Count how many messages contain the "£" symbol per class.
- Write 3 sentences summarising your insights. These go in your report, section 2.

---

## 4. Preprocessing (Day 2)

Open `spamclf/preprocess.py`. The function `clean_text` does:

| Step | Example | Why |
|---|---|---|
| lowercase | `WINNER` -> `winner` | "Free" and "free" should be one word |
| URL -> `urltoken` | `www.prize.com` -> `urltoken` | A link is a spam signal; the exact link is not |
| email -> `emailtoken` | | same idea |
| phone -> `phonetoken` | `08001234567` | Long numbers are suspicious; values are noise |
| £ $ € -> `currencytoken` | `£1000` | Money talk is a strong spam clue |
| digits -> `numtoken` | `2` `150` | Keep "there was a number" without memorising it |
| remove punctuation | `!!!` | simplifies tokens |

**Test it yourself:**
```python
from spamclf.preprocess import clean_text
print(clean_text("WIN £1000 now!! Call 08001234567 or visit www.prize.com"))
```
Expected: `win currencytoken numtoken now call phonetoken or visit urltoken`
(your phone number may print as `phonetoken`; digits inside are consumed.)

**Exercise B:** Edit `clean_text` to *not* replace numbers, retrain (Step 5) and compare F1. Did it help or hurt? Record the result, then undo the change. This is a real experiment: change one thing, measure, record.

> We deliberately do **not** remove stop words: words like "u", "to", "now", "your" are useful spam signals in SMS.

---

## 5. Train and compare models (Day 3-4)

```bash
python -m spamclf.train
```

What happens inside `spamclf/train.py`:
1. Loads and de-duplicates the data.
2. **Stratified 80/20 split**: the test set is locked away.
3. For each of 4 pipelines (word TF-IDF + NB, + Logistic Regression, + Linear SVM, and char TF-IDF + SVM):
   - 5-fold **cross-validation on the training set** (gives a reliable F1 estimate),
   - fits on the full training set,
   - evaluates **once** on the test set.
4. Picks the best model by **cross-validation F1**, not by test score. (If you choose using the test set you are "peeking" and your final number is optimistic.)
5. Saves everything:
   - `models/best_model.joblib`: the full pipeline (cleaning + TF-IDF + classifier)
   - `reports/model_comparison.csv` and `.png`
   - `reports/cm_<model>.png`: confusion matrices
   - `reports/misclassified.csv`: every mistake the best model made
   - `reports/top_features.csv`: most spam/ham-indicating words

**What numbers to expect on the real Kaggle data (approximate):**

| Metric | Typical range |
|---|---|
| Accuracy | 97-99% |
| Precision (spam) | 95-100% |
| Recall (spam) | 85-95% |
| F1 (spam) | 0.91-0.96 |

If you get **100%** or **below 90%**, something is wrong. See Troubleshooting.

### 5.1 Hyper-parameter tuning
```bash
python -m spamclf.train --tune
```
This runs `GridSearchCV` over settings such as NB `alpha`, LogReg/SVM `C`, and `ngram_range` (grids are in `spamclf/models.py`). It takes a few minutes.

- `alpha` (NB): smoothing; smaller = trusts rare words more.
- `C` (LogReg/SVM): inverse regularisation; larger = fits training data more tightly (risk of overfitting).

**Exercise C:** Record a table: baseline F1 vs tuned F1 for each model. Is tuning worth it? (Often the gain is small, which is a valid finding.)

### 5.2 Word-embedding experiment (optional, Day 6)
```bash
pip install -r requirements-optional.txt
python -m spamclf.train --embeddings
```
Compare `w2v_logreg` and `w2v_rbf_svm` to the TF-IDF models. In your report, explain the result. (Hint: tiny corpus, slang and misspellings, so Word2Vec has little to learn from, whereas TF-IDF counts exact tokens, which suits short spammy messages.)

---

## 6. Evaluate properly and do error analysis (Day 5)

### 6.1 Read the confusion matrix
Open `reports/cm_<best_model>.png`. Fill in:

|  | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | TN = ? | FP = ? |
| **Actual spam** | FN = ? | TP = ? |

### 6.2 Study the mistakes
Open `reports/misclassified.csv` (use Excel or pandas). For **each** error, ask:
- Why did the model get fooled? (Spam written like a normal chat? Ham containing words like "free" or "call"?)
- Is the label itself questionable? (Datasets contain noise.)

Group errors into categories, for example:
1. Spam with no obvious keywords ("Hey, it's Sarah, call me back on...")
2. Legit messages that mention prizes, offers or numbers
3. Very short messages
4. Messages with unusual spelling

### 6.3 Inspect the top words
Open `reports/top_features.csv`. Do the top spam words make sense? ("txt", "claim", "prize", "urltoken", "currencytoken" are typical.) If a weird word appears, investigate.

### 6.4 Business decision: threshold
`predict.py` flags SPAM when the score is >= 0.5. If false positives are costly, raise the threshold to 0.7 or 0.8: precision goes up, recall goes down.

**Exercise D:** Write a small script that loops through thresholds 0.3 to 0.9 and prints precision and recall on the test set. Plot precision vs recall. Recommend a threshold and justify it.

---

## 7. Use the model (Day 7)

### 7.1 Command line
```bash
python -m spamclf.predict "Congratulations! You have won a free £500 voucher. Call 08001234567 now"
python -m spamclf.predict "Hey, are we still on for lunch tomorrow?"
python -m spamclf.predict --file my_messages.txt      # one SMS per line
python -m spamclf.predict                             # interactive
```

### 7.2 From Python code
```python
from spamclf.predict import load_model, classify
model = load_model()
print(classify(model, ["URGENT! Your account is locked, click www.verify-now.com"]))
# [('SPAM', 0.97)]
```
The saved model includes the cleaning step, so you pass **raw text**.

### 7.3 Web demo (optional)
```bash
pip install streamlit
streamlit run app.py
```
Opens a browser page where you paste an SMS and see the prediction. Great for demos to your mentor.

### 7.4 Try to break it
Write 20 messages of your own (10 spam, 10 ham) including tricky ones:
- spam with spelling tricks: "Fr33 g1ft c@rd"
- legit bank OTP: "Your OTP is 482913. Do not share it."
- legit message with "free": "Are you free tonight?"

Record which ones the model gets wrong. This is the best way to understand its limits.

---

## 8. Final deliverables (Day 8)

Fill in `docs/REPORT_TEMPLATE.md` (copy it to `reports/REPORT.md`). Your submission checklist:

- [ ] Code runs from scratch with the commands in `README.md`
- [ ] `python -m pytest -q` passes
- [ ] EDA charts + 3-4 written insights
- [ ] Model comparison table (baseline and tuned)
- [ ] Confusion matrix of best model, with FP/FN counts
- [ ] Error analysis with at least 5 examples and explanations
- [ ] Top spam/ham words discussed
- [ ] Threshold recommendation
- [ ] Limitations and next steps
- [ ] Clean Git history

### Suggested daily plan
| Day | Goal |
|---|---|
| 1 | Setup, load data, EDA |
| 2 | Preprocessing, Exercise A/B |
| 3 | Baseline training, read metrics |
| 4 | Tuning, Exercise C |
| 5 | Error analysis, threshold (Exercise D) |
| 6 | Word2Vec experiment (optional), robustness tests |
| 7 | Prediction script, Streamlit demo |
| 8 | Report and presentation |

### Presentation outline (5 minutes)
1. The problem and why it matters (30s)
2. Data and one surprising insight (60s)
3. Method: pipeline diagram (60s)
4. Results and best model (60s)
5. Where it fails + what you'd do next (60s)

---

## 9. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError: Dataset not found` | File must be `data/spam.csv`; run commands from the project root |
| `UnicodeDecodeError` | Loader already tries utf-8 / latin-1 / cp1252. Check the file isn't corrupted or re-saved by Excel |
| `ValueError: Could not find label/text columns` | Print `pd.read_csv(path, encoding="latin-1").columns` and add your names to `LABEL_CANDIDATES` / `TEXT_CANDIDATES` in `data.py` |
| `ModuleNotFoundError: spamclf` | Run from the project root using `python -m spamclf.train` (not `python spamclf/train.py`) |
| `No module named sklearn` | Activate the venv, run `pip install -r requirements.txt` |
| F1 = 1.00 | Leakage: check duplicates were dropped, vectorizer is in the pipeline, labels not in text |
| F1 below 0.85 | Labels mapped wrongly (spam=1, ham=0), or the wrong column is used as text |
| `gensim` install fails | Skip the embedding step, or use Python 3.10-3.12 |
| Training slow | Remove `--tune`; or set `n_jobs=1` in `train.py` if memory is low |
| Streamlit says no model | Run `python -m spamclf.train` first |

---

## 10. Questions you should be able to answer (interview-style)

1. Why is accuracy a poor metric here? Which metric would you report and why?
2. What is TF-IDF and why is it better than raw counts?
3. Why must TF-IDF be fitted only on training data?
4. Why did we remove duplicates before splitting?
5. Why does Naive Bayes work well on text despite its "naive" assumption?
6. What does the `C` parameter do in SVM / Logistic Regression?
7. Why choose the best model by cross-validation rather than by test score?
8. Which is worse for users: a false positive or a false negative? How does that change your threshold?
9. Why might Word2Vec not beat TF-IDF on this dataset?
10. How would the model behave if spammers start using new wording? How would you monitor and retrain?

---

## 11. Extension ideas (if you finish early)

1. Add hand-made features: message length, digit count, uppercase ratio, `!` count, has-URL (use `FunctionTransformer` + `FeatureUnion`).
2. Calibrate SVM probabilities with `CalibratedClassifierCV`.
3. Try `ComplementNB`, `SGDClassifier`, or `RandomForest` and compare.
4. Use pretrained embeddings (GloVe via gensim downloader) and compare to your own Word2Vec.
5. Fine-tune a small transformer (DistilBERT) and compare accuracy vs. training time vs. model size.
6. Wrap the model in a FastAPI endpoint (`POST /predict`).
7. Add SHAP or LIME to explain individual predictions.
8. Build a small evaluation set of modern spam (crypto, delivery scams, UPI/OTP fraud) and measure how the old dataset's model copes. This shows *dataset drift*.

---

## 12. Glossary

- **Feature:** a number describing the text (e.g., TF-IDF weight of "free").
- **Pipeline:** chain of steps (vectorizer -> classifier) treated as one model.
- **Cross-validation:** splitting training data into K parts, training on K-1 and validating on the rest, rotating K times.
- **Stratified:** keeping the spam/ham ratio equal in every split.
- **Overfitting:** memorising training data and failing on new data.
- **Hyper-parameter:** a setting you choose before training (e.g., `C`, `alpha`).
- **Threshold:** the score above which a message is called spam.
- **Data drift:** real-world data slowly changing so the model gets worse over time.
