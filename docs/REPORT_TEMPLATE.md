# SMS Spam Classification: Project Report

**Author:** ______  **Date:** ______  **Mentor:** ______

## 1. Objective
(2-3 sentences: what problem, why it matters, what success looks like.)

## 2. Dataset
- Source, number of messages, spam percentage
- Duplicates found / removed
- 2-3 insights from EDA (message length, digits, URLs, common words). Include a chart.

## 3. Method
- Cleaning steps and why (placeholder tokens for URLs/phones/money/numbers)
- Features: TF-IDF (word 1-2 grams, char 2-5 grams), optional Word2Vec
- Models: Naive Bayes, Logistic Regression, Linear SVM
- Evaluation: stratified 80/20 split, 5-fold CV on training set, test set used once

## 4. Results
| Model | CV F1 | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| ... | | | | | |

Include the confusion matrix of the best model.

## 5. Error analysis
- Number of false positives (legit flagged as spam) and false negatives (spam missed)
- 3-5 example messages for each, with your explanation of why the model failed
- Which error is more costly for users and what threshold you would choose

## 6. Interpretation
- Top 10 spam words and top 10 ham words from the model; do they make sense?

## 7. Limitations and next steps
- Dataset age / language / size, class imbalance, adversarial spammers
- Ideas: transformer models, extra features, threshold tuning, monitoring in production

## 8. How to reproduce
(Exact commands.)
