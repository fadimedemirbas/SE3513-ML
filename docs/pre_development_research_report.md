# Pre-Development Research Report

**Project:** SE3513 Machine Learning — Fake Job Posting Detection
**Dataset:** Real / Fake Job Posting Prediction (EMSCAD), Kaggle, version 1
**Date:** 2026-11-25
**Updated:** 2026-11-26 (Sections 6 and 7 revised after Step 2)
**Purpose:** Review publicly available work on this dataset before the next development phase, separate reliable findings from flawed ones, and fix the methodology we will follow.

---

## 1. Summary

We reviewed the 11 highest-ranked public notebooks on the dataset's Kaggle Code page, reading both their code and their reported outputs. The main findings are:

1. **Most headline scores are not reliable.** Four of the eleven notebooks report results that are inflated by data leakage or by implementation bugs (reported fake-class F1 between 0.93 and 0.9996). Several others report accuracy only, which is uninformative at a 4.8% positive rate.
2. **Notebooks with a sound train/test separation reach a fake-class F1 of roughly 0.74–0.82**, and they do so on a *random* split that does not remove near-duplicate postings. That setting is easier than ours.
3. **Our current model (fake-class F1 0.80, ROC AUC 0.973 on a grouped, de-duplicated test set) is already competitive** with the best clean public results, under a stricter evaluation.
4. Several ideas from these notebooks are worth testing in our setting (Section 5). Several practices must be avoided (Section 4).
5. Our own results come from a **single split with only 128 fake postings in validation**. Differences of a few F1 points between experiments are within noise. The next phase therefore starts by strengthening the evaluation protocol (Section 6).

---

## 2. Dataset Characteristics Relevant to Method Choice

| Property | Value | Implication |
|---|---|---|
| Rows (raw / after de-duplication) | 17,880 / 17,598 | Small dataset; classical linear models remain strong baselines |
| Fake postings | 866 (4.84%) | Severe class imbalance; accuracy is not a useful metric |
| Near-duplicate groups (cosine radius 0.10 on description TF-IDF) | 12,372 groups, largest 414 rows | A random split places copies of the same posting in train and test |
| Missing values | `salary_range` 84%, `department` 65%, `required_education` 45%, `benefits` 40% | Missingness itself may carry signal |
| Masked tokens | Text contains placeholders such as `#URL_<hash>#` | Usually deleted during cleaning; could be counted as features instead |
| Source and period | Single source, 2012–2014 | Generalisation to current postings is not guaranteed |

---

## 3. Review of Public Notebooks

Notebooks are identified by number only, in the order in which they were ranked. "Clean" means the train/test separation was sound; it does not imply that near-duplicates were handled.

| # | Approach | Reported fake-class result | Assessment |
|---|---|---|---|
| 1 | Bag-of-words / TF-IDF + Multinomial NB; dense NN | NB F1 0.33 and 0.00; NN F1 0.64 | Title advertises 97% accuracy. `TfidfVectorizer(max_df=1)` passes an integer, which keeps only terms appearing in a single document, an implementation bug. |
| 2 | CountVectorizer + LR; GloVe + NN; frozen DistilBERT + LR | ROC AUC 0.85 / 0.81 / 0.77 | ROC AUC computed from hard 0/1 predictions instead of scores. DistilBERT applied to the first 2,000 rows only, truncated to 60 tokens. **Good practice:** stratified 5-fold CV with mean ± std. |
| 3 | Bi-LSTM on description | Not reported | Text truncated to 40 tokens, no class weighting, only one text field. |
| 4 | 1–3-gram bag-of-words on all text fields + `location`; LR / RF / SVC / XGBoost | LR recall 0.72 | Best simple clean result. Random split without de-duplication. |
| 5 | Bag-of-words + POS tags; several classifiers | F1 0.9996 | **Leakage:** the minority class is oversampled with replacement to the size of the majority class (13,937 rows) *before* the train/test split, so copies of the same posting appear in both. |
| 6 | TF-IDF + Multinomial NB | F1 0.49 (recall 0.32) | Vectoriser fitted on the full data before splitting (minor leakage). |
| 7 | Per-field TF-IDF + length/word-count features + one-hot categoricals; LR | ROC AUC 0.73 | ROC AUC from hard predictions. Word-count feature has a bug (`len(str(x.split()))` counts characters of the list representation). **Good ideas:** separate vectoriser for the title and text-length features. |
| 8 | Label-encoded text columns; LR / KNN / RF | Accuracy only | **Leakage:** the row index is used as a feature after oversampling and concatenation, so the index encodes the label. Free text is label-encoded as integers. |
| 9 | TF-IDF + SMOTE + RF; LSTM; GRU; Word2Vec + Bi-LSTM | F1 0.79 / 0.76 / 0.74 / 0.82 | **Cleanest methodology in the set:** stratified split first, resampling applied to training data only. Random split without de-duplication. |
| 10 | Per-class word-frequency features + RF | Recall 0.83 | **Target leakage:** word frequencies per class (and NB probabilities) are computed on the full dataset, including test labels, before splitting. |
| 11 | GloVe averages + label-encoded categoricals; XGBoost + SMOTETomek | CV F1 0.93 | **Bugs and leakage:** `str(data['text'])` turns the entire column into one string, so every row receives the same text. The computed GloVe features are never passed to the model. SMOTETomek is applied before cross-validation, so synthetic samples leak across folds. |

### Comparison with our current model

| Setting | Split | Fake-class F1 |
|---|---|---|
| Clean public notebooks (#4, #9) | Random, near-duplicates not removed | ~0.74–0.82 |
| Our class-weighted Linear SVM (TF-IDF + structured features) | Grouped by description similarity, de-duplicated | **0.80** (precision 0.86, recall 0.75) |

---

## 4. Practices We Will Avoid

| Pitfall observed | Rule for this project |
|---|---|
| Oversampling / SMOTE before splitting (#5, #8, #11) | Any resampling happens inside the training fold only, inside a `Pipeline`. |
| Fitting vectorisers or computing label statistics on all data (#6, #10) | Every fitted step (TF-IDF, scalers, encoders, frequency tables) is fitted on the training fold only. |
| ROC AUC from 0/1 predictions (#2, #7) | Ranking metrics are computed from `decision_function` or `predict_proba` scores. |
| Accuracy as the headline metric (#1, #5, #8) | Headline metrics are fake-class PR-AUC, F1, precision and recall. |
| Random split with near-duplicates (all) | Keep the grouped split; groups never cross fold boundaries. |
| Single split, no variance estimate (most) | Repeated grouped cross-validation with mean ± std; bootstrap CIs on the test set. |
| Excessive truncation of text for neural models (#2, #3) | If transformers are used, use the full training set and the longest feasible sequence length. |

---

## 5. Ideas Adopted for Testing

Each idea will be tested as an ablation against the current baseline under the protocol in Section 6. An idea is kept only if its improvement exceeds the fold-to-fold variation.

| Idea | Source | Rationale |
|---|---|---|
| Word n-grams (1–2, 1–3) and all text fields | #4 | Strongest clean simple result. Our earlier `requirements` ablation used unigrams only. |
| Character n-grams (`char_wb`, 3–5) | Not in any notebook | Robust to spelling variation and formatting artefacts. |
| Separate TF-IDF per field (title, company profile, description) | #7 | Keeps field-specific signals such as an empty company profile. |
| Text length, word count, empty-field indicators | #7 (bug fixed) | Fake postings differ in length and completeness. |
| Country extracted from `location` | #4, #5 (not measured there) | Plausibly strong signal. Risk of learning geography rather than fraud cues will be discussed. |
| Counts of masked tokens (`#URL_`, `#EMAIL_`, `#PHONE_`) | Observed in data | Usually discarded in cleaning; may indicate contact-heavy scam postings. |
| CV with mean ± std reporting | #2, #7 | Needed to judge small differences. |
| Sentence embeddings / DistilBERT on the full data | #2 (done poorly) | Captures semantic cues that TF-IDF misses (e.g. "pay a registration fee"). |

**Not adopted:** SMOTE on sparse text (no gain over class weighting in #9, and our own random-oversampling experiment lowered F1); recurrent networks trained from scratch (too little data; no advantage in #3 or #9).

---

## 6. Planned Methodology

### Step 1 — Evaluation framework
- Keep the existing grouped test set (2,597 postings, 131 fake) untouched until the final evaluation.
- On the remaining development data (training + validation, 15,001 postings), run **repeated `StratifiedGroupKFold`** (5 folds × 3 repeats) using the existing description-similarity groups.
- Choose the decision threshold **inside each training fold** (inner grouped CV), so reported F1 is not tuned on the fold it is measured on.
- Report per-model mean ± std of: fake-class PR-AUC (average precision), ROC AUC, F1, precision, recall.
- For the final test evaluation, report **group-level bootstrap 95% confidence intervals**.
- Re-measure the current final model under this protocol and fix it as the **baseline**.

### Step 2 — Feature engineering
Ablations of the ideas in Section 5, one change at a time.

**Completed** (`feature_engineering.ipynb`). Selected configuration: word 1–2 grams + `char_wb` 3–5 grams, `sublinear_tf=True`, `min_df=2`, `C=1`. Development CV PR-AUC 0.825 → 0.871 (better in 15/15 folds), F1 0.748 → 0.789 (14/15). Meta features, location, extra categorical and binary fields did not pass the decision rule.

### Step 3 — Model extensions
The Step 2 configuration is the new baseline. Two checks come first:
- **Re-check on fresh folds.** Compare the Step 2 configuration with the Step 1 baseline once more on folds from an unused CV seed. Step 2 selected the best of 14 variants on the same folds; if the gain also holds on new folds, it is not a product of that selection.
- **Label-noise check.** Review the 8 similarity groups that contain both labels (73 postings, 13 fake) and measure the effect of removing them from training (sensitivity analysis). Labels are not changed.

The candidates are fixed before running, and no variants are added after seeing the results: probability calibration of the SVM (`CalibratedClassifierCV`); sentence-embedding features; gradient boosting (LightGBM, strongly regularised) on out-of-fold text scores plus structured features.

### Step 4 — Optional transformer fine-tuning
DistilBERT with class weighting, evaluated under the same protocol, if GPU time allows.

### Step 5 — Final model and report
Select the model on development CV, fix the threshold, evaluate **once** on the held-out test set and report group-level bootstrap confidence intervals. In addition:
- **Interpretation.** Use SVM coefficients (or SHAP) to separate general fraud cues (e.g. "registration fee", "wire transfer") from dataset-specific traces such as company names.
- **Shortcut test.** Measure the performance drop when company names are masked or `company_profile` is removed. A large drop means the model memorises companies rather than learning fraud cues.
- **Error analysis.** Inspect the most confident errors; some may be label errors.
- **Limitations.** Report the CV–test gap and the limitations in Section 7 with the measurements above.

---

## 7. Risks and Limitations

| Risk | What we know | Mitigation | When |
|---|---|---|---|
| **Small positive class** | 855 fake postings, but only 518 independent similarity groups (436 in development data); any single split has high variance | Mean ± std over 15 folds, paired comparisons, strict decision rule, bootstrap CIs on the test set; prefer low-variance models | All steps |
| **Selection bias** | Step 2 chose the best of 14 variants on the same folds, so its CV score is slightly optimistic | Re-check on fresh folds; fix candidate lists before running; use the test set once; report the CV–test gap | Steps 3 and 5 |
| **Single source and period** | One platform, 2012–2014; no date column, so a temporal split is not possible | Cannot be solved with this data. Interpretation and the shortcut test show what the model relies on; stated clearly in the limitations | Step 5 |
| **Label noise** | 8 of 1,767 multi-posting groups have mixed labels (73 postings, 13 fake); one set of identical descriptions has conflicting labels | Manual review and sensitivity analysis without changing labels; confident-error analysis | Steps 3 and 5 |
| **Location features** | No measurable gain in Step 2; not adopted | Nothing further needed | — |
