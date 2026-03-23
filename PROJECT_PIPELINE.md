# Spam Detection Pipeline — Ethereum Token Transfer Data

**Project:** FinTech-540 | **Data:** 1,000-block window (~20M transfers) | **Status:** Preprocessing Complete

---

## Overview

Classify Ethereum tokens as **spam (0)** or **legitimate (1)** using token-level behavioral features engineered from raw transfer events. Labels are derived from:
- **Legit (1):** Contract address in `token_labels.csv` (verified)
- **Spam (0):** Same symbol as a verified token but different contract address (symbol collision)
- **Unlabeled (NaN):** Unverified, no collision — excluded from supervised training, candidate for semi-supervised extension

---

## Pipeline Stages

```
EDA                              ✅ Done
   ↓
Stage 1: Feature Engineering & Preprocessing   ✅ Done  →  preprocessing.ipynb
   ↓
Stage 2: Train/Test Split & Class Imbalance Handling   (folded into Stage 1 ✅)
   ↓
Stage 3: Baseline Model Training               ⬜ Next  →  modeling.ipynb
   ↓
Stage 4: Advanced Model Training & Hyperparameter Tuning   ⬜ To do
   ↓
Stage 5: Model Evaluation & Interpretation     ⬜ To do  →  evaluation.ipynb
   ↓
Stage 6: (Optional) Semi-Supervised Extension  ⬜ To do
   ↓
Stage 7: Final Report & Deliverables           ⬜ To do
```

---

## Stage 1 — Feature Engineering & Preprocessing ✅ Complete

**Notebook:** `preprocessing.ipynb` | **Output:** `data/processed/` (6 parquet splits + scaler + feature list)

**What was done:**
- Rebuilt `token_features` from raw transfer data (mirrors EDA exactly)
- Added 5 new features: `block_range`, `unique_values_count`, `zero_value_ratio`, `top1_sender_share`, `receiver_concentration` (Gini)
- Dropped leakage columns (`symbol_collision`, `is_verified`, `asset`) — 16 final features
- Median imputation (no nulls found in labeled set), `log1p` transform on 10 skewed features
- Stratified 70/15/15 split — Train: ~2,524 | Val: ~541 | Test: ~541
- `RobustScaler` fit on train only; saved scaled + unscaled versions
- All 13 cells ran without errors

**Goal (archived):** Finalize the feature matrix from EDA and prepare it for modeling.

### 1.1 Finalize Feature Set

Start from `token_features` DataFrame built in EDA. Confirmed features:

| Feature | Type | Notes |
|---|---|---|
| `n_transfers` | numeric | Log-transform (right-skewed) |
| `n_unique_senders` | numeric | Log-transform |
| `n_unique_receivers` | numeric | Log-transform |
| `sender_receiver_ratio` | numeric | Log-transform |
| `transfers_per_block` | numeric | Log-transform |
| `n_distinct_blocks` | numeric | Log-transform |
| `value_mean` | numeric | Log-transform |
| `value_std` | numeric | Log-transform |
| `value_null_ratio` | numeric | Already [0,1] |
| `category_entropy` | numeric | Already bounded |
| `sender_is_labeled` | numeric | Proportion [0,1] |
| `is_verified` | binary | Drop — leaks label |
| `symbol_collision` | binary | Drop — directly constructs label |

> **Important:** Drop `is_verified` and `symbol_collision` before training — they directly encode the label and would cause data leakage.

### 1.2 Additional Features to Engineer (Recommended)

- `receiver_concentration`: Gini coefficient of transfer counts per receiver (high = centralized airdrop)
- `block_range`: `max(blockNum) - min(blockNum)` for the token (narrow = burst)
- `unique_values_count`: Number of distinct transfer amounts (low = uniform airdrop)
- `top1_sender_share`: Fraction of transfers from the single most active sender
- `has_zero_value_transfers`: Binary, fraction of transfers with value = 0

### 1.3 Preprocessing

```python
# Recommended steps
1. Filter: keep only rows where label is not NaN (3,606 labeled tokens)
2. Log-transform skewed numeric features: log1p(x)
3. Impute missing values (value_mean, value_std): median imputation
4. Scale features: StandardScaler or RobustScaler (robust to outliers)
5. Encode any remaining categoricals if added
```

**Output:** `X_labeled` (3,606 × ~14 features), `y_labeled` (3,606 binary labels)

---

## Stage 2 — Train/Test Split & Class Imbalance ✅ Complete

**Folded into `preprocessing.ipynb` (Stage 1).**

**Actual results:**
- Legit (1): 1,599 (44.3%) | Spam (0): 2,007 (55.7%) — mild imbalance
- Stratified 70/15/15 split via `train_test_split(stratify=y)`
  - Train: 2,524 | Val: 541 | Test: 541 — class ratios preserved in all splits
- Test set is held out and untouched until final evaluation (Stage 5)
- Imbalance strategy: use `class_weight='balanced'` in all estimators; revisit SMOTE only if F1 on val set is poor

---

## Stage 3 — Baseline Models

Train simple models first to establish a performance floor. Use 5-fold stratified cross-validation on the training set.

### Models to Train

1. **Logistic Regression** — linear baseline, interpretable coefficients
2. **Decision Tree** — depth-limited for interpretability
3. **Naive Bayes** (GaussianNB) — probabilistic baseline

### Evaluation Metrics

Since both false positives (blocking legit tokens) and false negatives (missing spam) matter:

| Metric | Why |
|---|---|
| **F1 Score (macro)** | Primary metric — balances precision and recall across both classes |
| **ROC-AUC** | Threshold-independent ranking quality |
| **Precision** (per class) | Cost of false alarms |
| **Recall** (per class) | Cost of missed spam |
| **Confusion Matrix** | Visual breakdown |

```python
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
```

---

## Stage 4 — Advanced Models & Hyperparameter Tuning

### 4.1 Models to Train

| Model | Key Hyperparameters |
|---|---|
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **Gradient Boosting (XGBoost / LightGBM)** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| **Support Vector Machine** | `C`, `kernel`, `gamma` |
| **MLP (Neural Net)** | `hidden_layer_sizes`, `dropout`, `learning_rate` |

**Recommended primary model:** XGBoost or LightGBM — handles tabular data well, built-in feature importance, robust to scale.

### 4.2 Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV  # or Optuna

# Use RandomizedSearchCV with 5-fold CV on train set
# Optimize for: F1-macro
# Budget: ~50-100 iterations
```

Use `Optuna` for more efficient search if time permits:
```python
import optuna
# Define objective function, minimize 1 - val_f1_macro
```

### 4.3 Cross-Validation Setup

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])
```

Always include preprocessing inside the CV pipeline to prevent data leakage.

---

## Stage 5 — Model Evaluation & Interpretation

### 5.1 Final Test Set Evaluation

After selecting the best model from Stage 4, evaluate **once** on the held-out test set:

```python
# Report all metrics
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

### 5.2 Feature Importance

For tree-based models:
```python
# Built-in feature importance
xgb_model.feature_importances_

# SHAP values (recommended — model-agnostic, locally interpretable)
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Expected top features** (from EDA correlations):
1. `n_unique_senders` (r=0.61)
2. `n_transfers` (r=0.59)
3. `n_unique_receivers` (r=0.59)
4. `sender_receiver_ratio` (airdrop signal)
5. `n_distinct_blocks` (burst signal)

### 5.3 Error Analysis

Examine misclassified tokens:
- False positives: legit tokens flagged as spam — are they niche/low-activity tokens?
- False negatives: spam that passed — do they mimic legitimate activity patterns?

---

## Stage 6 — (Optional) Semi-Supervised Extension

**Motivation:** 10,757 unlabeled tokens remain. Semi-supervised learning can leverage these.

### Option A: Self-Training (Pseudo-Labeling)

```python
from sklearn.semi_supervised import SelfTrainingClassifier

# Assign -1 to unlabeled samples in y
# Train self-training wrapper on combined labeled + unlabeled X
```

### Option B: Label Propagation

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
```

### Option C: Confidence Thresholding

1. Train supervised model on labeled data
2. Predict probabilities on unlabeled tokens
3. Add high-confidence predictions (p > 0.90) to training set as pseudo-labels
4. Retrain and evaluate on held-out test set

**Caveat:** Pseudo-labels are noisy — monitor test set performance carefully.

---

## Stage 7 — Report & Deliverables

### Notebook Structure (Recommended)

```
EDA.ipynb              (done)
preprocessing.ipynb    → feature engineering, cleaning, splitting
modeling.ipynb         → baseline + advanced models, tuning
evaluation.ipynb       → final test evaluation, SHAP, error analysis
(optional) semi_supervised.ipynb
```

### Final Report Should Include

- [ ] Problem framing and label construction methodology
- [ ] EDA summary (key features, distributions, correlations)
- [ ] Feature engineering decisions and justification
- [ ] Model comparison table (CV F1, AUC across all models)
- [ ] Best model test set performance with confusion matrix
- [ ] SHAP feature importance plot
- [ ] Error analysis: what types of tokens are hardest to classify?
- [ ] Discussion: limitations, potential improvements, deployment considerations

---

## Quick Reference: Feature Leakage Checklist

Before training, verify these columns are **excluded** from `X`:

- [ ] `symbol_collision` — directly defines spam label
- [ ] `is_verified` — directly defines legit label
- [ ] `asset` (raw token symbol) — identifier, not a behavioral feature
- [ ] `label` — target variable

---

## Recommended Next Action

```
1. Open modeling.ipynb
2. Load data/processed/train_unscaled.parquet (tree models) or train.parquet (linear models)
3. Train baselines: Logistic Regression, Decision Tree, Naive Bayes
4. Train advanced models: Random Forest, XGBoost/LightGBM
5. Compare CV F1-macro scores across all models
```

---

*Pipeline version 1.0 | Created 2026-03-23*
