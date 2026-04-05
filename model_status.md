# Model Status — Post Issue #2, #3 & #4

## Best model: updated (Issue #2 — Feature Engineering Improvements)

`models/best_model.joblib` — tuned LightGBM trained on 2,524 labeled tokens with 27 behavioral features (expanded from 16 via Issue #2: graph-based, temporal, and MEV/transaction features).

| Metric | Value |
|---|---|
| F1-macro | *re-run needed* |
| ROC-AUC | *re-run needed* |
| Training set | 2,524 tokens |
| Features | 27 behavioral features (16 original + 11 from Issue #2) |

*Note: metrics will be populated after re-running `modeling.ipynb` → `evaluation.ipynb` with LightGBM selected on 27 features.*

---

## What Issues #3 and #4 attempted

### Issue #3 — Label Expansion via Account Labels (`label_expansion.ipynb`)

| Variant | F1-macro | Δ | Notes |
|---|---|---|---|
| Entity pseudo-labels v1 | 0.8658–0.8801 | −0.018 to −0.026 | 84.6% of pseudo-labels were null-address noise |
| Entity pseudo-labels v2 | 0.8801–0.8821 | −0.002 to −0.004 | Cleaner after fixes, but still degrades |
| Entity flags as features (Fix 3) | 0.8802 | −0.004 | `phishing_flag` and `hack_flag` scored zero feature importance — behavioral features already encode that signal |

### Issue #4 — Improved Semi-Supervised Learning (`improved_semi_supervised.ipynb`)

| Variant | F1-macro | Δ | Notes |
|---|---|---|---|
| Isolation Forest A1 (strict threshold) | — | — | Adds 0–5 tokens at useful thresholds — spam manifold too tight (*re-run needed with 27 features*) |
| Isolation Forest A2 (global anomaly) | 0.4314 | −0.452 | Conceptually wrong: IF flags the minority class; spam is the majority |
| Label Spreading (k=10, α=0.2) | 0.8563 | −0.028 | Feature-space overlap between spam and legit causes label blur |
| Sender-network propagation (share > 0.3) | 0.8696 | −0.014 | Best alternative across all semi-supervised stages; still below baseline |

---

## Why the model did not change

The 3,606 labeled tokens are already representative enough that the supervised LightGBM generalises well to the unlabeled pool. Every pseudo-labeling method — confidence thresholding (Stage 6), entity-based flagging (Issue #3), Isolation Forest, Label Spreading, and sender-network propagation (Issue #4) — introduced label noise faster than useful signal.

The model did not need more data. It needed *better* data. Neither issue provided that.

## What would actually move the needle

- Expanding ground-truth labels through manual annotation or a new labeling heuristic beyond symbol collision
- Adding `spam_sender_share` as a supervised feature (network topology signal orthogonal to the existing 27 behavioral features)
- Extending the transfer window beyond 20M blocks to capture slow-burn spam patterns
