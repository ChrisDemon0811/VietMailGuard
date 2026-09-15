# Version 2 model selection rationale

## Selection boundary

This decision was completed on 2026-09-14 before opening or evaluating
`data/splits/v2/test.csv`. It uses only saved Version 1 results, Version 2 TF-IDF
validation results, and Version 2 multilingual-embedding validation results.

Selected Version 2 production approach:

`Word TF-IDF + LinearSVC + group-aware sigmoid calibration`

The selected validation experiment is `word_tfidf__linear_svm`.

## Validation evidence

| Candidate | Validation Macro F1 | Phishing recall | English Macro F1 | Translated Vietnamese Macro F1* | Normal false-positive rate |
|---|---:|---:|---:|---:|---:|
| Word TF-IDF + LinearSVC | 0.986190 | 0.983660 | 0.988515 | 0.878835 | 0.000489 |
| Word + Character TF-IDF + LinearSVC | 0.985694 | 0.977124 | 0.988031 | 0.878835 | 0.000489 |
| Word + Character TF-IDF + Logistic Regression | 0.982534 | 0.980392 | 0.984769 | 0.905917 | 0.002610 |
| Best M1 embedding: LinearSVC | 0.910958 | 0.934641 | 0.912789 | 0.765564 | not reported in the embedding comparison |
| Best M2 embedding: LinearSVC | 0.903714 | 0.941176 | 0.904693 | 0.827586 | not reported in the embedding comparison |

*Translated-Vietnamese Macro F1 covers only the supported validation classes
`normal` and `spam`. The subset has no phishing rows, so it has no valid
three-class Macro F1 or phishing recall.

## Decision

The selected Word TF-IDF LinearSVC has the highest Version 2 validation Macro F1.
It also has higher phishing recall than the second-ranked Word + Character
LinearSVC and the Logistic Regression alternative. Its normal false-positive rate
is 0.000489, compared with 0.002610 for Word + Character Logistic Regression.

Word + Character Logistic Regression performs better on the small translated-
Vietnamese subset, but this evidence is based on 180 rows containing only 8 spam
and no phishing. That advantage does not override the primary Macro F1 result,
the phishing-recall priority, or the higher normal false-positive rate.

The multilingual embedding approaches are not selected. Their best combined
Macro F1 is 0.910958 for M1 and 0.903714 for M2, both materially below the selected
TF-IDF result. The embedding encoder also adds an approximately 457.6 MiB model,
Torch/Transformers dependencies, and encoding latency. Complexity does not buy a
validation improvement in this experiment.

Word TF-IDF is smaller and faster than the character and combined TF-IDF
representations while retaining the best primary and secondary validation
metrics. This satisfies the deployment-complexity and latency priorities without
selecting a more complex model automatically.

## Frozen production method

- Text composition: shared `EmailTextComposer`, `subject + "\n" + body`.
- Representation: word TF-IDF.
- Word n-grams: `(1, 2)`.
- Lowercasing: enabled.
- `min_df = 2`.
- `max_features = 60000`.
- Sublinear term frequency: enabled.
- TF-IDF dtype: `float32`.
- Classifier: `LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)`.
- Confidence: `CalibratedClassifierCV` with sigmoid calibration.
- Calibration folds: five-fold `StratifiedGroupKFold`, shuffled with seed 42,
  grouped by `final_group_id`.
- Calibration ensemble: disabled. The final base classifier is refitted on all
  train + validation rows after out-of-fold calibration predictions are created.
- Final fitting data: Version 2 train + validation only.
- Class order: `normal`, `spam`, `phishing`.

Calibration is required because raw LinearSVC decision scores are not
probabilities. The calibration method is frozen here before test access and must
not be changed in response to final test results.

## Post-test policy

The held-out test may be evaluated once after this document and the frozen JSON
snapshot exist. No hyperparameter, feature, calibration, or preprocessing change
is allowed after viewing test metrics. A serious defect must be documented for a
future Version 2.1 protocol with a new held-out test set.

