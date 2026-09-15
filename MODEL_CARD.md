# VietMailGuard V2 Model Card

Model card version: 1.0  
Model version: 2.0.0  
Status: frozen production model  
Evaluation date recorded by the artifact: 2026-09-14  

This card documents the saved VietMailGuard Version 2 artifacts. Metrics below were read from existing result files; they were not recalculated for this document.

## Model overview

VietMailGuard V2 is a bilingual English/Vietnamese three-class email classifier with a separate security-analysis and risk layer. The machine-learning classifier predicts exactly one of:

- `normal`
- `spam`
- `phishing`

Production artifact: `models/v2_bilingual/production_pipeline.joblib`  
Recorded SHA-256: `b2ebecf641f2a5fd3ce9f0d0680a90884850e46b59c48ce0ab5bd4fee64f87fa`

The production pipeline composes `subject + "\n" + body` through `vietmailguard.modeling.EmailTextComposer`; inference must load the serialized pipeline rather than rebuilding preprocessing or TF-IDF manually.

## Frozen production configuration

### Representation

| Parameter | Frozen value |
|---|---|
| Representation | Word TF-IDF |
| Analyzer | `word` |
| `ngram_range` | `(1, 2)` |
| `max_features` | `60000` |
| `min_df` | `2` |
| Lowercase | `true` |
| Sublinear TF | `true` |
| Matrix dtype | `float32` |

### Classifier

| Parameter | Frozen value |
|---|---|
| Estimator | `LinearSVC` |
| `C` | `1.0` |
| `class_weight` | `balanced` |
| `max_iter` | `5000` |
| `random_state` | `42` |

### Probability calibration

The LinearSVC is wrapped by sigmoid `CalibratedClassifierCV`:

| Parameter | Frozen value |
|---|---|
| Method | `sigmoid` |
| Cross-validation | `StratifiedGroupKFold` |
| Folds | `5` |
| Group key | `final_group_id` |
| Shuffle | `true` |
| Random seed | `42` |
| Ensemble | `false` |
| Jobs | `1` |

Calibration is group-aware so related duplicate, template and translation groups do not cross calibration folds. Displayed class probabilities come from the calibrated classifier's `predict_proba`; raw LinearSVC decision scores are not presented as probabilities.

The immutable configuration snapshot is [`config/v2_production_frozen.json`](config/v2_production_frozen.json).

## Intended uses

Suitable uses include:

- email triage into normal, spam and phishing queues;
- educational machine-learning and cybersecurity research;
- decision support for analysts or end users;
- local/offline analysis of email content, URLs and sender indicators;
- controlled comparison of model output and rule-based security findings.

The system should support, not replace, human judgment for consequential cases.

## Out-of-scope uses

VietMailGuard V2 is not:

- a malware or attachment scanner;
- a definitive URL/domain reputation service;
- proof that a URL or sender is malicious;
- a substitute for authentication controls, sandboxing or live threat intelligence;
- an autonomous high-stakes security gateway without human oversight;
- a validated detector for native Vietnamese phishing;
- suitable evidence for claiming real-world native Vietnamese accuracy.

## Training data

The final fit used saved Version 2 train and validation splits only: 38,013 rows, including 35,587 English rows and 2,426 translated Vietnamese rows.

| Class | Final-fit rows |
|---|---:|
| normal | 34,734 |
| spam | 1,546 |
| phishing | 1,733 |

English sources:

- CEAS_08
- Enron
- Ling
- Nazario
- SpamAssassin

Vietnamese sources:

- existing `data_vi` rows identified as translated rather than native;
- controlled local English-to-Vietnamese translations with English-parent provenance.

Only eligible high-confidence labels entered the modeling splits. Review, exclude, corrupt and weakly labeled rows were not used. Controlled translations preserve parent identifiers and remained train-only.

Full provenance, label semantics and redistribution status are documented in [`DATASETS.md`](DATASETS.md).

## Leakage prevention

The data pipeline grouped related rows before splitting. The final grouping accounts for:

- exact content duplicates;
- normalized duplicates;
- template-equivalent messages;
- known English parent and Vietnamese translation links;
- controlled translation descendants;
- known derivative samples.

Rows sharing `final_group_id` were constrained to the same split. The saved final evaluation reports zero cross-split overlap for `final_group_id`, `group_id`, `content_hash`, `normalized_content_hash`, `template_hash` and `translation_group_id`; it also reports zero test parent IDs in development data.

## Scientific evaluation protocol

1. Candidate representations and classifiers were compared using validation data only.
2. Selection was based primarily on Macro F1, followed by phishing recall, cross-language balance, normal false-positive behavior, latency and deployment complexity.
3. Word TF-IDF + LinearSVC was selected before test access.
4. Representation, hyperparameters, classifier, preprocessing and calibration were frozen.
5. The selected pipeline was refit on train + validation only.
6. Calibration used five-fold group-aware training data and did not use test rows.
7. The held-out test was evaluated once.
8. Post-test tuning is disabled and did not occur.

Saved protocol evidence:

- [`reports/v2/model_selection_rationale.md`](reports/v2/model_selection_rationale.md)
- [`reports/v2/final_model_evaluation.md`](reports/v2/final_model_evaluation.md)
- [`results/v2_bilingual/final_test_metrics.json`](results/v2_bilingual/final_test_metrics.json)

## Frozen held-out test results

Test rows: **6,704**  
Test groups: **6,696**

| Metric | Saved value |
|---|---:|
| Accuracy | 0.997763 |
| Macro Precision | 0.997992 |
| Macro Recall | 0.983992 |
| Macro F1 | 0.990900 |
| Weighted F1 | 0.997750 |
| Phishing Recall | 0.970588 |

These rounded values match `final_test_metrics.json`, `model_metadata.json` and `final_test_classification_report.csv`; no discrepancy was found.

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.997720 | 0.999837 | 0.998777 | 6,127 |
| spam | 0.996255 | 0.981550 | 0.988848 | 271 |
| phishing | 1.000000 | 0.970588 | 0.985075 | 306 |

### Confusion matrix

Rows are actual classes; columns are predicted classes.

| Actual \ Predicted | normal | spam | phishing |
|---|---:|---:|---:|
| normal | 6,126 | 1 | 0 |
| spam | 5 | 266 | 0 |
| phishing | 9 | 0 | 297 |

## Vietnamese evaluation boundary

The held-out translated Vietnamese subset contains:

| Class | Rows |
|---|---:|
| normal | 171 |
| spam | 8 |
| phishing | 0 |
| **Total** | **179** |

The saved Accuracy and Macro F1 are 1.000000 **only over the represented normal and spam classes**. The saved three-class Macro F1 is `null`.

Therefore:

- Vietnamese phishing recall is **N/A** and cannot be claimed;
- native Vietnamese accuracy cannot be claimed;
- no independent native Vietnamese benchmark currently exists;
- this subset cannot establish Vietnamese three-class generalization;
- Vietnamese support remains experimental and predominantly translation-dependent.

The English held-out subset contains 6,525 rows and all three classes. Results on English and translated Vietnamese must not be combined into a claim of native Vietnamese performance.

## Known failure modes and robustness limits

### Short promotional messages

Short promotional English and Vietnamese emails can be predicted `normal`, sometimes with high confidence. This behavior is documented by the separate short-form challenge set and production smoke examples.

The 90-row challenge artifact reports:

- overall accuracy `0.855556`;
- spam recall `0.700000`;
- 9 spam errors among 30 spam challenge examples;
- 7 cases where ML predicted normal while promotional security indicators were present;
- very-short-bucket accuracy `0.633333`.

These challenge results are **not** held-out test metrics and were not used to retrain or tune Version 2.

### Other limitations

- The phishing class is source-concentrated, principally in Nazario.
- Vietnamese training/evaluation data is predominantly translated and may reflect translation-domain artifacts.
- The translated Vietnamese held-out subset is small and has no phishing examples.
- Mixed-language content has not been independently benchmarked.
- A small separate robustness set showed degradation after Vietnamese accent removal; that set supports only normal and spam, not phishing.
- Short or out-of-distribution text may receive overconfident predictions.
- Offline URL and sender rules are heuristic signals, not live reputation evidence.
- Obfuscation, novel campaigns, image-only email and attachment-borne threats may evade text classification.

## ML prediction, confidence and security risk

The inference response deliberately keeps three concepts separate:

1. **Prediction** — the classifier's `normal`, `spam` or `phishing` output.
2. **Model Confidence** — calibrated class probability associated with that prediction.
3. **Overall Risk Score** — a separately configured 0–100 decision-support score based on ML output plus content, URL and sender indicators.

Security rules do not silently overwrite classifier predictions. For example:

```text
ML prediction: NORMAL
Security finding: Promotional language detected
Risk/recommendation: REVIEW
```

The result remains an ML prediction of `normal`; detecting a promotional indicator is not counted as classifier success and does not establish that the message is spam.

## Ethical, privacy and licensing considerations

- Email corpora may contain personal, sensitive or security-relevant information.
- Project MIT licensing does not automatically apply to third-party email content.
- Most row-level datasets and translations remain unsuitable for public redistribution until their rights and privacy status are resolved.
- Do not send corpus messages into live mail systems or external services.
- Do not describe translated Vietnamese data as native or real-world Vietnamese ground truth.
- Human review is recommended when model and security layers disagree or when consequences are significant.

See [`DATASETS.md`](DATASETS.md) for the dataset license/provenance audit.

## Reproducibility and deployment

- Supported Python range: `>=3.11`.
- Exact tested environment: [`ENVIRONMENT.md`](ENVIRONMENT.md).
- Human-readable dependency ranges: [`requirements.txt`](requirements.txt).
- Exact core/test dependency snapshot: [`requirements-lock.txt`](requirements-lock.txt).
- Runtime model metadata: [`models/v2_bilingual/model_metadata.json`](models/v2_bilingual/model_metadata.json).

Because the artifact uses joblib/pickle and scikit-learn estimator classes, it should be loaded only from a trusted release and with tested compatible package versions. Verify the recorded SHA-256 before deployment.

## Frozen-model change policy

VietMailGuard Version 2 is frozen. Do not retrain, refit TF-IDF, recalibrate, change hyperparameters or rerun the held-out test to improve its scores. Robustness and challenge findings are inputs to a separately versioned Version 2.1 research protocol with a new independent evaluation set.
