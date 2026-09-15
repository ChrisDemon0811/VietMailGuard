# Version 2 final model evaluation

## Locked selection and final refit

The production approach was selected and documented in `reports/v2/model_selection_rationale.md` before the held-out test was opened.

- Representation: Word TF-IDF, word n-grams (1, 2), 60,000 maximum features.
- Classifier: LinearSVC with C=1.0 and balanced class weights.
- Confidence: sigmoid CalibratedClassifierCV with 5 StratifiedGroupKFold folds and ensemble disabled.
- Refit data: train + validation, 38,013 rows.
- Refit time: 37.97 seconds.
- Test rows were not used to fit TF-IDF, LinearSVC, or calibration.

## One-shot combined held-out test

- Test rows: 6,704
- Accuracy: 0.997763
- Macro Precision: 0.997992
- Macro Recall: 0.983992
- Macro F1: 0.990900
- Weighted F1: 0.997750
- Prediction time: 2.722 seconds.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.997720 | 0.999837 | 0.998777 | 6127 |
| spam | 0.996255 | 0.981550 | 0.988848 | 271 |
| phishing | 1.000000 | 0.970588 | 0.985075 | 306 |

## Test by language and origin

| Scope | Rows | Accuracy | Macro F1* | Normal recall | Spam recall | Phishing recall |
|---|---:|---:|---:|---:|---:|---:|
| combined | 6704 | 0.997763 | 0.990900 | 0.999837 | 0.981550 | 0.970588 |
| english_test | 6525 | 0.997701 | 0.990774 | 0.999832 | 0.980989 | 0.970588 |
| translated_vietnamese_test | 179 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | N/A |
| original_english | 6525 | 0.997701 | 0.990774 | 0.999832 | 0.980989 | 0.970588 |
| existing_translated_vietnamese | 179 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | N/A |
| controlled_translation_augmentation_descendants | 0 | N/A | N/A | N/A | N/A | N/A |

*Subset Macro F1 covers supported classes only. Missing-class metrics are N/A.

Native Vietnamese benchmark is not available.

Controlled translated test descendants: 0; verified test parents: 0.

## Confusion matrix

| Actual / predicted | normal | spam | phishing |
|---|---:|---:|---:|
| normal | 6126 | 1 | 0 |
| spam | 5 | 266 | 0 |
| phishing | 9 | 0 | 297 |

## Version 1 comparison boundary

- V1 own English held-out Macro F1: 0.988711
- V1 own English held-out phishing recall: 1.000000
- V1 and V2 held-out figures are descriptive, not a paired comparison, because the held-out cohorts differ.
- V2 English test rows overlapping V1 development: 5530 / 6525.
- V1 cannot be evaluated fairly on the exact V2 English test subset because rows/groups from that subset occur in V1 development data.
- Vietnamese translated rows with known parents: 179 / 179.
- A leakage-safe V1 comparison on the exact translated-Vietnamese subset cannot be established because parent provenance is incomplete or a known parent/content/group appears in V1 development data.

## Reload smoke test

These handcrafted examples test artifact loading and output shape. They are not metrics.

| Scenario | Prediction | Calibrated confidence |
|---|---|---:|
| English normal example | normal | 0.998186 |
| English spam example | normal | 0.998157 |
| English phishing example | phishing | 0.999638 |
| Vietnamese normal example | normal | 0.999772 |
| Vietnamese spam-like example | normal | 0.503564 |
| Vietnamese phishing-like example | phishing | 0.974160 |

## Limitations and claim boundary

- Vietnamese evaluation uses translated data only. It does not establish native Vietnamese real-world performance.
- Any class absent from a subset has N/A metrics rather than an invented zero score.
- The phishing corpus remains source-concentrated, so source/class confounding is possible.
- Spam labels retain curated provenance and are not all individually human-confirmed.
- Mixed-language email has not been independently benchmarked.
- No post-test tuning or repeat held-out evaluation is permitted. A serious issue requires a Version 2.1 protocol and a new test set.
