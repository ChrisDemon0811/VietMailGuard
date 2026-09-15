# VietMailGuard Version 2 — Final Completion Audit

Audit date: 2026-09-15  
Scope: scientific methodology, dataset integrity, leakage, frozen model, calibration, inference, security/risk, Streamlit, claims, and reproducibility.  
Constraint observed: the production model was not retrained, TF-IDF was not refit, calibration was not rerun, and the held-out test was not reevaluated.

## Final status

**Version 2 status: COMPLETE WITH WARNINGS.**

No critical component has a `FAIL`. Warnings are scientific limitations, not hidden implementation failures.

| Component | Critical | Status | Evidence / limitation |
|---|:---:|:---:|---|
| Dataset integrity | Yes | PASS | Seven raw files match their recorded SHA-256 values; split rows have valid labels, eligibility, raw labels, and label provenance. |
| Leakage | Yes | PASS | Zero cross-split overlap for `final_group_id`, `content_hash`, `normalized_content_hash`, `template_hash`, `group_id`, and `translation_group_id`; all known parents and translations stay together. |
| Held-out test integrity | Yes | PASS | Evaluation count is 1; selection preceded test access; post-test tuning is disabled; final refit contains train + validation only. |
| Metrics consistency | Yes | PASS | Metrics were recomputed from the saved confusion matrix and agree with final JSON, model metadata, classification-report CSV, confusion CSV, and dashboard loader. |
| Production model artifact | Yes | PASS | Artifact loads successfully and SHA-256 matches model metadata. |
| Calibration/probability | Yes | PASS | Production classifier is `CalibratedClassifierCV`; probabilities align with estimator classes, remain in [0,1], and sum to one. |
| Inference API | Yes | PASS | English, Vietnamese, mixed-language, normal, spam, phishing, promotional disagreement, URL, and explanation smoke checks pass. |
| Security/risk | No | PASS | ML prediction, calibrated confidence, security findings, and risk score remain separate; rules do not relabel the classifier output. |
| Streamlit web | Yes | PASS | Four pages load in both interface languages, metrics come from V2 artifacts, UI-language switching preserves inference, and HTTP startup succeeds. |
| Scientific claims | Yes | PASS | UI, README, and reports identify Vietnamese data as translated/experimental and explicitly state that no native benchmark or Vietnamese phishing test exists. |
| Windows reproducibility | No | PASS | README now covers environment, install, audit, standardization, curation, augmentation, split, both experiment families, freeze/finalization boundary, robustness, web, and pytest. |
| Native Vietnamese evidence | No | WARNING | No independent native Vietnamese benchmark exists; the translated held-out subset has no phishing examples. |
| Robustness | No | WARNING | Removing Vietnamese accents collapses spam recall on the small robustness seed; very-short promotional email is a material failure mode. |
| External raw provenance | No | WARNING | Local hashes prove consistency with recorded repository audit baselines, not authenticity against an externally signed upstream release. |

## 1. Dataset composition and integrity

### Eligible bilingual master

| Dimension | Rows |
|---|---:|
| Total | 44,717 |
| English | 42,112 |
| Vietnamese, explicitly `data_origin=translated` | 2,605 |
| Normal | 40,861 |
| Spam | 1,817 |
| Phishing | 2,039 |
| Controlled translated augmentation included | 1,460 |
| Existing translated `data_vi` rows included | 1,145 |

### Rows held outside the master/splits

| Reason | Rows |
|---|---:|
| Vietnamese review | 1,330 |
| Vietnamese exclude | 1 |
| Vietnamese eligible label but unresolved parent linkage | 3,089 |
| Augmentation failed quality checks | 40 |

Checks on the actual split files confirm:

- labels are exactly `normal`, `spam`, or `phishing`;
- no `review` or `exclude` status enters a model split;
- every split row has a non-empty `raw_label` and `label_provenance`;
- every Vietnamese split row is marked `language=vi` and `data_origin=translated`;
- no row in the bilingual master is marked `native`;
- controlled translations are marked `augmentation_type=controlled_translation`, source `controlled_translation`, and `split_constraint=train_only`.

### Raw-file SHA-256 verification

| File | Recorded/current SHA-256 | Status |
|---|---|:---:|
| CEAS_08.csv | `22375e7d5f5a8229dbe987914ee9b3705656c590038662a7df6054629b376074` | PASS |
| Enron.csv | `99933d3233510cf5dc2ee7768ddc609425cb8002bf5d7691d9a7157ffa5fd318` | PASS |
| Ling.csv | `c133792260f18b251e9377b9cb31bef226322af6dc0d79841f61c67489929eca` | PASS |
| Nazario.csv | `b8fbc4158fbdfaff1ed98584c43d72e283c6352d7ade4d457d34c1d79488d184` | PASS |
| Nigerian_Fraud.csv | `f5d6930763cae6feeae6484131c79655a7440abbb97d6f51eab757aea4e6ab4b` | PASS |
| SpamAssasin.csv | `3bfe8f8abff89f69a98456be50413c2fcb20a476141116dd84ee1507b980c00e` | PASS |
| data_vi.csv | `d9fee4aee5d13126852d543758595d9ea78e67d074d5ab24eb0dbeb2f9ea2125` | PASS |

The audit performed no write under `data/raw/`.

## 2. Leakage audit

Actual saved split sizes are 31,304 train, 6,709 validation, and 6,704 test rows.

| Leakage identifier | Cross-split overlap |
|---|---:|
| `final_group_id` | 0 |
| `content_hash` / exact duplicate | 0 |
| `normalized_content_hash` | 0 |
| `template_hash` / template-equivalent | 0 |
| source `group_id` | 0 |
| `translation_group_id` | 0 |

Additional relationship checks:

- parent/translation pairs checked: 2,605;
- controlled train-only rows checked: 1,460;
- parent missing from all splits: 0;
- parent/translation split mismatch: 0;
- controlled translation outside train: 0;
- controlled test rows: 0.

The final grouping is therefore sufficient to bind exact, normalized, template-equivalent, English-parent, existing Vietnamese translation, and controlled-augmentation relationships.

## 3. Model selection and held-out integrity

The written selection rationale predates test access and selects `word_tfidf__linear_svm` using validation evidence:

- primary criterion: Macro F1;
- secondary criterion: phishing recall;
- additional criteria: language balance, normal false positives, latency, and deployment complexity;
- selected validation Macro F1: 0.986190;
- selected validation phishing recall: 0.983660;
- best multilingual embedding candidate was not selected because validation performance was materially lower and deployment cost was higher.

Frozen contract checks:

| Check | Stored value | Status |
|---|---|:---:|
| Held-out evaluation count | 1 | PASS |
| Selection basis | `validation_only` | PASS |
| Held-out accessed during selection | false | PASS |
| Selection completed before test access | true | PASS |
| Post-test tuning allowed | false | PASS |
| Final refit inputs | train + validation | PASS |
| Test used for calibration fitting | no | PASS |
| Risk rules frozen independently of held-out test | documented in `risk_rules_v2.json` | PASS |

No finalization/evaluation command was executed during this completion audit. The existing test result remains the one-shot guard.

## 4. Production architecture and calibration

- Representation: Word TF-IDF, n-grams `(1,2)`, maximum 60,000 features, `min_df=2`, sublinear TF, float32.
- Classifier: `LinearSVC(C=1.0, class_weight=balanced, max_iter=5000, random_state=42)`.
- Calibration: sigmoid `CalibratedClassifierCV`, five-fold shuffled `StratifiedGroupKFold`, seed 42, grouped by `final_group_id`, ensemble disabled.
- Shared text composer: `subject + "\n" + body` inside the serialized sklearn pipeline.
- Production artifact size: 2,199,876 bytes.
- Production SHA-256: `b2ebecf641f2a5fd3ce9f0d0680a90884850e46b59c48ce0ab5bd4fee64f87fa`.

The estimator's probability columns are paired with `classifier.classes_` by name. The reporting order (`normal`, `spam`, `phishing`) is independent of the estimator's internal array order, avoiding swapped probabilities. Seven smoke inputs produced finite probabilities in [0,1] whose sums satisfied the `1e-7` tolerance.

## 5. Held-out metric consistency

The following values were read from the final result and independently recomputed from its 3×3 confusion matrix. They agree with model metadata, classification CSV, confusion CSV, final evaluation report, and the Streamlit dashboard loader.

| Metric | Actual |
|---|---:|
| Test rows | 6,704 |
| Accuracy | 0.997763 |
| Macro Precision | 0.997992 |
| Macro Recall | 0.983992 |
| Macro F1 | 0.990900 |
| Weighted F1 | 0.997750 |
| Phishing recall | 0.970588 |

### Per-class consistency

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.997720 | 0.999837 | 0.998777 | 6,127 |
| spam | 0.996255 | 0.981550 | 0.988848 | 271 |
| phishing | 1.000000 | 0.970588 | 0.985075 | 306 |

Confusion matrix:

| Actual / predicted | normal | spam | phishing |
|---|---:|---:|---:|
| normal | 6,126 | 1 | 0 |
| spam | 5 | 266 | 0 |
| phishing | 9 | 0 | 297 |

### English test

| Metric | Actual |
|---|---:|
| Rows | 6,525 |
| Accuracy | 0.997701 |
| Macro F1 | 0.990774 |
| Normal recall | 0.999832 |
| Spam recall | 0.980989 |
| Phishing recall | 0.970588 |

### Translated Vietnamese test

| Item | Actual |
|---|---:|
| Rows | 179 |
| Normal | 171 |
| Spam | 8 |
| Phishing | 0 |
| Accuracy over represented rows | 1.000000 |
| Macro F1 over represented classes only | 1.000000 |
| Three-class Macro F1 | N/A |
| Vietnamese phishing recall | N/A |

**Native Vietnamese benchmark is not available.** The translated result is not native Vietnamese accuracy and does not validate Vietnamese phishing detection.

## 6. Inference, explainability, security, and risk

The production pipeline reloads successfully through the single shared inference API. Streamlit contains no direct `.predict()`, `predict_proba()`, TF-IDF fitting, or duplicate preprocessing call.

Observed smoke behavior:

| Scenario | Language/support | Prediction | Confidence | Risk | Action |
|---|---|---|---:|---:|---|
| English normal | en / supported | normal | 0.997545 | 0 LOW | ALLOW |
| English spam | en / supported | spam | 0.967619 | 20 LOW | MOVE_TO_SPAM |
| English phishing | en / supported | phishing | 0.999752 | 100 CRITICAL | QUARANTINE |
| Vietnamese normal | vi / experimental | normal | 0.999592 | 0 LOW | ALLOW |
| Vietnamese promotional | vi / experimental | normal | 0.503564 | 11 LOW | REVIEW |
| Vietnamese phishing-style | vi / experimental | phishing | 0.974160 | 100 CRITICAL | QUARANTINE |
| Mixed English/Vietnamese | mixed / experimental | phishing | 0.999554 | 69 HIGH | QUARANTINE |

The promotional disagreement demonstrates the required separation:

```text
ML prediction: normal
Base ML action: ALLOW
Security finding: commercial_promotion
Final recommendation: REVIEW
```

The prediction remains `normal`; the rule does not fabricate an ML `spam` prediction. Model Confidence is calibrated probability, while Risk Score is a separate 0–100 decision-support heuristic. URL rules report suspicious indicators, not confirmed malicious verdicts. Model explanations use observed TF-IDF feature contributions where the calibrated linear structure supports extraction.

## 7. Robustness and short-form challenge

These are post-freeze diagnostic tracks and remain separate from the one-shot held-out metrics.

### Vietnamese transformations

- Seed: 40 translated Vietnamese test parents, 32 normal and 8 spam; no phishing.
- Clean accented: Accuracy 1.0000, represented-class Macro F1 1.0000.
- Remove accents: Accuracy 0.8000, Macro F1 0.4444, spam recall 0.0000, prediction flip rate 20%.
- All non-clean transformations combined: Accuracy 0.9600, Macro F1 0.9322, spam recall 0.8000.
- All eight no-accent spam errors were high-confidence normal predictions.
- This seed is translated and test-derived; it is not a new independent or native held-out benchmark.

### Short-form challenge

- 90 controlled synthetic scenarios: 30 per class and 45 per language.
- Overall Accuracy 0.8556, Macro F1 0.8562, failure rate 14.44%.
- Spam misses: 9/30; phishing misses: 2/30.
- Very-short failure rate: 36.67%; very-short spam recall: 0.1000.
- Seven rows have `ML=normal` together with a promotional security signal.
- Security signals were present on 8/13 ML errors; review/quarantine mitigated 9/13, but these are not counted as classifier successes.

No post-test retraining, TF-IDF refit, recalibration, or threshold tuning followed these findings. They are requirements evidence for a future Version 2.1 protocol with a new held-out set.

## 8. Web and claim integrity

Streamlit verification covers:

- Email Analyzer, Model Dashboard, Dataset Dashboard, and Methodology/About;
- English and Vietnamese interface labels;
- identical internal prediction and confidence after changing UI language;
- V2 model name and language support loaded from metadata/inference;
- actual V2 result artifacts loaded by dashboards;
- translated-Vietnamese 179/171/8/0 presentation with immediate missing-phishing caveat;
- graceful states for missing result artifacts and missing production artifacts;
- explicit Model Confidence vs Overall Risk Score distinction;
- separate model explanation, content/sender security findings, and URL findings;
- no stale Version 1 unsupported-Vietnamese warning.

A repository claim scan found only negated/prohibited examples of “Vietnamese real-world accuracy”; no positive native-accuracy, Vietnamese-phishing-recall, or “Vietnamese phishing = 100%” claim was found in active UI/README/reports.

## 9. Reproducibility and tests

README now documents the complete Version 2 sequence and actual Windows CMD entry points. Training and one-shot finalization commands are retained for clean-workspace research reproduction but are clearly marked as commands that must not be rerun against the completed frozen repository.

Verification results:

| Verification | Result |
|---|---|
| Raw SHA audit | 7/7 PASS |
| Targeted dataset/leakage/model/inference/risk/web tests | 59 passed |
| Full pytest | 178 passed in 20.70 s |
| Production inference smoke | 7/7 scenarios completed |
| Streamlit AppTest | all four pages and required language/content scenarios passed |
| Streamlit server | HTTP 200 on localhost:8501 |

Commands used:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_completion_integrity.py tests\test_bilingual_dataset.py tests\test_v2_finalization.py tests\test_inference_v2.py tests\test_risk_engine.py tests\test_streamlit_app.py -q
.\.venv\Scripts\python.exe scripts\smoke_inference_v2.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app\app.py --server.headless true --server.port 8501
```

## 10. Remaining limitations and future work

1. Acquire and lock an independent native Vietnamese benchmark, especially phishing/social-engineering email.
2. Create new Version 2.1 development/test governance before changing the model or risk rules.
3. Improve no-accent Vietnamese and very-short commercial-email handling using development data, never the frozen V2 test/challenge labels as tuning ground truth.
4. Broaden phishing sources to reduce source/class confounding.
5. Complete human review of controlled-translation quality samples and unresolved Vietnamese linkage candidates.
6. Consider external-domain reputation only as an optional privacy-aware layer; core inference must remain offline and must not call heuristic suspicion “confirmed malicious.”

Because leakage, held-out integrity, model artifact, inference, web, and scientific-claims checks all pass, VietMailGuard Version 2 satisfies its completion gate. The `WARNING` items remain explicit constraints on what the system may scientifically claim.
