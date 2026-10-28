# VietMailGuard V2.1 Research Preview

Status: **local research preview, not V2.1 production** (2026-10-22). The frozen V2 production pipeline remains the default. No V2.1 final held-out evaluation has been run. This page summarizes saved results; it is not a new evaluation or a claim of verified English–Vietnamese–Chinese deployment.

## What can be demonstrated

| Component | Preview behavior | Boundary |
| --- | --- | --- |
| V2 Production | Default three-class ML, calibrated probabilities, offline security analysis, risk score and local mailbox routing | Frozen artifact; English evaluated, Vietnamese support experimental and predominantly translated |
| V2.1 Machine Learning | Opt-in, hash-verified Word+Character TF-IDF + LinearSVC candidate | Development/Validation only; **uncalibrated**: no model confidence or class probabilities |
| V2.1 Hybrid | The same V2.1 ML result plus a separate Gemini Reviewer after explicit per-email consent and button click | Reviewer can suggest `REVIEW`; it cannot rewrite ML prediction, risk or mailbox folder; a remote request may incur cost |
| TREC06C Chinese binary | Separate local review application and saved Ham/Spam research experiment | **Not** a three-class mode, not connected to the main mailbox router, no Chinese phishing metric |

The main interface languages are English and Vietnamese. This selector changes presentation, not the email language or prediction. V2.1 can process Chinese text as an unvalidated research input, but has no verified Chinese three-class support or Chinese-specific security rules. Native Vietnamese accuracy and Vietnamese phishing recall are also unavailable. The V2.1 development candidate was trained using English and *translated* Vietnamese, with zero Chinese supervised examples.

## Saved research results — do not mix evaluation tracks

| Track | Saved result | Interpretation |
| --- | --- | --- |
| Frozen V2 one-shot Test | 6,704 rows; Accuracy **0.997763**, Macro F1 **0.990900**, Phishing recall **0.970588** | V2 held-out result only; translated Vietnamese Test has 179 rows (171 Normal, 8 Spam, **0 Phishing**) |
| V2.1 Development candidate | Inherited Train **31,304**, Validation **6,709**; Validation Macro F1 **0.985852**, Phishing recall **0.970588** | V2 Train/Validation reuse, not an independent V2.1 Test; no native VI/ZH benchmark |
| TREC06C binary experiment | 64,620 raw Ham/Spam; 59,079 passed experiment-quality gates; Validation Macro F1 **0.993805** | Source-label Ham/Spam only; its binary Test remains sealed; **0** TREC three-class eligible decisions |
| Chinese synthetic exact-candidate A/B | 3,030 synthetic rows audited; 2,367 screened synthetic Train; A **0.985852** vs B **0.984599** Macro F1 on the unchanged EN/VI Development Validation | B added synthetic Chinese; delta **−0.001253**. The separate same-generator diagnostic is not real Chinese performance |
| V2.1 data audit and curation | 1,147,062 audit rows: 1,102,345 new-source rows + 44,717 read-only V2 reference rows | New-source approval: **0 eligible**, 517,564 review, 584,150 weak_label, 631 exclude. The 38,013 bootstrap rows are reused verified V2 Development data, not new approved sources |
| Independent Chinese real-mail benchmark | Candidate manifest **0 accepted rows**, status **pending** | No independently verified Chinese Normal/Spam/Phishing benchmark; no metric or final-evaluation authorization |

Sources: [`results/v2_bilingual/final_test_metrics.json`](results/v2_bilingual/final_test_metrics.json), local `results/v21_multilingual/development_validation_metrics.json`, and local reports `reports/v21/{trec06c_binary_experiment,research_synthetic_zh_current_svm,source_label_approval_report,chinese_real_world_benchmark_protocol}.md`. The V2.1 generated artifacts are local/Git-ignored; their presence is **not** assumed in a clean GitHub clone. The paired research A/B under a *different* shorter-text configuration is documented separately in `research_synthetic_zh_experiment.md` and must not be merged with the exact-current-candidate A/B values above.

## Windows PowerShell: run the local preview

Run from the repository root. Python 3.11+ is supported; the tested local V2 environment is Python 3.13.14. A clean clone does **not** contain the V2 model binary or V2.1 candidate. Obtain the V2 artifact from the official project Release and verify it with `scripts/download_production_model.py`; do not run a download without a real official URL. V2.1 candidate artifacts are local research outputs, not a published production release. Do not retrain merely to make a preview mode appear.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -m "not production_artifact" -q
.\.venv\Scripts\python.exe -m streamlit run app/app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. V2 is selected by default. The V2.1 ML selector appears only when `config/v21/runtime_model.json`, its referenced local candidate joblib, and the saved Validation JSON all pass hash/scope/pipeline checks. Run this **read-only** availability check:

```powershell
.\.venv\Scripts\python.exe -c "from vietmailguard.v21.runtime_inference import model_availability; print(model_availability())"
```

If V2.1 is absent or invalid, keep/select V2; the application fails closed instead of loading an unverified model. A V2 production artifact is still needed for V2 mode. To verify an official Release asset, set `V2_PRODUCTION_MODEL_URL` to its real HTTPS URL and run `scripts/download_production_model.py` according to `README.md`; the script checks the SHA-256 stored in V2 metadata. Never copy an unverified joblib into the candidate directory.

### Optional Gemini Reviewer

Gemini is **not required** for V2 or V2.1 ML. For the optional Hybrid mode:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gemini.txt
Copy-Item .env.example .env
# Edit only the local, Git-ignored .env: GEMINI_API_KEY and an accessible GEMINI_MODEL.
.\.venv\Scripts\python.exe scripts/v21/check_gemini.py
```

The check above is offline and prints configuration status without the key. No live request is needed for this preview. The app only calls Gemini after the user chooses Hybrid, consents to sending **that email**, and clicks Reviewer. Do not use real/sensitive mail without a separate privacy decision; a successful mock test does not prove live API availability, translation correctness, quota or cost. To return to V2, choose **V2 frozen production** in the mode selector. Reviewer output does not retroactively reroute or modify saved ML history.

### TREC06C research demo, isolated from the main app

Use the **separate** Vietnamese review UI only if its local review queues and original ZIP are already available. Configure the source directory without embedding a personal drive path in code:

```powershell
$env:VIETMAILGUARD_V21_DATA_DIR = '<your local dataset directory>'
.\.venv\Scripts\python.exe scripts/v21/curate_trec06c.py status
.\.venv\Scripts\python.exe -m streamlit run scripts/v21/trec06c_review_app.py --server.address 127.0.0.1 --server.port 8502
```

If queues are missing, consult `reports/v21/trec06c_human_review_report.md` and the `prepare`/`prepare-deep` CLI help before creating them. The UI shows original Chinese and optional Vietnamese translation; AI suggestions are weak labels. It does not approve all Ham/Spam, train a classifier or run the sealed TREC binary Test. The saved binary Validation report can be read without retraining; do **not** run `trec06c_binary_experiment.py train` for this preview.

## Limitations and next benchmark gates

| Status | Remaining work before any V2.1 production/final claims |
| --- | --- |
| **BLOCKED** | Independent, privacy/licensing-cleared Chinese **original email** benchmark with all three classes and qualified Chinese-language annotation; current candidate manifest is pending |
| **BLOCKED** | Independent native Vietnamese benchmark, including Phishing; translated VI Validation has no Phishing support |
| **BLOCKED** | New V2.1 final-Test protocol, source/label approval, provenance and parent/template leakage review, model/reviewer/risk freeze, explicit one-time Test access decision |
| **WARNING** | TREC binary and synthetic Chinese results are separate research tracks, not three-class or real-world Chinese phishing evidence |
| **WARNING** | V2.1 LinearSVC is uncalibrated; no probability-based confidence. Gemini availability, cost and privacy depend on the user's optional configuration and consent |
| **WARNING** | A clean clone lacks Git-ignored binaries and local datasets; current tracked V2 curated/evaluation CSVs contain email-text columns and need a separate publication/privacy decision |

See [`reports/v21/research_preview_qa.md`](reports/v21/research_preview_qa.md) for the exact local checks, Git comparison, safe-commit proposal and checks not completed. No commit, push, release, new training or final evaluation is part of this preview.
