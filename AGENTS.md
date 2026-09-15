# AGENTS.md

## 1. Project Identity

Project name: **VietMailGuard**

VietMailGuard is a Machine Learning web application for detecting and analyzing malicious email.

Current development target:

**Version 1 — English Email Detection with a Bilingual English/Vietnamese Web Interface**

The Machine Learning model in Version 1 is trained and evaluated using English email datasets.

However, the web application itself must support two interface languages from the beginning:

* English
* Vietnamese

The interface language is independent from the Machine Learning model language.

Changing the UI from English to Vietnamese must only translate interface text. It must not imply that the current Version 1 model has been trained for Vietnamese email classification.

Future target:

**Version 2 — English + Vietnamese bilingual email detection**

Version 1 must therefore be designed so Vietnamese datasets can later be added without rewriting the architecture or web interface.

---

## 2. Main Project Goal

Build a complete, reproducible Machine Learning pipeline and bilingual web application that can:

1. Standardize multiple heterogeneous email datasets.
2. Clean and audit data before training.
3. Prevent duplicate/template leakage between train, validation, and test data.
4. Extract text and security-related features.
5. Train multiple Machine Learning classifiers.
6. Compare models scientifically.
7. Save the selected production model.
8. Analyze individual emails from a web interface.
9. Explain why an email was classified as normal, spam, or phishing.
10. Produce a separate risk score.
11. Recommend an action to the user.
12. Display model evaluation results in a dashboard.
13. Allow the user to switch the entire web interface between English and Vietnamese.

Machine Learning must remain the central part of the project.

Do not turn this into only a web-development project.

---

## 3. Current Scope

### Version 1 — Machine Learning

Training language:

`en`

Supported ML classes:

* `normal`
* `spam`
* `phishing`

Primary input fields:

* Sender
* Subject
* Email body
* URLs extracted from content

Optional input:

* `.eml` file upload

Output:

* Predicted class
* Model confidence
* Risk score from 0 to 100
* Risk level
* Explanation
* Suspicious indicators
* Recommended action

Recommended actions:

* `normal` → ALLOW
* `spam` → MOVE TO SPAM
* `phishing` → QUARANTINE / WARNING

### Version 1 — Web Interface

The web application must support:

* English UI
* Vietnamese UI

Provide a clearly visible language selector such as:

`English | Tiếng Việt`

or a sidebar selectbox.

Changing interface language must immediately update all user-facing application text.

Examples:

English:

* Email Analyzer
* Sender
* Subject
* Email Content
* Analyze Email
* Normal
* Spam
* Phishing
* Model Confidence
* Overall Risk Score
* Recommended Action
* Why was this email classified this way?

Vietnamese:

* Phân tích Email
* Người gửi
* Tiêu đề
* Nội dung Email
* Phân tích Email
* Bình thường
* Thư rác
* Lừa đảo
* Độ tin cậy của mô hình
* Điểm rủi ro tổng thể
* Hành động đề xuất
* Tại sao email này được phân loại như vậy?

Do not duplicate the entire Streamlit application into separate English and Vietnamese files.

Use one application and a centralized translation system.

---

## 4. Important Language Distinction

The system must distinguish between:

### Interface Language

Controls what the user sees on the website.

Possible values:

* `en`
* `vi`

### Email Content Language

Represents the language of the email being analyzed.

In Version 1, the production Machine Learning model is officially trained for:

`en`

The interface may still be displayed in Vietnamese.

Example:

```text
Interface language: Vietnamese
Email being analyzed: English
ML model: English Version 1
```

This is valid.

Do not confuse:

```text
Vietnamese interface
```

with:

```text
Vietnamese-trained classifier
```

Version 1 documentation and the web application must clearly state this limitation.

---

## 5. Future Version 2

Later add:

`language = vi`

Version 2 will retrain using English + Vietnamese data.

Do not implement Vietnamese-specific ML training until Vietnamese datasets exist.

However:

* preserve the `language` field from Version 1,
* preserve Unicode-safe processing,
* keep the bilingual interface,
* design text resources so new UI languages can be added without changing business logic.

---

## 6. Technology Stack

Use Python as the primary language.

Target Python:

Python 3.11 or newer unless an installed dependency requires another supported version.

### Core

* Python
* pandas
* numpy
* scikit-learn
* scipy
* joblib

### Text processing

* Python `re`
* Python `unicodedata`
* BeautifulSoup4
* standard-library `email` package for `.eml`
* urllib.parse

### Security features

* tldextract
* regex
* urllib.parse

Do not require external threat-intelligence APIs for the core application.

The application must work offline after dependencies and models are installed.

### Visualization

* matplotlib
* plotly

### Web application

* Streamlit

The project is a web application even though Streamlit and Python are used instead of a JavaScript frontend framework.

### UI Internationalization

Use a lightweight Python-based translation layer.

Preferred design:

`config/translations.py`

or:

`config/i18n.json`

Do not install a heavy internationalization framework unless necessary.

### Testing

* pytest

### Optional persistence

* SQLite using Python `sqlite3`

SQLite may be used for scan history only after the core ML pipeline works correctly.

---

## 7. Machine Learning Task

This is a multiclass classification problem.

Do NOT use Linear Regression.

Required classifiers:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine

Optional experiments may later include:

* Random Forest
* PhoBERT / Transformer models

Do not add deep learning until the classical ML pipeline is complete and verified.

---

## 8. Feature Extraction

Implement experiments incrementally.

### Experiment A — Word TF-IDF

Use email subject + body.

Recommended starting configuration:

* word n-grams: `(1, 2)`
* lowercase normalization
* configurable `min_df`
* configurable `max_features`

Do not hardcode final hyperparameters before experiments.

### Experiment B — Character TF-IDF

Add character n-grams.

Recommended starting range:

`(3, 5)`

Character TF-IDF is important for robustness against:

* misspellings
* obfuscation
* unusual punctuation
* future Vietnamese text without accents
* phishing words deliberately modified to bypass filters

### Experiment C — Word + Character TF-IDF

Combine both representations with `FeatureUnion` or an equivalent sklearn pipeline.

### Experiment D — Security Features

Only after the text baseline works, evaluate whether adding deterministic security features improves results.

Possible features:

* URL count
* suspicious URL token count
* URL length
* host represented as an IP address
* HTTP vs HTTPS
* number of subdomains
* suspicious domain terms
* sender-domain information
* uppercase ratio
* exclamation mark count
* urgency indicators
* credential-request indicators
* account-suspension indicators
* financial bait indicators
* prize/winner indicators

Security features must never directly encode the ground-truth label.

---

## 9. Dataset Sources

The project may use multiple English datasets, including curated corpora such as:

* CEAS
* Enron
* Ling
* SpamAssassin
* Nazario
* Nigerian Fraud

Never assume two files are independent simply because the filenames differ.

Never include duplicate representations such as `.txt` and `.csv` versions of the same corpus.

Never merge an output/result file into the training corpus unless its provenance is verified.

---

## 10. Raw Data Safety

Raw datasets are immutable.

Never edit files inside:

`data/raw/`

All transformations must produce new files under:

`data/processed/`

Never overwrite the original datasets.

Use UTF-8 whenever possible.

Use `pathlib.Path`.

Never hardcode absolute Windows paths such as:

`C:\Users\...`

---

## 11. Standard Dataset Schema

All datasets must be standardized before merging.

Target schema:

* `id`
* `sender`
* `receiver`
* `date`
* `subject`
* `body`
* `label`
* `raw_label`
* `source`
* `language`
* `url_count`
* `content_hash`
* `group_id`

Required labels:

* `normal`
* `spam`
* `phishing`

Required languages for Version 1:

* `en`

Future:

* `vi`

If a field is unavailable, use an empty value.

Do not invent sender, receiver, subject, URL, or date values.

Always preserve:

* original source name
* original label

---

## 12. Label Mapping Policy

Dataset label mapping must be explicit and configurable.

Store mapping rules in:

`config/datasets.json`

Never silently guess what `0`, `1`, `ham`, `spam`, or another raw label means.

If label semantics are uncertain:

1. preserve the row,
2. mark it for review,
3. exclude it from final three-class training until its mapping is justified.

Dedicated phishing/scam corpora such as Nazario or Nigerian Fraud may be mapped to the phishing/social-engineering class when supported by dataset provenance.

Document any broad class definition in the README.

Do not claim that a spam/scam mixture is pure credential phishing.

---

## 13. Dataset Standardization Tool

Create a reusable Python standardization pipeline.

It must:

* load CSV files
* optionally load XLSX files
* normalize column names
* map source-specific columns into the standard schema
* normalize labels using configuration
* remove null/empty content
* detect placeholder strings such as `empty`
* normalize Unicode
* normalize whitespace
* extract URLs
* calculate deterministic metadata
* detect exact duplicates
* detect normalized/template duplicates
* generate content hashes
* preserve source provenance
* create cleaning reports
* create duplicate reports
* export a master dataset

Primary output:

`data/processed/English_Master_Dataset.csv`

Reports:

`reports/dataset_audit.csv`

`reports/removed_rows.csv`

`reports/duplicate_report.csv`

Never delete questionable rows without recording why they were removed.

---

## 14. Duplicate and Leakage Rules

Prevent data leakage aggressively.

Do not perform a naive random train/test split before duplicate analysis.

At minimum normalize text for duplicate detection by:

* Unicode normalization
* lowercasing for comparison
* whitespace normalization
* replacing URLs with a placeholder
* replacing email addresses with a placeholder

Create a group identifier for duplicated or near-template-equivalent samples.

Samples from the same duplicate/template group must never appear in different splits.

The same original/augmented email must remain within one split.

---

## 15. Train / Validation / Test

Target approximate ratio:

* Train: 70%
* Validation: 15%
* Test: 15%

Use reproducible seeds.

Default seed:

`42`

Maintain class distribution as closely as possible while respecting duplicate groups.

Fit all vectorizers and preprocessing learned from data ONLY on the training set.

Never fit TF-IDF on the complete dataset before the split.

Never use the test set for model selection or hyperparameter tuning.

The test set is for final evaluation only.

---

## 16. Evaluation Metrics

Do not judge the system only by accuracy.

Required metrics:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Weighted F1
* Precision per class
* Recall per class
* F1 per class
* Confusion Matrix

Pay special attention to:

* phishing recall
* phishing false negatives
* normal-email false positives

Primary selection metric:

`Macro F1`

Use phishing recall as an important secondary criterion.

Do not invent metrics.

All values displayed by the web application must come from actual saved experiment results.

---

## 17. Model Comparison

Generate a comparison table containing at least:

* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM

Evaluate:

* Word TF-IDF
* Character TF-IDF
* Word + Character TF-IDF

Do not automatically assume Linear SVM or Logistic Regression is best.

Choose based on measured validation results.

Store experiment results under:

`results/`

Example:

`results/model_comparison.csv`

---

## 18. Production Model

Save the complete sklearn pipeline, not only the classifier.

Artifacts may include:

* text preprocessing
* vectorizer
* classifier
* label encoder if required
* calibration component if required

Store under:

`models/`

Example:

`models/production_pipeline.joblib`

The inference application must not separately recreate a vectorizer with different parameters.

Training and inference preprocessing must be consistent.

---

## 19. Confidence

For Logistic Regression use calibrated class probabilities directly from the fitted pipeline.

For LinearSVC, do not treat the raw decision score as a probability.

If LinearSVC becomes the production model and confidence is required, use an appropriate probability calibration method such as:

`CalibratedClassifierCV`

Clearly distinguish:

**Model Confidence**

from:

**Overall Risk Score**

They are not the same number.

---

## 20. Risk Engine

Create a deterministic, documented risk engine.

Risk score range:

`0–100`

Potential inputs:

* ML phishing probability/confidence
* suspicious URL findings
* sender/domain findings
* credential-request language
* urgency
* financial lure
* account suspension language
* excessive formatting indicators

Keep weights in:

`config/risk_rules.json`

Do not scatter unexplained magic numbers throughout source files.

The risk score must be described as a decision-support heuristic, not as model probability.

Risk levels may be:

* 0–30: LOW
* 31–60: MEDIUM
* 61–80: HIGH
* 81–100: CRITICAL

These thresholds must remain configurable.

The translated UI labels may be:

English:

* LOW
* MEDIUM
* HIGH
* CRITICAL

Vietnamese:

* THẤP
* TRUNG BÌNH
* CAO
* NGHIÊM TRỌNG

Do not alter the underlying internal enum when switching interface language.

---

## 21. URL Analyzer

Extract URLs before destructive text cleaning.

Analyze URLs offline.

Possible checks:

* scheme
* HTTPS availability
* hostname
* hostname represented as raw IP
* unusual URL length
* excessive subdomains
* suspicious tokens such as `login`, `verify`, `secure`, `account`
* punycode marker
* URL shortener patterns when configured
* mismatch between claimed brand term and domain where feasible

Never claim a domain is malicious solely because the heuristic score is high.

Use wording such as:

`suspicious indicator`

rather than:

`confirmed malicious`

unless evidence actually supports it.

The Vietnamese UI may translate this as:

`dấu hiệu đáng ngờ`

rather than:

`đã xác nhận độc hại`

unless the system truly has evidence for that stronger claim.

---

## 22. Content Security Analyzer

Detect explainable social-engineering patterns.

Categories:

* urgency
* credential request
* account suspension
* financial bait
* prize/winner
* threatening language
* call-to-action
* excessive capitalization
* excessive punctuation

Rules should be configurable.

Do not let a rule directly determine the final ground-truth class during training.

Security rules supplement ML inference.

Version 1 security-language rules may initially target English email content.

Do not falsely claim Vietnamese phishing-language analysis is supported until Vietnamese rules or bilingual ML are implemented.

---

## 23. Explainability

The final prediction page must answer:

English:

**Why was this email classified this way?**

Vietnamese:

**Tại sao email này được phân loại như vậy?**

For linear classifiers, derive interpretable text contributions where feasible.

For Logistic Regression:

feature contribution can be based on TF-IDF feature value × class coefficient.

Show only a reasonable number of top contributing terms.

Prefer readable word features in the UI.

Character n-gram features may be used internally but should not flood the interface.

Combine model explanation with security-rule explanations.

Example output categories:

* suspicious phrase
* credential request
* account urgency
* suspicious URL
* sender/domain indicator
* ML-important terms

Translate category labels according to UI language.

Never fabricate explanations.

Do not translate the original analyzed email text unless explicitly requested.

---

## 24. Recommended Action

Internal values:

### Normal

`ALLOW`

### Spam

`MOVE_TO_SPAM`

### Phishing

`QUARANTINE`

Display labels:

English:

* Allow
* Move to Spam
* Quarantine

Vietnamese:

* Cho phép
* Chuyển vào thư rác
* Cách ly Email

The action must be presented as a recommendation, not as an irreversible automated security action.

---

## 25. Streamlit Web Application

Create a polished multipage Streamlit web application.

Required pages:

1. Email Analyzer
2. Model Dashboard
3. Dataset Dashboard
4. Methodology / About

Every page must support both interface languages.

### Global Language Selector

Provide a language selector available throughout the app.

Recommended placement:

* Streamlit sidebar
* or persistent top-level control

Options:

* English
* Tiếng Việt

Store selected language in:

`st.session_state`

The selected language must persist while moving between Streamlit pages during the current session.

Do not reload the ML model merely because UI language changes.

---

## 26. Translation Architecture

All user-facing strings must come from a centralized translation dictionary.

Recommended structure:

```python
TRANSLATIONS = {
    "en": {
        "app_title": "VietMailGuard",
        "email_analyzer": "Email Analyzer",
        "sender": "Sender",
        "subject": "Subject",
        "email_content": "Email Content",
        "analyze": "Analyze Email",
        "normal": "Normal",
        "spam": "Spam",
        "phishing": "Phishing",
        "model_confidence": "Model Confidence",
        "risk_score": "Overall Risk Score",
        "recommended_action": "Recommended Action"
    },
    "vi": {
        "app_title": "VietMailGuard",
        "email_analyzer": "Phân tích Email",
        "sender": "Người gửi",
        "subject": "Tiêu đề",
        "email_content": "Nội dung Email",
        "analyze": "Phân tích Email",
        "normal": "Bình thường",
        "spam": "Thư rác",
        "phishing": "Lừa đảo",
        "model_confidence": "Độ tin cậy của mô hình",
        "risk_score": "Điểm rủi ro tổng thể",
        "recommended_action": "Hành động đề xuất"
    }
}
```

Prefer a helper such as:

```python
t("email_analyzer")
```

instead of writing:

```python
if language == "vi":
    ...
else:
    ...
```

throughout every page.

Do not scatter translated strings throughout business logic.

Internal values must remain language-neutral.

Example:

```text
Internal:
phishing

English UI:
Phishing

Vietnamese UI:
Lừa đảo
```

The model must always return internal class identifiers, not translated labels.

---

## 27. Email Analyzer Page

Inputs:

* Sender
* Subject
* Email body
* optional `.eml` upload

Action:

`Analyze Email`

Vietnamese equivalent:

`Phân tích Email`

Results:

* predicted class
* model confidence
* risk score
* risk level
* explanation
* extracted URLs
* security findings
* recommended action

Translate UI labels but preserve:

* original sender
* original subject
* original email body
* URLs
* domains
* ML feature terms where translation would distort the actual model explanation

### Version 1 Model Language Notice

The analyzer must show a small, clear notice.

English:

`Version 1 is currently trained on English email datasets. The interface can be displayed in Vietnamese, but Vietnamese email classification will be added in Version 2.`

Vietnamese:

`Phiên bản 1 hiện được huấn luyện trên các bộ dữ liệu email tiếng Anh. Giao diện có thể sử dụng tiếng Việt, nhưng khả năng phân loại email tiếng Việt sẽ được bổ sung ở Phiên bản 2.`

Do not present this as an error.

It is a model-scope notice.

---

## 28. Model Dashboard

Display actual saved experiment results:

* selected model
* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* class-specific metrics
* phishing recall
* confusion matrix
* model comparison table

Translate headings and explanatory text.

Do not translate mathematical metric names unnecessarily.

For example both interfaces may retain:

* Accuracy
* Precision
* Recall
* F1-score

Vietnamese descriptions may explain their meaning.

Never hardcode fake metrics.

---

## 29. Dataset Dashboard

Display:

* number of samples
* class distribution
* source distribution
* duplicate-removal statistics
* train/validation/test counts

The underlying dataset remains English in Version 1.

The Vietnamese interface only translates the dashboard descriptions.

Example:

English:

`Class Distribution`

Vietnamese:

`Phân bố lớp`

Do not translate actual dataset source names such as:

* CEAS
* Enron
* Nazario
* SpamAssassin

---

## 30. Methodology / About

Explain:

* task definition
* datasets
* preprocessing
* TF-IDF
* classifiers
* evaluation metrics
* risk score distinction
* known limitations
* future Vietnamese model

Provide this information in both English and Vietnamese.

The Vietnamese page must use clear university-level Vietnamese rather than machine-translated awkward terminology.

Technical names should remain intact where appropriate:

* TF-IDF
* Logistic Regression
* Linear SVM
* Naive Bayes
* Macro F1

---

## 31. Streamlit UI Requirements

The UI should have a professional cybersecurity + Machine Learning appearance.

Use:

* cards
* columns
* metrics
* progress indicators
* charts
* tables
* clear status badges

Avoid:

* excessive animation
* excessive emojis
* duplicated pages
* hardcoded text in one language
* overly dense layout

The application must remain usable on a standard laptop display.

Changing interface language must not:

* retrain the model
* reload datasets unnecessarily
* change prediction
* change confidence
* change risk score
* change internal class IDs

Only presentation text should change.

---

## 32. Optional Scan History

Only after the core system works:

Use SQLite to store:

* scan id
* timestamp
* sender
* subject
* predicted class
* confidence
* risk score

Store internal prediction values:

`normal`
`spam`
`phishing`

Do not store translated class labels.

This allows the same history entry to render correctly in either English or Vietnamese.

Do not store full sensitive email content by default.

If body persistence is added, make it explicit and configurable.

---

## 33. Robustness Tests

Prepare a small robustness-test framework.

Version 1 may test English modifications such as:

* casing changes
* punctuation changes
* URL obfuscation
* spelling mutations
* spacing changes

Future Version 2 must support tests for:

* Vietnamese with accents
* Vietnamese without accents
* obfuscated Vietnamese
* English/Vietnamese mixed text

Keep robustness results separate from normal held-out test metrics.

---

## 34. Future Bilingual ML Upgrade

When Vietnamese datasets are added:

1. Standardize them to the same master schema.
2. Set `language = vi`.
3. Preserve Unicode NFC.
4. Merge English and Vietnamese training data.
5. Refit the TF-IDF vocabulary.
6. Retrain the classifiers.
7. Preserve the existing bilingual web interface.

Do NOT simply call `.fit()` on Vietnamese data using an English-only TF-IDF model and assume it becomes bilingual.

Evaluate separately:

* English test
* Vietnamese test
* combined test

The application should then support:

* English email
* Vietnamese email
* mixed English/Vietnamese email

The existing UI language selector must continue to operate independently from email-content language.

Example future state:

```text
UI language: Vietnamese
Email language: English
Prediction model: Bilingual
```

or:

```text
UI language: English
Email language: Vietnamese
Prediction model: Bilingual
```

Both must work.

---

## 35. Recommended Repository Structure

Use a structure similar to:

```text
VietMailGuard/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── app.py
│   ├── i18n.py
│   └── pages/
│
├── config/
│   ├── datasets.json
│   ├── risk_rules.json
│   └── translations.json
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── models/
│
├── reports/
│
├── results/
│
├── scripts/
│   ├── standardize_datasets.py
│   ├── build_splits.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── src/
│   └── vietmailguard/
│       ├── preprocessing.py
│       ├── dataset_standardizer.py
│       ├── duplicate_detector.py
│       ├── feature_extraction.py
│       ├── url_analyzer.py
│       ├── content_analyzer.py
│       ├── risk_engine.py
│       ├── explainability.py
│       ├── inference.py
│       ├── eml_parser.py
│       └── database.py
│
└── tests/
```

Do not put all project logic into `app.py`.

The Streamlit layer should call reusable functions from `src/vietmailguard`.

The translation layer belongs to presentation/UI code and must not contaminate ML training logic.

---

## 36. Coding Rules

Prefer readable student-level production code.

Requirements:

* clear function names
* type hints for public functions
* concise docstrings
* `pathlib`
* reusable modules
* deterministic seeds
* no duplicated preprocessing logic
* no hardcoded absolute file paths
* no secrets in source code
* no hidden network dependency
* no unexplained magic constants
* UTF-8-safe source files

Use English for:

* Python variable names
* Python function names
* class names
* filenames
* internal enum values
* internal model labels

Use the translation layer for user-facing Vietnamese.

Avoid variable names such as:

```python
nguoi_gui
thu_rac
lua_dao
```

Prefer:

```python
sender
spam
phishing
```

and translate only for display.

Avoid unnecessary architecture.

Do not add Docker, Redis, Celery, React, Node.js, cloud databases, or microservices unless explicitly requested.

This is primarily a Machine Learning course project.

---

## 37. Notebook Policy

Notebooks may be used for exploration.

However:

* training logic must also exist in Python modules/scripts,
* inference must never depend on a notebook,
* final reproducibility must not require manually executing notebook cells.

---

## 38. Testing Rules

Use pytest.

At minimum test:

* text normalization
* URL extraction
* empty-email handling
* label mapping
* duplicate hashing/grouping
* risk-score bounds
* risk-level thresholds
* `.eml` parsing
* inference output schema
* English translation keys
* Vietnamese translation keys
* fallback behavior for missing translation key
* switching UI language does not change internal prediction
* internal class IDs remain `normal`, `spam`, `phishing`

Tests must not require internet access.

After meaningful changes run the smallest relevant test suite.

Before declaring a milestone complete run the full test suite.

---

## 39. Scientific Integrity Rules

Never:

* fabricate Accuracy, Precision, Recall, F1, or confusion-matrix values,
* use test data during feature fitting,
* leak duplicates across splits,
* call spam automatically phishing without a documented label policy,
* treat SVM decision score as probability,
* call risk score model probability,
* report training performance as test performance,
* hide removed/corrupt rows,
* silently change dataset labels,
* invent dataset provenance,
* claim Version 1 supports Vietnamese email classification merely because the UI supports Vietnamese.

If results are weaker than expected, report them accurately.

Correct methodology is more important than an artificially high score.

---

## 40. Documentation

README must be bilingual or provide clearly separated:

* English documentation
* Vietnamese documentation

At minimum README must include:

* project purpose
* supported classes
* current ML language support
* supported UI languages
* technology stack
* directory structure
* dataset setup
* environment setup
* standardization command
* training command
* evaluation command
* Streamlit command
* explanation of risk score
* known limitations
* future English/Vietnamese ML plan

Clearly state:

```text
Version 1 ML model:
English email detection

Version 1 user interface:
English + Vietnamese
```

Include Windows CMD examples because the project must be easy to run on Windows.

Example commands must follow the actual repository implementation.

---

## 41. Expected Workflow

Typical pipeline:

```text
Raw English datasets
        ↓
Dataset audit
        ↓
Standardization
        ↓
Cleaning
        ↓
Duplicate grouping
        ↓
English master dataset
        ↓
Leakage-safe splitting
        ↓
TF-IDF experiments
        ↓
Model training
        ↓
Validation comparison
        ↓
Final model selection
        ↓
Held-out test evaluation
        ↓
Saved production model
        ↓
Inference API
        ↓
English/Vietnamese Streamlit UI
        ↓
Dashboard
```

Do not build the visual web application before the ML pipeline can complete successfully.

However, translation architecture must already be considered when web development begins.

---

## 42. Verification Before Completion

Before claiming Version 1 is complete:

* confirm standardization runs from raw input
* confirm processed dataset is generated
* confirm no duplicate group crosses data splits
* confirm training script runs
* confirm all required models are compared
* confirm actual result files exist
* confirm production model loads
* confirm a sample normal email can be analyzed
* confirm a sample spam email can be analyzed
* confirm a sample phishing email can be analyzed
* confirm Streamlit launches
* confirm every required page supports English
* confirm every required page supports Vietnamese
* confirm changing UI language does not alter prediction results
* confirm model-language limitation notice appears correctly
* confirm pytest passes
* confirm README commands match the implementation

Report failed checks honestly.

---

## 43. Code Review Rules

Flag any change that:

* introduces train/test leakage,
* fits TF-IDF before splitting,
* removes provenance,
* silently changes labels,
* uses synthetic metrics,
* makes risk score equal confidence without justification,
* duplicates preprocessing between training and inference,
* uses external network calls for core inference,
* mixes raw and generated datasets,
* uses ambiguous dataset mapping without documenting it,
* duplicates the entire application to create a Vietnamese version,
* embeds Vietnamese display strings inside ML business logic,
* changes internal class IDs based on UI language,
* implies Vietnamese UI means Vietnamese ML support.

Prefer a smaller scientifically valid implementation over a feature-rich but unreliable one.

---

## 44. Definition of Done for Version 1

Version 1 is complete only when:

* English datasets can be standardized automatically.
* A leakage-safe English master dataset exists.
* Three ML classifiers have been trained and evaluated.
* Word and character TF-IDF have been tested.
* A production model has been selected from real validation results.
* Final held-out test results have been generated.
* URL/security analysis works.
* Explainability works.
* Risk score works.
* Streamlit web application works.
* User can switch between English and Vietnamese interface.
* All major pages are translated.
* Prediction results remain identical regardless of selected interface language.
* The interface clearly explains that Version 1 ML is trained on English email.
* Model and dataset dashboards use real data.
* Tests pass.
* README contains reproducible Windows setup and run commands.
* README clearly separates UI language support from ML language support.

# 45. Version 2 — Bilingual English/Vietnamese ML Upgrade

Version 2 extends the existing English-only Machine Learning system into a bilingual English/Vietnamese email detection system.

Do not rewrite Version 1 from scratch.

Preserve:

* existing dataset audit pipeline
* existing English master dataset
* leakage-safe splitting logic
* existing model comparison framework
* URL analyzer
* security analyzer
* explainability
* risk engine
* Streamlit bilingual interface
* existing Version 1 results and artifacts

Version 2 must be implemented as an extension of the validated Version 1 architecture.

---

# 46. Language Support Model

The application has two independent language concepts:

## UI Language

Possible values:

* `en`
* `vi`

Controls interface text only.

## Email Content Language

Possible values:

* `en`
* `vi`
* `mixed`
* `unknown`

Controls model-support status and evaluation grouping.

Never infer Machine Learning language support from UI language.

---

# 47. Vietnamese Dataset Origin

Every Vietnamese dataset row must contain:

`data_origin`

Allowed values:

* `native`
* `translated`
* `synthetic`

Definitions:

### native

Originally written as Vietnamese email content.

### translated

Originally from another language and translated into Vietnamese.

### synthetic

Artificially generated email content.

Do not merge these categories without preserving their origin.

Do not present translated or synthetic Vietnamese data as native Vietnamese data.

---

# 48. Current Vietnamese Dataset Policy

The current `data_vi.csv` dataset must be treated as:

`language = vi`

`data_origin = translated`

unless stronger provenance later proves that it is native Vietnamese.

It may be used for:

* bilingual training
* augmentation
* feature experiments
* robustness experiments

It must NOT be used as the sole Vietnamese final benchmark.

Do not claim:

`Vietnamese real-world accuracy`

based only on translated Vietnamese data.

---

# 49. Vietnamese Dataset Standard Schema

Vietnamese datasets must use the same standard schema as English datasets.

Required common fields include:

* `id`
* `sender`
* `receiver`
* `date`
* `subject`
* `body`
* `label`
* `raw_label`
* `source`
* `language`
* `data_origin`
* `url_count`
* `content_hash`
* `group_id`

If sender, receiver, subject, or date are unavailable:

leave them empty.

Never invent missing email metadata.

---

# 50. Vietnamese Text Processing

All Vietnamese text processing must preserve semantic information.

Required:

* UTF-8
* Unicode NFC normalization
* whitespace normalization
* safe HTML cleanup
* URL extraction before destructive cleaning

Do not remove Vietnamese diacritics from the primary training text.

Do not convert all Vietnamese text into unaccented Vietnamese.

Do not aggressively remove Vietnamese stopwords by default.

Character-level features should be used to improve robustness to:

* Vietnamese without accents
* spelling variations
* intentional character substitutions
* punctuation obfuscation
* mixed English/Vietnamese content

---

# 51. Translated Data Leakage

Cross-language semantic leakage must be considered.

Example risk:

English original:

`Your account has been suspended`

appears in English training data.

Vietnamese translation:

`Tài khoản của bạn đã bị khóa`

must not automatically be considered an independent Vietnamese test sample.

Translated versions of an English source must remain linked to the original source whenever possible.

Add fields when provenance permits:

* `parent_id`
* `translation_source_id`

Samples derived from the same original email must remain in the same split.

Never place:

English original → Train

Vietnamese translation → Test

when they originate from the same email.

---

# 52. Cross-Language Overlap Detection

Before creating bilingual splits:

attempt to detect overlap between English and translated Vietnamese corpora.

Use available provenance first.

Possible supporting methods:

* source identifiers
* subject similarity
* structural patterns
* preserved names/entities
* deterministic hashes before translation when available
* multilingual semantic similarity as an audit aid

Semantic similarity must only assist leakage detection.

Do not silently delete data based only on an uncertain similarity score.

Flag uncertain cases for review.

---

# 53. Bilingual Training Strategy

Version 2 must compare multiple training strategies.

Required baseline:

## Model A — English-only

Existing Version 1 model.

## Model B — English + translated Vietnamese

Use:

* Word TF-IDF
* Character TF-IDF
* Logistic Regression
* Linear SVM
* Naive Bayes where technically compatible

TF-IDF vocabulary must be refit using bilingual training data.

Do not reuse the English-only fitted vocabulary as if it already supports Vietnamese.

## Model C — Multilingual Representation Experiment

Add one optional but recommended cross-lingual experiment using multilingual sentence/text embeddings.

Possible architecture:

Email text
→ multilingual embedding
→ Logistic Regression or Linear SVM
→ normal / spam / phishing

Use a pretrained multilingual sentence embedding model compatible with English and Vietnamese.

Do not fine-tune a large Transformer model unless explicitly required.

The multilingual embedding experiment must remain separable from the classical TF-IDF baseline.

---

# 54. Model Comparison for Version 2

Compare at least:

* English-only TF-IDF baseline
* bilingual Word TF-IDF
* bilingual Character TF-IDF
* bilingual Word + Character TF-IDF
* multilingual embedding + Logistic Regression
* multilingual embedding + Linear SVM, if practical

Do not assume multilingual embeddings are automatically better.

Select models based on measured results.

---

# 55. Evaluation by Language and Origin

Never report only one overall bilingual Accuracy.

At minimum report:

## English Test

* Accuracy
* Macro F1
* phishing recall
* spam recall

## Vietnamese Translated Test

Clearly label as:

`Translated Vietnamese Test`

Do not call it native Vietnamese test.

## Vietnamese Native Test

Only when actual native Vietnamese benchmark data exists.

## Combined Test

English + Vietnamese.

Also report performance grouped by:

`language`

and when possible:

`data_origin`

---

# 56. Native Vietnamese Benchmark

A small native Vietnamese test set is preferred over a large translated test set.

Suggested target:

300–500 native Vietnamese emails if feasible.

Possible approximate distribution:

* normal
* spam
* phishing/scam

Do not require exact balance if real data availability prevents it.

Native benchmark data must remain completely outside training.

Never translate English training samples into Vietnamese and include those translations in the native benchmark.

---

# 57. Version 2 Claim Policy

Do not claim:

`VietMailGuard supports Vietnamese email detection`

until the bilingual model has been evaluated on Vietnamese data.

Prefer staged wording:

### Before bilingual training

`Vietnamese UI supported; Vietnamese ML classification experimental.`

### After translated Vietnamese training only

`Vietnamese classification supported experimentally using bilingual training with translated Vietnamese data.`

### After native Vietnamese evaluation

`Vietnamese classification evaluated on an independent native Vietnamese benchmark.`

Always state which stage applies.

---

# 58. Language Detection

Add lightweight email-language detection.

Possible internal values:

* `en`
* `vi`
* `mixed`
* `unknown`

Language detection must be separate from classification.

Do not use detected language as a direct proxy for:

* spam
* phishing
* normal

Language detection is used for:

* support-status messaging
* routing evaluation
* analytics
* UI notices

If confidence in language detection is low:

return `unknown`.

---

# 59. Model Support Metadata

Store production-model metadata.

Recommended file:

`models/model_metadata.json`

Include:

* model version
* training languages
* training dataset sources
* training data origins
* feature representation
* classifier
* training timestamp
* supported classes
* evaluation summary
* supported content languages

The web application should read this metadata.

Do not hardcode:

`English-only`

or:

`Bilingual`

inside UI business logic.

---

# 60. Versioned Model Artifacts

Never overwrite validated Version 1 artifacts.

Use versioned directories such as:

`models/v1_english/`

`models/v2_bilingual/`

Likewise use:

`results/v1_english/`

`results/v2_bilingual/`

Preserve previous evaluation results.

---

# 61. Version 2 Risk Engine

The risk engine must support English and Vietnamese signals separately.

Add Vietnamese security-language indicators such as:

* khẩn cấp
* xác minh
* xác nhận tài khoản
* tài khoản bị khóa
* đăng nhập
* mật khẩu
* cập nhật thông tin
* trúng thưởng
* nhận thưởng
* chuyển khoản
* thông tin ngân hàng
* thanh toán
* trong vòng 24 giờ

These rules supplement Machine Learning.

They must not replace the classifier.

Do not use one keyword alone to determine phishing.

Keep English and Vietnamese rule lists configurable.

---

# 62. Vietnamese Explainability

When a Vietnamese email is analyzed:

* preserve original Vietnamese text
* display Vietnamese feature terms as they actually appear
* do not translate model-important terms back into English
* translate only UI explanation labels

Example:

Internal feature:

`xác minh tài khoản`

Vietnamese UI:

`Cụm từ có ảnh hưởng đến dự đoán: "xác minh tài khoản"`

English UI may display:

`Prediction-influencing phrase: "xác minh tài khoản"`

Do not alter the actual feature text.

---

# 63. Vietnamese Robustness Evaluation

Create a dedicated robustness set.

Required transformations:

### Original

`Tài khoản của bạn đã bị khóa.`

### Without accents

`Tai khoan cua ban da bi khoa.`

### Character substitution

`T@i kh0an cua ban da bi kh0a.`

### Spacing obfuscation

`T à i  k h o ả n của bạn đã bị khóa.`

### Mixed language

`Your account cần được xác minh ngay.`

Keep robustness evaluation separate from normal test metrics.

Report degradation relative to clean Vietnamese text.

---

# 64. Translation as Inference Fallback

Do NOT make online translation a required component of core inference.

Do not send potentially sensitive email content to third-party translation APIs by default.

If translation-based fallback is ever implemented:

* make it optional
* make privacy implications explicit
* do not treat it as the main bilingual solution
* keep it outside baseline scientific evaluation

The preferred Version 2 approach is a bilingual model, not runtime translation.

---

# 65. Bilingual Web Behavior

When Version 2 production model is active:

English email:

`Supported`

Vietnamese email:

`Supported`

Mixed English/Vietnamese:

`Experimental` unless properly evaluated.

Unknown language:

show a clear support warning.

The selected UI language must still remain independent.

Examples:

Vietnamese UI + English email → valid

English UI + Vietnamese email → valid

Vietnamese UI + Vietnamese email → valid

---

# 66. Version 2 Definition of Done

Version 2 is complete only when:

* Vietnamese dataset has been audited
* translated/native origin is preserved
* cross-language leakage has been checked
* Vietnamese data has been standardized
* bilingual master dataset exists
* bilingual leakage-safe splits exist
* TF-IDF vocabulary has been refit
* bilingual classifiers have been trained
* multilingual embedding experiment has been evaluated
* English performance has been measured
* Vietnamese translated performance has been measured
* native Vietnamese performance is reported if native benchmark exists
* Version 1 model/results remain preserved
* language detection works
* Vietnamese security rules work
* Streamlit reads model metadata
* UI correctly reports support status
* robustness tests exist
* README documents the bilingual methodology and limitations

Do not claim Version 2 completion if Vietnamese evaluation is missing.

# 67. Version 2 Final Evaluation Contract

The Version 2 held-out test set is locked.

During model development:

- use only train data for fitting,
- use only validation data for model/hyperparameter selection,
- do not repeatedly evaluate the held-out test set,
- do not inspect individual test errors to redesign the model before final selection.

The final test set may be evaluated only after the Version 2 production candidate
has been selected using validation results.

Translated augmentation derived from an English parent must inherit the same
split/group assignment as that parent.

Never place:
English parent -> train
Vietnamese translation -> validation/test

or:
English parent -> validation
Vietnamese translation -> train

All descendants of one source email belong to one leakage group.

Performance must be reported separately for:
- English,
- translated Vietnamese,
- combined bilingual data.

Do not call translated-Vietnamese performance "native Vietnamese performance".

If no independent native Vietnamese benchmark exists, state this explicitly.

Model selection priority for Version 2:

1. Macro F1
2. phishing recall
3. performance balance across languages
4. false-positive behavior on normal email
5. inference cost / deployability

A model with slightly higher overall Accuracy must not automatically win if it
has materially worse phishing recall or severe language imbalance.

# 68. VietMailGuard Mail Client

VietMailGuard Version 2 production classifier is frozen.

The next application layer turns VietMailGuard from a copy/paste analyzer into
a local email-security client.

This work MUST NOT retrain or modify the frozen Version 2 classifier.

The mailbox layer sits above:

Frozen ML Model
→ Security Analysis
→ Risk Engine
→ Mail Routing
→ SQLite
→ Streamlit Mail Client

---

# 69. Product Identity

The application is called:

`VietMailGuard Mail`

It may use a familiar three-pane webmail layout inspired by modern email clients,
but must not copy Gmail branding, logos, trademarks, or claim to be Gmail.

Preferred product description:

`AI-Powered Bilingual Email Security Client`

---

# 70. Mailbox Scope

Version 2 Mail Client initially operates locally.

Supported input sources:

1. seeded demonstration emails
2. imported `.eml` files
3. existing VietMailGuard analyzer input where useful

Do NOT integrate Gmail API, OAuth, IMAP, or real external mail accounts in this
phase unless explicitly requested later.

The local mailbox must be stable before external-account integration.

---

# 71. SQLite Naming Convention

ALL SQLite table names and column names must use Vietnamese without diacritics.

Use:

- lowercase
- snake_case
- ASCII only
- meaningful Vietnamese names

Examples:

GOOD:

`thu`
`phan_tich_thu`
`phan_hoi_nguoi_dung`
`lich_su_thu_muc`

`id_thu`
`nguoi_gui`
`nguoi_nhan`
`tieu_de`
`noi_dung`
`ngay_gui`
`thu_muc`
`da_doc`
`da_gan_sao`
`diem_rui_ro`

BAD:

`emails`
`sender`
`subject`
`risk_score`

BAD:

`người_gửi`
`tiêu_đề`

Vietnamese diacritics must not appear in SQL identifiers.

Python module/class/function names may remain English when that matches the
existing codebase.

---

# 72. SQLite Database Location

Runtime database:

`data/runtime/vietmailguard_mail.db`

The runtime database must NOT be committed to Git.

Add:

`data/runtime/*`

to `.gitignore`, while preserving a `.gitkeep` if useful.

Database creation must be reproducible from migration/schema code.

---

# 73. Core SQLite Schema

The initial schema should contain at least the following tables.

## Table: `thu`

Recommended fields:

- `id_thu`
- `ma_thu_ngoai`
- `nguoi_gui`
- `nguoi_nhan`
- `cc`
- `tieu_de`
- `noi_dung`
- `noi_dung_html`
- `ngay_gui`
- `thu_muc`
- `da_doc`
- `da_gan_sao`
- `nguon`
- `ma_nguon`
- `thoi_gian_tao`
- `thoi_gian_cap_nhat`

Primary key:

`id_thu`

Allowed internal folder values:

- `hop_thu_den`
- `thu_rac`
- `cach_ly`
- `da_xoa`

Do not use localized display strings as database state.

---

## Table: `phan_tich_thu`

Recommended fields:

- `id_phan_tich`
- `id_thu`
- `phien_ban_mo_hinh`
- `ngon_ngu_phat_hien`
- `nhan_du_doan`
- `do_tin_cay`
- `diem_rui_ro`
- `muc_rui_ro`
- `hanh_dong_goc`
- `hanh_dong_de_xuat`
- `co_canh_bao`
- `so_phat_hien_bao_mat`
- `ket_qua_json`
- `thoi_gian_phan_tich`

`nhan_du_doan` must preserve the frozen ML internal classes:

- `normal`
- `spam`
- `phishing`

Do NOT translate these database values.

---

## Table: `phan_hoi_nguoi_dung`

Recommended fields:

- `id_phan_hoi`
- `id_thu`
- `nhan_du_doan_ban_dau`
- `thu_muc_truoc`
- `thu_muc_sau`
- `hanh_dong_nguoi_dung`
- `ghi_chu`
- `thoi_gian_phan_hoi`

Examples of user actions:

- `khong_phai_thu_rac`
- `danh_dau_thu_rac`
- `bao_cao_lua_dao`
- `chuyen_vao_hop_thu_den`
- `chuyen_vao_cach_ly`
- `xoa`

User feedback must NOT automatically retrain the production model.

---

## Table: `lich_su_thu_muc`

Recommended fields:

- `id_lich_su`
- `id_thu`
- `thu_muc_cu`
- `thu_muc_moi`
- `ly_do`
- `thoi_gian_thay_doi`

This provides an auditable routing history.

---

# 74. SQLite Integrity

Enable:

`PRAGMA foreign_keys = ON`

Use transactions for:

- importing email
- analysis
- routing
- moving messages

Add appropriate indexes, including where useful:

- `thu(thu_muc, ngay_gui)`
- `thu(da_doc)`
- `thu(da_gan_sao)`
- `phan_tich_thu(id_thu)`
- `phan_tich_thu(diem_rui_ro)`
- `phan_tich_thu(nhan_du_doan)`

Avoid unnecessary denormalization.

---

# 75. Automatic Mail Routing

Automatic routing is based on the frozen ML prediction.

Default routing:

`normal`
→ `hop_thu_den`

`spam`
→ `thu_rac`

`phishing`
→ `cach_ly`

Do NOT automatically delete phishing or spam.

---

# 76. Security Disagreement Policy

Security rules must never silently rewrite the ML class.

Example:

ML prediction:

`normal`

Security engine:

promotional indicators detected

Result:

- `nhan_du_doan = normal`
- `thu_muc = hop_thu_den`
- `co_canh_bao = true`
- recommendation may be `REVIEW`

The UI must display the disagreement.

---

# 77. Phishing User Experience

A phishing prediction must:

1. route the email to `cach_ly`
2. display a prominent warning banner
3. show Risk Score
4. show relevant security indicators
5. show suspicious URL findings
6. recommend avoiding links and credential entry

Never claim a URL is definitively malicious based only on offline heuristics.

Use wording such as:

`suspicious indicator`

---

# 78. Spam User Experience

Spam prediction:

- automatically routes to `thu_rac`
- remains recoverable
- supports `Khong phai thu rac`
- keeps original ML prediction for audit history

Moving the message back to Inbox must not rewrite historical model output.

---

# 79. Mail Client Layout

Preferred Streamlit layout:

Left sidebar:
- Hop thu den
- Gan sao
- Thu rac
- Cach ly
- Da xoa
- Security View

Center:
- message list

Right:
- selected email content
- VietMailGuard analysis

Do not reproduce Gmail branding exactly.

---

# 80. Mail List

Each mail-list item should show when practical:

- sender
- subject
- preview
- date/time
- unread/read status
- star status
- prediction indicator
- risk indicator

Avoid overwhelming the list with every security detail.

---

# 81. Mail Detail

The selected email should show:

- sender
- recipient
- date
- subject
- body

Then a clearly separated:

`VietMailGuard Security Analysis`

showing:

- prediction
- calibrated confidence
- Risk Score
- Risk Level
- recommended action
- model explanation
- content findings
- sender findings
- URL findings
- support/limitation notices

---

# 82. HTML Email Safety

Never render arbitrary email HTML directly with unrestricted
`unsafe_allow_html=True`.

Email HTML may contain unsafe or tracking content.

Preferred behavior:

- display sanitized/plain-text body
- extract links separately
- do not execute scripts
- do not load remote images automatically
- do not execute embedded content

Remote images should remain disabled in the first mailbox version.

---

# 83. Attachments

The current VietMailGuard system is NOT a malware attachment scanner.

If attachments are shown:

- display metadata only
- filename
- MIME type
- size where available

Do not claim attachments were scanned for malware.

Do not automatically execute/open attachments.

---

# 84. Importing `.eml`

Imported `.eml` files must use the existing parser where possible.

Import flow:

`.eml`
→ parse
→ store email
→ frozen inference
→ security/risk analysis
→ route
→ save analysis
→ show in mailbox

Duplicate imports should be detected when practical.

Do not create duplicate mailbox rows every time the same `.eml` is imported.

---

# 85. Demo Seed Emails

Provide a reproducible seed dataset for UI demonstration.

Seed examples should cover:

- normal English
- spam English
- phishing English
- normal Vietnamese
- spam-like Vietnamese
- phishing-like Vietnamese
- mixed language
- short promotional email

These are DEMO emails.

They are not evaluation data.

Do not report seed behavior as scientific metrics.

---

# 86. Mail Service Layer

Streamlit must NOT execute SQL directly throughout UI pages.

Create a service/repository layer such as:

`mail_repository.py`
`mail_service.py`
`mail_router.py`

Preferred architecture:

Streamlit
→ MailService
→ Repository / Inference
→ SQLite

---

# 87. Inference Reuse

There must be only one production inference implementation.

Reuse:

`vietmailguard.inference`

Do not duplicate:

- TF-IDF
- classifier logic
- URL analyzer
- Risk Engine
- language detector

in mailbox code.

---

# 88. Analysis Persistence

Store the analysis output used when the email entered the mailbox.

This allows historical review even if future model versions change.

Store:

- model version
- prediction
- confidence
- risk
- language
- recommendation
- security result JSON
- analysis timestamp

Do not silently recompute old mail with a new model and overwrite history.

---

# 89. Reanalysis Policy

If manual reanalysis is added later:

create a new analysis record or preserve previous history.

Never destroy historical predictions silently.

---

# 90. User Feedback

Actions such as:

- Not Spam
- Mark as Spam
- Report Phishing
- Move to Inbox

affect mailbox state.

They do NOT change frozen model weights.

Store feedback for future Version 2.1 research.

---

# 91. Search and Filtering

Mailbox search should support at minimum:

- sender
- subject
- body text where practical

Filters may include:

- folder
- unread
- starred
- prediction
- risk level

Use parameterized SQL.

Never build SQL by concatenating untrusted user input.

---

# 92. Security View

Add a dedicated `Security View`.

It may sort/filter mail by:

- CRITICAL
- HIGH
- MEDIUM
- LOW

Show:

- email
- ML prediction
- confidence
- risk score
- number of findings
- folder

This is a view of stored analysis.

It must not create new scientific metrics.

---

# 93. Existing Quick Analyzer

Do not delete the existing Email Analyzer.

It may remain as:

`Quick Analyzer`

or an advanced/debug page.

The Mail Client becomes the main user-facing workflow.

---

# 94. No External Gmail Yet

Do not implement Gmail OAuth/API in this phase.

Future integration may be:

Gmail API
→ fetch
→ local VietMailGuard analysis
→ labels/actions

But it requires a separate privacy and permission design.

---

# 95. Mail Client Definition of Done

The local mail client is complete only when:

- SQLite schema exists
- migrations/schema initialization works
- Vietnamese-no-diacritic SQL naming policy is followed
- seeded demo mailbox works
- `.eml` import works
- emails automatically route to Inbox/Spam/Quarantine
- phishing warning works
- spam recovery works
- model/security disagreement is visible
- mailbox state persists after restart
- user feedback persists
- security view works
- existing production inference remains frozen
- old Quick Analyzer still works
- tests pass


luôn luôn trả lời bằng tiếng Việt