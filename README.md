# VietMailGuard

## English

VietMailGuard Version 2 is a reproducible three-class email security project. Its frozen production model analyzes English and Vietnamese content and returns one internal label: `normal`, `spam`, or `phishing`.

Current claim boundary:

- English content: supported and evaluated on an independent held-out English subset.
- Vietnamese content: supported experimentally; training and evaluation data is predominantly translated.
- Mixed English/Vietnamese: experimental and not independently benchmarked.
- Native Vietnamese benchmark: not available.
- The translated Vietnamese held-out subset has 179 rows (171 normal, 8 spam, 0 phishing). It cannot establish Vietnamese phishing recall or native Vietnamese accuracy.

The English/Vietnamese interface language changes presentation only. It never changes the model version, prediction, confidence, risk score, or internal class identifiers.

### Production architecture

```text
Sender + Subject + Body / optional .eml
                  |
                  v
Shared vietmailguard.inference API
      |                         |
      v                         v
Frozen Word TF-IDF        Offline security analysis
+ calibrated LinearSVC    content + URL + sender rules
      |                         |
      v                         v
ML prediction/confidence  separate 0-100 risk score
                  |
                  v
English/Vietnamese Streamlit UI
```

The web application never calls the classifier or recreates TF-IDF directly. It loads the complete frozen pipeline from `models/v2_bilingual/production_pipeline.joblib` through `src/vietmailguard/inference.py`.

Production configuration:

- Word TF-IDF `(1, 2)`, `max_features=60000`.
- LinearSVC `C=1.0`, `class_weight=balanced`.
- Sigmoid `CalibratedClassifierCV` using five-fold `StratifiedGroupKFold` on `final_group_id`.
- Model Confidence comes from calibrated `predict_proba`; raw SVM decision scores are not shown as probability.
- Overall Risk Score is a separate configurable decision-support heuristic. Security rules never overwrite the ML prediction.

### Scientific data pipeline

```text
English + translated Vietnamese raw datasets
        -> dataset and label audit
        -> English/Vietnamese standardization
        -> label curation
        -> cross-language parent/overlap audit
        -> controlled Vietnamese translation augmentation
        -> exact/normalized/template/translation grouping
        -> leakage-safe bilingual train/validation/test split
        -> bilingual TF-IDF experiments
        -> multilingual embedding experiments
        -> validation-only model selection
        -> immutable configuration freeze
        -> one-shot held-out test evaluation
        -> inference + security rules + risk engine
        -> separate robustness/challenge evaluation
        -> Streamlit web application
```

Translated descendants keep `parent_id`, `translation_source_id`, provenance, origin, and group membership. English parents and all known translations/derivatives remain in the same `final_group_id` and split.

### Dataset schema and composition

The bilingual master schema includes at least:

`id`, `parent_id`, `translation_source_id`, `source`, `parent_source`, `language`, `data_origin`, `augmentation_type`, `subject`, `body`, `label`, `raw_label`, `label_status`, `label_provenance`, `content_hash`, `group_id`, `translation_group_id`, `final_group_id`, `training_eligible`.

Unavailable source metadata remains empty; it is never invented. Review and excluded rows do not enter model splits.

Current eligible bilingual master:

| Dimension | Count |
|---|---:|
| Total | 44,717 |
| English | 42,112 |
| Translated Vietnamese | 2,605 |
| Normal | 40,861 |
| Spam | 1,817 |
| Phishing | 2,039 |
| Train | 31,304 |
| Validation | 6,709 |
| Test | 6,704 |

### Frozen held-out results

These values are stored in `results/v2_bilingual/final_test_metrics.json`; the dashboard reads that artifact rather than embedding values in UI code.

| Metric | Value |
|---|---:|
| Accuracy | 0.997763 |
| Macro Precision | 0.997992 |
| Macro Recall | 0.983992 |
| Macro F1 | 0.990900 |
| Weighted F1 | 0.997750 |
| Phishing recall | 0.970588 |

The held-out test was evaluated once after validation selection. It was not used for feature fitting, model/hyperparameter selection, calibration fitting, or risk-threshold tuning. Robustness and short-form challenge results are separate and are not part of these held-out metrics.

### Windows CMD reproducibility workflow

Run commands from the repository root. Raw files under `data\raw\` are immutable.

1. Environment and installation:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

2. Place the English source files and `data_vi.csv` in `data\raw\`, then audit:

```bat
python scripts\audit_datasets.py
python scripts\audit_vietnamese_dataset.py
```

3. Standardize English and translated Vietnamese data:

```bat
python scripts\standardize_datasets.py
python scripts\standardize_vietnamese_dataset.py
```

4. Audit cross-language overlap and curate labels:

```bat
python scripts\audit_cross_language_overlap.py
python scripts\build_label_review_pools.py
python scripts\build_spam_curation_candidates.py
python scripts\curate_vietnamese_labels.py
```

Optional resumable human spam review:

```bat
python scripts\review_spam_candidates.py --reviewer reviewer_name
```

5. Build controlled translation augmentation. `--offline` requires the pinned translation model to already exist in the local cache and prevents email content from being sent to an online API:

```bat
python scripts\build_translated_augmentation.py --offline
```

6. Build the bilingual master and leakage-safe V2 splits:

```bat
python scripts\build_bilingual_dataset.py
```

7. Historical experiment reproduction in a fresh research workspace only—these commands fit models on train and compare on validation, never on test:

```bat
python scripts\train_v2_tfidf_models.py
python scripts\train_v2_multilingual_embeddings.py --device auto
```

The comparison artifacts are `results\v2_bilingual\tfidf_model_comparison.csv` and `results\v2_bilingual\multilingual_embedding_validation.csv`. Selection rationale is frozen in `reports\v2\model_selection_rationale.md` and `config\v2_production_frozen.json`.

8. One-shot finalization—historical command, **do not rerun in the completed repository**:

```bat
python scripts\finalize_v2_model.py
```

The existing result file is the one-shot guard. Do not delete it or bypass the guard to retest/tune Version 2.

9. Run frozen inference and separate post-freeze robustness evaluation:

```bat
python scripts\smoke_inference_v2.py
python scripts\evaluate_v2_robustness.py
```

The robustness command does not train or modify the production model. Its metrics stay under separate robustness/challenge artifacts.

10. Launch the web application and run tests:

```bat
python -m streamlit run app\app.py
python -m pytest -q
```

If the project directory was moved after `.venv` was created, use the environment interpreter directly:

```bat
.venv\Scripts\python.exe -m streamlit run app\app.py
```

### Limitations and future work

- No independent native Vietnamese benchmark exists.
- The translated Vietnamese test subset has no phishing examples.
- Short promotional emails can be classified as normal; security indicators may separately recommend `REVIEW` without relabeling the ML prediction.
- Vietnamese phishing examples in training are translated, not native ground truth.
- The phishing class remains source-concentrated, creating possible source/class confounding.
- Offline rules do not provide live domain reputation or confirm a URL as malicious.
- Version 2.1 needs a new, untouched native Vietnamese benchmark and a separately governed protocol for short, obfuscated, and mixed-language email. The frozen Version 2 test must not be reused for tuning.

## Tiếng Việt

VietMailGuard Version 2 là hệ thống phân loại email ba lớp có khả năng tái lập: `normal`, `spam`, `phishing`.

Phạm vi tuyên bố hiện tại:

- Email tiếng Anh: đã được hỗ trợ và đánh giá trên held-out subset độc lập.
- Email tiếng Việt: hỗ trợ ở mức thử nghiệm; dữ liệu training và evaluation chủ yếu là dữ liệu dịch.
- Email pha trộn Anh/Việt: thử nghiệm, chưa có benchmark độc lập.
- Chưa có benchmark email tiếng Việt bản địa.
- Held-out subset tiếng Việt được dịch có 179 mẫu: 171 normal, 8 spam, 0 phishing. Vì vậy không thể báo cáo phishing recall tiếng Việt hoặc accuracy tiếng Việt bản địa.

Ngôn ngữ giao diện độc lập với ngôn ngữ email. Đổi giao diện Anh/Việt chỉ thay đổi phần trình bày, không thay đổi model version, prediction, confidence, risk score hoặc nhãn nội bộ.

### Kiến trúc và phương pháp khoa học

Luồng hoàn chỉnh là:

```text
Dataset tiếng Anh + tiếng Việt được dịch
-> audit dữ liệu và nhãn
-> chuẩn hóa
-> curation
-> phát hiện overlap Anh/Việt
-> augmentation dịch có kiểm soát
-> nhóm exact/normalized/template/parent-translation
-> split song ngữ chống leakage
-> thực nghiệm TF-IDF
-> thực nghiệm multilingual embedding
-> chọn model chỉ bằng validation
-> đóng băng cấu hình
-> đánh giá held-out test đúng một lần
-> inference + security + risk
-> robustness/challenge riêng
-> Streamlit song ngữ
```

Production dùng Word TF-IDF `(1,2)` và LinearSVC đã sigmoid-calibrate bằng `CalibratedClassifierCV` với `StratifiedGroupKFold` 5 fold theo `final_group_id`. `Model Confidence` lấy từ `predict_proba`. `Overall Risk Score` là heuristic hỗ trợ quyết định độc lập 0–100; đây không phải xác suất mô hình. Rule bảo mật không âm thầm đổi prediction của classifier.

### Chạy trên Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts\audit_datasets.py
python scripts\audit_vietnamese_dataset.py
python scripts\standardize_datasets.py
python scripts\standardize_vietnamese_dataset.py
python scripts\audit_cross_language_overlap.py
python scripts\build_label_review_pools.py
python scripts\build_spam_curation_candidates.py
python scripts\curate_vietnamese_labels.py
python scripts\build_translated_augmentation.py --offline
python scripts\build_bilingual_dataset.py
```

Hai lệnh experiment sau chỉ dùng để tái lập nghiên cứu trong workspace sạch; không chạy để thay đổi production V2 đã đóng băng:

```bat
python scripts\train_v2_tfidf_models.py
python scripts\train_v2_multilingual_embeddings.py --device auto
```

Lệnh finalization lịch sử sau là bước one-shot và **không được chạy lại trong repository đã hoàn tất**:

```bat
python scripts\finalize_v2_model.py
```

Kiểm tra inference, robustness, web và test:

```bat
python scripts\smoke_inference_v2.py
python scripts\evaluate_v2_robustness.py
python -m streamlit run app\app.py
python -m pytest -q
```

### Giới hạn

- Chưa có benchmark tiếng Việt bản địa độc lập.
- Held-out tiếng Việt được dịch không có phishing.
- Email quảng cáo ngắn có thể bị ML dự đoán `normal`; lớp security có thể đề xuất `REVIEW` nhưng không thay nhãn ML.
- Kết quả challenge/robustness tách biệt hoàn toàn với held-out metrics.
- Version 2.1 cần test set tiếng Việt bản địa mới và protocol mới; không dùng lại test Version 2 để tuning.
