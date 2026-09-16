# VietMailGuard

## Project documentation

- [VietMailGuard V2 Model Card](MODEL_CARD.md)
- [Dataset provenance and licensing](DATASETS.md)
- [Tested environment and reproducibility](ENVIRONMENT.md)

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

### VietMailGuard Mail local client

VietMailGuard Mail is the primary local mailbox workflow. It reuses the frozen
inference layer and keeps SQL outside Streamlit:

```text
Email / .eml import
        -> frozen ML + security/risk analysis
        -> prediction-based routing
        -> SQLite persistence
        -> Inbox / Spam / Quarantine mailbox
```

The concrete software boundary is:

```text
Streamlit -> MailService -> MailRepository / frozen inference -> SQLite
```

Folders have distinct, reversible meanings:

- **Inbox** (`hop_thu_den`): messages predicted `normal`. A security disagreement can add a warning or `REVIEW` recommendation without silently changing the ML class or folder.
- **Spam** (`thu_rac`): messages predicted `spam`. Users can use **Not spam** to restore a message while preserving its original analysis.
- **Quarantine** (`cach_ly`): messages predicted `phishing`. Quarantine is protective isolation, **not deletion**; messages remain available for review and recovery.
- **Deleted** (`da_xoa`): a user-controlled mailbox state. VietMailGuard never automatically deletes spam or phishing messages.

Search covers sender, subject, and plain body text. Filters cover folder,
read/star state, prediction, risk level, and detected language. Security View
uses persisted analysis and sorts by risk; rendering a mailbox list never runs
inference again. Imported HTML is converted to plain text, remote images are not
loaded, links are not opened automatically, and attachments are not claimed as
malware-scanned.

The mailbox now uses two full-width navigation modes rather than a permanent
split pane:

```text
Mailbox list -> select a message -> full-width reading view
             <- Back returns to the originating folder/view
```

Inbox, Spam, Quarantine, Deleted, Starred, and Security View all open the same
stored-message detail renderer. Detail lookup is by email id through
`MailService`, so active search/filter results do not control whether an email
can be read. Spam and Quarantine content remains readable; warnings and security
analysis use progressive disclosure below the plain-text message body. Moving or
deleting a message returns to the originating list, while read/star changes stay
on the open message.

The reproducible seed contains demonstration messages only. They are not an
evaluation dataset and their predictions are not scientific metrics:

```bat
python scripts\seed_demo_mailbox.py
python -m streamlit run app\app.py
```

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
python run.py
```

The launcher above is equivalent to the direct Streamlit command and safely
handles project paths containing spaces or Unicode characters. Optional
Streamlit arguments can be appended, for example `python run.py --server.port 8502`.

Direct command and tests:

```bat
python -m streamlit run app\app.py
python -m pytest -q
```

If the project directory was moved after `.venv` was created, use the environment interpreter directly:

```bat
.venv\Scripts\python.exe -m streamlit run app\app.py
```

### Continuous integration

The normal workflow at `.github/workflows/ci.yml` runs on push and pull requests with Python 3.11. It installs only the core requirements and runs:

```bat
python -m pytest -m "not production_artifact" -q
```

This lane does not download the production model, raw datasets, translation models or embedding models. Tests for optional local generated datasets skip with an explicit reason when those artifacts are absent.

The manually dispatched `.github/workflows/production-integration.yml` is the separate release-artifact lane. Before running it, configure the repository Actions variable `V2_PRODUCTION_MODEL_URL` with the official HTTPS URL of the `production_pipeline.joblib` GitHub Release asset. The workflow fails clearly when the variable is absent. It downloads through `scripts/download_production_model.py`, verifies SHA-256 against `model_metadata.json`, then runs:

```bat
python -m pytest -m "production_artifact" -q
python scripts\smoke_inference_v2.py
```

No placeholder or unofficial model URL is embedded in the repository.

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

### VietMailGuard Mail — hộp thư cục bộ

VietMailGuard Mail là luồng sử dụng chính ở máy cá nhân. Streamlit không chạy
SQL hay gọi classifier trực tiếp:

```text
Email / file .eml
-> ML đã đóng băng + phân tích security/risk
-> định tuyến theo prediction
-> lưu SQLite
-> Hộp thư đến / Thư rác / Cách ly
```

Ranh giới phần mềm là:

```text
Streamlit -> MailService -> MailRepository / frozen inference -> SQLite
```

- **Hộp thư đến** (`hop_thu_den`) nhận email được ML dự đoán `normal`. Dấu hiệu bảo mật vẫn có thể tạo cảnh báo hoặc khuyến nghị `REVIEW` nhưng không âm thầm đổi prediction.
- **Thư rác** (`thu_rac`) nhận email được dự đoán `spam`. Người dùng có thể chọn **Không phải thư rác** mà không sửa lịch sử ML.
- **Cách ly** (`cach_ly`) nhận email được dự đoán `phishing`. Cách ly **không phải xóa**; email vẫn được giữ để kiểm tra hoặc khôi phục.
- **Đã xóa** (`da_xoa`) chỉ thay đổi bởi thao tác người dùng. Hệ thống không tự động xóa spam hoặc phishing.

Tìm kiếm áp dụng cho người gửi, tiêu đề và plain-text body; bộ lọc hỗ trợ thư
mục, trạng thái đọc/gắn sao, prediction, risk level và ngôn ngữ phát hiện.
Security View đọc analysis đã lưu và sắp xếp theo rủi ro, không chạy inference
lại khi render danh sách. HTML được chuyển thành plain text, ảnh từ xa không
được tải, URL không tự mở và hệ thống không tuyên bố đã quét malware trong file
đính kèm.

Mailbox dùng hai chế độ toàn chiều rộng, không còn khung danh sách và nội dung
đặt cố định cạnh nhau:

```text
Danh sách thư -> chọn thư -> trang đọc thư toàn chiều rộng
               <- Quay lại đúng thư mục/góc nhìn ban đầu
```

Hộp thư đến, Thư rác, Cách ly, Đã xóa, Gắn sao và Security View đều dùng chung
một trang chi tiết. Thư được mở trực tiếp bằng id qua `MailService`, nên bộ lọc
hoặc kết quả tìm kiếm hiện tại không quyết định thư có đọc được hay không. Thư
rác và thư cách ly vẫn đọc được; phần nội dung plain text được ưu tiên, còn giải
thích mô hình, dấu hiệu bảo mật, URL, người gửi, giới hạn và lịch sử được thu gọn
theo progressive disclosure. Chuyển/xóa thư quay về danh sách ban đầu; thao tác
đọc và gắn sao giữ nguyên trang chi tiết.

Các email do script seed tạo chỉ dùng để trình diễn giao diện, không phải dữ
liệu evaluation và không tạo metric khoa học:

```bat
python scripts\seed_demo_mailbox.py
python -m streamlit run app\app.py
```

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
python run.py
```

`python run.py` tự dùng đúng Python đang active và xử lý an toàn project path
có khoảng trắng hoặc ký tự Unicode. Có thể truyền thêm tham số Streamlit, ví dụ
`python run.py --server.port 8502`.

Lệnh Streamlit trực tiếp và pytest:

```bat
python -m streamlit run app\app.py
python -m pytest -q
```

### Continuous integration

Workflow thường tại `.github/workflows/ci.yml` chạy source-level/unit tests trên Python 3.11 khi push hoặc tạo pull request. Lane này không tải production model, raw dataset, translation Transformer hoặc embedding Transformer:

```bat
python -m pytest -m "not production_artifact" -q
```

Workflow `.github/workflows/production-integration.yml` chỉ chạy thủ công. Trước khi chạy, cần cấu hình Actions repository variable `V2_PRODUCTION_MODEL_URL` thành URL HTTPS chính thức của GitHub Release asset `production_pipeline.joblib`. Workflow sẽ fail rõ ràng nếu chưa cấu hình URL, kiểm tra SHA-256 theo `model_metadata.json`, rồi chạy production-artifact tests và inference smoke test.

### Giới hạn

- Chưa có benchmark tiếng Việt bản địa độc lập.
- Held-out tiếng Việt được dịch không có phishing.
- Email quảng cáo ngắn có thể bị ML dự đoán `normal`; lớp security có thể đề xuất `REVIEW` nhưng không thay nhãn ML.
- Kết quả challenge/robustness tách biệt hoàn toàn với held-out metrics.
- Version 2.1 cần test set tiếng Việt bản địa mới và protocol mới; không dùng lại test Version 2 để tuning.
