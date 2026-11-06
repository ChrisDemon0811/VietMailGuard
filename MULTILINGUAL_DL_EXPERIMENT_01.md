# Multilingual Deep Learning Experiment 01

Ngày: 2026-10-29. Chỉ Development Validation.

## Dataset và integrity

Train: 31,304; Validation: 6,709. Giữ nguyên assignment bootstrap V2.
Validation kế thừa từng dùng cho lựa chọn model V2; đây không phải Final Test độc lập.
Không dùng TREC06C, Chinese synthetic hoặc Gemini làm ground truth. Không mở held-out Test.
Exact/content, normalized, template, final group và parent/translation kiểm tra lại trước khi fit.

| Split | Ngôn ngữ | Normal | Spam | Phishing |
|---|---|---:|---:|---:|
| train | en | 27340 | 782 | 936 |
| train | vi | 1264 | 491 | 491 |
| train | zh | 0 | 0 | 0 |
| validation | en | 5958 | 265 | 306 |
| validation | vi | 172 | 8 | 0 |
| validation | zh | 0 | 0 | 0 |

## Architecture và protocol

A: load TF-IDF Word+Character + LinearSVC hiện tại read-only, không fit lại.
B: frozen multilingual MiniLM, L2-normalized 384D embeddings → balanced LinearSVC.
C: cùng embeddings → StandardScaler fit Train → MLP 128-ReLU → 64-ReLU → 3 softmax.
MLP dùng balanced sample weights từ Train; không dùng Validation cho scaler, encoder, optimizer hoặc early stopping.
Early stopping nội bộ random split bị tắt; tối đa 120 epoch, dừng theo training loss nếu hội tụ.
Đầu MLP huấn luyện CPU; GPU chỉ phục vụ encoder. Softmax chưa calibrated, không là model confidence.
Cấu hình và seed được lưu trước khi chạy; không sweep hyperparameter theo Validation.
Subject/body qua EmailTextComposer hiện tại, giới hạn 4096 ký tự mỗi field trong development loader; MiniLM tiếp tục truncate ở 128 tokens. Baseline TF-IDF có thể giữ thông tin dài hơn — đây là giới hạn so sánh cố định này, không phải bằng chứng mọi neural model đều kém hơn.

## Tài nguyên và dependency

```json
{
  "resources": {
    "python": "3.13.14",
    "cuda": true,
    "free_ram_gib": 1.896,
    "total_ram_gib": 15.63,
    "free_disk_gib": 105.89,
    "torch_version": "2.8.0+cu128",
    "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "total_vram_gib": 4.0,
    "free_vram_gib": 3.222,
    "process_rss_gib": null
  },
  "packages": {
    "scikit-learn": "1.9.1",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "joblib": "1.6.0",
    "torch": "2.8.0+cu128",
    "sentence-transformers": "5.7.0",
    "transformers": "4.56.2",
    "huggingface-hub": "0.36.2",
    "threadpoolctl": "3.6.0"
  },
  "encoder_allowed": true,
  "encoder_reason": "resource gate passed",
  "encoder_cached": true,
  "mlp_sample_weight_supported": true
}
```

## Validation comparison

| Candidate | Status | Accuracy | Macro F1 | Phishing Recall | Normal→Spam/Phishing | Fit s | Full-Val ms/email* | End-to-end smoke ms/email |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_tfidf_svc | PASS | 0.996572 | 0.985852 | 0.970588 | 3.000000 | N/A | 1.242480 | 1.800312 |
| B_minilm_svc | PASS | 0.976599 | 0.903714 | 0.941176 | 99.000000 | 2.842887 | 0.001686 | 3.373331 |
| C_minilm_mlp | PASS | 0.988970 | 0.951334 | 0.937908 | 20.000000 | 7.342584 | 0.001994 | 3.410544 |

| Candidate | RAM trống trước fit (GiB) | RAM trống sau (GiB) | Iterations | Fit warnings |
|---|---:|---:|---:|---|
| A_tfidf_svc | N/A | 1.344000 | N/A | none recorded |
| B_minilm_svc | 1.605000 | 1.643000 | 10 | none recorded |
| C_minilm_mlp | 1.674000 | 1.627000 | 30 | none recorded |

RAM ở đây là snapshot khả dụng của hệ thống, không phải peak RSS. psutil không có trong môi trường; không tạo số đo peak memory giả.
Do RAM hạn chế, xác minh encoder/cache, fit heads và benchmark end-to-end chạy thành các process riêng. Thời gian fit head không bao gồm thời gian sinh embeddings; cache metadata ghi riêng thời gian sinh gốc.

*Full-Val: A gồm compose/TF-IDF/classifier; B/C chỉ classifier trên embedding đã cache.
Cột end-to-end dùng cùng mẫu EN/VI, bao gồm encoder cho B/C, warm-up và median các lần lặp.
Độ trễ batch không tương đương request đơn; xem JSON để biết batch size, device và từng lần đo.

### A_tfidf_svc

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996746 | 0.999511 | 0.998127 | 6130 |
| spam | 0.996198 | 0.959707 | 0.977612 | 273 |
| phishing | 0.993311 | 0.970588 | 0.981818 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6127, 1, 2], [11, 262, 0], [9, 0, 297]]
```

| Ngôn ngữ | Rows | Accuracy | Macro F1 ba lớp | Macro F1 lớp có mặt | Phishing Recall |
|---|---:|---:|---:|---:|---:|
| en | 6529 | 0.997090 | 0.988214 | 0.988214 | 0.970588 |
| vi | 180 | 0.977778 | N/A | 0.878835 | N/A |
| zh | 0 | N/A | N/A | N/A | N/A |

| Ngôn ngữ | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| en | normal | 5958 | 0.997154 | 0.999664 | 0.998408 |
| en | spam | 265 | 0.996124 | 0.969811 | 0.982792 |
| en | phishing | 306 | 0.996644 | 0.970588 | 0.983444 |
| vi | normal | 172 | 0.982759 | 0.994186 | 0.988439 |
| vi | spam | 8 | 1.000000 | 0.625000 | 0.769231 |
| vi | phishing | 0 | N/A | N/A | N/A |
| zh | normal | 0 | N/A | N/A | N/A |
| zh | spam | 0 | N/A | N/A | N/A |
| zh | phishing | 0 | N/A | N/A | N/A |

### B_minilm_svc

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.991615 | 0.983850 | 0.987717 | 6130 |
| spam | 0.763934 | 0.853480 | 0.806228 | 273 |
| phishing | 0.894410 | 0.941176 | 0.917197 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6031, 70, 29], [35, 233, 5], [16, 2, 288]]
```

| Ngôn ngữ | Rows | Accuracy | Macro F1 ba lớp | Macro F1 lớp có mặt | Phishing Recall |
|---|---:|---:|---:|---:|---:|
| en | 6529 | 0.976566 | 0.904693 | 0.904693 | 0.941176 |
| vi | 180 | 0.977778 | N/A | 0.827586 | N/A |
| zh | 0 | N/A | N/A | N/A | N/A |

| Ngôn ngữ | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| en | normal | 5958 | 0.992042 | 0.983384 | 0.987694 |
| en | spam | 265 | 0.760797 | 0.864151 | 0.809187 |
| en | phishing | 306 | 0.894410 | 0.941176 | 0.917197 |
| vi | normal | 172 | 0.977273 | 1.000000 | 0.988506 |
| vi | spam | 8 | 1.000000 | 0.500000 | 0.666667 |
| vi | phishing | 0 | N/A | N/A | N/A |
| zh | normal | 0 | N/A | N/A | N/A |
| zh | spam | 0 | N/A | N/A | N/A |
| zh | phishing | 0 | N/A | N/A | N/A |

### C_minilm_mlp

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.991722 | 0.996737 | 0.994223 | 6130 |
| spam | 0.933333 | 0.871795 | 0.901515 | 273 |
| phishing | 0.979522 | 0.937908 | 0.958264 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6110, 15, 5], [34, 238, 1], [17, 2, 287]]
```

| Ngôn ngữ | Rows | Accuracy | Macro F1 ba lớp | Macro F1 lớp có mặt | Phishing Recall |
|---|---:|---:|---:|---:|---:|
| en | 6529 | 0.989432 | 0.953943 | 0.953943 | 0.937908 |
| vi | 180 | 0.972222 | N/A | 0.765564 | N/A |
| zh | 0 | N/A | N/A | N/A | N/A |

| Ngôn ngữ | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| en | normal | 5958 | 0.992313 | 0.996643 | 0.994473 |
| en | spam | 265 | 0.932540 | 0.886792 | 0.909091 |
| en | phishing | 306 | 0.979522 | 0.937908 | 0.958264 |
| vi | normal | 172 | 0.971751 | 1.000000 | 0.985673 |
| vi | spam | 8 | 1.000000 | 0.375000 | 0.545455 |
| vi | phishing | 0 | N/A | N/A | N/A |
| zh | normal | 0 | N/A | N/A | N/A |
| zh | spam | 0 | N/A | N/A | N/A |
| zh | phishing | 0 | N/A | N/A | N/A |

## Kết luận và bước nghiên cứu tiếp theo

MLP Macro F1 trừ TF-IDF baseline = -0.034518. MLP không cải thiện baseline trên Development Validation này.

Chỉ EN và VI dịch có dữ liệu development; VI Validation không có phishing, ZH không có hàng.
Không có native Vietnamese benchmark hoặc benchmark Chinese phishing độc lập; không tuyên bố EN/VI/ZH Production hay Final Evaluation PASS.
XLM-RoBERTa chưa được chạy trong thí nghiệm này. Điều kiện tài nguyên đã lưu:
```json
{
  "allowed": false,
  "reason": "insufficient free RAM",
  "trained": false
}
```
Nên ưu tiên benchmark bản địa và kiểm duyệt nhãn trước một thí nghiệm Transformer lớn. Cần nhiệm vụ riêng, cache/dependency và tài nguyên đạt gate để tiếp tục XLM-R.

## Tái lập

```powershell
.\.venv\Scripts\python.exe scripts/v21/multilingual_dl_experiment_01.py --preflight
.\.venv\Scripts\python.exe scripts/v21/multilingual_dl_experiment_01.py --run
.\.venv\Scripts\python.exe scripts/v21/multilingual_dl_experiment_01.py --benchmark
```
Candidate và metric lưu riêng trong models/results/v21_multilingual/experiments/multilingual_dl_01/.
Không ghi đè candidate/runtime hiện tại. JSON lưu config/hash, cache provenance, resource snapshots, convergence, class ordering và per-language per-class metrics.
Cache mới ghi batch + checkpoint riêng; cache cũ chỉ đọc. Candidate checkpoint cho phép tiếp tục sau gián đoạn, không fit lại candidate đã hoàn tất.

## Kiểm thử và bảo toàn — kết quả thực tế

- Nhánh local: `v21/experiment-multilingual-dl-01`; giữ nguyên `.env.example` untracked đã có. Không commit/push.
- **PASS** — 31.304 Train, 6.709 Validation giữ nguyên SHA-256 và assignment. Kiểm tra lại content/normalized/template fingerprints, final_group_id, nhãn xung đột và parent/translation bằng validator development hiện có: 0 crossing, 0 conflict, 0 parent crossing.
- **PASS** — encoder cache được đối chiếu model/revision, ordered content keys, shape, float32, L2 và bốn vector encode lại mỗi split. Receipt `embedding_verification.json` khóa SHA-256 cho cả matrix và metadata trước khi process fit riêng tái sử dụng.
- **PASS** — hai candidate mới reload được. Thứ tự cột classifier được lưu theo `classifier.classes_` (`normal`, `phishing`, `spam`); confusion matrix dùng thứ tự được ghi riêng (`normal`, `spam`, `phishing`). Hoàn thiện metadata thứ tự nhãn không thay learned parameters hoặc dự đoán trên toàn bộ 6.709 Validation.
- **PASS** — MLP 57.731 tham số, dừng ở 30 iterations theo training loss, không có fit warning trên dữ liệu development thật. Encoder không được fine-tune. Softmax không được công bố như calibrated confidence.
- **PASS** — 27 unit/integration tests liên quan. Bao gồm synthetic end-to-end, classifier reload, checkpoint/resume, cache hỏng, resource gate, chặn socket/DNS/Test, training-only scaler, nhãn thiếu trả N/A và benchmark không fit.
- **PASS** — regression có guard: **638 passed, 3 skipped, 3 deselected**. Không có lần truy cập held-out hoặc kết nối ngoài bị ghi nhận. Đây không phải toàn bộ suite không lọc: ba test đọc CSV held-out thật được chủ động deselect để tuân thủ nhiệm vụ.
- Ba skipped là các bài opt-in với ZIP TREC thật, curation TREC local và review UI TREC local. Không liên quan chất lượng model ba lớp của thí nghiệm này.
- Một `ConvergenceWarning` trong test fixture MLP cố ý giới hạn 5 iterations; không phải cảnh báo của candidate C thật.
- **PASS** — 315 file source/config/model/results và development inputs đã có giữ nguyên hash so với đầu nhiệm vụ. Backup Functional Research Preview xác minh **400/400** file, không thay đổi. Không mở record Final Test để thực hiện phép kiểm tra này.
- **PASS** — không sửa runtime/UI/Gemini/V2 Production/candidate preview; không tải dependency hoặc model, không gọi API, không sử dụng TREC/synthetic Chinese làm nhãn ba lớp.
- **PASS** — `git diff --check`. Model và kết quả generated tiếp tục được Git ignore.
- **BLOCKED** — XLM-RoBERTa không đủ RAM trống cho gate 8 GiB; GPU 4 GiB cũng dưới yêu cầu 8 GiB VRAM của cấu hình hiện có. Không tải hoặc train XLM-R.
- **N/A** — Phishing tiếng Việt/Trung và hiệu năng Chinese ba lớp: Validation VI chỉ có 172 normal + 8 spam, ZH 0. Không thể kết luận hiệu năng trên email bản địa VI/ZH.

Các lệnh kiểm thử thực tế:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v21_multilingual_dl_experiment.py tests/test_v21_development_training.py tests/test_embedding_modeling.py -q
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py --include-dashboard-page-test -q -rs
.\.venv\Scripts\python.exe .local-test-deps/dl-experiment-01/preserve.py before
.\.venv\Scripts\python.exe .local-test-deps/dl-experiment-01/preserve.py after
git diff --check
git status --short
```

Guard regression sử dụng runtime SQLite tạm, chặn kết nối ngoài, chặn đọc file Test thật; chạy cả các production-artifact tests có artifact local, ngoại trừ ba test đọc held-out nêu trên. Unit tests có thể fit pipeline nhỏ synthetic. Harness và receipt nằm trong `.local-test-deps/` bị ignore, không phải thay đổi production.

## File được tạo

Source/config/test:

- `config/v21/experiments/multilingual_dl_01.json`
- `src/vietmailguard/v21/multilingual_dl_experiment.py`
- `scripts/v21/multilingual_dl_experiment_01.py`
- `tests/test_v21_multilingual_dl_experiment.py`
- `MULTILINGUAL_DL_EXPERIMENT_01.md`

Artifact local: `models/v21_multilingual/experiments/multilingual_dl_01/B_minilm_svc.joblib` và `C_minilm_mlp.joblib`. Cấu hình encoder/preprocessing, class ordering, source hashes, model hashes, fit/Validation/latency và cache lineage được lưu trong thư mục results riêng cùng tên. Không đưa hai candidate này vào Mail Routing.

Kết luận: hoàn thành thí nghiệm development được giao; giữ baseline TF-IDF đang có. Chưa nên triển khai XLM-R trên cấu hình tài nguyên này hoặc đổi model runtime. Cần benchmark bản địa và dữ liệu các lớp còn thiếu trước khi mở rộng kết luận. **Không có tuyên bố Production hoặc Final Evaluation PASS.**
