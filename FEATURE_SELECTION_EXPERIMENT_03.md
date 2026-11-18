# Experiment 03 — Chi-square Feature Selection

Thí nghiệm Development riêng, không thay runtime, không đánh giá Final Test.

**Kết luận:** Macro F1 tốt nhất của feature selection = 0.985775288; baseline = 0.985852233; chênh lệch = -0.000076946.
Giữ nguyên mô hình chính. Đây là so sánh quan sát trên Development Validation đã dùng trong phát triển, không phải kiểm định cải thiện có ý nghĩa thống kê hoặc kết quả Test độc lập.

## Dữ liệu và protocol

- Train: 31,304; Validation: 6,709; assignment kế thừa được giữ nguyên.
- Chỉ đọc Development Master/Train/Validation. Kiểm tra hash, provenance/eligibility, exact/normalized/template fingerprint, final_group_id và parent/translation: PASS.
- TF-IDF frozen baseline đã fit trên đúng Train: Word (1,2), tối đa 60.000; char_wb (3,5), tối đa 80.000; min_df=2, sublinear_tf=True, float32.
- Subject và body cắt tối đa 4.096 ký tự mỗi trường theo preprocessing Development hiện tại; ghép subject + newline + body. Không đổi tiền xử lý giữa baseline và candidate.
- A tái sử dụng TF-IDF baseline chỉ học từ Train để cô lập tác động selection; chi2 và classifier mới chỉ fit Train. Sparse TF-IDF không chuyển toàn bộ thành dense.
- LinearSVC: C=1.0, class_weight=balanced, max_iter=5000, random_state=42. CPU, giới hạn threadpool=2; xử lý transform theo lô 256 dòng.
- B (nếu qua resource gate): StratifiedGroupKFold 3 folds, union các quan hệ group/hash/template/parent; k × C∈{0.1,1,10}; fit lại vectorizer, selector và SVM trong từng fold; chọn bằng Macro F1, tie-break Phishing Recall rồi k/C nhỏ hơn. Chỉ đánh giá candidate đã chốt trên Validation.
- Các mức k của A được cố định trước; không lặp tối ưu bằng Validation. Không có calibration/probability.

## Kết quả Development Validation

| Cấu hình | Chiều trước → sau | Accuracy | Macro F1 | Δ Macro F1 | Phishing Recall | Normal → Spam/Phishing |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 140,000 → 140,000 | 0.996572 | 0.985852233 | +0.000000000 | 0.970588 | 3 |
| A_chi2_15000 | 140,000 → 15,000 | 0.995081 | 0.979370293 | -0.006481940 | 0.970588 | 7 |
| A_chi2_30000 | 140,000 → 30,000 | 0.996125 | 0.983872630 | -0.001979603 | 0.973856 | 4 |
| A_chi2_60000 | 140,000 → 60,000 | 0.996572 | 0.985775288 | -0.000076946 | 0.973856 | 3 |
| A_chi2_100000 | 140,000 → 100,000 | 0.996572 | 0.985775288 | -0.000076946 | 0.973856 | 3 |

## Thời gian, RAM và artifact

| Cấu hình | Fit selector + SVM (s) | Warm E2E median (ms/email) | Peak RSS lấy mẫu (GiB) | Native process peak (GiB) | Artifact (MiB) |
|---|---:|---:|---:|---:|---:|
| baseline | không refit | 1.274 | 0.568 | 0.603 | 7.534 |
| A_chi2_15000 | 4.927 | 2.048 | 1.404 | 1.405 | 3.694 |
| A_chi2_30000 | 8.152 | 2.014 | 1.711 | 1.711 | 4.021 |
| A_chi2_60000 | 6.996 | 2.016 | 2.078 | 2.080 | 4.617 |
| A_chi2_100000 | 7.961 | 1.977 | 2.222 | 2.224 | 5.325 |

Fit trên chỉ gồm chi2 + SVM; không gồm đọc/kiểm tra dữ liệu, transform TF-IDF, serialize hoặc chi phí fit TF-IDF baseline trước đây. Không dùng số này làm tổng thời gian huấn luyện từ đầu.
Latency cùng 16 dòng (8 EN + 8 VI), warm-up một lần, 5 lần lặp, threadpool 2: gồm ghép nội dung → TF-IDF → selection nếu có → SVM; không gồm disk I/O/SQLite/UI, cold model loading hoặc truncation CSV. Đây là throughput mỗi email trong batch nhỏ, không phải SLA cho email dài bất kỳ.
RSS lấy mẫu 100 ms trong mỗi subprocess riêng; native peak tính cả khởi tạo/đọc dữ liệu. Baseline không huấn luyện lại nên peak của baseline là reload/Validation; không so nó với peak fit candidate để kết luận giảm hay tăng RAM huấn luyện baseline. Ứng dụng khác có thể làm nhiễu đo thời gian/RAM. Artifact candidate nén joblib mức 3, baseline dùng kích thước file hiện có; chênh lệch kích thước chịu ảnh hưởng serialization.

## Đặc trưng được giữ

| Cấu hình | Word / 60.000 | Character / 80.000 | Giữ abs(coef) baseline Normal | Spam | Phishing |
|---|---:|---:|---:|---:|---:|
| A_chi2_15000 | 6,474 | 8,526 | 28.94% | 29.96% | 34.26% |
| A_chi2_30000 | 11,461 | 18,539 | 46.63% | 48.23% | 53.39% |
| A_chi2_60000 | 23,766 | 36,234 | 68.55% | 69.84% | 73.72% |
| A_chi2_100000 | 43,095 | 56,905 | 84.94% | 85.38% | 87.01% |

Chi2 xếp hạng liên hệ đơn biến với nhãn trên Train; không đo quan hệ nhân quả hoặc chất lượng từng lớp. Các tỷ lệ |coef| trên là tổng trị tuyệt đối trọng số baseline còn lại tại các cột được chọn, không phải tỷ lệ confidence, explained variance hay đóng góp xác suất. Chi2 score quantiles và SHA-256 indices lưu trong JSON; không xuất token/email thô.
Selector giảm số cột đầu vào SVM nhưng vẫn giữ toàn bộ vocabulary TF-IDF để transform rồi lọc cột. Vì vậy không kỳ vọng E2E latency giảm tỷ lệ với k; preprocessing vẫn tốn phần lớn công việc.

## Per-class và confusion matrix

Thứ tự hàng thật/cột dự đoán: normal, spam, phishing.

### baseline

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996746 | 0.999511 | 0.998127 | 6130 |
| spam | 0.996198 | 0.959707 | 0.977612 | 273 |
| phishing | 0.993311 | 0.970588 | 0.981818 | 306 |

```text
[6127, 1, 2]
[11, 262, 0]
[9, 0, 297]
```

### A_chi2_15000

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.995772 | 0.998858 | 0.997312 | 6130 |
| spam | 0.984615 | 0.937729 | 0.960600 | 273 |
| phishing | 0.990000 | 0.970588 | 0.980198 | 306 |

```text
[6123, 4, 3]
[17, 256, 0]
[9, 0, 297]
```

### A_chi2_30000

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996422 | 0.999347 | 0.997882 | 6130 |
| spam | 0.996154 | 0.948718 | 0.971857 | 273 |
| phishing | 0.990033 | 0.973856 | 0.981878 | 306 |

```text
[6126, 1, 3]
[14, 259, 0]
[8, 0, 298]
```

### A_chi2_60000

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996746 | 0.999511 | 0.998127 | 6130 |
| spam | 0.996183 | 0.956044 | 0.975701 | 273 |
| phishing | 0.993333 | 0.973856 | 0.983498 | 306 |

```text
[6127, 1, 2]
[12, 261, 0]
[8, 0, 298]
```

### A_chi2_100000

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996746 | 0.999511 | 0.998127 | 6130 |
| spam | 0.996183 | 0.956044 | 0.975701 | 273 |
| phishing | 0.993333 | 0.973856 | 0.983498 | 306 |

```text
[6127, 1, 2]
[12, 261, 0]
[8, 0, 298]
```

## Theo ngôn ngữ

| Cấu hình | EN Macro F1 (6.529) | VI translated Accuracy (180) | VI Spam Recall (8) | VI Phishing Recall | ZH |
|---|---:|---:|---:|---|---|
| baseline | 0.988214 | 0.977778 | 0.625000 | N/A (0 mẫu) | N/A (0 mẫu) |
| A_chi2_15000 | 0.982281 | 0.972222 | 0.500000 | N/A (0 mẫu) | N/A (0 mẫu) |
| A_chi2_30000 | 0.986875 | 0.972222 | 0.500000 | N/A (0 mẫu) | N/A (0 mẫu) |
| A_chi2_60000 | 0.988125 | 0.977778 | 0.625000 | N/A (0 mẫu) | N/A (0 mẫu) |
| A_chi2_100000 | 0.988125 | 0.977778 | 0.625000 | N/A (0 mẫu) | N/A (0 mẫu) |

VI chỉ có 172 Normal, 8 Spam, 0 Phishing và đều là bản dịch; không có benchmark VI bản địa. Không báo Macro F1 ba lớp VI hoặc khả năng ZH. JSON giữ per-language/per-origin/support đầy đủ; origin trống của EN được ghi not_declared, không suy diễn native.

## B — Train group-aware CV

**BLOCKED**.
RAM trống khi kiểm tra: 2.970 GiB; ngưỡng cấu hình trước chạy: 6.0 GiB. Không fit fold thật, không có score Train CV, không có candidate B hoặc lựa chọn k/C tối ưu. Ngưỡng là headroom bảo thủ cho nhiều lần fit vocabulary trong fold, không phải khẳng định mức RAM tối thiểu tuyệt đối của sklearn. Không thay bằng CV dùng vectorizer toàn Train.

## Diễn giải và bảo toàn

Ở k=60.000 và 100.000, confusion matrix giống nhau: bắt thêm 1 Phishing nhưng bỏ sót thêm 1 Spam so với baseline, làm Macro F1 thấp hơn dù Accuracy bằng nhau. k nhỏ hơn làm mất thêm tín hiệu phân biệt Spam. Đây là bằng chứng trade-off quan sát; chưa chứng minh nguyên nhân của từng lỗi.
Không có cấu hình A vượt baseline về Macro F1. Không chọn/thay model chính từ những kết quả này. CV B chưa thực hiện nên không kết luận mọi k/C khác đều thất bại.
V2, V2.1 runtime, Experiment 01/02 và dữ liệu được giữ nguyên; không truy cập Final Test, không Transformer mới, không download, không Gemini, không commit/push.

## Tái lập

```powershell
.\.venv\Scripts\python.exe -u scripts/v21/feature_selection_03.py --run
.\.venv\Scripts\python.exe -m pytest tests/test_v21_feature_selection.py -q
```

Checkpoint được xác minh hash config/Train/Validation/runtime/artifact, không tự ghi đè model có sẵn. BLOCKED CV cũng là checkpoint của lần đo này; thử CV ở điều kiện khác cần run namespace/config riêng, không sửa kết quả đã đo. Chi tiết phiên bản, hash code/artifact: run_manifest.json.

## File của nhiệm vụ

- config/v21/experiments/feature_selection_03.json — grid và resource gates.
- src/vietmailguard/v21/feature_selection_experiment.py — training, CV, kiểm tra và báo cáo.
- scripts/v21/feature_selection_03.py — CLI và subprocess từng candidate.
- tests/test_v21_feature_selection.py — sparse safety, Train-only fitting, folds, reload/resume/integrity.
- FEATURE_SELECTION_EXPERIMENT_03.md — báo cáo này.
- models/v21_multilingual/experiments/feature_selection_03/ — bốn artifact A riêng.
- results/v21_multilingual/experiments/feature_selection_03/ — checkpoints, metrics, diagnostics, manifest và verification.

## Kiểm thử và bảo toàn

- Unit/integration riêng: 12 passed (2.60 s).
- Regression offline: 690 passed, 3 skipped, 3 deselected, 1 warning (97.95 s).
- 3 skipped: real TREC archive/curation/review UI tests cần opt-in; 3 deselected: các test đọc held-out thật, được loại để tuân thủ không mở Test. Không coi đây là full regression bao gồm held-out.
- 1 ConvergenceWarning của fixture MLP chỉ chạy 5 iterations, không phải candidate Chi-square.
- SHA-256 trước/sau: 263/263 file bảo vệ không đổi.
- git diff --check: PASS (exit 0; pre-existing LF/CRLF notices); không có external-network/held-out access attempt trong regression.

Lệnh thực tế:

```powershell
.\.venv\Scripts\python.exe -u scripts/v21/feature_selection_03.py --run
.\.venv\Scripts\python.exe -m pytest tests/test_v21_feature_selection.py -q
.\.venv\Scripts\python.exe -u .local-test-deps/gemini-reviewer/run_offline_tests.py --include-dashboard-page-test -q -rs
.\.venv\Scripts\python.exe -u scripts/v21/feature_selection_03.py --report-only
git diff --check
```

Nhánh local: v21/experiment-feature-selection-03. Các thay đổi có sẵn trước nhiệm vụ được bảo toàn.
