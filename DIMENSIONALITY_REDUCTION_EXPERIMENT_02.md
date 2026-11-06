# Dimensionality Reduction Experiment 02

## Phạm vi và protocol

Thí nghiệm Development Validation, seed 42. Không thay V2 Production, V2.1 Preview hoặc LinearSVC runtime.
Train 31,304; Validation 6,709; giữ nguyên assignment và SHA-256 bootstrap.
Validation này từng phục vụ phát triển V2, không phải Final Test độc lập. Không mở held-out Test, không dùng TREC/synthetic Chinese, không gọi Gemini hoặc tải dependency/model.
Đã kiểm tra lại exact/content, normalized, template, group và parent/translation bằng validator development hiện có.
```json
{
  "passed": true,
  "cross_split_groups": 0,
  "conflicting_label_groups": 0,
  "parent_crossings": 0
}
```

| Split | Language | Data origin | Class | Rows |
|---|---|---|---|---:|
| train | en | not_declared | normal | 27340 |
| train | en | not_declared | phishing | 936 |
| train | en | not_declared | spam | 782 |
| train | vi | translated | normal | 1264 |
| train | vi | translated | phishing | 491 |
| train | vi | translated | spam | 491 |
| validation | en | not_declared | normal | 5958 |
| validation | en | not_declared | phishing | 306 |
| validation | en | not_declared | spam | 265 |
| validation | vi | translated | normal | 172 |
| validation | vi | translated | spam | 8 |

## Pipeline

- A: frozen TF-IDF Word+Character + balanced LinearSVC, load read-only; không refit TF-IDF hoặc classifier.
- B: cùng TF-IDF fitted trên Train → TruncatedSVD randomized (100/200/300) → balanced LinearSVC C=1. TF-IDF luôn CSR sparse; chỉ latent output được dense.
- C: cache MiniLM L2 384D, revision cố định → PCA (50/100/200, whiten=False) → balanced LinearSVC C=1. Có đối chứng MiniLM 384D + LinearSVC từ Experiment 01, không fit lại.
- D: LDA solver=svd, tối đa 2 chiều với 3 lớp. So sánh LDA classifier và LDA 2D → balanced LinearSVC. Biểu đồ dùng trục LDA học từ Train, áp dụng nguyên trạng lên Validation.
D_lda_classifier vẫn nhận input 384D và dùng predict của LDA; n_components=2 chỉ giới hạn transform. D_lda2_svc mới dùng tọa độ 2D làm input LinearSVC. Không coi hai cách dự đoán là cùng một classifier.
Tất cả reducer/scaler/classifier mới chỉ fit trên Train. LDA chỉ nhận y_train, không nhận nhãn Validation khi fit. PCA không whitening hoặc refit trên Validation; encoder không fine-tune.
Chọn LinearSVC cho PCA để tập trung vào ảnh hưởng phép giảm chiều và tiết kiệm tài nguyên; không sweep MLP/hyperparameter ngoài cấu hình đã khai báo.
Mọi phép fit dùng CPU và thread limit trong config. Cache embeddings được kiểm tra ordered content keys, model/revision, finite/L2/shape/dtype và SHA-256 đối chiếu receipt encoder đã xác minh ở Experiment 01; không nạp encoder mới.

## So sánh Validation

| Candidate | Status | Dim | Macro F1 | Phishing Recall | Normal→Spam/Phishing | Variance* | Fit s | ms/email** | RSS GiB*** | Artifact MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | PASS | 140000.000000 | 0.985852 | 0.970588 | 3.000000 | N/A | N/A | 2.913400 | 0.465565 | 7.533783 |
| B_svd_100 | PASS | 100.000000 | 0.957809 | 0.967320 | 43.000000 | 0.180419 | 74.169078 | 5.802531 | 1.022949 | 57.739227 |
| B_svd_200 | PASS | 200.000000 | 0.969704 | 0.970588 | 28.000000 | 0.231793 | 122.905980 | 7.107056 | 1.230427 | 111.148422 |
| B_svd_300 | PASS | 300.000000 | 0.974017 | 0.970588 | 23.000000 | 0.270171 | 171.230337 | 9.843256 | 1.462372 | 164.557632 |
| C_embedding_384_control | PASS | 384.000000 | 0.903714 | 0.941176 | 99.000000 | N/A | N/A | 0.011475 | 0.521347 | 0.010044 |
| C_pca_50 | PASS | 50.000000 | 0.825098 | 0.862745 | 195.000000 | 0.646958 | 1.242674 | 0.034031 | 0.655701 | 0.078468 |
| C_pca_100 | PASS | 100.000000 | 0.865919 | 0.915033 | 155.000000 | 0.843384 | 2.257244 | 0.058281 | 0.660667 | 0.153435 |
| C_pca_200 | PASS | 200.000000 | 0.890730 | 0.947712 | 127.000000 | 0.976969 | 4.330250 | 0.052956 | 0.731808 | 0.303352 |
| D_lda_classifier | PASS | 2.000000 | 0.906882 | 0.885621 | 42.000000 | N/A | 1.629453 | 0.015050 | 0.500900 | 0.014856 |
| D_lda2_svc | PASS | 2.000000 | 0.894620 | 0.931373 | 101.000000 | N/A | 1.659712 | 0.030719 | 0.593853 | 0.015480 |

*Variance là tổng explained_variance_ratio_ trên Train của PCA/SVD; không đồng nghĩa phần thông tin phân loại được bảo toàn. LDA discriminant variance là đại lượng khác nên không điền chung cột này.
**Median 3 lần sau warm-up, cùng batch tối đa 8 EN + 8 VI. A/B gồm compose+TF-IDF+projection/classifier; C/D và embedding control chỉ gồm pipeline trên embeddings đã cache, không gồm encoder. Không so trực tiếp hai scope này như độ trễ end-to-end tương đương.
***RSS peak lấy mẫu 100 ms, gồm resident input/cache và quá trình reload/inference; không phải phép đo allocation riêng reducer hay peak tuyệt đối. Native process peak và RAM trước/sau lưu trong JSON. Artifact B chứa cả fitted TF-IDF; C/D cần encoder local đúng revision ngoài artifact head.
RSS của D_lda_classifier còn bao gồm dựng hình bằng Matplotlib; không quy toàn bộ phần chênh lệch đó cho thuật toán LDA.
Fit s gồm reducer.fit_transform(Train) + classifier.fit; không gồm tạo TF-IDF/embeddings. Baseline/control được tái sử dụng nên fit N/A.

### A_baseline

Accuracy: 0.996572.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996746 | 0.999511 | 0.998127 | 6130 |
| spam | 0.996198 | 0.959707 | 0.977612 | 273 |
| phishing | 0.993311 | 0.970588 | 0.981818 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6127, 1, 2], [11, 262, 0], [9, 0, 297]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.988214 | 0.970588 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

### B_svd_100

Accuracy: 0.989715.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.995910 | 0.992985 | 0.994445 | 6130 |
| spam | 0.898601 | 0.941392 | 0.919499 | 273 |
| phishing | 0.951768 | 0.967320 | 0.959481 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6087, 29, 14], [15, 257, 1], [10, 0, 296]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.960676 | 0.967320 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

Validation squared-energy lost theo lớp (PCA: tâm là mean Train):
```json
{
  "normal": 0.791572093963623,
  "spam": 0.7346274852752686,
  "phishing": 0.7971289157867432
}
```

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 54,
  "baseline_wrong_candidate_correct": 8
}
```

### B_svd_200

Accuracy: 0.992547.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996408 | 0.995432 | 0.995920 | 6130 |
| spam | 0.945455 | 0.952381 | 0.948905 | 273 |
| phishing | 0.958065 | 0.970588 | 0.964286 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6102, 15, 13], [13, 260, 0], [9, 0, 297]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.972377 | 0.970588 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

Validation squared-energy lost theo lớp (PCA: tâm là mean Train):
```json
{
  "normal": 0.7469933032989502,
  "spam": 0.6811907887458801,
  "phishing": 0.7393507361412048
}
```

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 34,
  "baseline_wrong_candidate_correct": 7
}
```

### B_svd_300

Accuracy: 0.993591.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.996736 | 0.996248 | 0.996492 | 6130 |
| spam | 0.959707 | 0.959707 | 0.959707 | 273 |
| phishing | 0.961165 | 0.970588 | 0.965854 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6107, 11, 12], [11, 262, 0], [9, 0, 297]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.976124 | 0.970588 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

Validation squared-energy lost theo lớp (PCA: tâm là mean Train):
```json
{
  "normal": 0.7151588797569275,
  "spam": 0.6470763087272644,
  "phishing": 0.7004310488700867
}
```

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 29,
  "baseline_wrong_candidate_correct": 9
}
```

### C_embedding_384_control

Accuracy: 0.976599.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.991615 | 0.983850 | 0.987717 | 6130 |
| spam | 0.763934 | 0.853480 | 0.806228 | 273 |
| phishing | 0.894410 | 0.941176 | 0.917197 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6031, 70, 29], [35, 233, 5], [16, 2, 288]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.904693 | 0.941176 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 142,
  "baseline_wrong_candidate_correct": 8
}
```

### C_pca_50

Accuracy: 0.954986.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.983919 | 0.968189 | 0.975991 | 6130 |
| spam | 0.617211 | 0.761905 | 0.681967 | 273 |
| phishing | 0.776471 | 0.862745 | 0.817337 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[5935, 125, 70], [59, 208, 6], [38, 4, 264]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.825529 | 0.862745 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

Validation squared-energy lost theo lớp (PCA: tâm là mean Train):
```json
{
  "normal": 0.35348057746887207,
  "spam": 0.3765884041786194,
  "phishing": 0.3058534562587738
}
```

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 283,
  "baseline_wrong_candidate_correct": 4
}
```

### C_pca_100

Accuracy: 0.965569.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.988420 | 0.974715 | 0.981520 | 6130 |
| spam | 0.657817 | 0.816850 | 0.728758 | 273 |
| phishing | 0.861538 | 0.915033 | 0.887480 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[5975, 113, 42], [47, 223, 3], [23, 3, 280]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.866265 | 0.915033 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

Validation squared-energy lost theo lớp (PCA: tâm là mean Train):
```json
{
  "normal": 0.1578749120235443,
  "spam": 0.17282982170581818,
  "phishing": 0.13503597676753998
}
```

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 217,
  "baseline_wrong_candidate_correct": 9
}
```

### C_pca_200

Accuracy: 0.972574.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.991740 | 0.979282 | 0.985472 | 6130 |
| spam | 0.709480 | 0.849817 | 0.773333 | 273 |
| phishing | 0.881459 | 0.947712 | 0.913386 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6003, 93, 34], [36, 232, 5], [14, 2, 290]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.892366 | 0.947712 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

Validation squared-energy lost theo lớp (PCA: tâm là mean Train):
```json
{
  "normal": 0.023256178945302963,
  "spam": 0.02533356472849846,
  "phishing": 0.02088221162557602
}
```

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 170,
  "baseline_wrong_candidate_correct": 9
}
```

### D_lda_classifier

Accuracy: 0.978983.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.984476 | 0.993148 | 0.988793 | 6130 |
| spam | 0.863636 | 0.765568 | 0.811650 | 273 |
| phishing | 0.957597 | 0.885621 | 0.920204 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6088, 30, 12], [64, 209, 0], [32, 3, 271]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.909951 | 0.885621 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 122,
  "baseline_wrong_candidate_correct": 4
}
```

### D_lda2_svc

Accuracy: 0.974661.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.989984 | 0.983524 | 0.986743 | 6130 |
| spam | 0.757576 | 0.824176 | 0.789474 | 273 |
| phishing | 0.885093 | 0.931373 | 0.907643 | 306 |

Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.
```json
[[6029, 69, 32], [43, 225, 5], [18, 3, 285]]
```

| Language | Rows | Macro F1 (3 lớp) | Phishing Recall |
|---|---:|---:|---:|
| en | 6529 | 0.897513 | 0.931373 |
| vi | 180 | N/A | N/A |
| zh | 0 | N/A | N/A |

So sánh lỗi theo cùng dòng Validation, không xuất nội dung email:
```json
{
  "baseline_correct_candidate_wrong": 153,
  "baseline_wrong_candidate_correct": 6
}
```

## Trực quan LDA

![LDA Train và Development Validation](results/v21_multilingual/experiments/dimensionality_reduction_02/lda_projection.png)
Lấy mẫu phân tầng cố định tối đa 1.000 điểm/lớp/panel chỉ để dễ nhìn. Không thay đổi dữ liệu đánh giá. Trục dùng nhãn Train nên hình Train có thể tách đẹp hơn; không coi biểu đồ là bằng chứng hiệu năng độc lập.

## Kết luận

Baseline thực tế: 0.985852; reduced candidate tốt nhất: B_svd_300.
Mọi candidate giảm chiều đã hoàn tất đều kém baseline về Macro F1 Development. Giữ nguyên model chính; không thay runtime.

| Phép giảm chiều | Δ Macro F1 so với control cùng representation | Δ Phishing Recall |
|---|---:|---:|
| B_svd_100 | -0.028044 | -0.003268 |
| B_svd_200 | -0.016149 | +0.000000 |
| B_svd_300 | -0.011835 | +0.000000 |
| C_pca_50 | -0.078616 | -0.078431 |
| C_pca_100 | -0.037795 | -0.026144 |
| C_pca_200 | -0.012984 | +0.006536 |
| D_lda_classifier | +0.003168 | -0.055556 |
| D_lda2_svc | -0.009094 | -0.009804 |
Giảm chiều có thể loại các hướng ít variance nhưng chứa tín hiệu hiếm của Spam/Phishing. Bảng per-class, energy loss và số baseline-correct→candidate-wrong cho biết trade-off quan sát được; không đủ để khẳng định một từ khóa cụ thể bị mất là nguyên nhân.
PCA center theo mean Train và giữ phương sai tổng thể; TruncatedSVD không center sparse TF-IDF, xấp xỉ ma trận gốc; LDA là phép supervised dùng nhãn Train. Cả ba mục tiêu đều không bảo đảm Macro F1/Phishing Recall tốt hơn classifier gốc.
Baseline tuyến tính ba lớp chỉ cần sparse dot với các hướng phân loại. Thêm SVD 100–300 chiều phải tính thêm sparse×dense projection và lưu dense components; số chiều đầu vào classifier giảm không bảo đảm artifact nhỏ hơn hay inference nhanh hơn. PCA cũng thêm phép chiếu; mọi lợi ích trên head còn phải tính chi phí MiniLM encoder khi triển khai thực tế.
So sánh PCA với control 384D trong cùng họ embeddings, SVD với TF-IDF cùng vectorizer. Khác biệt TF-IDF/MiniLM còn bao gồm encoder và truncation 128 tokens; không quy mọi thay đổi cho giảm chiều.
VI Validation chỉ 172 normal + 8 spam, 0 phishing; ZH không có hàng. Không có native Vietnamese/Chinese three-class benchmark độc lập. Không tuyên bố Phishing VI/ZH, Production hoặc Final Evaluation PASS.

## Tái lập

```powershell
.\.venv\Scripts\python.exe scripts/v21/dimensionality_reduction_02.py --preflight
.\.venv\Scripts\python.exe scripts/v21/dimensionality_reduction_02.py --run
```
Mỗi candidate chạy process riêng để giải phóng workspace. Stage hoàn tất có hash/checkpoint được tái sử dụng; artifact hiện có không bị ghi đè. Candidate thiếu RAM/cache ghi BLOCKED, không tự hạ n_components hoặc đổi model.
Config: config/v21/experiments/dimensionality_reduction_02.json. Models/results/cache lưu riêng dưới namespace experiments/dimensionality_reduction_02. Không chứa raw email trong report/plot.

## Kiểm thử, bảo toàn và file tạo mới

```json
{
  "branch": "v21/experiment-dimensionality-02",
  "targeted_tests": {
    "status": "PASS",
    "passed": 34,
    "warnings": 1,
    "command": ".\\.venv\\Scripts\\python.exe -m pytest tests/test_v21_dimensionality_experiment.py tests/test_v21_multilingual_dl_experiment.py tests/test_v21_development_training.py tests/test_embedding_modeling.py -q"
  },
  "guarded_regression": {
    "status": "PASS",
    "passed": 645,
    "skipped": 3,
    "deselected": 3,
    "command": ".\\.venv\\Scripts\\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py --include-dashboard-page-test -q -rs",
    "receipt": {
      "exit_code": 0,
      "arguments": [
        "-q",
        "-rs",
        "--deselect=tests/test_split_builder.py::test_generated_repository_splits_have_disjoint_groups",
        "--deselect=tests/test_v2_completion_integrity.py::test_actual_v2_splits_preserve_eligibility_provenance_and_origin",
        "--deselect=tests/test_dashboard_data.py::test_dataset_dashboard_uses_only_eligible_v2_splits"
      ],
      "guarded_access_attempts": [],
      "isolated_runtime": ".local-test-deps\\gemini-reviewer\\runtime-outbrq1t",
      "excluded_real_heldout_readers": [
        "tests/test_split_builder.py::test_generated_repository_splits_have_disjoint_groups",
        "tests/test_v2_completion_integrity.py::test_actual_v2_splits_preserve_eligibility_provenance_and_origin",
        "tests/test_dashboard_data.py::test_dataset_dashboard_uses_only_eligible_v2_splits"
      ],
      "synthetic_fixture_fitting": "Existing unit tests may fit tiny synthetic pipelines, never actual development or production artifacts."
    },
    "skips": "Three existing TREC local-data/UI opt-in integration tests",
    "warning": "Existing Experiment 01 tiny MLP fixture deliberately limited to 5 iterations; no new experiment fit warnings"
  },
  "preservation": {
    "preserved_files": 417,
    "unchanged": true,
    "changed": []
  },
  "preservation_command": ".\\.venv\\Scripts\\python.exe .local-test-deps/dimensionality-02/preserve.py after",
  "scope_check": {
    "git_diff_check": "PASS",
    "existing_files_changed": false,
    "publication_or_commit": false
  },
  "visual_review": {
    "status": "PASS",
    "figure": "lda_projection.png",
    "shared_axes": true,
    "raw_email_content": false
  },
  "reload_checks": {
    "SVD": "16 fixed stratified validation rows for saved full text pipeline",
    "embedding_candidates": "All 6,709 Validation rows: saved model metrics identical after reload"
  },
  "heldout_final_test_opened": false,
  "model_or_dependency_downloaded": false,
  "gemini_called": false
}
```
Source/config/test mới: config/v21/experiments/dimensionality_reduction_02.json; src/vietmailguard/v21/dimensionality_experiment.py; scripts/v21/dimensionality_reduction_02.py; tests/test_v21_dimensionality_experiment.py; DIMENSIONALITY_REDUCTION_EXPERIMENT_02.md.
Các harness/receipt local nằm trong .local-test-deps/dimensionality-02/ (Git ignored). Không sửa source/UI/runtime cũ hoặc các file Experiment 01. Không commit/push.
