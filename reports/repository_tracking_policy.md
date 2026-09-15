# Chính sách Git tracking cho VietMailGuard

Ngày cập nhật: 2026-09-15  
Phạm vi: Git tracking policy và release preparation  
Ngưỡng mặc định cho artifact được track trực tiếp: 10 MB

## 1. Nguyên tắc

- Source, config, tests và scientific artifacts nhỏ có giá trị tái lập được phép nằm trên `main`.
- Dataset có nội dung email, cache, virtual environment và generated artifacts lớn tiếp tục được giữ ngoài Git.
- Production/candidate `.joblib` không nằm trực tiếp trong Git history, bất kể file hiện nhỏ hơn 10 MB.
- Production model V2 chỉ được phân phối dưới dạng Release asset có version, metadata và SHA-256.
- Exception trong `.gitignore` dùng allowlist theo từng file. Việc mở một thư mục con không tự động mở toàn bộ nội dung của thư mục đó.
- Task này không stage, commit, push, xóa local dataset hoặc thay đổi trạng thái index của dataset đã tracked.

## 2. Newly trackable files

Tất cả file dưới đây tồn tại và nhỏ hơn 10 MB:

| Path | Size (bytes) | Purpose |
|---|---:|---|
| `models/v2_bilingual/model_metadata.json` | 12,567 | Model version, architecture, limitations, checksum và frozen metrics |
| `results/v2_bilingual/final_test_metrics.json` | 15,757 | Frozen one-shot held-out metrics |
| `results/v2_bilingual/final_test_classification_report.csv` | 3,335 | Per-class held-out metrics |
| `results/v2_bilingual/final_confusion_matrix.csv` | 88 | Held-out confusion matrix |
| `results/v2_bilingual/tfidf_model_comparison.csv` | 8,675 | TF-IDF validation comparison |
| `results/v2_bilingual/multilingual_embedding_validation.csv` | 3,943 | Multilingual embedding validation comparison |
| `reports/v2/model_selection_rationale.md` | 3,991 | Selection rationale viết trước final test |
| `reports/v2/final_model_evaluation.md` | 3,928 | Final evaluation methodology và limitations |
| `reports/v2/version2_completion_report.md` | 15,701 | Final scientific/engineering audit |
| `reports/repository_tracking_policy.md` | generated | Chính sách tracking hiện tại |

`.gitignore` mở đúng các file trên. Các file chưa được stage; `git status` hiển thị thư mục untracked ở chế độ mặc định và hiển thị từng file khi dùng `--untracked-files=all`.

## 3. Still ignored

Các nhóm sau tiếp tục bị ignore:

- `.venv/`, `venv/`;
- mọi `__pycache__/`, `*.pyc`, `*.egg-info/`;
- `.pytest_cache/`, editor state và local test dependencies;
- `data/raw/*`;
- `data/processed/*`;
- `data/splits/*`;
- `data/cache/*`;
- `models/v2_bilingual/production_pipeline.joblib`;
- mọi file dưới `models/v2_bilingual/candidates/`, bao gồm candidate `.joblib`;
- legacy/intermediate model binaries dưới `models/`;
- mọi V2 result/report không có tên trong allowlist;
- report chứa body/subject excerpt, review pool hoặc row-level email data;
- temporary generated artifacts.

Các kiểm tra đại diện xác nhận production pipeline, candidate pipeline, raw corpus, embedding cache, bilingual master và V2 train split vẫn bị Git ignore.

## 4. Release-only files

| Path | Size (bytes) | Policy |
|---|---:|---|
| `models/v2_bilingual/production_pipeline.joblib` | 2,199,876 | V2 Release asset; không track trực tiếp |
| `models/production_pipeline.joblib` | 7,895,127 | Legacy V1 Release asset riêng; không track trực tiếp |
| `reports/v2/cross_language_overlap.csv` | 7,061,615 | Scientific supplement tùy chọn sau content/license review |

Ngưỡng 10 MB là trần mặc định, không phải điều kiện đủ để track. Binary model tiếp tục là `RELEASE_ONLY` dù nhỏ hơn ngưỡng. Mọi artifact vượt 10 MB phải được giữ ngoài main và đánh giá cho Release hoặc local storage.

Release model phải kèm:

- `models/v2_bilingual/model_metadata.json`;
- SHA-256 của artifact;
- tag/version khớp source;
- dependency compatibility;
- model card và scientific limitations;
- cảnh báo chỉ load `joblib` từ release đáng tin cậy.

## 5. Large tracked files requiring review

Git index hiện có một file vượt ngưỡng 10 MB:

| Tracked path | Size (bytes) | Status | Required review |
|---|---:|---|---|
| `data/curated/v2/Vietnamese_Curated_Dataset.csv` | 35,380,746 | Already tracked, unchanged | Dataset license, privacy, redistribution và Git-history exposure |

Các tracked corpus-derived files nhỏ hơn 10 MB vẫn cần license/privacy review:

- `data/curated/spam_manual_labels.csv`;
- `data/curated/v2/vietnamese_positive_review.csv`;
- `data/evaluation/v2/vietnamese_robustness.csv`.

Task này không xóa các file local, không chạy `git rm --cached`, không rewrite history và không thay đổi index. Trước khi repository public, cần audit cả Git history; một commit xóa file trong tương lai không xóa nội dung khỏi commit cũ.

`data/evaluation/v2/short_form_challenge.csv` tiếp tục được track vì đây là controlled synthetic challenge set có provenance rõ, không phải native/raw email corpus.

## 6. `.gitignore` implementation

Policy dùng thứ tự sau cho mỗi vùng:

1. ignore toàn bộ nội dung generated;
2. unignore đúng thư mục con cần đi qua;
3. re-ignore toàn bộ nội dung trong thư mục con;
4. unignore từng approved file.

Ví dụ cho model:

```gitignore
models/*
!models/.gitkeep
!models/v2_bilingual/
models/v2_bilingual/*
!models/v2_bilingual/model_metadata.json
```

Cấu trúc tương tự được dùng cho selected files dưới `results/v2_bilingual/` và `reports/v2/`. Vì vậy việc thêm file mới vào các thư mục này không tự động làm file đó trackable.

## 7. Verification

Đã kiểm tra:

- 9/9 artifact được yêu cầu đều tồn tại;
- 9/9 artifact đều nhỏ hơn 10 MB;
- 9/9 artifact không còn bị ignore;
- production V2 `.joblib` vẫn bị ignore;
- candidate `.joblib` vẫn bị ignore;
- raw, processed, split và cache artifacts đại diện vẫn bị ignore;
- không có file nào được stage;
- không commit hoặc push.

Các lệnh kiểm tra:

```text
git check-ignore -v -- <path>
git ls-files
git status --short
git status --short --untracked-files=all
```

## 8. Kết luận

Repository hiện có allowlist hẹp cho metadata, frozen metrics và scientific reports quan trọng. Production/candidate model binaries và generated datasets vẫn không được đưa lên main bằng một thao tác Git thông thường.

Trước khi public repository, hai việc vẫn bắt buộc:

1. giải quyết tracked curated dataset và lịch sử chứa corpus-derived content;
2. bổ sung repository/dataset license documentation.
