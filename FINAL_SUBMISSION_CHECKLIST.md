# VietMailGuard V2.2 — Final Submission Readiness

Kiểm tra ngày **21/11/2026**, branch `v22/large-scale-dataset`. Không training, thí nghiệm mới, Final Evaluation, Gemini live, commit hoặc push.

**Kết luận:** kết quả nghiên cứu và demo local có thể dùng cho đồ án theo [hướng dẫn riêng](V22_SUBMISSION_GUIDE.md). **Chưa sẵn sàng công khai nguyên repository hoặc gửi nguyên backup chứa dữ liệu.** Cài đặt trên máy mới và browser E2E trên máy nhận chưa được xác minh trong nhiệm vụ này.

## 1. Trạng thái bàn giao

| Hạng mục | Trạng thái | Bằng chứng / điều kiện |
| --- | --- | --- |
| Báo cáo V2.2, bảng số liệu và hình | **READY** | [Báo cáo](reports/v22/V22_FINAL_RESEARCH_REPORT.md), [bảng](reports/v22/V22_RESEARCH_RESULTS.md); khớp metrics/manifest/predictions đã lưu |
| Integrity artifact nghiên cứu | **PASS** | 859 file lịch sử và snapshot 273 file được đối chiếu; không thay model/config/scientific results |
| Dependency môi trường local | **PASS** | Python 3.13.14; core lock 63/63 khớp; `pip check`: no broken requirements |
| Chạy app/default V2/chuyển V2.1 ML/Hybrid/Back | **PASS — AppTest** | Entry thật, database demo riêng; không traceback, 0 external-network attempt, 0 Gemini constructor, 0 Final Test record read |
| Kiểm tra launcher/UI/runtime | **PASS** | 30 tests, 0 failed/skipped sau thiết lập UI import path |
| Demo trực tiếp model V2.2 trong Streamlit | **NOT IMPLEMENTED** | Runtime giữ V2/V2.1; V2.2 trình bày bằng báo cáo/artifact, không giả đổi nhãn phiên bản |
| Clean-install trên máy người nhận | **NOT VERIFIED** | Chưa tạo venv/cài dependency mới; cần wheelhouse hoặc nguồn tải được chấp thuận và model local phù hợp |
| Browser E2E trên máy người nhận | **NOT VERIFIED** | AppTest không thay thế kiểm tra bằng trình duyệt thật |
| Source snapshot có thể xem xét đưa vào repo | **CONDITIONAL** | 112 file nhóm A, vẫn cần owner/third-party rights và quyết định công bố phù hợp |
| Artifacts nghiên cứu lưu riêng | **LOCAL** | 88 file nhóm B; không tự phân phối model/pretrained bundle chỉ vì đã backup |
| Dữ liệu/lineage riêng tư | **DO NOT DISTRIBUTE** | 73 file nhóm C; không gửi cùng source-only hoặc lên GitHub |
| Template secret | **FIXED** | Để trống `GEMINI_API_KEY` trong `.env.example`; `.env` và credential thực tế giữ nguyên |
| Công khai toàn repository/history | **BLOCKED** | Bốn CSV V2 chứa các trường email vẫn tracked; quyền/privacy/history chưa giải quyết |
| Production / Final Evaluation / publication | **NOT APPROVED** | Không suy từ kết quả pilot hoặc từ test phần mềm |

## 2. README, cài đặt và chạy demo

- [x] Đã rà soát README, ENVIRONMENT, hướng dẫn V2.1, `run.py`, dependency declarations và wiring ứng dụng.
- [x] Thêm mục V2.2 ở đầu README, liên kết [V22_SUBMISSION_GUIDE.md](V22_SUBMISSION_GUIDE.md); giữ nội dung lịch sử V2/V2.1 và toàn bộ runtime/model nguyên trạng.
- [x] Hướng dẫn mới không yêu cầu audit, training, seed dữ liệu thật hoặc finalization để trình diễn.
- [x] Dùng interpreter `.venv\Scripts\python.exe`, địa chỉ `127.0.0.1`, database `submission_demo.sqlite3` riêng; không đưa mailbox cá nhân vào demo/bộ nộp.
- [x] ML không cần Gemini. Hybrid demo chọn **MOCK OFFLINE**, không bật consent/send. Biến HF offline không được mô tả sai thành firewall cho mọi dịch vụ.
- [x] Nêu rõ model artifact/dependency không tự có trong clone hoặc archive 273 file; không thay model khác hoặc tạo prediction giả khi thiếu artifact.
- [ ] Trên máy mới: hoàn tất cài đặt độc lập, kiểm tra hash model, `pip check`, test và trình duyệt. Không dùng PASS của môi trường local hoặc Clean Export cũ như bằng chứng cho máy nhận.

Lần chạy pytest độc lập đầu tiên có **1 failed/29 passed**: thiếu `app` trên import path nên không tìm được `i18n`. Sau khi đặt `PYTHONPATH=src;app`, cùng test selection đạt **30 passed trong 11,85 s**, không bỏ test/xfail hoặc sửa runtime. Hướng dẫn mới ghi rõ setup này. Chưa sửa `pyproject.toml` hoặc các test V2/V2.1 đã đóng băng.

## 3. Kết quả và tài liệu nghiên cứu

| Cấu hình — cùng Validation 1.938 | Macro F1 |
| --- | ---: |
| TF-IDF + LinearSVC Supervised | **0.974976** |
| MiniLM + MLP Supervised | 0.949472 |
| TF-IDF + LinearSVC SSL | 0.957240 |
| MiniLM + MLP SSL | 0.917176 |

- [x] Cohort 9.686, Train 7.748, Validation 1.938; SSL dùng **952** pseudo, weight **0,1**, 0 Verified mới.
- [x] Dùng run SSL `exploratory_ssl_pilot_01_token_dedup`; không lấy run 954 superseded vì điểm MLP cao hơn.
- [x] Đối chiếu Accuracy/Macro/Weighted/per-class metrics, confusion matrices và nhãn/split từ artifacts thật; không chạy lại training.
- [x] Link biểu đồ trong project hoạt động; PNG/SVG chứa số liệu tổng hợp, không có nội dung thư.
- [x] Nêu rõ 33 Spam/13 nhóm, precision pseudo chưa biết, 342+20 cảnh báo trong ledger, thiếu benchmark native VI/ZH và không dùng Final Test cũ.
- [x] Phân biệt giả thuyết SSL giảm hiệu năng với nguyên nhân đã chứng minh; không kết luận SSL nói chung kém hiệu quả.

**Đóng gói tài liệu:** bản `report_materials` trong backup cũ không phải một cây project độc lập; copy riêng Markdown có thể làm hỏng relative image links. Bản tài liệu nộp mới phải giữ cấu trúc `reports/v22/` và `results/.../figures/` như project, hoặc sửa link trong bản copy. Không sửa report gốc đã niêm phong. Thư mục tài liệu mới `VietMailGuard_V22_Submission_Documents_20261121/` nằm cùng cấp project và không chứa email/model/credential.

## 4. Phân loại đúng 273 file ngoài Git

| Nhóm | Số file | Xử lý |
| --- | ---: | --- |
| **A — Source/config/docs candidate** | **112** | Code, cấu hình và tài liệu tổng hợp để xem xét version control; không phải quyền công bố đã được tự phê duyệt |
| **B — Research artifacts** | **88** | Models, pretrained encoder/tokenizer, aggregate results và receipts; lưu riêng, kiểm tra phạm vi phân phối trước khi chia sẻ |
| **C — Local only** | **73** | Cohort/pseudo CSV, audit cache, row/group lineage, private manifests hoặc nội dung cần rà soát; không public |
| **Tổng** | **273** | Không đổi category thành Approved hoặc xóa file gốc |

Danh sách từng đường dẫn: [SUBMISSION_FILE_CLASSIFICATION.md](SUBMISSION_FILE_CLASSIFICATION.md). Bản máy đọc `results/v22_multilingual/submission_readiness_01/file_classification_273.json` giữ SHA-256 và lý do.

Phạm vi này là **273 entry của ZIP niêm phong**, không phải toàn source app và không gồm mọi tài liệu tạo sau ZIP. Snapshot thiếu toàn bộ `app/`, `run.py`, README và một số tài liệu/dependency ngoài phạm vi research. Không dùng nó như một ứng dụng source-only đầy đủ hoặc một Git repository sạch.

Các nhóm chỉ là phân loại bàn giao. Không tạo phê duyệt owner, quyền tái phân phối email hoặc giấy phép mới. Các script `.local-test-deps` thuộc nhóm A vẫn cần chuẩn hóa dependency/layout nếu được chọn đưa vào repo, thay vì force-add cả thư mục local.

## 5. Secrets, .gitignore và dữ liệu email

### Kết quả quét thực tế

Gitleaks **8.30.1** do người dùng cung cấp; staging ngoài Git, ACL chỉ tài khoản local hiện tại và SYSTEM. Quét `gitleaks dir`, `--redact=100`, không allowlist rộng hoặc bỏ qua `gitleaks:allow`. Không in key/email vào terminal hay báo cáo.

| Phạm vi | Scanner raw findings | Kết luận theo bằng chứng |
| --- | ---: | --- |
| Snapshot đầy đủ 273 entry | **42** | 34 checksum tokenizer/model-support khớp file thật; 4 mô tả encoder; **4 occurrences trong 2 CSV pseudo chưa xác minh**, giữ local |
| 112 source candidates của snapshot | **4** | Cùng các mô tả normalization/dimension encoder; không phải credential theo đối chiếu nội dung/config |
| Source hiện tại trước fix | **5** | 3 checksum, 1 mô tả encoder, **1 giá trị cấu hình nhạy cảm trong `.env.example`** |
| Source sau fix template | **4** | Chỉ còn checksum/mô tả encoder nêu trên; raw scanner exit=1 vẫn được lưu, không giả báo 0 findings |

`.env.example` trước sửa chứa giá trị trùng khóa đang cấu hình trong `.env` local. Đã xóa **giá trị ở template**, không đổi/rotate credential thực tế và không gọi API xác minh tính hoạt động. `.env.example` hiện untracked ở branch; không đủ căn cứ kết luận file chưa từng được gửi hoặc xuất hiện trong lịch sử. Nếu từng chia sẻ file mẫu có khóa, chủ dự án cần kiểm tra exposure và thu hồi/rotate bằng kênh quản lý phù hợp.

Bốn finding trong dataset là **occurrences**, không chứng minh bốn credential còn hoạt động hoặc bốn email độc lập. Không tự gán false positive, không gọi URL/API để xác minh; các CSV vốn thuộc nhóm C vẫn giữ local. Chi tiết scanner có thể chứa ngữ cảnh dữ liệu, được giữ **ngoài payload**, không gửi kèm tài liệu nộp.

Giới hạn quét: file cap **32 MiB** loại riêng cache SQLite và encoder binary lớn; detector không bảo đảm mọi nội dung binary/unknown được quét. Không quét lại Git history và không tuyên bố không thể còn secret. Source gate có bằng chứng đối chiếu; **full backup không được xem là gói public sạch**.

### Git và quyền phân phối

- [x] `.env`, mailbox DB/WAL trong runtime, `data/v22`, model/results V2.2, local cache/checkpoint, cấu hình `.local.*` và EML được ignore. `.env.example` là ngoại lệ cố ý, vì vậy phải không chứa key.
- [x] `.gitignore` giữ nguyên để bảo toàn trạng thái trước; không làm mất ngoại lệ metadata V2.
- [ ] Bốn CSV đã tracked: `data/curated/v2/Vietnamese_Curated_Dataset.csv`, `data/curated/v2/vietnamese_positive_review.csv`, `data/evaluation/v2/short_form_challenge.csv`, `data/evaluation/v2/vietnamese_robustness.csv`. Kiểm tra header xác nhận có các trường email/original/transformed body; không đọc records để phân tích lại lỗi/mô hình.
- [x] `data/curated/spam_manual_labels.csv` là file metadata review riêng, không schema chứa body; không đồng nhất với bốn CSV nội dung thư.
- [ ] `.gitignore` không xóa file tracked hoặc lịch sử public. Privacy/redistribution/history của repository cũ còn chưa giải quyết; **không gửi nguyên clone/history như một source-only sạch**.
- [x] Không untrack, xóa/sửa V2, rewrite, commit, push hoặc tự phê duyệt publication.

## 6. Còn thiếu trước bàn giao khác máy hoặc publication

1. Xác nhận phạm vi chia sẻ của nhóm A/B; nhóm C, `.env` và mailbox cá nhân giữ local. Nếu cần artifact cho giảng viên, phải có quyền và phạm vi chia sẻ cụ thể; không gửi cả backup 273 file.
2. Chuẩn bị dependency/wheelhouse và model được phép cung cấp; clean-install + browser walkthrough trên máy nhận chưa thực hiện. Không tải thêm hoặc thay phiên bản/model trong task này.
3. Nếu yêu cầu đồ án có nút demo **model V2.2 trực tiếp trong app**, đây là phần chưa tích hợp; cần nhiệm vụ riêng, không gọi model V2.1 là V2.2.
4. Giải quyết riêng tracked email/history và kiểm tra exposure của template key nếu đã chia sẻ, trước bất kỳ công bố GitHub. Nhiệm vụ này không đưa ra quyết định history hoặc distribution thay chủ dự án.

**LOCAL_RESEARCH_DEMO_READY: YES theo phạm vi đã kiểm tra.**  
**NEW_MACHINE_HANDOFF_VERIFIED: NO.**  
**WHOLE_REPOSITORY_PUBLIC_READY: NO.**  
**NEW_REPO_PUBLICATION_APPROVED: NO.**  
**OLD_REPOSITORY_PRIVACY_RESOLVED: NO.**

Các file thay đổi: README (chỉ thêm điều hướng V2.2), `.env.example` (xóa giá trị key), checklist này, hướng dẫn V2.2 và bảng phân loại. Bằng chứng mới tại `results/v22_multilingual/submission_readiness_01/`; helper một lần tại `.local-test-deps/v22/submission-readiness-01/`. Không thay model/runtime, dữ liệu hay số liệu nghiên cứu đã niêm phong.
