# VietMailGuard V2.2 — Hướng dẫn bàn giao và demo local

Ngày kiểm tra: 21/11/2026. Dùng hướng dẫn này cho đồ án; các phần V2/V2.1 trong README và ENVIRONMENT là lịch sử riêng, không phải chuỗi lệnh phải chạy lại.

## 1. Phân biệt hai phần trình diễn

- **Nghiên cứu V2.2:** đọc [báo cáo](reports/v22/V22_FINAL_RESEARCH_REPORT.md), [bảng kết quả](reports/v22/V22_RESEARCH_RESULTS.md) và hình PNG/SVG. So sánh bốn cấu hình trên cùng Validation, không chạy training hoặc Final Test.
- **Ứng dụng:** `python run.py` chạy sản phẩm V2/V2.1 hiện có. Không gọi kết quả trên giao diện là inference V2.2: model V2.2 chưa tích hợp vào runtime. Giao diện hiện mặc định tiếng Việt; ngôn ngữ email EN/VI/ZH là vấn đề khác.
- Local Git clone, archive 273 file và backup nghiên cứu không đồng nghĩa một bộ cài ứng dụng đầy đủ. Archive 273 file không có toàn bộ `app/`, `run.py`, `.env.example`, raw source pool hoặc mọi cache lịch sử.

## 2. Chạy ngay trên máy đã xác minh

Mở PowerShell tại **project gốc có `.venv`, `app/app.py` và model local**. Nếu thư mục được chuyển vị trí, luôn gọi interpreter trực tiếp thay vì launcher `.exe` bên trong venv.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check

# Database riêng cho thư giả; không dùng mailbox cá nhân khi trình diễn.
$env:VIETMAILGUARD_MAIL_DB = Join-Path (Get-Location) 'data\runtime\submission_demo.sqlite3'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
.\.venv\Scripts\python.exe run.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Mở `http://127.0.0.1:8501`. V2 là mặc định. Chọn V2.1 Traditional ML hoặc Multilingual Experimental chỉ khi selector cho phép model đã được xác minh hash. Nếu artifact thiếu, giữ thông báo unavailable; không retrain, không chép một model khác vào để giả làm candidate.

Các biến HF ở trên chỉ ngăn download Hugging Face; **không phải firewall chặn mọi dịch vụ**. Demo không dùng API: chọn Reviewer **MOCK OFFLINE**, không chọn Gemini và không bật consent gửi thư. Gemini là tùy chọn, không bắt buộc cho ML. Không cần API key để trình diễn Mock. Không chép `.env` vào bộ nộp.

## 3. Kịch bản demo ngắn, không có dữ liệu thật

1. Mở Hộp thư đến trong database demo riêng. Chọn chế độ V2.1 ML nếu khả dụng.
2. Dùng **Soạn thư demo**: sender `sender@example.com`, recipient `reader@example.org`, subject `Lịch họp`, body `Cuộc họp dự án sẽ bắt đầu lúc 9 giờ sáng mai.` Bấm phân tích/lưu, mở lại thư và xem folder/analysis đã lưu.
3. Dùng Quick Analyzer với một thông báo EN hoặc ZH tự viết; ZH phải hiện experimental/unvalidated. Đây là kiểm tra chức năng, không benchmark accuracy hoặc nhãn đúng đã được bảo đảm.
4. Có thể nhập EML giả UTF-8 chứa cùng nội dung. Không dùng email thật hoặc attachment. Không mở URL; nếu cần minh họa liên kết, chỉ dùng tên miền reserved như `example.invalid`.
5. Chọn Hybrid và Reviewer **MOCK OFFLINE**. Bấm mock review, chỉ rõ kết quả LLM/mock tách khỏi nhãn ML và không đổi routing. Không diễn giải mock như đánh giá chất lượng Gemini.
6. Quay lại các trang báo cáo V2.2 và giải thích vì sao giữ TF-IDF + LinearSVC Supervised. Không dùng dashboard V2/V2.1 như bảng số liệu V2.2 nếu chưa có phần tích hợp riêng.

Không cam kết một thư giả nhất định phải được mô hình xếp đúng một lớp. Demo sai dự đoán cũng phải trình bày trung thực; không sửa dữ liệu/nhãn/model để làm đẹp kết quả.

## 4. Dependency và model cho máy khác

Môi trường hiện tại: Python **3.13.14**, core lock **63/63** package khớp, `pip check` PASS. Core ML/Streamlit dùng `requirements-lock.txt`; embeddings là stack tùy chọn riêng `requirements-embeddings.txt`; Gemini là tùy chọn `requirements-gemini.txt`. Phiên bản thí nghiệm nằm trong protocol và báo cáo, không chỉ trong core lock.

Chưa thực hiện clean-install trên máy mới trong nhiệm vụ này. Không sao chép nguyên `.venv` sang máy khác hoặc coi core lock là wheelhouse đầy đủ có hash. Dùng một venv mới và nguồn package/wheelhouse được chấp thuận; không tự tải dependency trong lần kiểm tra bàn giao.

Ví dụ cài **offline**, chỉ thực hiện khi đã có wheelhouse tương thích và được phép dùng:

```powershell
py -3.13 -m venv .venv-submission
.\.venv-submission\Scripts\python.exe -m pip install --no-index --find-links '<approved-wheelhouse>' -r requirements-lock.txt
.\.venv-submission\Scripts\python.exe -m pip install --no-index --find-links '<approved-wheelhouse>' .
.\.venv-submission\Scripts\python.exe -m pip check
```

Giữ build isolation chuẩn; wheelhouse phải có build dependencies từ `pyproject.toml` và các dependencies của nó. `<approved-wheelhouse>` là placeholder, không phải một thư mục đã được cung cấp ở project này. Embedding/CUDA dependencies cần bộ wheel tương thích riêng nếu trình diễn MiniLM; không coi lệnh core trên là đủ cho mọi mode.

Model là local artifact tin cậy, không đi kèm Git tự động:

- V2 cần pipeline và `model_metadata.json` tương ứng trong `models/v2_bilingual/`.
- V2.1 Traditional cần artifact/config/Validation receipt được tham chiếu bởi `config/v21/runtime_model.json`.
- V2.1 Multilingual cần artifact, encoder snapshot và hashes trong `config/v21/runtime_multilingual.json`.
- V2.2 được bảo quản riêng dưới `models/v22_multilingual/experiments/`; chỉ dùng làm kết quả nghiên cứu, không đổi config runtime.

Joblib/pickle không là định dạng model an toàn cho nguồn không tin cậy hoặc đảm bảo tương thích mọi phiên bản. Chỉ dùng artifact đã đối chiếu SHA-256 và môi trường phù hợp. Không tải model lớn hoặc chạy lại training để lấp chỗ thiếu.

## 5. Tái lập kết quả mà không training

Hai nguồn metrics chính:

- `results/v22_multilingual/experiments/group_aware_supervised_pilot_01/metrics.json`.
- `results/v22_multilingual/experiments/exploratory_ssl_pilot_01_token_dedup/metrics.json`.

Kiểm tra cohort/pseudo manifests, model SHA, saved predictions và confusion matrices. `results/v22_multilingual/research_consolidation_01/` giữ bảng tổng hợp, seal và verification receipt. Không chạy lại helper tổng kết lên output đã niêm phong chỉ để xem bảng; nó có thể ghi lại file được sinh. Không chạy `finalize_v2_model.py`, các lệnh train hoặc Final Evaluation cho demo.

Tập test demo có thể cần UI import path khi chạy độc lập:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$(Get-Location)\app"
.\.venv\Scripts\python.exe -m pytest tests/test_run_launcher.py tests/test_streamlit_app.py tests/test_streamlit_module_refresh.py tests/test_v21_runtime_contract.py tests/test_v21_runtime_integration.py -q
```

Lần chạy này đạt **30 passed**. Lần đầu chưa đặt import path có **1 failed/29 passed** do `i18n` không tìm thấy; không sửa runtime/model để xử lý lỗi setup test. AppTest entry thật còn kiểm tra default V2, chuyển V2.1 ML/Hybrid/Back, 0 Gemini constructor, 0 external-network attempt, 0 Final Test record read. Đây không phải browser E2E trên máy mới.

## 6. Nộp những gì?

- Tài liệu: báo cáo V2.2, bảng, PNG/SVG và checklist; bản copy standalone cần sửa link ảnh trong **bản copy**, không sửa báo cáo gốc đã niêm phong.
- Code/config/docs: theo [phân loại 273 file](SUBMISSION_FILE_CLASSIFICATION.md), kiểm tra owner/third-party rights còn áp dụng; không `git add .` hoặc lấy cả backup làm source-only release.
- Artifacts model/metrics: lưu riêng, chỉ cung cấp khi có quyết định phạm vi chia sẻ phù hợp. Backup private gồm cả dữ liệu email, không là gói công khai.
- Không nộp `.env`, key, mailbox DB cá nhân, raw email/dataset chưa có quyền phân phối hoặc Git history cũ trong một bộ source-only mới.

Bốn CSV V2 chứa các trường nội dung email vẫn được Git theo dõi. `.gitignore` không xóa file đã tracked hoặc lịch sử. Chưa có quyết định phân phối/history riêng nên **không đưa nguyên repository hoặc lịch sử lên public**. Task này không xóa/sửa V2, untrack, rewrite, commit hay push.
