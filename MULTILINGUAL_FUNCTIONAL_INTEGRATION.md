# VietMailGuard V2.1 — Multilingual Functional Integration

Ngày kiểm chứng: 2026-10-29. Nhánh local: `v21/multilingual-functional-integration`.

**MULTILINGUAL_FUNCTIONAL_COMPLETE=YES** — EN/VI/ZH đã chạy xuyên suốt bằng
encoder MiniLM và MLP thật, trên email giả, với giới hạn hiển thị trong ứng dụng.
Đây là hoàn thành chức năng experimental, **không phải** xác minh độ chính xác.

- VALIDATED_MULTILINGUAL_ACCURACY=NO
- MULTILINGUAL_PRODUCTION_READY=NO
- FINAL_EVALUATION_PERFORMED=NO
- NEW_REPO_PUBLICATION_APPROVED=NO
- OLD_REPOSITORY_PRIVACY_RESOLVED=NO

## Chức năng đã triển khai

Sidebar vẫn mặc định **V2 frozen production**. Các lựa chọn V2.1:

| Lựa chọn | Classifier thực sự chạy | Confidence |
|---|---|---|
| V2.1 Traditional ML | TF-IDF word/character + LinearSVC hiện có | Không có, chưa calibrated |
| V2.1 Multilingual Experimental | MiniLM 384 chiều → StandardScaler đã fit trên Train → MLP 128/64 → ba lớp | Không hiển thị xác suất MLP chưa calibrated |
| V2.1 Hybrid | Người dùng chọn một trong hai classifier trên; Reviewer là bước riêng | Không thay confidence/nhãn của ML |

Không ensemble, bỏ phiếu, fallback sang classifier khác hay dùng V2 giả làm V2.1.
Chuyển classifier nền của Hybrid làm ẩn kết quả Quick Analyzer không còn phù hợp
cho tới khi bấm Analyze mới; không thay đổi bản phân tích đã lưu.

### Inference theo ngôn ngữ

EN/VI/ZH đều sử dụng cùng một nhánh MiniLM + MLP, không dịch ngầm:

```text
Email thủ công hoặc MIME đã parse
  → subject/body giới hạn 4.096 ký tự mỗi trường, giống Experiment 01
  → EmailTextComposer (subject + newline + body)
  → MiniLM local, revision cố định, tối đa 128 token, L2-normalized float32
  → StandardScaler đã fit + MLP đã fit
  → normal / spam / phishing
```

Sender, URL, luật bảo mật và Risk Score được xử lý từ **email gốc đầy đủ** qua
inference/security pipeline dùng chung. Email gốc trong SQLite không bị dịch hay
cắt ngắn. Cả giới hạn ký tự lẫn token đều được ghi trong kết quả/provenance.
Không tạo đóng góp TF-IDF giả cho MLP; phần giải thích nêu rõ không hỗ trợ.

Traditional ML vẫn dùng nguyên pipeline, preprocessing và artifact trước nhiệm vụ.
Không triển khai Trung→Anh→LinearSVC: cache/config dịch đã kiểm tra chỉ có
`Helsinki-NLP/opus-mt-en-vi`, không có translator Trung→Anh phù hợp. Không tải mới.

### Artifact, hash và dependency

Runtime mới đọc `config/v21/runtime_multilingual.json`, kiểm tra SHA-256 trước
deserialize, kiểm tra provenance Experiment 01 và contract của head/encoder:

- Head: `models/v21_multilingual/experiments/multilingual_dl_01/C_minilm_mlp.joblib`.
- SHA-256: `4594b919c56218a8bae1eea649a950bf4e21b4fafe126d487a653d9f85c453ff`.
- Receipt: `results/v21_multilingual/experiments/multilingual_dl_01/model_metadata.json`.
- Training config: `config/v21/experiments/multilingual_dl_01.json`.
- Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Revision: `e04d38a7d68c60a7a95390045400a555127ab033`.
- Chín file weight/tokenizer/config của encoder được đối chiếu hash với manifest
  trong cấu hình runtime mới. Không ghi đường dẫn cá nhân vào cấu hình.
- Thứ tự class thực tế: `normal, phishing, spam`; đối chiếu với metadata head,
  không giả định theo thứ tự hiển thị Normal/Spam/Phishing.

Kiểm tra thư viện tùy chọn trước khi cho chọn chế độ. Thiếu file, dependency,
encoder revision hoặc sai hash thì chặn kích hoạt và hiển thị nguyên nhân an toàn.
Không giả lập prediction, không tự cài dependency hoặc tải model. Preflight không
nạp Torch/encoder; encoder chỉ nạp khi yêu cầu inference thật, được cache và dùng
CPU. Từng lần encode được khóa để tránh chạy chồng trên cùng encoder.

Môi trường thực tế: Python 3.13.14; torch 2.8.0+cu128; sentence-transformers 5.7.0;
scikit-learn 1.9.1; numpy 2.5.3; joblib 1.6.0; Streamlit 1.63.0. Không kiểm chứng
lại tổ hợp Python/dependency khác trong nhiệm vụ này.

## Mailbox, persistence và Reviewer

`Streamlit → MailService → inference / MailRepository → SQLite` được giữ nguyên.
Không đổi schema, routing, feedback hay code V2 inference.

- Import EML hoặc nhận thư mới chạy đúng classifier được chọn một lần.
- Save từ Quick Analyzer lưu chính kết quả đã phân tích, không chạy lại model.
- JSON phân tích chứa version, candidate ID, artifact SHA, encoder ID/revision,
  training-config SHA, encoder-file hashes và preprocessing.
- Mở thư hoặc đổi model không chạy lại ML và không ghi đè lịch sử.
- Routing theo nhãn ML: normal→Inbox, spam→Spam, phishing→Quarantine.
  Người dùng vẫn có thể chuyển thư; không sửa prediction/confidence/risk gốc.
- Hybrid dùng nguyên Reviewer hiện có; mặc định mock offline, chỉ chạy sau bấm nút.
  Gemini yêu cầu chọn provider, đồng ý gửi sender/subject/body, rồi bấm Review.
  Không có key thì không chạy, kể cả khi checkbox consent đã bật.
- Nhận xét Reviewer được lưu riêng, không đổi nhãn, Risk Score hay routing ML.
  Các test timeout/API error/invalid JSON/consent hiện có vẫn PASS.
- Không thực hiện Gemini live hoặc xác minh quyền truy cập model Gemini từ xa.

Đã sửa lỗi vòng đời widget: classifier nền Hybrid được giữ trong state độc lập
`hybrid_classifier`; chuyển Quick Analyzer ↔ mailbox không làm nó quay về LinearSVC.

## Kiểm thử thực tế

Tất cả email đầu vào mới là synthetic với domain reserved. Đây là kiểm thử chức
năng; không tính accuracy hoặc biến các ví dụ thành benchmark.

### Pytest

Các lệnh đã chạy từ project gốc bằng môi trường có sẵn:

```powershell
.\.venv\Scripts\python.exe -u .local-test-deps/gemini-reviewer/run_offline_tests.py tests/test_v21_multilingual_runtime.py tests/test_v21_runtime_integration.py tests/test_v21_runtime_contract.py tests/test_v21_application_qa.py -q -x
.\.venv\Scripts\python.exe -u .local-test-deps/gemini-reviewer/run_offline_tests.py tests/test_v21_multilingual_runtime.py tests/test_v21_email_input.py -q -x
.\.venv\Scripts\python.exe -u .local-test-deps/gemini-reviewer/run_offline_tests.py --include-dashboard-page-test -q -rs
git diff --check
```

- Nhóm integration ban đầu: **32 passed**.
- Nhóm runtime + email input sau sửa navigation: **19 passed**; sau đó bổ sung
  test từ chối label-order/revision sai vào regression cuối.
- Regression cuối: **657 passed, 3 skipped, 3 deselected**, 47,29 giây.
- Một warning đã có: tiny synthetic MLP test chỉ chạy 5 iteration chưa hội tụ;
  không phải huấn luyện lại candidate thực tế.
- `git diff --check`: **PASS** (Git chỉ nhắc LF→CRLF theo cấu hình local).

Ba skipped là integration TREC ZIP thật, curation local và TREC review UI yêu cầu
opt-in. Ba deselected là những test cũ đọc các dòng held-out V2:

1. `test_split_builder.py::test_generated_repository_splits_have_disjoint_groups`.
2. `test_v2_completion_integrity.py::test_actual_v2_splits_preserve_eligibility_provenance_and_origin`.
3. `test_dashboard_data.py::test_dataset_dashboard_uses_only_eligible_v2_splits`.

Đây là regression có guard, gồm test artifact V2/V2.1 có sẵn, **không phải** toàn
bộ pytest không lọc. Không sửa/xfail các test bị loại. Guard chặn đọc Test và
kết nối ngoài loopback; receipt ghi nhận **0 lần truy cập bị thử**. Database test
được cô lập; không dùng mailbox cá nhân. Một lần chạy đầu thất bại ở 6 test AST
do harness chưa đưa helper mới vào scope; đã sửa harness, giữ nguyên assertions,
chạy lại 8 email-input test PASS và regression cuối PASS.

### Browser E2E thật

Playwright core có sẵn + **Microsoft Edge 156.0.4314.50, headful**, không dùng
AppTest/health endpoint thay cho E2E. Server kiểm thử chỉ cho loopback, bỏ key khỏi
môi trường tiến trình và dùng SQLite riêng. **20 tình huống PASS**:

| Nhóm quan sát trực tiếp | Kết quả |
|---|---|
| Phiên mới mặc định V2, không có prediction trước Analyze | PASS |
| EN/VI/ZH: MiniLM thật → Analyze → Save → Open | PASS ×3 |
| Chuyển giao diện Việt/Anh giữ bản nháp và kết quả | PASS |
| EN/VI/ZH qua Traditional ML thật | PASS ×3 |
| Hybrid MiniLM mock chỉ chạy khi bấm, ML không đổi | PASS |
| ML và Reviewer được lưu/mở lại riêng | PASS |
| Spam và Quarantine vẫn đọc được nội dung gốc, có cảnh báo | PASS ×2 |
| Classifier nền Hybrid giữ nguyên sau chuyển trang | PASS |
| Không key: Gemini vẫn bị chặn khi đã consent | PASS |
| Upload EML EN/VI/ZH → MLP thật → Save → Open | PASS ×3 |
| HTML thành plain text, không script/remote image | PASS |
| EML rỗng: lỗi an toàn, không giữ prediction cũ | PASS |
| Inject thiếu encoder: đúng thông báo, ẩn chế độ experimental, V2/Traditional còn hoạt động | PASS |

Tình huống thiếu encoder dùng failure injection ở preflight, không di chuyển hoặc
đổi tên artifact. Hash sai/thiếu dependency/revision sai/label order sai và output
embedding không hợp lệ được kiểm tra bằng pytest. Browser ghi **0 request ngoài
loopback**. Log/ảnh/input giả nằm local trong `.local-test-deps/multilingual-integration/`.
Các lần thử harness ban đầu gặp locator của icon/checkbox và thời điểm dropdown;
đã dùng locator theo nhãn và thao tác keyboard hợp lệ, không force-click hay bỏ
kiểm tra consent. Lỗi Hybrid reset model thực tế đã được sửa và có regression.

Lệnh E2E local đã chạy:

```powershell
.\.venv\Scripts\python.exe -u .local-test-deps/multilingual-integration/serve.py 8528
node .local-test-deps/multilingual-integration/browser.cjs
.\.venv\Scripts\python.exe -u .local-test-deps/multilingual-integration/serve.py 8529 missing-multilingual
node .local-test-deps/multilingual-integration/browser_unavailable.cjs
```

Các harness `.local-test-deps` tái sử dụng công cụ đã có trên máy và không phải
dependency/runtime bắt buộc của ứng dụng.

## Bảo toàn và file thay đổi

Đã tạo nhánh phát triển riêng, giữ nguyên các file untracked của Experiment 01/02
và `.env.example`. SHA-256 trước/sau của **221 file được bảo vệ đều không đổi**:
models/results hiện có, reports V2, cấu hình V2 và Traditional runtime, inference
V2, cùng Train/Validation/manifest development. Không mở file Final Test để hash
hay đánh giá, không huấn luyện lại, không cập nhật artifact/metrics đã đóng băng.
Các unit test có fit fixture giả nhỏ không tác động model nghiên cứu.

File chức năng được tạo:

- `src/vietmailguard/v21/multilingual_runtime.py`.
- `config/v21/runtime_multilingual.json`.
- `tests/test_v21_multilingual_runtime.py`.

File được sửa:

- `src/vietmailguard/v21/runtime_modes.py`: chọn classifier riêng, fail closed.
- `app/app.py`: lựa chọn experimental, Hybrid base và thông báo preflight.
- `app/views/email_analyzer.py`: dùng đúng engine, state kết quả theo classifier.
- `app/views/mail_client.py`: nhận/import bằng mode đã chọn, đọc history nguyên vẹn.
- `config/translations.json`: presentation EN/VI và giới hạn experimental.
- `tests/test_v21_application_qa.py`: trường hợp cả hai artifact không khả dụng.
- `tests/test_v21_email_input.py`: harness AST nạp helper state mới.

Tài liệu bàn giao duy nhất mới: `MULTILINGUAL_FUNCTIONAL_INTEGRATION.md`.
Harness/receipt/ảnh/SQLite test là file local ignored, không phải dataset hay
model mới. Không sửa Clean Export, không commit/push/publication.

## Khởi động và trình diễn local

Tại thư mục project gốc, với `.venv` và cache/artifact đã có:

Nếu Streamlit đang chạy từ trước cập nhật, dừng bằng Ctrl+C rồi khởi động lại
để nạp các module và lựa chọn model mới trong cùng một phiên bản mã nguồn.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

1. Chọn **V2.1 Multilingual Experimental — MiniLM + MLP** ở sidebar.
2. Mở Quick Analyzer, nhập sender/subject/body giả hoặc upload `.eml`, bấm Analyze.
3. Kiểm tra tên MiniLM + MLP, ngôn ngữ, status experimental/unvalidated và không
   có confidence %. Bấm Save rồi Open để đọc bản phân tích đã lưu.
4. Chọn Hybrid, chọn MiniLM + MLP làm classifier nền, giữ provider Offline mock,
   Analyze rồi bấm Run offline mock review để trình diễn không phát sinh phí.
5. Gemini thật: cấu hình `GEMINI_API_KEY` và `GEMINI_MODEL` trong `.env` hoặc
   environment; chọn Google Gemini, đọc disclosure, consent rồi chủ động Review.
   Nhiệm vụ này không chạy API; lựa chọn provider/đổi trang không gửi email.
6. Có thể quay về V2 Production hoặc Traditional ML. Thư đã lưu giữ nguyên model
   provenance cũ. Lần mở email không yêu cầu model cũ còn được chọn.

Máy khác phải được cung cấp các artifact/receipt/config nêu trên và encoder đúng
revision qua quy trình local được cho phép. Dependency tùy chọn được khai báo ở
`requirements-embeddings.txt`; không tự cài/tải khi chế độ bị chặn. Joblib chỉ nạp
artifact tin cậy, không nhận file model tùy ý từ người dùng.

## Giới hạn và đánh giá độc lập còn thiếu

- English: có Development Validation, không thay thế test V2.1 độc lập.
- Vietnamese: dữ liệu development dịch; Validation có 180 mẫu (172 normal,
  8 spam, **0 phishing**). Không có native Vietnamese/phishing recall được xác minh.
- Chinese: encoder có thể xử lý và trả một trong ba lớp, nhưng head không được
  train trên Chinese. **Chưa xác minh phân loại ba lớp thực tế**, kể cả khi ví dụ
  giả cho kết quả có vẻ hợp lý. Không báo accuracy Chinese.
- TREC06C binary và Chinese synthetic không được ghép thành ground truth Phishing,
  không được dùng để chứng minh production readiness; không tích hợp binary vào routing.
- MiniLM 128 token có thể bỏ phần quan trọng ở cuối thư. Security rules dùng toàn
  văn nhưng chưa có bộ luật Chinese chuyên biệt và không thay nhãn ML.
- Saved Macro F1 Development: Traditional **0,985852**, MiniLM + MLP **0,951334**.
  Không tuyên bố Deep Learning tốt hơn baseline; Traditional vẫn được bảo toàn.
- Cần benchmark EN/VI/ZH độc lập, native VI/ZH đủ ba lớp, nhãn/provenance/rights
  được xác minh, dedup/translation grouping với toàn bộ development, protocol và
  cấu hình khóa trước lần đánh giá cuối có phê duyệt riêng. Cần đánh giá calibration,
  per-class/per-language recall, false positives và latency thực tế.

**Functional complete không đồng nghĩa validated accuracy hoặc production ready.**
