# VietMailGuard V2.1 — Functional Local Completion

## Kết luận và phạm vi

`FUNCTIONAL_V21_COMPLETE: YES`

Trạng thái này chỉ xác nhận **ứng dụng local Research Preview** hoạt động với
candidate V2.1 thật, trong phạm vi tiếng Anh và tiếng Việt dịch mang tính thử
nghiệm. Không phải phê duyệt V2.1 Production, chất lượng phát hiện phishing
tiếng Trung, publication, hay Final Evaluation.

Thực hiện trong project gốc, trên nhánh `v21/research-preview`; giữ nguyên những
thay đổi chưa commit có sẵn. Không sửa Clean Export, commit/push, tải dependency,
gọi Gemini thật, huấn luyện/refit/calibrate candidate hoặc đánh giá held-out.

## 1. Model thật và hợp đồng inference

| Thành phần | Đã xác minh |
| --- | --- |
| Runtime config | `config/v21/runtime_model.json` |
| Candidate | `models/v21_multilingual/candidates/tfidf_word_character_linear_svm.joblib` |
| Phiên bản/phạm vi | `2.1-development` / `development_candidate_only` |
| Kích thước | 7.899.744 byte |
| SHA-256 artifact | `1050eb9cde01610dda3d2f6a34bdaffda90f8462a08e8648cb7fa9310de28af3` |
| Pipeline | EmailTextComposer → word/character TF-IDF FeatureUnion → LinearSVC |
| Features | Word `(1,2)`: 60.000; `char_wb (3,5)`: 80.000 |
| Classifier | LinearSVC, `C=1.0`, `class_weight=balanced` |
| Thứ tự `classes_` thật | `normal`, `phishing`, `spam` |
| Confidence | Chưa calibrated: `None`, probabilities `{}`; không đổi decision score thành % |

Runtime kiểm tra SHA-256 của artifact, training config và saved Validation
metadata; đối chiếu tham số, text composer, lớp và kích thước feature đã fit.
Artifact thiếu, sai hash hoặc sai hợp đồng sẽ không được sử dụng; không lấy V2
để giả làm V2.1. Cache cũng nhận diện thay đổi config/rules.

Đã sửa sai khác preprocessing: training giới hạn mỗi trường subject/body ở
4.096 ký tự nhưng inference cũ chưa làm vậy. V2.1 giờ áp dụng cùng giới hạn cho
ML và feature explanation; security/URL analyzer vẫn nhận nội dung đầy đủ.
Không refit vectorizer. Smoke trên 11 mẫu Development Train/Validation hợp lệ
(gồm nội dung dài) cho kết quả runtime trùng representation training trực tiếp
11/11. Đây là kiểm tra tương thích, **không phải metric chất lượng mới**.

V2 Production vẫn là chế độ mặc định, sử dụng artifact/inference riêng.
SHA-256 của **103/103 file V2 được snapshot trước công việc** không đổi, bao gồm
models, kết quả/reports, splits/curated/evaluation, config frozen và các module
inference/preprocessing/security/risk được bảo vệ. Test CSV chỉ được stream
byte để kiểm tra hash, không parse nhãn/mẫu hoặc chạy dự đoán.

## 2. Chức năng đã hoàn thành

- Quick Analyzer: nhập thủ công hoặc EML; chỉ nút Analyze kích hoạt inference.
  EML cung cấp trọn sender/subject/body, không trộn với bản nháp nhập trước đó.
  Lỗi parse/empty input xóa kết quả cũ để không hiển thị prediction lỗi thời.
- Bản nháp Unicode giữ khi đổi EN/VI và điều hướng trong cùng session; sửa
  việc gán lại widget mỗi rerun có thể làm mất chỉnh sửa mới. Không hứa giữ
  bản nháp chưa lưu qua restart server.
- Analyze → Save → Open: lưu đúng snapshot ML và nội dung đã phân tích, không
  inference lại; lưu/import lặp mở mail hiện có. Có thể dùng DB demo riêng bằng
  `VIETMAILGUARD_MAIL_DB`; mặc định DB hiện có không đổi.
- Mailbox: Inbox/Spam/Quarantine/Trash/Starred/Security dùng cùng full-width
  detail, mở trực tiếp ID, không phụ thuộc mail còn nằm trong bộ lọc. Back,
  chuyển thư mục, Not Spam, xóa mềm, star và feedback giữ nguyên ML lịch sử.
- Sửa Mark unread: mở lần đầu có thể đánh dấu đã đọc, nhưng rerun detail không
  được tự đánh dấu đã đọc lại sau thao tác Unread của người dùng.
- SQLite: lưu chính xác analysis, provenance artifact và Reviewer riêng;
  không thêm migration trong nhiệm vụ này, không sửa schema để xóa dữ liệu cũ.
- EML: EN/VI/ZH, UTF-8/MIME/multipart/HTML an toàn; body plain/sanitized,
  không execute script, tải ảnh từ xa, mở URL hoặc quét attachment giả.
- Sửa cold-start deep-link có modal “Page not found”: Streamlit đồng thời
  nhận thư mục legacy `app/pages` và `st.navigation`. Chuyển module sang
  `app/views`, giữ nguyên public URL. Không tái tạo `app/pages` vì sẽ kích hoạt
  legacy page discovery trở lại. Các dashboard không được thiết kế lại.
- Dùng favicon SVG local thay Material icon gây request font ngoài khi trình
  duyệt mở ứng dụng. Browser lượt cuối không ghi nhận host ngoài localhost.

Kiến trúc giữ nguyên: Streamlit → MailService → Repository / inference → SQLite.
UI không chứa SQL hoặc bản sao classifier; mở/list mail đọc analysis đã lưu.

## 3. Hybrid và giới hạn ngôn ngữ

Hybrid chọn Reviewer rõ ràng: **offline mock mặc định** hoặc Gemini. Mock chỉ
kiểm thử tích hợp, không phải phân tích ngữ nghĩa thực và không làm nhãn chuẩn.

Gemini cần cấu hình hợp lệ, người dùng chọn provider, đọc disclosure về việc
gửi sender/subject/body cho Google và chi phí có thể phát sinh, tích consent,
rồi chủ động bấm Review. Đổi provider reset consent. Không tạo provider request
chỉ vì render/chọn Hybrid; ML hoạt động khi không có key. Output được validate,
evidence giữ riêng, text Reviewer không render như HTML/Markdown tin cậy.
Timeout, lỗi provider và invalid JSON trả trạng thái lỗi/REVIEW, giữ nguyên ML,
Risk Score và routing. Không gọi Gemini thật trong công việc này.

| Nội dung | Trạng thái đúng |
| --- | --- |
| English | Candidate Development, experimental; không phải Final Test độc lập |
| Vietnamese | Experimental, chủ yếu dữ liệu dịch; chưa benchmark bản địa độc lập |
| Chinese | Có thể chạy kỹ thuật nhưng `unvalidated`; không có bằng chứng phishing ba lớp thực tế |
| Mixed/unknown | Hiển thị giới hạn; language detection không thay nhãn ML |

TREC06C binary và Chinese synthetic không được dùng để tuyên bố Chinese
production hoặc thay routing ba lớp. Model Confidence khác Risk Score; linear
feature contribution không phải phân rã calibrated probability.

## 4. File code/config/test thay đổi trong nhiệm vụ

Không đồng nhất danh sách này với toàn bộ `git status`: worktree đã có nhiều
thay đổi V2.1 từ trước và được giữ nguyên.

- `app/app.py`, `app/mailbox_ui.py`: wiring mode, mailbox và navigation.
- `app/runtime_ui.py` (mới): drafts, DB path, consent/provider/render Reviewer.
- `app/assets/mail.svg` (mới): favicon local tự tạo.
- Di chuyển nguyên cấu trúc `app/pages/` sang `app/views/`: `.gitkeep`,
  `__init__.py`, `dataset_dashboard.py`, `model_dashboard.py`, `methodology.py`,
  `email_analyzer.py`, `mail_client.py`; sửa chức năng hai trang cuối như trên.
- `src/vietmailguard/v21/runtime_inference.py`, `config/v21/runtime_model.json`:
  hợp đồng artifact và preprocessing V2.1.
- `src/vietmailguard/mail_service.py`: `save_analyzed_email(...)`, kiểm tra
  snapshot và lưu kết quả/Reviewer riêng mà không inference lại.
- `config/translations.json`: EN/VI cho luồng save, provider/consent, EML và
  thông báo Research Preview/uncalibrated đúng nghĩa.
- Test mới: `tests/test_v21_runtime_contract.py`,
  `tests/test_v21_saved_analysis.py`, `tests/test_v21_functional_local.py`,
  `tests/test_v21_functional_ui.py`, `tests/test_v21_email_input.py`.
- Test cập nhật: `tests/test_mail_architecture.py`,
  `tests/test_streamlit_app.py`, `tests/test_v21_application_qa.py`,
  `tests/test_v21_bootstrap_development.py`,
  `tests/test_v21_development_dataset.py`. Hai test cuối đóng SQLite connection
  đúng cách trước hash để tránh WAL làm snapshot nondeterministic; không hạ
  assertion hoặc sửa dataset thật.
- Tài liệu bàn giao mới duy nhất: `V21_FUNCTIONAL_COMPLETION.md`.

Helper, receipt, screenshot và SQLite chỉ chứa email giả phục vụ QA giữ local
trong `.local-test-deps/functional-v21/`, ngoài source payload công bố.

## 5. Pytest và bảo toàn

Môi trường đã dùng: Windows, Python 3.13.14; scikit-learn 1.9.1, numpy 2.5.3,
pandas 3.0.5, scipy 1.18.1, joblib 1.6.0, Streamlit 1.63.0, pytest 9.1.1.

| Kiểm tra | Kết quả thực tế |
| --- | --- |
| Toàn bộ suite trong phạm vi không đọc held-out | **488 passed, 3 skipped, 4 deselected** |
| Source-level (`not production_artifact`) cùng guard | **462 passed, 3 skipped, 30 deselected** |
| Production-artifact an toàn với model V2 local thật | **26 passed, 469 deselected** |
| `pip check` | PASS — no broken requirements |
| `git diff --check` | PASS; có cảnh báo LF/CRLF, không có lỗi whitespace |
| Hash V2 trước/sau | PASS — 103 file, không thay đổi |

3 skipped là integration TREC loader/curation/review opt-in chưa bật, không
phải bỏ qua vì model V2.1 thiếu. Không gọi đây là full regression không điều
kiện: 4 test đọc CSV held-out thật được chủ động deselect để tuân thủ yêu cầu:

1. `tests/test_split_builder.py::test_generated_repository_splits_have_disjoint_groups`
2. `tests/test_v2_completion_integrity.py::test_actual_v2_splits_preserve_eligibility_provenance_and_origin`
3. `tests/test_dashboard_data.py::test_dataset_dashboard_uses_only_eligible_v2_splits`
4. `tests/test_streamlit_app.py::test_all_streamlit_pages_render_saved_artifacts`

30 source deselected = 26 production-artifact + 4 test trên. Unit tests có thể
fit pipeline nhỏ bằng dữ liệu giả trong temp dir; không huấn luyện/refit bất
kỳ candidate hoặc production artifact thật nào.

Lệnh thực tế (helper local chặn non-loopback network, đọc test CSV thật, cô lập
DB và thêm đúng bốn deselect trên):

```powershell
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/run_regression.py -q -rs --basetemp=.local-test-deps/functional-v21/pytest-delivery-all --junitxml=.local-test-deps/functional-v21/delivery-all.xml
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/run_regression.py -q -m 'not production_artifact' -rs --basetemp=.local-test-deps/functional-v21/pytest-delivery-source --junitxml=.local-test-deps/functional-v21/delivery-source.xml
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/run_regression.py -q -m production_artifact -rs --basetemp=.local-test-deps/functional-v21/pytest-delivery-production --junitxml=.local-test-deps/functional-v21/delivery-production.xml
.\.venv\Scripts\python.exe -m pip check
git diff --check
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/verify_local.py compare
```

Guarded regression không ghi nhận attempt truy cập held-out hoặc mạng ngoài.
Không chạy `pytest -q` không guard vì các test đã nêu có đọc Test CSV thật.

## 6. Browser E2E thật

**34 tình huống riêng biệt PASS**, Edge 155.0.4283.62 ở chế độ hiển thị, điều
khiển bằng Playwright core 1.62.1 đã có sẵn sau khi người dùng cho phép.
Browser connector/Computer Use ban đầu không kết nối được; không cài thêm công
cụ, không dùng AppTest hoặc HTTP health để thay Browser E2E.

Đã quan sát trực tiếp:

- V2 mặc định; V2.1 Analyze thật cho EN, VI, ZH/mixed có giới hạn phù hợp;
  không có prediction trước Analyze, không có % confidence SVM giả.
- Save/Open, Inbox/Spam/Quarantine full-width, Back đúng thư mục, Not Spam,
  star/unread, Security/Starred, chuyển Inbox, Trash vẫn đọc được.
- EN/VI switch giữ bản nháp/kết quả; EML tiếng Việt HTML và tiếng Trung MIME;
  empty/unsupported file; import lặp mở bản cũ, không tạo analysis mới.
- Hybrid mock chỉ chạy sau click; lưu và mở lại review riêng; thiếu key vẫn
  chặn Gemini dù tích consent; ML không bị thay nhãn.
- Restart Streamlit thật rồi đọc SQLite: toàn bộ logical-row hashes giữ
  nguyên; 10 mail, 10 analysis, 25 feedback, 6 folder history, 1 review;
  `integrity_check=ok`, FK violations=0. Đây là dữ liệu QA giả, không phải metric ML.
- Missing-model, invalid JSON, timeout và provider error được kiểm thử bằng
  **fault injection trên server QA riêng**, không đổi/xóa artifact thật và
  không gọi Gemini. UI hiển thị lỗi an toàn, không traceback/giả prediction.

Browser/server được cô lập localhost, dùng DB QA riêng. Lượt browser cuối không
ghi nhận host ngoài; server không ghi nhận external request attempt. Screenshot
và 34 receipt giữ local tại `.local-test-deps/functional-v21/browser-e2e.json`
và các `browser-*.png`; nội dung chỉ là fixture giả.

Lệnh đã dùng cho E2E (các script là helper QA local, không phải launcher bắt buộc):

```powershell
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/serve.py 8521
D:\Apps\node.exe .local-test-deps/functional-v21/edge_host.cjs
D:\Apps\node.exe .local-test-deps/functional-v21/browser_cases.cjs
D:\Apps\node.exe .local-test-deps/functional-v21/browser_remaining.cjs
D:\Apps\node.exe .local-test-deps/functional-v21/browser_mailbox.cjs
D:\Apps\node.exe .local-test-deps/functional-v21/browser_faults_restart.cjs
```

## 7. Khởi động local trên Windows

Môi trường `.venv` và candidate đã có, không cần tải/cài thêm để chạy trên máy
đã kiểm thử:

```powershell
Set-Location 'D:\Study\Macchine Learrning\VietMailGuard'
.\.venv\Scripts\python.exe -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
```

Mở `http://127.0.0.1:8501`. Mặc định V2 Production. Chọn V2.1 ML Research
Preview, vào Quick Analyzer, nhập sender/subject/body hoặc EML, bấm Analyze →
Save → Open. Trong mailbox cũng có nhập demo trực tiếp và import EML; model
đang chọn quyết định route khi tiếp nhận, không thay lịch sử mail cũ.

DB mặc định: `data/runtime/vietmailguard_mail.db`. Muốn tách dữ liệu demo:

```powershell
$env:VIETMAILGUARD_MAIL_DB = Join-Path (Get-Location) 'data/runtime/v21_demo_mail.db'
.\.venv\Scripts\python.exe -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
# Sau khi dừng app, bỏ override; không xóa file DB:
Remove-Item Env:VIETMAILGUARD_MAIL_DB
```

Hybrid: chọn Hybrid rồi **Offline mock** để kiểm thử không mạng/không key. Muốn
dùng Gemini sau khi được phép riêng, cấu hình `.env` ở root với
`GEMINI_API_KEY` và `GEMINI_MODEL`, không đưa secret vào Git/log. SDK/key không
chứng minh tài khoản có quota/model access; live Gemini chưa kiểm thử lần này.
Chọn Gemini và consent không tự gọi API; vẫn phải bấm Review. Không gửi email
nhạy cảm nếu chưa quyết định rõ về dữ liệu/chính sách nhà cung cấp.

Nếu V2.1 lỗi, chọn V2 Production; không thay runtime config để giả mạo phiên
bản. Chỉ cung cấp joblib local từ nguồn tin cậy, khớp SHA/config (joblib không
phải định dạng an toàn cho artifact không tin cậy).

## 8. Phần chưa xác minh / ngoài phạm vi

- Gemini live, quota và chất lượng Reviewer thực: **NOT RUN**, cần quyền riêng;
  mock/fault injection không thay thế kiểm chứng dịch vụ thật.
- Hiệu năng độc lập tiếng Việt bản địa và phishing tiếng Trung ba lớp:
  **UNVALIDATED**; chưa đủ benchmark. Chinese UI output không chứng minh chất
  lượng, không dùng kết quả synthetic/TREC binary để lấp khoảng trống này.
- Candidate V2.1 chưa có calibrated probabilities và chưa được phát hành
  Production; không tạo confidence giả để che giới hạn.
- Final Evaluation V2/V2.1/TREC: **NOT RUN**; không dùng Test để sửa ứng dụng.
- Đây là mail client local, không gửi/nhận Internet bằng SMTP/IMAP/Gmail và
  không scan malware attachment. Không phê duyệt public release trong task.

Không còn blocker chức năng nghiêm trọng trong phạm vi local Research Preview
đã công bố và kiểm thử ở trên. Giới hạn nghiên cứu không được chuyển thành
tuyên bố hiệu năng Production.

## 9. Freeze Functional Research Preview — 2026-10-23 (Asia/Saigon)

`FUNCTIONAL_RESEARCH_PREVIEW_STATUS: FROZEN_LOCAL`

`PUBLICATION_APPROVAL: NO`

`NEW_REPO_PUBLICATION_APPROVED: NO`

`OLD_REPOSITORY_PRIVACY_RESOLVED: NO`

Đã ghi nhận mốc chức năng hiện tại theo yêu cầu chủ dự án. Đây là snapshot
local có manifest SHA-256, không phải Git commit/tag, production freeze hay
giấy phép phát hành. Các kết quả 488 passed / 3 skipped / 4 deselected và
34/34 Edge E2E ở trên là bằng chứng từ lượt kiểm thử chức năng trước; nhiệm vụ
backup này không chạy lại Training, Gemini live hoặc held-out Final Test.

Backup nằm ngoài Git working tree:

```text
D:\Study\Macchine Learrning\VietMailGuard_Local_Backups\V21_Functional_Research_Preview_20261023_204040\
├── source/                 # Source, tests, cấu hình an toàn, dependency/docs
├── protected_artifacts/    # Model local, kết quả/reports và bằng chứng QA riêng
├── metadata/               # Git state, thay đổi chưa commit, hash V2, freeze record
├── freeze_snapshot.py
├── backup_manifest.json
├── backup_manifest.sha256
└── verification.json
```

Snapshot lấy cả source chưa được Git theo dõi và ghi nhận các đường dẫn đã xóa
do chuyển `app/pages` sang `app/views`; không chỉ sao chép các file tracked.
Không thay branch, staged/index hoặc lịch sử Git. Bản tài liệu trước khi thêm
mục freeze này cũng được giữ trong metadata của backup.

Thư mục backup được giới hạn ACL local cho tài khoản đang thao tác và SYSTEM;
model và kết quả nghiên cứu không nằm trong `source/`. Đây là hạn chế quyền
truy cập local, không phải mã hóa và không phải bản sao được phép chia sẻ.

Không sao chép `.env` thật, API key, Git history, email dataset, EML riêng tư,
SQLite, venv, cache/checkpoint hoặc pretrained model downloads. Review queue
có cột nội dung email được loại khỏi backup. Các dữ liệu đó vẫn giữ nguyên tại
project gốc; đây **không phải** backup toàn bộ ổ đĩa/dataset/database cá nhân.

Manifest là danh sách file/size/hash chuẩn để xác minh backup; sidecar kiểm tra
hash của chính manifest. Receipt `verification.json` ghi kết quả so khớp từng
file backup với project, bảo toàn Git state và 103 hash V2. Secret-scan receipt
riêng chỉ phục vụ an toàn bản sao local, không thay đổi publication approval.
Nếu cần khôi phục, trước tiên dùng thư mục local mới để đối chiếu; không tự
ghi đè V2 frozen, dataset hoặc database hiện có. Các thay đổi chức năng tiếp
theo cần yêu cầu riêng và không được ghi đè snapshot này.

## 10. Gemini Reviewer thật — hoàn thiện sau snapshot theo yêu cầu riêng

### Kiểm tra thực tế, không gọi API

Gemini provider đã tồn tại tại `src/vietmailguard/v21/gemini_reviewer.py`, dùng
SDK chính thức `google-genai`, được tái sử dụng; không tạo adapter trùng.
Kiểm tra offline lần này cho thấy:

- Root `.env`: có; `GEMINI_API_KEY`: đã cấu hình, không hiển thị giá trị.
- `google-genai`: 2.32.0; `python-dotenv`: 1.2.4.
- `GEMINI_MODEL`: `gemini-3.5-flash-lite`, hợp lệ về cú pháp cấu hình.
- `model_access_status=not_checked_offline`, `network_called=false`.

Không thể kết luận quyền truy cập, quota, billing hoặc khả năng đáp ứng hiện
tại chỉ từ tên model/key có mặt. **Chưa gọi models.list hoặc generateContent**
trong nhiệm vụ này. Live smoke PASS ở giai đoạn trước không được coi là kiểm
tra tài khoản/quota mới. Không sửa `.env` thật hoặc chọn model khác tự động.

`project_environment()` đọc `.env` ở root bằng `dotenv_values`, không đưa key
vào session state hoặc log. Biến môi trường process có ưu tiên cao hơn `.env`,
kể cả giá trị rỗng được dùng để vô hiệu hóa trong test. `.env` tiếp tục ignored.

### Cách bật trong Streamlit

Khởi động project gốc như mục 7, rồi:

1. Trong sidebar, chọn **V2.1 Hybrid — Research Preview**. Cần candidate V2.1
   hợp lệ; không dùng V2 giả làm V2.1 nếu artifact lỗi.
2. Ở **Nhà cung cấp Reviewer**, chọn **Google Gemini (bên ngoài, cần đồng ý)**,
   không chọn Mock offline. Mock vẫn là mặc định an toàn và được giữ để test.
3. Trong Quick Analyzer, nhập email hoặc EML rồi bấm Analyze. Có thể mở mail
   đã lưu và review trực tiếp mà không phân tích ML lại.
4. Trong phần **LLM Reviewer (tách biệt với ML)**, đọc disclosure về nội dung
   gửi đến Google và chi phí. Xác nhận đồng ý cho email đang xem/đã phân tích.
5. Bấm **Chạy Gemini review**. Chỉ lúc này mới tạo request thật. Chọn provider,
   đổi trang, render, Analyze hoặc chỉ tích consent đều không tự gọi Gemini.

Trong `.env` local, cấu hình key trực tiếp trên máy; không dán key vào chat:

```dotenv
GEMINI_API_KEY=<key-cua-ban-chi-luu-local>
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_REVIEW_TIMEOUT_SECONDS=30
```

Biến timeout là tùy chọn, hợp lệ `(0,60]` giây. App mặc định chờ 30 giây thay
vì timeout ngầm 5 giây trước đây; mỗi lần bấm chỉ có **một attempt, retry=0**.
Timeout không bảo đảm hủy xử lý/billing phía nhà cung cấp. Không bấm gửi lại
liên tiếp nếu chưa kiểm tra lỗi. `VIETMAILGUARD_V21_LLM_PROVIDER` vẫn dùng cho
API Python lấy provider từ environment; Streamlit dùng lựa chọn provider rõ
ràng trong sidebar, không âm thầm bật Gemini theo file cấu hình.

Request chứa tối đa 20.000 ký tự đầu của từng trường sender/subject/body,
bao gồm URL dạng văn bản trong đó; không truy cập URL hoặc gửi attachment.
Prompt xem email là untrusted data, không cấp tools và không gửi nhãn ML cho
Reviewer. Structured JSON được kiểm tra schema và evidence quote cục bộ.
Nhận xét Gemini, model thực sự được gọi, trạng thái và REVIEW được hiển thị/lưu
riêng. Không biến LLM thành ground truth/calibrated confidence, không thay ML,
Risk Score hay routing. Đổi provider reset consent.

### Lỗi đã sửa và file thay đổi

Root cause: transport đã có chẩn đoán Gemini redacted nhưng `review_email()`
bỏ metadata khi bắt exception, chỉ trả `provider_error`. Nay giữ diagnostic
allowlist (không message/request/header/key), phân biệt xác thực, quyền model,
model/endpoint không tồn tại, quota/rate limit, lỗi dịch vụ, kết nối, timeout,
cấu hình và JSON/schema. UI có giải thích EN/VI và HTTP code an toàn; lỗi không
làm mất kết quả ML hoặc đổi thư mục. Model được lưu theo request thật, không
gán tên cấu hình mới cho review lịch sử cũ.

File thay đổi trong nhiệm vụ Reviewer này:

- `src/vietmailguard/v21/gemini_reviewer.py`
- `src/vietmailguard/v21/llm_reviewer.py`
- `src/vietmailguard/mail_service.py` (forward timeout/retry cho review riêng)
- `app/runtime_ui.py`
- `app/views/email_analyzer.py`
- `app/views/mail_client.py`
- `config/translations.json`
- `scripts/v21/check_gemini.py`
- `.env.example` (timeout mẫu, không chứa secret)
- `tests/test_v21_gemini_reviewer.py`
- `tests/test_v21_gemini_smoke_cli.py`
- `tests/test_v21_gemini_ui.py` (mới)
- `V21_FUNCTIONAL_COMPLETION.md`

### Lệnh kiểm tra và kết quả

Đã chạy kiểm tra cấu hình **offline**:

```powershell
.\.venv\Scripts\python.exe scripts/v21/check_gemini.py
```

Hai lệnh sau **chỉ chuẩn bị, chưa chạy**. Chỉ thực hiện khi chủ dự án cho phép
gọi API tương ứng. Lệnh đầu gọi metadata, không gửi email/generateContent;
được listed không chứng minh quota. Lệnh thứ hai có thể phát sinh chi phí,
chỉ gửi câu nhân tạo “Reply with the single word OK.”, không phải email thật:

```powershell
.\.venv\Scripts\python.exe scripts/v21/check_gemini.py --list-models
.\.venv\Scripts\python.exe scripts/v21/check_gemini.py --live --confirm-synthetic-request
```

Kiểm thử tự động dùng fake/mock transport, dữ liệu giả, SQLite tạm; không dùng
key thật hoặc dịch vụ Gemini thật:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v21_gemini_reviewer.py tests/test_v21_llm_reviewer.py tests/test_v21_gemini_smoke_cli.py -q
# 59 passed
.\.venv\Scripts\python.exe -m pytest tests/test_v21_gemini_ui.py -q
# 7 passed; AppTest, không phải browser E2E mới
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py -q -rs --basetemp=.local-test-deps/gemini-reviewer/pytest-regression --junitxml=.local-test-deps/gemini-reviewer/regression.xml
# 512 passed, 3 skipped, 4 deselected
git diff --check
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/verify_local.py compare
```

Regression helper được sao chép sang vùng QA local riêng, không ghi đè receipts
của snapshot chức năng trước. Guard chặn non-loopback network/đọc held-out và
cô lập DB; receipt ghi **0 guarded access attempts**. Ba skip TREC opt-in và
bốn deselect đọc Test CSV vẫn như mục 5, không gọi đây là full suite không
điều kiện. Chưa thực hiện live Gemini hay browser E2E mới trong nhiệm vụ này.

Kiểm tra V2 frozen vẫn 103/103 hash không đổi; không train/refit/calibrate,
không thay model, risk rules, held-out metrics hoặc dataset. Snapshot backup ở
mục 9 giữ nguyên; đây là thay đổi Reviewer tiếp theo được người dùng yêu cầu,
không ghi đè snapshot cũ hoặc đồng bộ Clean Export. Publication approval vẫn NO;
không commit/push.

## 11. Sửa ImportError khi Streamlit còn giữ module UI cũ

Đã tái hiện đúng `cannot import name 'reviewer_request_options' from
'runtime_ui'`: file hiện tại có helper, fresh Python import thành công, nhưng
module UI trước cập nhật còn trong `sys.modules` không có helper đó. Hai page
mới đều lỗi khi import từ module cũ. Test process mới trước đây chưa bao phủ
tình huống server chạy xuyên qua lần cập nhật này.

`app/app.py` nay kiểm tra API UI trước khi chạy page và chỉ `importlib.reload`
module `runtime_ui` nếu helper bắt buộc còn thiếu/không callable. Không clear
model/resource cache, reload inference/provider, xóa session state hoặc restart
tiến trình của người dùng. Nếu backend Reviewer vẫn là code cũ trong bộ nhớ,
UI chặn request và hiện hướng dẫn restart thay vì KeyError hoặc gửi API.

File sửa trong lần fix này: `app/app.py`, `app/runtime_ui.py`,
`config/translations.json`, `tests/test_streamlit_module_refresh.py` (mới),
và mục tài liệu này.

Đã chạy:

```powershell
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py tests/test_streamlit_module_refresh.py tests/test_v21_gemini_ui.py tests/test_v21_functional_ui.py -q --basetemp=.local-test-deps/gemini-reviewer/pytest-module-refresh --junitxml=.local-test-deps/gemini-reviewer/module-refresh.xml
# 15 passed
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py -q -rs --basetemp=.local-test-deps/gemini-reviewer/pytest-import-fix-regression --junitxml=.local-test-deps/gemini-reviewer/import-fix-regression.xml
# 517 passed, 3 skipped, 4 deselected
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/verify_local.py compare
# 103 V2 frozen files unchanged
git diff --check
# PASS (LF/CRLF warnings only)
```

Skip/deselect vẫn đúng phạm vi ở mục 5; guard không ghi nhận attempt mạng ngoài
hoặc đọc held-out. Chưa gọi Gemini thật, train hoặc Final Evaluation. Để nạp
toàn bộ code backend mới trong server đang mở, hãy giữ lại nội dung nháp cần
thiết, dừng terminal Streamlit bằng Ctrl+C và chạy lại lệnh ở mục 7. Không xóa
database, `.env`, model, hoặc backup để xử lý cache code.

## 12. Reviewer evidence rubric — cập nhật theo xác nhận, kiểm thử offline

### Nguyên nhân và phạm vi sửa

Prompt Reviewer trước đây yêu cầu JSON và trích dẫn, nhưng chưa định nghĩa rõ
taxonomy hoặc mức bằng chứng cần để nhận định phishing. Schema hợp lệ và câu
trích dẫn có thật không chứng minh suy luận là đúng. `status=ok` chỉ xác nhận
nhận được phản hồi hợp lệ, không xác nhận email thực sự lừa đảo. ML và LLM có
thể bất đồng; bản thân sự bất đồng cũng không chứng minh bên nào đúng.

Đã bổ sung policy `v21-reviewer-rubric-2` với các nguyên tắc tổng quát:

- Tách quan sát từ nội dung và giả định chưa kiểm chứng; không suy đoán xác
  thực người gửi, quyền sở hữu domain, quan hệ với dịch vụ bên thứ ba, reputation
  URL hoặc những trường nhập trên form chưa được mở.
- Form bên thứ ba, domain khác, quyền lợi tài chính/học bổng, thời hạn và yêu
  cầu xác nhận không tự chứng minh phishing. Khi chưa có cơ chế lừa dối được
  hỗ trợ bằng nội dung, dùng `review` và nêu điều còn cần xác minh.
- Phishing cần bằng chứng về cơ chế lừa dối nhằm dẫn đến hành động có hại;
  không chỉ dựa vào từ khóa, nhưng cũng không giới hạn phishing vào mật khẩu
  hay OTP. Không tự quy mọi lời mời tài chính/scam mơ hồ thành phishing.
- Không whitelist tổ chức/domain, không ép khớp ML và không hậu xử lý đổi
  `phishing` thành `normal` hoặc `review`. Gemini vẫn đánh giá độc lập, không
  nhận nhãn ML và không có quyền thay dự đoán/risk/routing.

Adapter gửi đúng system instruction được tạo cho request, lưu `prompt_version`
và SHA-256 của chuỗi prompt thực gửi trong kết quả Reviewer. Không lưu thêm
nội dung email vào metadata. Metadata được reset mỗi request để lỗi preflight
không mang phiên bản/model của lần trước. Hash policy hiện tại:

```text
0f47e644a2fe6f50d07af770470a503769a78942b228499f0081e51334d9dee6
```

UI EN/VI đổi cách diễn đạt nhận định LLM `phishing` thành **nghi ngờ lừa đảo**,
giải thích nghĩa của `ok`, và nêu giới hạn phân tích chỉ từ nội dung. Internal
assessment vẫn là `phishing`; nhãn ML không đổi. Review lịch sử thiếu phiên bản
prompt được ghi chú, không tự chạy lại hoặc sửa kết quả đã lưu.

Email người dùng gửi để giải thích tình huống chỉ được xem trong hội thoại;
không sao chép email, địa chỉ, số điện thoại hoặc URL thật vào code/tests/tài
liệu. Không mở liên kết, gọi Gemini hay xác minh danh tính bên ngoài. Chưa đủ
bằng chứng để kết luận email cụ thể chắc chắn an toàn hoặc lừa đảo.

### File thay đổi trong nhiệm vụ này

- `src/vietmailguard/v21/llm_reviewer.py`: rubric, policy version/hash và
  metadata kết quả tùy chọn; giữ schema bốn trường và validation evidence.
- `src/vietmailguard/v21/gemini_reviewer.py`: system instruction và provenance
  đúng request; không đổi consent, timeout, API model hoặc retry policy.
- `app/runtime_ui.py`: diễn đạt trạng thái/nhận định và phiên bản policy;
  không sửa navigation hoặc kết quả ML.
- `config/translations.json`: năm thông báo mới EN/VI riêng cho Reviewer.
- `tests/test_v21_reviewer_policy.py`: 16 test mới với dữ liệu synthetic,
  fake Gemini và network guard; có kiểm tra renderer bằng AppTest.
- `V21_FUNCTIONAL_COMPLETION.md`: mục này.

### Verification thực tế

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v21_gemini_reviewer.py tests/test_v21_llm_reviewer.py tests/test_v21_gemini_smoke_cli.py -q
# Baseline trước sửa policy: 59 passed
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py tests/test_v21_reviewer_policy.py tests/test_v21_gemini_reviewer.py tests/test_v21_llm_reviewer.py tests/test_v21_gemini_ui.py tests/test_streamlit_module_refresh.py -q --basetemp=.local-test-deps/gemini-reviewer/pytest-policy-targeted --junitxml=.local-test-deps/gemini-reviewer/policy-targeted.xml
# PASS: 76 passed
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py -q -rs --basetemp=.local-test-deps/gemini-reviewer/pytest-policy-regression --junitxml=.local-test-deps/gemini-reviewer/policy-regression.xml
# PASS: 533 passed, 3 skipped, 4 deselected
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/verify_local.py compare
# PASS: 103 V2 frozen files unchanged
git diff --check
```

Ba skip là integration TREC opt-in chưa được yêu cầu. Bốn deselect vẫn là các
test đọc dữ liệu held-out đã liệt kê ở mục 5; đây không phải full suite không
điều kiện. Guard cô lập runtime DB, chặn mạng ngoài/đọc held-out và ghi nhận
**0 guarded access attempts**. Unit tests hiện có chỉ fit pipeline nhỏ trên dữ
liệu giả; không huấn luyện model nghiên cứu/production.

Kiểm tra bảo toàn độc lập bằng byte-hash: **103/103** file V2 frozen không đổi,
**11/11** binary V2.1 không đổi; runtime config, training config và saved
validation metrics **3/3** không đổi. Immutable backup **400/400** file khớp
manifest/size/hash, manifest vẫn
`bf2cf51491faf613112eec8c769101dd495218fa569861166ec74ef848cbe4e0`.
Không đọc dữ liệu Test để tuning/evaluation, không sửa Clean Export, không
commit/push; publication approval vẫn NO.

**Giới hạn kiểm chứng:** mock/fake client chứng minh contract, schema, consent,
provenance, ML independence và UI; không chứng minh chất lượng ngữ nghĩa của
Gemini hoặc tỷ lệ false positive đã giảm. Chưa gọi Gemini thật và chưa có
browser E2E mới trong nhiệm vụ này. Các review cũ giữ nguyên.

Để nạp prompt/backend mới, giữ lại bản nháp cần thiết, dừng terminal Streamlit
bằng Ctrl+C rồi chạy lại trong project gốc:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Sau restart, review mới chỉ xảy ra nếu người dùng chọn Gemini, đồng ý gửi nội
dung email đang xem và chủ động bấm nút review. Kết quả mới sẽ có phiên bản
policy ở phần Reviewer. Không tự gửi lại email hoặc làm mới lịch sử khi restart.

## 13. Dataset Dashboard và Model Dashboard V2.1 đầy đủ

Cập nhật ngày 2026-10-28 trong project gốc. Hai dashboard nay mặc định mở
**V2.1 Research Preview**, có mục V2 frozen riêng, và đọc các artifact tổng hợp
đã lưu. Không gọi inference, fit model, đánh giá held-out, mở email CSV hoặc
gọi Gemini để tạo số liệu dashboard.

### Nội dung đã triển khai

Dataset Dashboard có năm mục:

- **Development Bootstrap:** 31.304 Train, 6.709 Validation, 38.013 tổng;
  phân bố nhãn/ngôn ngữ, nguồn, data origin, parent và nhóm trùng; trạng thái
  sẵn sàng development và final evaluation từ manifest. English origin chưa
  khai báo vẫn giữ `not_declared`; Vietnamese dịch vẫn là `translated`.
- **Audit & curation:** 1.147.062 hàng audit, gồm 1.102.345 nguồn mới và
  44.717 V2 tham chiếu; snapshot source approval có 584.150 weak_label,
  517.564 review, 631 exclude, 0 eligible mới. Có bảng từng dataset, trạng
  thái curation, ngôn ngữ gốc, nhóm trùng, xung đột và quy tắc chờ duyệt.
  Không cộng V2 tham chiếu hoặc các nguồn có chung lineage thành dữ liệu mới.
- **TREC06C:** 64.620 nhãn nguồn Ham/Spam; 59.079 đạt bộ lọc nghiên cứu nhị
  phân. Hiển thị số loại bỏ, split, leakage, human-review snapshot và tập Ham
  có điều kiện chưa được phê duyệt. Không biến quality pass thành eligible.
- **Chinese synthetic:** 3.030 hàng nguồn, 2.959 qua sàng lọc, 2.367 Train,
  592 challenge; phân bố nhãn nguồn, cờ sàng lọc và nhóm trùng. Đây không phải
  email thực/human-verified hoặc Final Test.
- **Benchmark & khoảng trống:** Chinese real-world benchmark pending, 0
  accepted; giới hạn native Vietnamese/Chinese, cùng số lượng PhishVN trong
  mục URL riêng. Không nhập URL PhishVN vào số email huấn luyện.

Model Dashboard có các phần độc lập:

- Candidate V2.1 trong runtime config, recommendation Validation, bảng so
  sánh Word+Character TF-IDF/LinearSVC và embedding classifiers; per-class
  P/R/F1/support, confusion matrix, EN/VI/ZH, data origin, thời gian fit và
  inference; cấu hình và trạng thái XLM-R resource gate.
- TREC06C Ham/Spam với ba candidate và lỗi Ham→Spam/Spam→Ham. Character SVC
  có Validation Macro F1 **0,993805**; không tạo phishing metric nhị phân.
- Cặp LinearSVC hiện tại A/B cùng cấu hình: Macro F1 **0,985852 → 0,984599**,
  chênh lệch từ số liệu lưu **−0,001253**. Các cặp SVC/LR cấu hình nhỏ hơn có
  bảng riêng; synthetic-only challenge có cảnh báo và vùng hiển thị riêng.
- Kết quả V2 held-out, robustness và short-form challenge vẫn hiển thị từ
  artifact V2 đã lưu, không đánh giá lại hoặc trộn với V2.1 Validation.

Đã sửa cách hiểu chiều lỗi trong bảng V2.1 cũ: trường JSON
`normal_false_positives` thực tế đếm **Spam/Phishing → Normal** (20 ở candidate
hiện tại); `non_normal_false_positives` đếm **Normal → Spam/Phishing** (3).
Dashboard ghi đúng chiều lỗi theo schema và confusion matrix đã lưu; không
sửa JSON, CSV kết quả hoặc training logic. Độ trễ embedding dùng số end-to-end
smoke đã lưu, không hiển thị classifier-only latency như toàn bộ inference.

### Nguồn dữ liệu và hành vi khi thiếu file

Loader mới dùng allowlist 18 artifact: JSON aggregate/metadata, các summary
audit/curation, manifest và CSV `label_curation_counts.csv` chỉ có dataset,
label_status, rows. Giới hạn 2 MB mỗi file; không theo đường dẫn dataset/model
trong metadata. File thiếu, JSON lỗi, sai scope, NaN hoặc CSV aggregate sai
cấu trúc chỉ làm mục tương ứng unavailable. Không thay bằng zero hay metric
giả. Cache dùng mtime/size của allowlist để nhận biết artifact thay đổi.

Dataset V2 trước đây mở trực tiếp split CSV và curated CSV để tạo bảng. Trang
nay dùng bốn báo cáo V2 frozen: bilingual dataset, bilingual split, Vietnamese
curation và translated augmentation, giữ nguyên nội dung/ngôn ngữ gốc trong
expander. Vì vậy đọc dashboard không yêu cầu mở `data/splits/v2/test.csv`.
UI và header bảng mới hỗ trợ EN/VI; metric hiển thị sáu chữ số thập phân, giá
trị thiếu là N/A. Báo cáo curation là snapshot có thời gian, không phải bộ đếm
annotation trực tiếp. Không tuyên bố benchmark tiếng Việt bản địa hoặc
phishing tiếng Trung đã hoàn thành.

### File source/test/documentation thay đổi

- `app/views/dataset_dashboard.py`
- `app/views/model_dashboard.py`
- `app/research_dashboard_ui.py` (mới)
- `src/vietmailguard/v21/dashboard_artifacts.py`
- `config/translations.json`
- `tests/test_v21_dashboards.py` (mới, 20 test cases)
- `tests/test_streamlit_app.py` (assertion trang dataset theo saved reports)
- `V21_FUNCTIONAL_COMPLETION.md` (mục này)

Helper QA local `.local-test-deps/gemini-reviewer/run_offline_tests.py` có flag
`--include-dashboard-page-test` để chạy lại integration page test nay không
còn đọc held-out. Server/browser harness và ảnh QA riêng ở
`.local-test-deps/dashboard-qa/`, dùng SQLite thử nghiệm riêng, không dùng
mailbox cá nhân và không thuộc payload công bố.

### Kiểm thử thực tế

```powershell
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py tests/test_v21_dashboards.py tests/test_v21_runtime_integration.py -q --basetemp=.local-test-deps/gemini-reviewer/pytest-dashboard-targeted
# PASS: 28 passed
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py --include-dashboard-page-test -q -rs --basetemp=.local-test-deps/gemini-reviewer/pytest-dashboard-regression --junitxml=.local-test-deps/gemini-reviewer/dashboard-regression.xml
# PASS: 554 passed, 3 skipped, 3 deselected
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py --include-dashboard-page-test tests/test_v21_dashboards.py tests/test_streamlit_app.py::test_all_streamlit_pages_render_saved_artifacts -q --basetemp=.local-test-deps/gemini-reviewer/pytest-dashboard-final-pages
# PASS: 21 passed
.\.venv\Scripts\python.exe .local-test-deps/gemini-reviewer/run_offline_tests.py tests/test_v21_dashboards.py -q --basetemp=.local-test-deps/gemini-reviewer/pytest-dashboard-final
# PASS: 20 passed, gồm hardening CSV aggregate thiếu trường
.\.venv\Scripts\python.exe .local-test-deps/dashboard-qa/serve.py
node .local-test-deps/dashboard-qa/browser.cjs
# PASS: Edge 156.0.4314.50, headless=false, 22/22 observed cases
.\.venv\Scripts\python.exe .local-test-deps/functional-v21/verify_local.py compare
# PASS: 103/103 V2 frozen files unchanged
git diff --check
# PASS; cảnh báo LF/CRLF sẵn có
```

Regression vẫn chặn mạng ngoài và đọc held-out, guard ghi 0 access attempts.
Ba skip là TREC integration opt-in. Ba deselect còn lại đọc split email thật:
split builder disjoint groups, V2 eligibility/provenance integrity và V2
dataset loader test. Page integration trước đây bị deselect đã được chạy lại
thành công. Đây không phải full pytest không điều kiện.

Computer Use native pipe không kết nối được; đã dùng Playwright core có sẵn
và Edge local theo quyền kiểm thử trước đó, không tải/cài công cụ. E2E thực tế
kiểm tra các tab dataset/model, đổi EN/VI, V2 riêng, mở per-class/matrix và
hai trang ở 375/768/1024/1440px. Bảng rộng cuộn ngang trong vùng bảng. Viewport
check chờ resize ổn định; không coi kích thước trung gian trong lúc resize là
kết quả layout cuối. Browser receipt có 0 external requests; không nhập email
thật, không Gemini live hoặc Final Test. Server QA riêng đã được dừng sau test.

Để nạp toàn bộ code dashboard mới, giữ bản nháp cần thiết và restart Streamlit
trong project gốc. Có thể dùng launcher hiện có:

```powershell
.\.venv\Scripts\python.exe run.py
```

Mở **Dataset Dashboard** hoặc **Model Dashboard**, chọn tab V2.1 Research
Preview và mục nghiên cứu cần xem. Publication approval vẫn NO; không commit,
push, sửa Clean Export, train, refit hoặc recalibrate model.
