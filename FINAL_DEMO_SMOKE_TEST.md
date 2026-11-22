# FINAL DEMO SMOKE TEST — VietMailGuard V2.2

Ngày: 22/11/2026, Asia/Saigon.

## Kết luận

**DEMO_LOCAL_TRADITIONAL_ML: PASS.** Ứng dụng local với model hiện có thực hiện được Analyze → lưu SQLite → mở thư trong Inbox/Spam/Quarantine, không cần Gemini.

**DEMO_LOCAL_MULTILINGUAL_FUNCTIONAL: PASS, có giới hạn.** MiniLM + MLP thực sự load encoder/head local và chạy inference, gồm ví dụ Việt/Trung. Tuy nhiên mẫu quảng cáo tiếng Anh bị dự đoán Normal thay vì Spam; không đánh dấu mọi dự đoán là đúng.

**README_CHECK: PASS sau bản sửa cuối.** Mục 8 của README bàn giao đã được khôi phục UTF-8. Launcher chủ động cấu hình stdout/stderr và môi trường Python con thành UTF-8, không cần workaround encoding bên ngoài. Xem phần kiểm tra lại ở cuối tài liệu; các lỗi ban đầu dưới đây được giữ để truy vết.

**RECEIVER_MACHINE_TESTED: NO.** Đây là smoke test local, không phải đánh giá accuracy, Final Test, clean-install mới hoặc kiểm thử trên máy nhận.

## Môi trường và phạm vi

- Project phát triển: nhánh `v22/large-scale-dataset`, giữ nguyên các thay đổi chưa commit có sẵn.
- Chạy project gốc bằng `.venv` hiện có để sử dụng model local. Không dùng kết quả này thay cho bằng chứng clean-install của gói bàn giao.
- Gói `VietMailGuard_V22_Clean_Submission` chạy riêng bằng venv độc lập đã xác minh ở nhiệm vụ trước để kiểm tra trạng thái thiếu model. Không sao chép model, encoder, `.env` hoặc database vào gói.
- Trình duyệt thật: Microsoft Edge 158.0.4370.33, headless, điều khiển bằng Playwright core đã có sẵn. Không tải/cài công cụ.
- SQLite, cache, log và helper smoke test tạm ở thư mục verification riêng ngoài project/gói. Database chỉ chứa email synthetic của đợt kiểm tra này.
- Python runtime được gắn guard chặn network ngoài, truy cập held-out data và ghi ngoài vùng tạm. Trình duyệt chỉ cho phép request loopback. API keys được để trống trong environment của tiến trình kiểm thử; không lấy giá trị khóa để đưa vào báo cáo.

## Model thực sự sử dụng

| Model | Artifact local | SHA-256 | Phạm vi |
| --- | --- | --- | --- |
| V2 Production | `models/v2_bilingual/production_pipeline.joblib` | `b2ebecf641f2a5fd3ce9f0d0680a90884850e46b59c48ce0ab5bd4fee64f87fa` | Inference thực, không sửa frozen |
| V2.1 Traditional | `models/v21_multilingual/candidates/tfidf_word_character_linear_svm.joblib` | `1050eb9cde01610dda3d2f6a34bdaffda90f8462a08e8648cb7fa9310de28af3` | Development Candidate, chưa calibrated |
| V2.1 MiniLM + MLP | `models/v21_multilingual/experiments/multilingual_dl_01/C_minilm_mlp.joblib` | `4594b919c56218a8bae1eea649a950bf4e21b4fafe126d487a653d9f85c453ff` | Multilingual Experimental; encoder trong cache local, không download |
| V2.2 Supervised TF-IDF | `models/v22_multilingual/experiments/group_aware_supervised_pilot_01/tfidf_linearsvc.joblib` | `485c07a94990d53d3af6b64f82051881f16a287073829083af1537e7a2b4211b` | Inference offline riêng; không đưa vào routing/runtime V2/V2.1 |

V2.1 và MiniLM kiểm tra cấu hình/hash qua loader hiện có. Artifact V2.2 được đối chiếu checksum trong metadata trước khi load; dùng cùng text contract subject/body, mỗi trường tối đa 4.096 ký tự. Không fit hoặc chỉnh thành phần nào.

## Inference thực trên nội dung giả

Ba mẫu EN được xác định trước khi chạy, chỉ chứa địa chỉ/domain reserved:

- `meeting`: thông báo họp dự án, mang agenda và báo cáo tiến độ; kỳ vọng Normal.
- `promotion`: quảng cáo giảm giá 70%, mua hàng, free shipping và unsubscribe; kỳ vọng Spam.
- `credential_request`: đe dọa khóa tài khoản, yêu cầu nhập password/OTP qua `email-verify.example.invalid` và tránh kênh hỗ trợ; kỳ vọng Phishing.

Không lấy mẫu từ Train/Validation/Test hoặc hộp thư thật. Không mở các URL.

| Mẫu | V2 Production | V2.1 Traditional | V2.2 Supervised TF-IDF | V2.1 MiniLM + MLP |
| --- | --- | --- | --- | --- |
| meeting | Normal — PASS | Normal — PASS | Normal — PASS | Normal — PASS |
| promotion | Spam — PASS | Spam — PASS | Spam — PASS | **Normal — FAIL so với kỳ vọng mẫu** |
| credential_request | Phishing — PASS | Phishing — PASS | Phishing — PASS | Phishing — PASS |

MiniLM còn chạy hai mẫu thông báo họp synthetic: tiếng Việt → Normal, tiếng Trung → Normal. Trạng thái tiếng Trung trả `unvalidated`. Đây chỉ là kiểm tra Unicode/inference, không phải nhãn được kiểm duyệt cho dataset hay bằng chứng chất lượng ngôn ngữ.

V2.1 Traditional và MiniLM trả `confidence = None`; không chuyển decision score thành phần trăm. V2.2 TF-IDF không cung cấp calibrated probability. Các số confidence có sẵn của V2 giữ nguyên ngữ nghĩa model, không dùng làm Risk Score.

Sai khác MiniLM ở mẫu quảng cáo được giữ nguyên: không đổi email mẫu, prompt, threshold, model hoặc metrics để làm kết quả đẹp hơn. Không tính accuracy/Macro F1 từ vài email synthetic này.

## Kiểm thử thực tế qua Edge

### Project gốc — 16/16 assertion PASS

| Tình huống quan sát | Kết quả |
| --- | --- |
| Inbox khởi động, không cần Gemini | PASS |
| Mở Quick Analyzer | PASS |
| V2 Production mặc định, Analyze mẫu họp bằng model thật | PASS |
| V2.1 Traditional Analyze ba mẫu Normal/Spam/Phishing | PASS, 3 assertion |
| Lưu và mở lại từng mẫu, đúng Inbox/Spam/Quarantine | PASS, 3 assertion |
| Đọc được toàn bộ body đã lưu trong Spam/Quarantine | PASS trong các assertion lưu/mở |
| Hybrid mock chạy offline và giữ nguyên nhãn ML | PASS |
| Gemini không có key/consent: nút review bị vô hiệu hóa | PASS; không bấm gọi API |
| Upload EML synthetic hợp lệ và inference thật | PASS |
| EML rỗng: lỗi phù hợp, xóa kết quả cũ, không traceback | PASS |
| So sánh mô hình, Dashboard dữ liệu, Tổng quan/phương pháp | PASS, 3 assertion; không traceback |

### MiniLM trong UI — 4/4 assertion PASS

1. Mode Multilingual xuất hiện khi dùng đúng encoder cache local.
2. Analyze EN chạy MiniLM + MLP thật; không hiện confidence đã hiệu chỉnh giả.
3. Analyze ZH chạy thật, hiển thị trạng thái chưa được xác minh.
4. Save/Open thư ZH giữ nguyên subject/body Unicode.

### Gói source-only thiếu model — 6/6 assertion PASS

Inbox, Quick Analyzer, Analyze báo thiếu model và không tạo prediction giả; ba màn hình thông tin mở được không traceback. Các màn hình có thể báo thiếu artifact local đúng phạm vi gói. Không cung cấp model để lách bài kiểm tra này.

Tổng cộng 26 assertion trình duyệt PASS trong các lượt cuối. Điều đó không phủ nhận FAIL của dự đoán MiniLM ở mẫu Spam trong kiểm tra inference riêng.

Không dùng AppTest hoặc health endpoint thay cho thao tác Edge. Health của các server cũng PASS như kiểm tra bổ sung. Các server và browser kiểm thử đã được dừng sau khi hoàn thành.

## Vấn đề phát hiện trong lượt smoke ban đầu

| Vấn đề | Kết quả / cách xử lý, không sửa source |
| --- | --- |
| `run.py` in tiếng Việt khi stdout bị chuyển hướng và Python dùng CP1252 | Lượt khởi động đầu FAIL với `UnicodeEncodeError`. Đặt `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1` cho tiến trình rồi launcher/Streamlit PASS. Đây là caveat launcher, không phải lỗi model. |
| TLD extractor muốn ghi cache ngoài vùng tạm | Guard chặn hai lần ghi trước khi có thay đổi. Đặt `TLDEXTRACT_CACHE` trong thư mục tạm; các lượt inference tiếp theo PASS, không truy cập URL/mạng. |
| Đổi XDG cache sang vùng tạm làm MiniLM không tìm thấy cache gốc | Đặt `HF_HOME` trỏ cache local hiện có cho tiến trình kiểm thử, giữ HF offline. Loader xác minh encoder/head rồi inference PASS; không sao chép/download model. |
| RAM khả dụng đầu phiên khoảng 1,14 GiB | Chạy TF-IDF trước, dừng server ban đầu. RAM sau đó khoảng 2,09 GiB; mới load MiniLM trên CPU. Không đóng ứng dụng của người dùng. |
| Driver dùng full-page navigation hoặc selector option quá chặt | Đã sửa helper tạm: điều hướng bằng link Streamlit, chờ rerun hoàn tất và chọn theo option thực. Không sửa routing/widget của ứng dụng. |
| README gói, mục 8 có ký tự `?` thay chữ Việt | FAIL ở lượt ban đầu; đã sửa UTF-8 và kiểm tra lại trong bản sửa cuối. |

Không phát hiện traceback hiển thị cho người dùng trong các luồng Edge cuối cùng. Không gọi Gemini live, không có request trang ra ngoài hoặc event network/held-out bị chặn trong các lượt cuối. Hai event guard ban đầu chỉ là ghi cache ngoài vùng tạm, không phải API call.

## Hướng dẫn demo đã đối chiếu

README gói mô tả đúng source-only: phải có model/config/receipt hợp lệ cung cấp riêng mới chạy ML. Không hứa một bundle không có model sẽ phân loại được; không cần Gemini cho demo core.

Trên project gốc đã có model và dependency, mở PowerShell tại project, chạy:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GEMINI_API_KEY = ''
$env:GOOGLE_API_KEY = ''
$env:VIETMAILGUARD_MAIL_DB = Join-Path $env:TEMP 'vietmailguard_final_demo_synthetic.sqlite3'
$env:TLDEXTRACT_CACHE = Join-Path $env:TEMP 'vietmailguard_final_demo_tld_cache'
.\.venv\Scripts\python.exe run.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Mở `http://127.0.0.1:8501`. Chọn V2.1 Traditional → nhập một mẫu giả → Phân tích → lưu → mở lại. Có thể chọn Hybrid với Mock offline, không chọn/chạy Gemini. Với MiniLM, giữ nguyên cache encoder hiện có; nếu tự đổi `XDG_CACHE_HOME`, phải chỉ rõ `HF_HOME` tới cache đúng, không tải thêm model. Các biến offline không phải firewall tổng quát; phiên kiểm chứng này có guard riêng để chặn mạng ngoài.

Trên máy nhận, làm theo README để chuẩn bị Python 3.13, wheelhouse đã được phép và model bundle riêng có checksum/config/receipt khớp. Không chuyển `.venv`, cache người dùng hay `.env` thật từ máy phát triển để thay thế cài đặt. Không chạy training/finalization hoặc mở Final Test để tạo artifact thiếu.

## Bảo toàn và những phần chưa kiểm chứng

- Snapshot của lượt smoke ban đầu đối chiếu **322 file source/config/model/results** của project gốc: SHA-256 giữ nguyên, kể cả `.env` được đối chiếu hash mà không tiết lộ giá trị. Bản sửa cuối chỉ thay launcher/test và tài liệu đã được cho phép.
- Ở lượt smoke ban đầu, **90/90 file gói source-only giữ nguyên**, manifest có SHA-256 `7fcd636260d3c83f7fa8897e8bdea1c156f117386bb158c5dd1fb48a3f8abe8a`. Bản sửa cuối cập nhật manifest để phản ánh các file bàn giao đã sửa.
- Ở lượt smoke ban đầu chỉ thêm tài liệu `FINAL_DEMO_SMOKE_TEST.md` vào project gốc. Không thêm báo cáo này vào payload. Bản sửa cuối cập nhật launcher/test, README và các receipt/manifest bàn giao theo yêu cầu riêng; xem phần cuối tài liệu. Helper/log/database synthetic ở workspace tạm ngoài payload, không phải code bàn giao mới.
- Lượt smoke ban đầu không sửa model, source, README, config, routing, metrics hoặc dataset. Bản sửa cuối không thay đổi model/config/routing/metrics/dataset; không train, đọc held-out Final Test, gọi API, commit/push hay đổi Git history.
- Lượt smoke ban đầu không chạy lại pytest. Bản sửa cuối đã chạy kiểm tra launcher và subset source-only mới, ghi riêng ở phần cuối; không coi đó là full regression của project nghiên cứu.
- Chưa thử trên máy nhận/OS khác; chưa xác minh cài optional embeddings stack và model bundle trên máy đó. Chưa chạy Gemini live hoặc xác minh quyền truy cập model/quota Google.
- Không smoke-test riêng V2.2 MiniLM + MLP hoặc SSL artifacts; inference V2.2 TF-IDF được kiểm tra riêng, không tích hợp runtime mới.
- Hai lỗi README/launcher đã được xử lý trong nhiệm vụ sửa cuối. Vẫn chưa có cơ sở tuyên bố accuracy thực tế, Chinese production hoặc Final Evaluation PASS. Lỗi MiniLM dự đoán mẫu quảng cáo thành Normal được giữ nguyên, không chỉnh model hoặc số liệu.

**NEW_REPO_PUBLICATION_APPROVED: NO**  
**OLD_REPOSITORY_PRIVACY_RESOLVED: NO**

## Kiểm tra lại sau bản sửa cuối

Hai lỗi đã được xử lý theo nhiệm vụ FINAL MINOR FIXES:

- `run.py` tự cấu hình UTF-8 cho stdout/stderr và Python con. Kiểm tra thực bằng CP1252 + file log đã PASS, có dòng tiếng Việt và không `UnicodeEncodeError`; không cần hai biến workaround encoding trong hướng dẫn smoke ban đầu.
- Mục 8 README của Clean Submission và báo cáo audit được phục hồi UTF-8.
- Bổ sung hai subprocess regression tests; launcher tests **4 passed**. Venv mới cài offline 63 wheel với build isolation: PASS; pip check/import PASS. Source subset **100 passed, 4 skipped, 5 deselected** (31.46 giây), không phải full regression.
- Chạy lại Edge: **16/16** model-local, **4/4** MiniLM và **6/6** source-only thiếu model đều PASS; không request ngoài/API. Log/tests nằm trong workspace kiểm chứng ngoài payload, không phân phối email/model.
- Đồng bộ launcher/test vào gói, cập nhật provenance, receipt và manifest/checksum. Xem `SUBMISSION_PACKAGE_AUDIT.md` và `SUBMISSION_VERIFICATION.json` trong gói để biết danh sách file cùng checksum cuối.
- Sai nhãn MiniLM ở mẫu quảng cáo vẫn là hạn chế được ghi nhận; không chỉnh model hoặc metrics.

**Kết luận cuối:** không còn blocker kỹ thuật đã biết trong phạm vi Windows/source-only đã kiểm tra. Model phải được cung cấp riêng; chưa kiểm thử trên máy nhận, chưa xác minh accuracy đa ngôn ngữ độc lập hoặc Gemini live. Publication/privacy approval vẫn NO.

Manifest sau bản sửa cuối: `7e4e8370750d408f12c46d6c4659db084703b9ab91c8c794ea8406bc4fa6f366`. Đối chiếu 88/88 entry và hai file điều khiển: PASS. Quét cuối bằng Gitleaks 8.30.1 không có finding mới; raw exit 1 vẫn do bảy checksum/revision đã xác minh riêng, không phải credential. Gói vẫn có đúng 90 file, không có model/email thật/secrets/Git history. `git diff --check`: PASS.
