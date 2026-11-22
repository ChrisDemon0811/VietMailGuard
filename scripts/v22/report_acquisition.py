"""Summarize actual Phase 3.2 data acquisition and human-review readiness."""
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v22.acquisition import connect,report
from vietmailguard.v22.dataset_review import DatasetReview
from vietmailguard.v22.quality import digest


def main():
    folder=ROOT/'reports/v22';area=ROOT/'data/v22/phase32'
    acquisition=report(ROOT)
    verified=json.loads((area/'verified/manifest.json').read_text(encoding='utf-8'))
    pseudo=json.loads((area/'pseudo_labeled/manifest.json').read_text(encoding='utf-8'))
    review=DatasetReview(ROOT).progress()
    evidence=json.loads((area/'license_evidence/manifest.json').read_text(encoding='utf-8'))
    with connect(ROOT) as conn:
        files=[dict(zip(('path','source','url','sha256','size_bytes','license','version','status'),row)) for row in conn.execute('SELECT * FROM tep')]
        candidates=[list(row) for row in conn.execute('SELECT nguon,nhan_ung_vien,ngon_ngu_goi_y,trang_thai,count(*) FROM mau GROUP BY 1,2,3,4')]
        reasons=dict(conn.execute('SELECT ly_do,count(*) FROM mau GROUP BY ly_do'))
    sources=[
        {'source':'nazario_modern','status':'downloaded','original_email':True,'license':'CC-BY-4.0',
         'evidence':'https://monkey.org/~jose/phishing/README.txt','labels':'publisher hand-classified phishing, new project source-quality approval pending'},
        {'source':'phishing_pot','status':'downloaded','original_email':True,'license':'CC-BY-NC-4.0',
         'evidence':'https://github.com/rf-peixoto/phishing_pot/blob/78be141545ed9e56eefb857808ac78e83eb81b81/DATASET_CARD.md',
         'labels':'mixed phishing/spam/fraud collection; not automatically three-class phishing'},
        {'source':'VNSED','status':'not_acquired','evidence':'https://vjs.ac.vn/jcc/article/view/22392',
         'reason':'article describes 6008 Vietnamese emails; dataset access and dataset-specific permission not verified; no author contacted'},
        {'source':'CSRI_CNEPD','status':'rejected_not_email_content','evidence':'https://csri.scu.edu.cn/info/1012/1084.htm',
         'reason':'probing/network records, no subject/body; cannot count as native Chinese email'},
        {'source':'IWSPA_AP_v2','status':'not_acquired','evidence':'https://www2.cs.uh.edu/~rmverma/book.html',
         'reason':'institutional identity/NDA required; no bypass or document submission performed'},
        {'source':'translated/synthetic/SMS/web mirrors','status':'not_acquired','reason':'not independent native email data, uncertain upstream permissions or overlap'}]
    summary={'verified_reused':verified,'acquisition':acquisition,'review':review,'pseudo_labeled':pseudo,
        'downloaded_artifacts':files,'license_evidence':evidence,'new_source_candidates':sources,
        'candidate_distributions_not_groundtruth':candidates,'screening_reasons':reasons,
        'approved_total_unique':verified['approved_reused_rows'],
        'approved_new_independent':0,'remaining_to_300k':300000-verified['approved_reused_rows'],
        'native_vi_approved':0,'native_zh_approved':0,'dataset_finalized':False,'training_performed':False,
        'gemini_called':False,'publication_approved':False}
    snapshot=folder/'phase32_snapshot_integrity.json'
    if snapshot.exists():summary['pinned_archive_verification']=json.loads(snapshot.read_text(encoding='utf-8'))
    ui=folder/'phase32_ui_smoke.json'
    if ui.exists():summary['ui_smoke']=json.loads(ui.read_text(encoding='utf-8'))
    (folder/'DATASET_ACQUISITION_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# V2.2 Phase 3.2 — Dataset Acquisition','',
        'Nhánh v22/large-scale-dataset. Kết quả acquisition thực tế; không train, không dùng Final Test cũ, không gọi Gemini.',
        '', '## Dữ liệu đã có thể dùng theo phê duyệt local','',
        f'Đã áp dụng đúng xác nhận của chủ dự án cho **{verified["approved_reused_rows"]:,} mẫu bootstrap CC BY 4.0**. Đây là dữ liệu kế thừa, không phải acquisition độc lập mới. Train/Validation cũ: {verified["split_rows"]}; nhãn: {verified["by_label"]}; ngôn ngữ: {verified["by_language"]} (VI đều translated).',
        'Hai CSV verified giữ nguyên ID, source, parent_id, final_group_id, label_provenance và assignment; bổ sung attribution, license, reference đến phê duyệt owner. Bản gốc V2/V2.1 không đổi. Dữ liệu có thể chứa PII: restricted local, không gọi API/không public, không giả định đã ẩn danh.',
        'Thư mục data/v22/phase32 được giới hạn ACL cho tài khoản local hiện tại và SYSTEM; bind công cụ review vào loopback. Quyền phần mềm MIT không áp dụng cho email. Nazario yêu cầu attribution và ghi nhận bản dịch/adaptation. Publication approval vẫn NO.',
        'Bootstrap còn lại 36.280 hàng: tiếp tục giữ riêng 13.238 hàng có research-use statement của CMU (không phải blanket license redistribution) và 23.042 hàng chưa đủ evidence chain. Chưa tự phê duyệt các hàng này.',
        '', '## Nguồn mới đã tải','',
        f'Đã tải {acquisition["downloaded_files"]} archive, **{acquisition["downloaded_bytes"]:,} byte**, đọc **{acquisition["rows_received"]:,} bản ghi email**. Receipt từng file gồm URL gốc, version/commit, kích thước và SHA-256; tài liệu license/README được lưu riêng với SHA-256.',
        '- [Nazario 2015–2025](https://monkey.org/~jose/phishing/): 11 hộp thư công khai từ inbox của nhà phát hành, tác giả mô tả hand classification; [README](https://monkey.org/~jose/phishing/README.txt) và LICENSE ghi CC BY 4.0. Không tải file có tên private hoặc các mbox cũ đã dùng V2. Nhãn mới vẫn chờ source-quality/project approval.',
        '- [Phishing Pot](https://github.com/rf-peixoto/phishing_pot): khóa commit 78be141545ed9e56eefb857808ac78e83eb81b81; [LICENSE](https://github.com/rf-peixoto/phishing_pot/blob/78be141545ed9e56eefb857808ac78e83eb81b81/LICENSE) và dataset card cùng nêu CC BY-NC 4.0. Chỉ dùng nghiên cứu phi thương mại, giữ attribution. Corpus là thư nhận ở địa chỉ do tác giả kiểm soát/honeypot, không phải mailbox lấy từ breach; việc địa chỉ honeypot từng xuất hiện trong leaked lists không được dùng làm nguồn tải leaked data.',
        'Snapshot Phishing Pot chứa 12.864 email; toàn bộ 12.864 Git blob SHA và kích thước khớp cây GitHub đã khóa, 0 mismatch. Không clone lịch sử Git, không giải nén archive ra filesystem, không mở URL hoặc thực thi attachment. Các mẫu spam/fraud trong collection không được gọi chung là Phishing ba lớp.',
        '', '| Trạng thái acquisition | Hàng |','|---|---:|']
    lines+=['| '+key+' | '+f'{value:,}'+' |' for key,value in sorted(acquisition['status_counts'].items())]
    lines+=['',f'Near screening chạy trên {acquisition["near_checked_rows"]:,} hàng mới không empty/exact-duplicate; {acquisition["near_matched_rows"]:,} hàng có match near và {acquisition["near_overflow_rows"]:,} hàng gặp bucket lớn đều giữ review. Không tính chúng vào approved. Dùng lại MinHash/index có sẵn, không re-audit corpus 1,1 triệu hàng.',
        'Raw, parsed counts, weak, review và approved là các số khác nhau. Cả exact/normalized overlap với pre-split fingerprint index cũ và near/template overlap đều được ghi lý do. Không đọc file Final Test cũ hoặc dùng label/error của Test để lựa chọn.',
        '', '## Verified / Weak / Pseudo và khả năng semi-supervised','',
        f'- Verified: {verified["approved_reused_rows"]:,}, toàn bộ Phishing; 1.242 EN và 491 VI dịch. Train 1.427, Validation 306, không đổi assignment cũ. Chưa đủ ba lớp từ partition đã phê duyệt này để bắt đầu một model ba lớp mới.',
        f'- Weak-labeled mới: {acquisition["status_counts"].get("weak_label",0):,} hàng qua screening sơ bộ; candidate distributions trong JSON không phải ground truth. Review/quarantine mới: {acquisition["status_counts"].get("review",0):,}. Cần kiểm định nguồn, language và các quan hệ chưa giải quyết trước admission.',
        f'- Pseudo-labeled: {pseudo["rows"]}. Chưa chạy classifier/LLM để tạo pseudo-label; không có confidence giả.',
        '- Thí nghiệm semi-supervised sau này chỉ dùng weak/pseudo trong Train pool riêng có quyền phù hợp, khóa nhóm/provenance và source-specific semantics. Chọn threshold bằng Train CV có nhãn verified; LinearSVC margin không là probability. Cần verified Normal/Spam trước khi dùng teacher ba lớp. Mẫu positive hỗn hợp không tự trở thành Phishing.',
        '- Validation/Test chuẩn chỉ có nhãn đủ bằng chứng; không đẩy weak/pseudo vào đó. Không coi Validation development kế thừa là benchmark mới độc lập. Chưa tạo split hoặc fit teacher trong phase này.',
        '', '## Công cụ kiểm duyệt','',
        f'Đã chuẩn bị {review["content_ready"]}/{review["samples"]} nội dung gốc, {review["strata"]} tầng. Hiện {review["reviewed_samples"]} mẫu có quyết định và {review["decision_events"]} sự kiện quyết định thật. Không giả định đã có reviewer EN/VI/ZH.',
        'UI tách riêng khỏi VietMailGuard Mail: trước/sau, lọc nguồn/chưa review, giữ nguyên raw label/candidate và nội dung gốc plain text, ẩn contact mặc định, chọn/sửa nhãn, xác nhận ngôn ngữ, loại mẫu và lưu lịch sử append-only. Không tự cấp eligible dù người dùng chọn nhãn. Attestation là tự khai, không phải hệ thống đã xác minh năng lực reviewer. Phishing/nhãn mơ hồ vẫn cần hai reviewer/adjudication trước source approval.',
        'Lưu tiến độ ở data/v22/phase32/review.sqlite3 và human_review_decisions.csv. Không ghi vào cache/quyết định Phase 3 để bảo toàn snapshot cũ. Văn bản email là dữ liệu không đáng tin cậy; URL hiển thị trong ô text, không auto-open, không render HTML hoặc tải ảnh remote.',
        '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -m streamlit run scripts/v22/dataset_review_app.py --server.address 127.0.0.1 --server.port 8512 --browser.gatherUsageStats false',
        '```','',
        'Cần người hiểu ngôn ngữ gốc thực hiện review, ghi reviewer/competence/evidence. Người chưa hiểu Chinese đánh dấu cần xác minh; không dùng việc đọc bản dịch hoặc đồng ý AI thay ground truth. Seed/38 tầng/7.585 mẫu và tiêu chí chấp nhận Phase 3 giữ nguyên.',
        '', '## Khoảng cách tới 300.000','',
        f'Approved V2.2 local hiện **{summary["approved_total_unique"]:,}**, còn thiếu **{summary["remaining_to_300k"]:,}**. Normal 0, Spam 0, Phishing 1.733 trong partition đã phê duyệt; native VI 0 và native ZH 0 được phê duyệt. Không gọi 34 language signals zh trong acquisition là Chinese benchmark/ground truth đã xác minh.',
        '- [VNSED](https://vjs.ac.vn/jcc/article/view/22392) là ứng viên 6.008 email tiếng Việt; dataset-specific license/access chưa đủ rõ. Không tải hoặc liên hệ tác giả tự động. Cần quyền truy cập, consent/redaction và taxonomy ba lớp riêng trước nhận.',
        '- [CSRI_CNEPD](https://csri.scu.edu.cn/info/1012/1084.htm) có quyền nghiên cứu nhưng là record dò tài khoản/network, không có subject/body; không tải để cộng 764k record thành email. Corpus SMS/web/AI-generated cũng không thay native email.',
        '- [IWSPA-AP v2](https://www2.cs.uh.edu/~rmverma/book.html) yêu cầu institutional identity/NDA; không vượt quyền hoặc gửi giấy tờ thay chủ dự án. Chinese/Vietnamese native phishing có license và nhãn phù hợp vẫn là khoảng trống.',
        '', 'Chưa đủ dữ liệu đủ chuẩn để cam kết 300.000. Đã mở rộng dữ liệu thực tế, nhưng số tải về không thay thế bằng chứng nhãn, quyền, ngôn ngữ và grouping.', '']
    checks=folder/'phase32_verification.json'
    if checks.exists():lines+=['## Kiểm thử và bảo toàn','',*['- '+x for x in json.loads(checks.read_text(encoding='utf-8'))['checks']],'']
    lines+=['Files: config/v22/acquisition.json; src/vietmailguard/v22/acquisition.py; src/vietmailguard/v22/dataset_review.py; scripts/v22/acquire_public_data.py; scripts/v22/prepare_dataset_review.py; scripts/v22/dataset_review_app.py; scripts/v22/report_acquisition.py; tests/test_v22_acquisition_review.py. Chỉ dữ liệu/artifact mới trong data/v22/phase32 và báo cáo reports/v22.','']
    (folder/'PHASE3_2_ACQUISITION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=[ROOT/name for name in ['config/v22/acquisition.json','src/vietmailguard/v22/acquisition.py','src/vietmailguard/v22/dataset_review.py',
        'scripts/v22/acquire_public_data.py','scripts/v22/prepare_dataset_review.py','scripts/v22/dataset_review_app.py','scripts/v22/report_acquisition.py','tests/test_v22_acquisition_review.py']]
    paths+=list((area/'verified').glob('*'))+list((area/'weak_labeled').glob('*.csv'))+list((area/'license_evidence').glob('*'))
    paths+=[folder/name for name in ('DATASET_ACQUISITION_SUMMARY.json','PHASE3_2_ACQUISITION_REPORT.md','phase32_verification.json','phase32_snapshot_integrity.json','phase32_ui_smoke.json')]
    (folder/'phase32_manifest.json').write_text(json.dumps({'sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths if p.is_file()},
        'publication_approved':False,'raw_artifact_hashes_in_summary':True},indent=2),encoding='utf-8')
    print(json.dumps({'approved_local_reused':verified['approved_reused_rows'],'acquired_rows':acquisition['rows_received'],
        'weak_pool_rows':acquisition['status_counts'].get('weak_label',0),'human_reviewed':review['reviewed_samples']}))


if __name__=='__main__':main()
