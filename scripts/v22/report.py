"""Render actual Phase 1/2 measurements; no approvals, split or training."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    report=ROOT/'reports/v22'
    inventory=json.loads((report/'phase1_inventory.json').read_text())
    quality=json.loads((report/'quality_summary.json').read_text())
    audit=inventory['historical_audit']; cur=inventory['curation']; bootstrap=inventory['bootstrap']
    resources=inventory['resources']
    lines=['# VietMailGuard V2.2 — Phase 1/2', '', 'Ngày báo cáo: 2026-11-18. Nhánh: `v22/large-scale-dataset`.',
        '', '**Phạm vi:** audit và pipeline chất lượng; chưa mở Phase 3–7. Không huấn luyện, split, gọi API hoặc mở Final Test cũ.',
        '', '## Kết quả thực tế', '',
        f"- Inventory: {inventory['inventory_files']} file nguồn tại các dataset roots của project và biến môi trường V21/V22 được cấu hình; đã SHA-256 từng file. Không quét toàn bộ ổ đĩa hoặc thư cá nhân ngoài các roots này.",
        '- SHA-256 local xác định phiên bản/kiểm tra bất biến, không tự chứng minh nguồn gốc, giấy phép hoặc khớp checksum nhà phát hành. Các chuỗi provenance chưa đủ bằng chứng vẫn pending.',
        f"- Audit SQLite cũ đọc read-only: {audit['total_rows']:,} hàng; 44.717 là V2 reference, không phải dữ liệu mới và không được đưa vào V2.2 Train.",
        f"- Nội dung nguồn mới đã kiểm tra: {quality['rows']:,} hàng; unique normalized: {quality['normalized_unique']:,}.",
        f"- Exact unique: {quality['exact_unique']:,}; nhóm lặp exact/normalized/template: {quality['exact_duplicate_groups']:,}/{quality['normalized_duplicate_groups']:,}/{quality['template_duplicate_groups']:,}. Các loại nhóm có giao nhau, không cộng để suy ra số email độc lập.",
        f"- Quarantine V2.2: {quality['states']}; phê duyệt mới: {quality['newly_approved_rows']}; quyết định người kiểm duyệt: {quality['human_decisions']}.",
        f"- Trạng thái curation V2.1 được bảo toàn: {cur['status_counts']}. Weak label không trở thành ground truth; V2.2 đưa mọi hàng chưa đủ bằng chứng vào review.",
        f"- Near pass hoàn thành: {quality['near_pass_complete']}; xử lý {quality['near_processed_unique']:,} fingerprint; {quality['near_candidate_edges']:,} cạnh gợi ý, {quality['near_overflow_rows']:,} fingerprint gặp bucket quá tải.",
        f"- Xung đột kế thừa: {quality['inherited_conflict_rows']} hàng; {quality['old_reference_overlap_rows']:,} hàng chồng fingerprint/provenance group với V2 reference, lưu guard riêng. Không truyền nhãn V2 qua các match này.",
        '', '| Nguồn | Hàng cũ | Hàng V2.2 | Quyết định |', '|---|---:|---:|---|']
    for source,count in quality['source_counts'].items():
        lines.append(f"| {source} | {audit['by_dataset'][source]:,} | {count:,} | chưa phê duyệt ba lớp |")
    lines += ['', '## Tính khả thi 300.000 mẫu', '',
        f"Nguồn mới có {audit['new_source_unique_normalized']:,} normalized fingerprint duy nhất, không phải {sum(quality['source_counts'].values()):,} email độc lập. Còn quan hệ template, near-duplicate, bản dịch, nguồn chồng lặp và chất lượng nhãn cần giải quyết. Không cộng Merged + Balanced như nguồn độc lập.",
        'V2 bootstrap có 38.013 hàng development (31.304 Train / 6.709 Validation), 35.539 final groups: Normal 34.734, Spam 1.546, Phishing 1.733; EN 35.587, VI translated 2.426, ZH 0. Đây là bằng chứng nhãn development đã kế thừa, chưa phải phê duyệt V2.2 hoặc benchmark mới. Không cộng lại 44.717 V2 reference.',
        'Nếu cho phép tái sử dụng bootstrap sau kiểm tra quyền/group, chênh lệch số hàng tới 300.000 vẫn là 261.987; số hàng độc lập thấp hơn do bản dịch/nhóm liên quan. Hiện **V2.2 approved = 0**, không tuyên bố đạt 300.000.',
        'Metadata nguồn mới có EN 599.576 hàng / 323.699 normalized fingerprint; VI 2/2; ZH 64.935/34.740. Đây là ngôn ngữ khai báo, chưa xác minh bởi người bản ngữ. Các signal nội dung mới nằm trong quality_summary.json/distribution.csv, không được gọi là language ground truth.',
        'VI thiếu dữ liệu bản địa quy mô lớn. ZH thiếu ít nhất 25.260 fingerprint so với mục tiêu 60.000 ngay cả khi bỏ qua chất lượng; còn chưa có Phishing ba lớp được xác minh. TREC chỉ có Ham/Spam. Merged/MeAJOR positive trộn Spam/Scam/Phishing; Enron Original không có nhãn. Không thể suy ra đủ 75.000 Phishing từ các nhãn binary.',
        '', '## Quyền sử dụng / provenance', '',
        '- [NIST Spam Track](https://trec.nist.gov/data/spam.html), kiểm tra trực tiếp ngày 2026-11-18: TREC 2006/2007 không còn được phân phối vì PII. Không phân phối lại TREC06C; cần quyết định quyền/privacy riêng trước sử dụng rộng hơn. Việc đang có ZIP local không chứng minh quyền phân phối.',
        '- [CMU Enron](https://www.cs.cmu.edu/~enron/) mô tả tài nguyên nghiên cứu và cảnh báo quyền riêng tư; không chứng minh email đều Normal hoặc đã được cấp quyền phân phối vô điều kiện.',
        '- Merged/Balanced/MeAJOR: xem DATASETS.md và reports/v21/source_approval_matrix.md; license từng thành phần và chuỗi quyền còn NOT VERIFIED. Không phê duyệt bằng filename hoặc metadata article license.',
        '- Synthetic ZH: 3.030 hàng, Apache-2.0 được source card khai báo, không đồng nghĩa mọi quyền nội dung đã được xác minh. Tách research-only, không tính là email thật được xác minh. PhishVN là URL dataset, không phải email. Phishing_validation giữ challenge-only. Các corpus V2 đã có trong bootstrap không được cộng lại từ raw.',
        'Source policy chi tiết: config/v22/source_policy.json. Inventory chỉ chứa đường dẫn tương đối theo input_root, kích thước/hash; không có nguyên văn email hoặc credential.',
        '', '## Pipeline và giới hạn kỹ thuật', '',
        '1. Stream CSV từng part; đối chiếu size/mtime với audit cũ, SHA-256 checkpoint V2.2, kiểm tra lại exact/normalized/template fingerprint từng hàng. Không rewrite raw hoặc V2/V2.1.',
        '2. NFC, loại control chars, plain-text HTML an toàn, whitespace dùng cho quality/review; giữ subject/body gốc trong nguồn. URL không được truy cập. Fingerprint so trùng kế thừa dùng NFKC/casefold và URL/email placeholders đủ ngữ cảnh.',
        '3. Quality flags: rỗng, decode/MIME/missing content, replacement character, ngắn, language signal không rõ hoặc khác metadata. Ngắn không có nghĩa Spam/Phishing; language signal dựa script/lexicon chỉ là gợi ý, cần xác minh độc lập.',
        '4. SimHash 64 bit, 4 bands × 16 bit; prefix 512 ký tự, bottom-64 char 5-gram CRC sketch; Hamming ≤3 và sketch Jaccard ≥0,8 tạo cạnh review. Đây KHÔNG phải exact full-text Jaccard, không exhaustive và chưa đo recall near-duplicate. Bucket giới hạn 100; overflow được ghi rõ. Có thể bỏ sót thay đổi ngoài prefix, cross-language semantic duplicates và collision. Không tự xóa, gộp hoặc cấp nhãn.',
        '5. Giữ parent_id/provenance groups, các fingerprint/cạnh near và old-reference guard; trước split phải union toàn bộ quan hệ đã xác minh, resolve conflict, kiểm tra translation lineage và phủ đầy đủ protected-test fingerprint manifests. Guard hiện tại phủ fingerprint/provenance group V2 reference; chưa chứng nhận phủ near/cross-language hay toàn bộ TREC binary / các test nghiên cứu cũ. Thiếu coverage phải tiếp tục chặn admission, không mở Test để bù. Chưa chứng nhận leakage-free cho một tập Train V2.2 vì tập đó chưa được tạo.',
        '6. Review queue phân tầng nguồn/raw label/signal/quality flags, tối đa 5 hàng mỗi tầng, metadata-only. Đây là queue công việc, không phải mẫu ngẫu nhiên đại diện để ước lượng sai nhãn nguồn. Human decision có reviewer/evidence/competence/time UTC; ghi quyết định không tự kích hoạt bulk approval. Muốn eligible cần quyền sử dụng, nhãn/language có bằng chứng, curation_rule_id, group/overlap/conflict được giải quyết và phê duyệt riêng.',
        '7. SQLite giao dịch từng part và từng 1.000 near fingerprints; rollback part đang dở, resume không nhân đôi. Input/config/audit hash thay đổi sẽ bị chặn. Cache RAM giới hạn (64 MiB output, 192 MiB read-only audit); không có dense TF-IDF hoặc encoder trong phase này.',
        '', '## Tài nguyên', '',
        f"CPU i5-12500H, 12 cores / {resources['logical_cpus']} logical CPUs; RAM tổng {resources['total_ram_gib']:.2f} GiB, trống lúc inventory {resources['free_ram_gib']:.2f} GiB; SSD trống {resources['disk_free_gib']:.2f} GiB.",
        'nvidia-smi xác nhận RTX 3050 Laptop 4.096 MiB VRAM, trống 2.712 MiB tại thời điểm kiểm tra. Không dùng GPU cho audit này. RAM trống thấp là lý do tiếp tục stream/disk-backed; chưa đủ cơ sở hứa full embedding/MLP training 300k với VRAM này. Embedding cache hiện có của V2.1 giữ nguyên, tương lai phải khóa encoder/config/content hashes trước tái sử dụng.',
        '', '## Lệnh Windows', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/v22/inventory.py',
        '.\\.venv\\Scripts\\python.exe -u scripts/v22/audit_quality.py --run',
        '.\\.venv\\Scripts\\python.exe scripts/v22/audit_quality.py --summary',
        '.\\.venv\\Scripts\\python.exe scripts/v22/audit_quality.py --review-pack "SOURCE|ID"',
        '.\\.venv\\Scripts\\python.exe scripts/v22/audit_quality.py --review-id "SOURCE|ID" --label review --reviewer "NAME" --evidence "EVIDENCE" --language-competence "QUALIFICATION"',
        '.\\.venv\\Scripts\\python.exe -m pytest tests/test_v22_quality.py tests/test_v21_dedup_audit.py tests/test_v21_label_curation.py -q',
        '```', '',
        'Queue, review packs/decisions, database và reports/v22 bị gitignore. Review pack chứa plain email riêng tư, chỉ mở local; không đưa lên Git. Không nhập evidence/PII nhạy cảm vào CLI history nếu môi trường không được bảo vệ. Người không hiểu ngôn ngữ gốc chuyển chuyên gia; AI không là ground truth.',
        '', '## Điều kiện trước bước tiếp theo', '',
        '- Giải quyết quyền/privacy từng nguồn và xác nhận source-level policy đủ bằng chứng; không bulk approve weak labels.',
        '- Review language/nhãn ba lớp, nhất là nguồn binary positive và Chinese Phishing; thu thập native Vietnamese có consent/provenance.',
        '- Hoàn tất grouping có near/provenance/translation, đo khả năng phát hiện overlap và resolve mọi conflict trước Train.',
        '- Chỉ sau phê duyệt dataset mới thực hiện Phase 3/4; tỷ lệ 80/10/10 là đề xuất tương lai, không đổi assignment cũ. Test mới phải độc lập với development đã dùng và chỉ đánh giá sau freeze.',
        '- Training, model selection, dashboard/README/MODEL_CARD update và integration chưa được thực hiện trong nhiệm vụ này.', '']
    verification=report/'verification.json'
    if verification.exists():
        data=json.loads(verification.read_text())
        lines += ['## Kiểm thử và bảo toàn', '',*['- '+item for item in data['checks']], '',
            'Files: .gitignore; config/v22/quality.json; config/v22/source_policy.json; src/vietmailguard/v22/__init__.py; src/vietmailguard/v22/quality.py; scripts/v22/inventory.py; scripts/v22/audit_quality.py; scripts/v22/report.py; tests/test_v22_quality.py. Tất cả kết quả nằm riêng data/v22 và reports/v22.', '']
    (report/'PHASE1_PHASE2_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=[ROOT/'.gitignore',ROOT/'tests/test_v22_quality.py']
    for area,pattern in [('src/vietmailguard/v22','*.py'),('scripts/v22','*.py'),('config/v22','*.json'),('reports/v22','*.md'),('reports/v22','*.csv'),('reports/v22','*summary.json'),('reports/v22','phase1_inventory.json'),('reports/v22','verification.json')]:
        paths.extend((ROOT/area).glob(pattern))
    hashes={}
    for path in sorted(set(paths)):
        with path.open('rb') as handle:
            hashes[path.relative_to(ROOT).as_posix()]=hashlib.file_digest(handle,'sha256').hexdigest()
    (report/'run_manifest.json').write_text(json.dumps({'sha256':hashes,
        'source_audit_sha256':audit['audit_sha256'],'scope':'phase_1_2_only',
        'publication_approved':False,'training_performed':False,'old_final_test_opened':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()
