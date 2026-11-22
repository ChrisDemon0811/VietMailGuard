"""Phase 3.1 measured recovery and source-approval evidence; private outputs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .quality import digest
from .recovery import connect


def bootstrap_evidence(root: Path) -> dict:
    """Reuse already-verified metadata; distinguish local use from publication."""
    area=root/'data/v21/curated/bootstrap_v2_development'
    manifest=json.loads((area/'manifest.json').read_text(encoding='utf-8'))
    policy=json.loads((root/'config/v21/bootstrap_development_policy.json').read_text(encoding='utf-8'))
    if manifest['leakage']['passed'] is not True:raise ValueError('Bootstrap guard evidence not PASS')
    for split in ('train','validation'):
        if digest(area/(split+'.csv'))!=manifest['output_sha256'][split]:raise ValueError('Bootstrap changed')
    proposal=root/'data/v22/phase3/bootstrap_proposal.csv'
    evidence=json.loads((root/'reports/v22/phase3_manifest.json').read_text(encoding='utf-8'))
    if digest(proposal)!=evidence['sha256'][proposal.relative_to(root).as_posix()]:raise ValueError('Bootstrap proposal changed')
    with proposal.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    records={row['id']:row for row in rows};groups={};counts=Counter();langs=Counter();origins=Counter();splits=Counter();rights=Counter()
    cc_counts=Counter();cc_languages=Counter();statements=Counter()
    destination=root/'data/v22/phase31/bootstrap_reuse_manifest.csv'
    with destination.open('w',encoding='utf-8',newline='') as out:
        fields=list(rows[0])+['local_use_evidence','attribution','adoption_status'];writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for row in rows:
            if row['split'] not in ('train','validation'):raise ValueError('Non-development bootstrap assignment')
            if policy['allowed_labels_by_provenance'].get(row['label_provenance'])!=row['label']:raise ValueError('Unverified bootstrap provenance')
            value=(row['split'],row['label']);key=row['final_group_id']
            if key in groups and groups[key]!=value:raise ValueError('Bootstrap group/split conflict')
            groups[key]=value
            parent=records.get(row['parent_id']) if row['parent_id'] else None
            if row['source']=='controlled_translation' and (not parent or parent['label']!=row['label'] or parent['final_group_id']!=key or parent['split']!=row['split']):
                raise ValueError('Controlled translation parent integrity failed')
            original=parent if row['source']=='controlled_translation' else row
            use='not_verified';attribution=''
            if original['source']=='Nazario' and original['label_provenance']=='nazario_phishing':
                use='cc_by_4_0_explicit';attribution='Jose Nazario, phishing corpus, https://monkey.org/~jose/phishing/README.txt; retain attribution; Vietnamese controlled translations are adaptations'
                cc_counts[row['label']]+=1;cc_languages[row['language']]+=1
            elif original['source']=='Enron' and original['label_provenance']=='verified_legitimate_source':
                use='publisher_research_use_statement';attribution='CMU Enron resource for researchers, https://www.cs.cmu.edu/~enron/; privacy obligations remain; not a blanket redistribution license'
                statements[row['label']]+=1
            rights[use]+=1;counts[row['label']]+=1;langs[row['language']]+=1;origins[(row['language'],row['data_origin'] or 'not_declared')]+=1;splits[row['split']]+=1
            writer.writerow({**row,'local_use_evidence':use,'attribution':attribution,'adoption_status':'owner_adoption_pending; development only; not new source approval'})
    if any(splits[key]!=count for key,count in manifest['output_rows'].items()):raise ValueError('Bootstrap row counts mismatch')
    return {'verified_inherited_development_rows':len(rows),'groups':len(groups),'by_label':dict(counts),'by_language':dict(langs),
        'by_language_origin':{lang+'|'+origin:n for (lang,origin),n in origins.items()},'split_rows':dict(splits),
        'use_evidence_counts':dict(rights),'explicitly_licensed_inherited_rows':sum(cc_counts.values()),
        'explicitly_licensed_by_label':dict(cc_counts),'explicitly_licensed_by_language':dict(cc_languages),
        'publisher_research_statement_rows':sum(statements.values()),'new_source_rows':0,'owner_adoption_confirmed':False,
        'independent_benchmark':False,'old_test_opened':False}


def export(conn,path,query):
    cursor=conn.execute(query)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow([c[0] for c in cursor.description]);writer.writerows(cursor)


def generate(root: Path,config: dict) -> dict:
    report=root/'reports/v22';output=root/'data/v22/phase31';output.mkdir(parents=True,exist_ok=True)
    with connect(root) as conn:
        complete=conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='recovery_complete'").fetchone()
        if not complete:raise ValueError('Incomplete recovery; cannot issue approval summary')
        summary={'technical_recovery':json.loads(complete[0]),'new_source_rows_approved':conn.execute("SELECT count(*) FROM phuc_hoi WHERE nhan_duoc_duyet!=''").fetchone()[0],
            'human_decisions':conn.execute('SELECT count(*) FROM ba.quyet_dinh').fetchone()[0],
            'source_review_sample_rows':conn.execute('SELECT count(*) FROM ba.lay_mau').fetchone()[0],
            'source_review_strata':conn.execute('SELECT count(DISTINCT tang) FROM ba.lay_mau').fetchone()[0],
            'source_review_policy_sha256':digest(root/'config/v22/expansion.json'),
            'source_review_queue_sha256':digest(root/'data/v22/phase3/review_queue.csv'),
            'strict_approved_by_label':{key:0 for key in config['target_labels']},
            'strict_approved_by_language':{key:0 for key in config['target_languages']},
            'source_approval_activated':False,'dataset_finalized':False,'phase4_ready':False}
        summary['quarantine_before']=conn.execute('SELECT count(*) FROM phuc_hoi WHERE cach_ly_truoc=1').fetchone()[0]
        summary['original_quarantine_recovered']=conn.execute("SELECT count(*) FROM phuc_hoi WHERE trang_thai='technical_recovered_label_pending'").fetchone()[0]
        summary['original_quarantine_remaining']=summary['quarantine_before']-summary['original_quarantine_recovered']
        summary['all_quarantined_now']=conn.execute("SELECT count(*) FROM phuc_hoi WHERE trang_thai='quarantined'").fetchone()[0]
        summary['newly_quarantined_from_previous_clear']=summary['all_quarantined_now']-summary['original_quarantine_remaining']
        summary['net_quarantine_reduction']=summary['quarantine_before']-summary['all_quarantined_now']
        summary['technical_clear_fingerprints_label_pending']=conn.execute("SELECT count(*) FROM phuc_hoi WHERE trang_thai!='quarantined'").fetchone()[0]
        bits={1:'legacy_overflow',2:'minhash_overflow',4:'unverified_prefix_relation',8:'protected_reference_overlap',
            16:'candidate_label_conflict',32:'insufficient_short_context',64:'decode_or_mime_quality'}
        summary['remaining_reason_counts']={name:conn.execute('SELECT count(*) FROM phuc_hoi WHERE ma_ly_do & ? != 0',(bit,)).fetchone()[0] for bit,name in bits.items()}
        summary['lexical_retrieval_validation']=[]
        for (stratum,) in conn.execute('SELECT DISTINCT tang FROM cap_kiem_dinh'):
            pairs=conn.execute('SELECT jaccard,truy_hoi,cach_ly FROM ket_qua_cap WHERE tang=?',(stratum,)).fetchall()
            positives=[row for row in pairs if row[0]>=config['minhash_similarity']]
            expected=conn.execute('SELECT count(*) FROM cap_kiem_dinh WHERE tang=?',(stratum,)).fetchone()[0]
            summary['lexical_retrieval_validation'].append({'cohort':stratum,'selected_pairs':expected,'checked_pairs':len(pairs),
                'unscored_pairs':expected-len(pairs),'lexical_positive_pairs':len(positives),
                'retrieved_positive_pairs':sum(p[1] for p in positives),
                'candidate_recall':sum(p[1] for p in positives)/len(positives) if positives else None,
                'missed_positives_in_prior_quarantine':sum(not p[1] and p[2] for p in positives),
                'semantic_duplicate_human_recall':None})
        summary['source_strata_progress']=[]
        for stratum,population,sample in conn.execute('SELECT tang,max(dan_so),count(*) FROM ba.lay_mau GROUP BY tang'):
            reviewed=conn.execute('''SELECT count(DISTINCT q.id_mau) FROM ba.quyet_dinh q JOIN ba.lay_mau s USING(id_mau)
                WHERE s.tang=?''',(stratum,)).fetchone()[0]
            summary['source_strata_progress'].append({'stratum':stratum,'population_groups':population,'sample_groups':sample,
                'human_reviewed':reviewed,'outstanding':sample-reviewed,'observed_label_error_rate':None,
                'acceptance':'pending competent human review/adjudication; no activation'})
        summary['source_samples_need_human_review']=sum(row['outstanding'] for row in summary['source_strata_progress'])
        export(conn,output/'recovery_manifest.csv','SELECT * FROM phuc_hoi ORDER BY id_van_ban')
        export(conn,output/'source_review_precheck.csv','''SELECT s.id_mau AS audit_id,s.tang AS stratum,m.nguon AS source,
            m.nhan_goc AS raw_label,m.nhan_ung_vien AS candidate_label,m.ngon_ngu_goi_y AS language_signal,m.ly_do AS source_flags,
            p.id_nhom AS recovered_group,p.trang_thai AS technical_status,p.ma_ly_do AS technical_reason_mask
            FROM ba.lay_mau s JOIN ba.mau m USING(id_mau) LEFT JOIN phuc_hoi p USING(ma_bam) ORDER BY s.tang,s.id_mau''')
        export(conn,output/'candidate_label_conflicts.csv',"SELECT id_mau,nguon,nhan_goc,nhan_ung_vien,ma_bam,id_nhom,ly_do FROM ba.mau WHERE ly_do LIKE '%candidate_label_conflict%'")
        export(conn,output/'bucket_recovery.csv','SELECT * FROM khoi_kiem_dinh ORDER BY loai,ma_khoi')
        export(conn,output/'lexical_validation_pairs.csv','SELECT * FROM ket_qua_cap ORDER BY tang,id_truoc,id_sau')
    summary['bootstrap']=bootstrap_evidence(root)
    summary['remaining_to_300k_strict_approved']=config['target_rows']-summary['new_source_rows_approved']
    summary['conditional_gap_if_all_bootstrap_adopted']=config['target_rows']-summary['bootstrap']['verified_inherited_development_rows']
    summary['conditional_gap_if_explicitly_licensed_subset_adopted']=config['target_rows']-summary['bootstrap']['explicitly_licensed_inherited_rows']
    (report/'SOURCE_APPROVAL_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    write_reports(root,summary)
    paths=[root/'src/vietmailguard/v22/recovery.py',root/'src/vietmailguard/v22/recovery_report.py',root/'scripts/v22/recover_dataset.py',root/'tests/test_v22_recovery.py']
    paths+=list(output.glob('*.csv'))+[report/name for name in ('PHASE3_1_APPROVAL_REPORT.md','SOURCE_APPROVAL_SUMMARY.json','QUARANTINE_RECOVERY_REPORT.md','DATASET_GAP_ANALYSIS.md','phase31_verification.json')]
    (report/'phase31_manifest.json').write_text(json.dumps({'sha256':{p.relative_to(root).as_posix():digest(p) for p in paths if p.is_file()},
        'publication_approved':False,'dataset_finalized':False},indent=2),encoding='utf-8')
    return {key:summary[key] for key in ('new_source_rows_approved','original_quarantine_recovered','original_quarantine_remaining','source_samples_need_human_review')}


def write_reports(root: Path,s: dict) -> None:
    folder=root/'reports/v22';b=s['bootstrap'];t=s['technical_recovery']
    approval=['# Phase 3.1 — Dataset Approval & Recovery','',
        f'Nhánh v22/large-scale-dataset. Nhãn mới được phê duyệt: **{s["new_source_rows_approved"]:,}**. Quyết định người thật: {s["human_decisions"]}. Chưa chốt dataset hoặc chuyển Phase 4.',
        '', '## Bootstrap đã kiểm tra','',
        f'- {b["verified_inherited_development_rows"]:,} hàng / {b["groups"]:,} nhóm có provenance và nhãn kế thừa; SHA-256 file khớp manifest. Train/Validation giữ nguyên: {b["split_rows"]}.',
        f'- Theo lớp: {b["by_label"]}; theo ngôn ngữ: {b["by_language"]}; origin: {b["by_language_origin"]}. Không đổi thành dữ liệu mới hoặc benchmark độc lập.',
        f'- **{b["explicitly_licensed_inherited_rows"]:,} hàng kế thừa có chuỗi CC BY 4.0 rõ**: {b["explicitly_licensed_by_language"]}, {b["explicitly_licensed_by_label"]}. Bao gồm bản dịch được đánh dấu adaptation, parent/group/split khớp. Đây là subset đề xuất nhận lại, chưa tự kích hoạt adoption V2.2.',
        f'- {b["publisher_research_statement_rows"]:,} hàng có bằng chứng nhà phát hành cung cấp cho nghiên cứu, phân biệt với giấy phép redistribution chưa rõ. Các nguồn còn lại: {b["use_evidence_counts"]}. Không kết luận quyền local bị cấm chỉ vì chưa xác minh quyền public.',
        '- [Nazario README](https://monkey.org/~jose/phishing/README.txt) ghi hand-classified và CC BY 4.0; cần attribution, ghi nhận bản dịch/biến đổi và bảo vệ PII. [CMU Enron](https://www.cs.cmu.edu/~enron/) mô tả tài nguyên cho nghiên cứu và nghĩa vụ thận trọng privacy; không đồng nghĩa mọi thư Enron Original là Normal.',
        '', '## Source approval theo sample đã khóa','',
        f'Giữ nguyên {s["source_review_strata"]} tầng và {s["source_review_sample_rows"]:,} mẫu. Policy SHA-256: `{s["source_review_policy_sha256"]}`. Không thay sample, seed, giới hạn lỗi hay tiêu chí sau kiểm định.',
        f'Đã chạy precheck kỹ thuật trên toàn bộ mẫu; còn **{s["source_samples_need_human_review"]:,}** mẫu cần người có năng lực ngôn ngữ gốc đánh giá. Observed label error rate vẫn null khi chưa có quyết định/adjudication. Không giả lập người kiểm duyệt.',
        'Giữ quy tắc tối đa 299 nhóm/tầng, 0 lỗi/0 unresolved và upper bound một phía 95% dưới 1% cho tầng đủ cỡ mẫu; tầng nhỏ kiểm tra census. Chỉ kết luận theo từng tầng, không bảo đảm đồng thời toàn corpus. Phishing/nhãn mơ hồ cần hai reviewer và adjudication; binary positive không mặc định là Phishing.',
        'TREC Ham, Merged raw 0, MeAJOR raw 0 vẫn cần kết quả kiểm định và bằng chứng quyền thích hợp. TREC Spam/mixed positive cần phân biệt Spam/Scam/Phishing. Enron Original không có nhãn nguồn an toàn nên chỉ duyệt các mẫu/nhóm có bằng chứng riêng. Balanced không tăng quota độc lập.',
        '', '## Các quyết định cần chủ dự án xác nhận','',
        '1. Có nhận subset bootstrap có CC BY 4.0 đã liệt kê để dùng development local không? Giữ assignment cũ, attribution và bản dịch; không dùng làm Final Test mới.',
        '2. Người kiểm duyệt EN/VI/ZH là ai, ai xử lý adjudication? Hoàn thành 38 tầng với tiêu chí cũ; owner không chỉ bấm đồng ý thay cho việc đọc và kiểm định thực tế.',
        '3. Với nguồn chưa có đủ bằng chứng quyền hoặc nhãn: cần xác nhận bằng chứng phạm vi nghiên cứu cụ thể, không chỉ phê duyệt toàn nguồn bằng tên file.',
        '', '## Lệnh tiếp tục review đã có','', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/v22/audit_quality.py --review-pack "SOURCE|ID"',
        '.\\.venv\\Scripts\\python.exe scripts/v22/expand_dataset.py --review-id "SOURCE|ID" --label normal --reviewer "NAME" --language-competence "QUALIFICATION" --evidence "EVIDENCE"',
        '.\\.venv\\Scripts\\python.exe scripts/v22/recover_dataset.py --report-only', '```','',
        'Queue gốc và precheck ở data/v22/phase3/review_queue.csv và data/v22/phase31/source_review_precheck.csv. Email/derived text được giữ trong cache local bị ignore; không công khai hoặc gửi API. Ghi quyết định không tự cấp source approval.', '']
    recovery=['# Quarantine Recovery — Phase 3.1','',
        f'- Quarantine đầu vào: {s["quarantine_before"]:,} fingerprint. Khôi phục kỹ thuật: **{s["original_quarantine_recovered"]:,}**; còn cách ly trong cohort cũ: **{s["original_quarantine_remaining"]:,}**. Các con số là fingerprint, không phải số email có nhãn mới được duyệt.',
        f'- Trạng thái toàn bộ fingerprint: {t["status_counts"]}. Recovery giữ đầy đủ constraint groups cũ và bổ sung quan hệ mới. Không split dữ liệu.',
        f'- Có thêm {s["newly_quarantined_from_previous_clear"]:,} fingerprint trước đây clear được cách ly do quan hệ/quality blocker mới; tổng cách ly hiện tại {s["all_quarantined_now"]:,}. Mức giảm ròng là {s["net_quarantine_reduction"]:,}, không gọi số khôi phục gross là mức giảm ròng. {s["technical_clear_fingerprints_label_pending"]:,} fingerprint clear kỹ thuật vẫn cần nhãn/quyền/language review.',
        f'- Hoàn thành signature toàn văn cho {t["long_signatures_completed"]} email dài (cả email trên 4 triệu ký tự), bằng các lô vị trí shingle 4.096. Không cắt prefix; kết quả trùng thuật toán cũ trên miền ≤50.000 ký tự.',
        f'- Hoàn thành {t["short_signatures_completed"]} signature email ngắn; dưới 5 ký tự không đủ char-5 signature. Nội dung dưới 20 ký tự vẫn cần ngữ cảnh. Join char-3 exact trong cohort short dùng prefix filtering đầy đủ, tìm {t["short_exact_edges"]:,} cạnh; không chứng nhận recall toàn corpus qua mọi miền độ dài.',
        f'- Kiểm tra đầy đủ {t["overflow_blocks_completely_checked"]} overflow blocks chọn theo seed/budget trước khi đọc nội dung; {t["overflow_blocks_still_pending"]} block đã chọn còn pending. Mỗi block được xét toàn bộ thành viên bằng Jaccard char-5 exact/prefix join, không lấy vài thành viên thay cả bucket. Các bucket khác vẫn giữ cách ly.',
        f'- {t["exact_prefix_edges_checked"]:,} cạnh prefix tồn đọng đã so sánh exact; {t["exact_prefix_positive_edges"]:,} cạnh đạt ngưỡng được giữ cùng group. Các quan hệ ngoài cache chưa được suy rộng là đã giải quyết.',
        f'- Nguyên nhân còn lại trên toàn bộ fingerprint (có giao nhau, đã propagate theo nhóm): {s["remaining_reason_counts"]}. Xung đột candidate giữ nguyên, xuất riêng để adjudication; không tự chọn label.',
        '', '## Kiểm định retrieval trên dữ liệu thật','',
        '| Cohort | Cặp chọn/đã xét | Lexical positive | Thu hồi | Recall |','|---|---:|---:|---:|---:|']
    for row in s['lexical_retrieval_validation']:
        recall='N/A' if row['candidate_recall'] is None else f'{row["candidate_recall"]:.6f}'
        recovery.append(f'| {row["cohort"]} | {row["selected_pairs"]}/{row["checked_pairs"]} | {row["lexical_positive_pairs"]} | {row["retrieved_positive_pairs"]} | {recall} |')
    recovery+=['','Đây là lexical retrieval QA trên cặp email thật: ground relation là full-set char-5 Jaccard ≥0,8125, không phải nhãn email và không phải duplicate semantic đã được người xác minh. Cohort legacy được lấy từ candidate prefix cũ, nên recall có điều kiện; random control quá ít positive có thể N/A. Không gộp thành global recall hoặc dùng kết quả này để hạ threshold. Cặp quá dài chưa tính exact được báo unscored trong JSON. Semantic human-verified recall vẫn chưa có.',
        '', 'Khôi phục kỹ thuật chỉ gỡ lý do cách ly đã giải quyết đầy đủ trong nhóm. Quyền sử dụng, nhãn ba lớp, language verification và protected-test fingerprint coverage vẫn là gate riêng. Không mẫu nào tự vào supervised training.', '']
    gap=['# Dataset Gap Analysis — Phase 3.1','',
        f'Strict V2.2 approved mới = {s["new_source_rows_approved"]}; còn thiếu {s["remaining_to_300k_strict_approved"]:,} so với mục tiêu 300.000. Bootstrap kế thừa có {b["verified_inherited_development_rows"]:,} nhãn verified-development, chưa phải phê duyệt mới.',
        f'Nếu toàn bộ bootstrap được nhận lại sau kiểm tra quyền/phạm vi: thiếu {s["conditional_gap_if_all_bootstrap_adopted"]:,}. Nếu chỉ nhận subset có chuỗi CC BY 4.0: thiếu {s["conditional_gap_if_explicitly_licensed_subset_adopted"]:,}. Đây là kịch bản, không số approved đã đạt.',
        '', '| Ngôn ngữ | Normal kế thừa | Spam kế thừa | Phishing kế thừa | Origin |','|---|---:|---:|---:|---|',
        '| EN | 33.298 | 1.047 | 1.242 | nguồn gốc theo từng corpus; không tự đổi origin rỗng thành native |',
        '| VI | 1.436 | 499 | 491 | translated, không phải native |','| ZH | 0 | 0 | 0 | chưa có ba lớp verified-development |',
        '', 'Native VI và Chinese Phishing vẫn thiếu nghiêm trọng; TREC Ham/Spam không xác minh Phishing. Dữ liệu dịch và duplicate không tăng nguồn native độc lập. Không gán sai nhãn để đủ mục tiêu Normal 150k/Spam 75k/Phishing 75k hoặc EN 150k/VI 90k/ZH 60k.',
        '', '## Nguồn bổ sung được khảo sát, chưa tải','',
        '- [VNSED, UIT/VNU-HCM, JCC 2026](https://vjs.ac.vn/jcc/article/download/22392/2543256498/2543290161): tác giả mô tả 6.008 email tiếng Việt (3.368 Ham/2.640 Spam), thu từ hộp thư và báo cáo người dùng. Positive gồm nhiều loại, không tự là Phishing. Ưu tiên liên hệ tác giả xin quyền dữ liệu/nghiên cứu, consent/redaction, dữ liệu trước preprocessing và nhãn type. Chưa thấy bằng chứng dataset license/access đủ để nhận nguồn; license bài báo không tự áp dụng cho email. 0 hàng đã nhận.',
        '- [Phishing Pot](https://github.com/rf-peixoto/phishing_pot/blob/main/DATASET_CARD.md): email thực, license CC BY-NC 4.0 cho nghiên cứu phi thương mại theo card/license. Cần khóa commit, kiểm kê VI/ZH, taxonomy, PII và overlap trước khi xin tải. Chưa có số native VI/ZH được xác minh; không được hứa 75k Phishing.',
        '- [DataCon Chinese email challenge](https://github.com/yaoyue123/datacon2023-spoof-email): liên hệ xin row-level labels và quyền; public samples chưa đủ nhãn/quyền cho ba lớp. Không dùng số lượng giả hoặc coi model prediction là ground truth.',
        '- [STING9 dataset](https://sting9.org/dataset) / [license](https://sting9.org/license): prior audit ghi điều khoản mâu thuẫn; lần kiểm tra này endpoint timeout. Chưa thể giải quyết quyền hoặc lượng Chinese/Vietnamese email verified; giữ pending, chưa tải.',
        '- Thu thập native VI/ZH có consent từ tổ chức/người tham gia: ghi quyền nghiên cứu, ẩn PII, hai reviewer cho Phishing/nhãn mơ hồ và một benchmark độc lập chưa dùng phát triển. Không sử dụng corpus SMS, URL, synthetic hoặc dữ liệu rò rỉ làm email thật.',
        '', 'Chưa đủ cơ sở xác nhận 300.000 mẫu đạt chuẩn từ nguồn hiện tại. Tăng số lượng test/code hoặc khôi phục cache không thay thế bằng chứng nhãn và dữ liệu native.', '']
    verification=folder/'phase31_verification.json'
    if verification.exists():approval+=['## Kiểm thử / bảo toàn','',*['- '+line for line in json.loads(verification.read_text(encoding='utf-8'))['checks']],'']
    approval+=['File code mới: src/vietmailguard/v22/recovery.py, src/vietmailguard/v22/recovery_report.py, scripts/v22/recover_dataset.py, tests/test_v22_recovery.py. Không sửa runtime, V2/V2.1 hoặc cache/quyết định Phase 3.','']
    for name,lines in [('PHASE3_1_APPROVAL_REPORT.md',approval),('QUARANTINE_RECOVERY_REPORT.md',recovery),('DATASET_GAP_ANALYSIS.md',gap)]:
        (folder/name).write_text('\n'.join(lines),encoding='utf-8')
