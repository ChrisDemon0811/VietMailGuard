"""Measured Phase 3 reports and private metadata exports; never activate rules."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from .expansion import connect, readonly, minhash
from .quality import digest


def inspect_bootstrap(root: Path) -> dict:
    """Reverify existing development-only files without opening any old Test."""
    area=root/'data/v21/curated/bootstrap_v2_development'
    manifest_path=area/'manifest.json'
    if not manifest_path.is_file():
        return {'rows':0,'status':'missing','new_v22_approval':False,'development_only':True}
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    policy=json.loads((root/'config/v21/bootstrap_development_policy.json').read_text(encoding='utf-8'))
    if manifest.get('leakage',{}).get('passed') is not True:
        raise ValueError('Bootstrap inherited leakage evidence is not PASS')
    counts=Counter();languages=Counter();origins=Counter();splits=Counter();groups={};ids={}
    files={}
    destination=root/'data/v22/phase3/bootstrap_proposal.csv'
    fields=['id','source','label','language','data_origin','parent_id','final_group_id','label_provenance','curation_rule_id','split','v22_approved']
    csv.field_size_limit(100_000_000)
    with destination.open('w',encoding='utf-8',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for split in ('train','validation'):
            path=area/(split+'.csv')
            if path.is_symlink() or digest(path)!=manifest['output_sha256'][split]:
                raise ValueError('Bootstrap output differs from approved development manifest')
            files[split]=digest(path)
            with path.open(encoding='utf-8-sig',newline='') as handle:
                for row in csv.DictReader(handle):
                    if (row.get('label') not in ('normal','spam','phishing') or not row.get('label_provenance')
                        or not row.get('curation_rule_id') or not row.get('final_group_id') or not row.get('content_hash')
                        or row.get('label_status') not in policy['allowed_label_status'] or row.get('split')!=split
                        or policy['allowed_labels_by_provenance'].get(row['label_provenance'])!=row['label']
                        or row.get('training_eligible','').casefold()!='true'):
                        raise ValueError('Bootstrap row lacks verified provenance/schema')
                    identifier=row['id'];group=row['final_group_id']
                    if identifier in ids:raise ValueError('Repeated bootstrap ID')
                    if group in groups and groups[group]!=(split,row['label']):
                        raise ValueError('Bootstrap group crosses development splits or conflicts')
                    ids[identifier]=(split,group,row['label']);groups[group]=(split,row['label'])
                    counts[row['label']]+=1;languages[row['language']]+=1
                    origins[row['language']+'|'+(row['data_origin'] or 'not_declared')]+=1;splits[split]+=1
                    out={key:row.get(key,'') for key in fields};out['v22_approved']='false';writer.writerow(out)
            if splits[split]!=manifest['output_rows'][split]:raise ValueError('Bootstrap count mismatch')
    return {'rows':sum(counts.values()),'groups':len(groups),'by_label':dict(counts),'by_language':dict(languages),
        'by_language_origin':dict(origins),'split_rows':dict(splits),'new_v22_approval':False,
        'evidence_sha256':digest(manifest_path),'verified_file_sha256':files,'development_only':True,
        'old_test_opened':False,'inherited_leakage_pass':True,'status':'conditional_reuse_proposal'}


def detector_check(config: dict) -> dict:
    """Independent synthetic retrieval diagnostic, not corpus recall evidence."""
    rng=random.Random(config['seed'])
    positives=hits=negatives=false_hits=0
    for index in range(60):
        alphabet='abcdefghijklmnopqrstuvwxyz' if index<20 else ('中文邮件研究质量审核内容测试会议安全普通信息通知' if index<40 else 'thư điện tử kiểm tra nghiên cứu nội dung tiếng Việt')
        base=''.join(rng.choice(alphabet) for _ in range(1600))
        changed=list(base)
        for position in rng.sample(range(len(base)),10):changed[position]=rng.choice(alphabet)
        positive=''.join(changed)
        negative=base[:512]+''.join(rng.choice(alphabet) for _ in range(1088))
        for other in (positive,negative):
            # Synthetic alphabet contains no URL/email; same normalization.
            a=' '.join(base.casefold().split());b=' '.join(other.casefold().split())
            left={a[i:i+5] for i in range(len(a)-4)};right={b[i:i+5] for i in range(len(b)-4)}
            truth=len(left&right)/len(left|right)>=.85
            s1=np.frombuffer(minhash(base,config)[0],dtype='<u4')
            s2=np.frombuffer(minhash(other,config)[0],dtype='<u4')
            width=config['minhash_rows_per_band']
            collision=any(np.array_equal(s1[i:i+width],s2[i:i+width]) for i in range(0,len(s1),width))
            predicted=collision and float(np.mean(s1==s2))>=config['minhash_similarity']
            if truth:positives+=1;hits+=int(predicted)
            else:negatives+=1;false_hits+=int(predicted)
    return {'scope':'synthetic functional retrieval only, NOT measured real-corpus recall',
        'positive_pairs':positives,'retrieved_positive_pairs':hits,'negative_pairs':negatives,
        'accepted_negative_pairs':false_hits,'recall':hits/positives if positives else None,
        'false_positive_rate':false_hits/negatives if negatives else None,
        'real_corpus_recall':None}


def _export(conn: sqlite3.Connection, path: Path, query: str) -> int:
    cursor=conn.execute(query)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow([c[0] for c in cursor.description])
        count=0
        for row in cursor:writer.writerow(row);count+=1
    return count


def generate(root: Path, config: dict) -> dict:
    report=root/'reports/v22';report.mkdir(parents=True,exist_ok=True)
    output=root/'data/v22/phase3';output.mkdir(parents=True,exist_ok=True)
    with connect(root) as conn:
        complete=conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='groups_complete'").fetchone()
        if not complete:raise ValueError('Incomplete Phase 3 groups; cannot report dataset ready')
        status=dict(conn.execute('SELECT trang_thai,count(*) FROM mau GROUP BY trang_thai'))
        if not status:raise ValueError('Missing Phase 3 classification')
        for key in ('approved','pending_review','weak_label','duplicate','excluded'):status.setdefault(key,0)
        sources=[dict(zip(('source','raw_label','candidate_label','declared_language','signal','origin','status','rows'),row))
            for row in conn.execute('''SELECT nguon,nhan_goc,nhan_ung_vien,ngon_ngu_khai_bao,ngon_ngu_goi_y,
            nguon_goc,trang_thai,count(*) FROM mau GROUP BY 1,2,3,4,5,6,7''')]
        _export(conn,output/'decisions_manifest.csv','''SELECT id_mau AS audit_id,nguon AS source,thanh_phan AS component,
            nhan_goc AS raw_label,nhan_ung_vien AS candidate_label,nhan AS label,ngon_ngu_khai_bao AS declared_language,
            ngon_ngu_goi_y AS language_signal,nguon_goc AS data_origin,ma_bam AS content_hash,id_nhom AS constraint_group_id,
            id_dai_dien AS canonical_id,trang_thai AS status,ly_do AS reason,bang_chung AS label_provenance,quy_tac AS curation_rule_id
            FROM mau ORDER BY id_mau''')
        _export(conn,output/'review_queue.csv','''SELECT s.id_mau AS audit_id,s.tang AS stratum,s.dan_so AS population_groups,
            s.so_lay AS sampled_groups,s.xac_suat AS inclusion_probability,m.nguon AS source,m.thanh_phan AS component,
            m.nhan_goc AS raw_label,m.nhan_ung_vien AS candidate_label,m.ngon_ngu_goi_y AS language_signal,
            m.id_nhom AS constraint_group_id,m.ly_do AS review_reason FROM lay_mau s JOIN mau m USING(id_mau) ORDER BY s.tang,s.id_mau''')
        _export(conn,output/'human_decisions.csv','SELECT * FROM quyet_dinh ORDER BY id_mau,nguoi_duyet')
        _export(conn,output/'anomaly_queue.csv','''SELECT id_mau AS audit_id,nguon AS source,
            nhan_goc AS raw_label,ngon_ngu_goi_y AS language_signal,id_nhom AS constraint_group_id,ly_do AS reason
            FROM (SELECT *,row_number() OVER (PARTITION BY nguon,ly_do ORDER BY id_mau) thu_tu
            FROM mau WHERE trang_thai='pending_review') WHERE thu_tu<=5 ORDER BY nguon,ly_do,id_mau''')
        strata=[]
        for stratum,population,n in conn.execute('SELECT tang,max(dan_so),count(*) FROM lay_mau GROUP BY tang'):
            reviewed=conn.execute('SELECT count(DISTINCT q.id_mau) FROM quyet_dinh q JOIN lay_mau s USING(id_mau) WHERE s.tang=?',(stratum,)).fetchone()[0]
            strata.append({'stratum':stratum,'population_groups':population,'sampled_groups':n,'reviewed_groups':reviewed,
                'observed_label_error_rate':None,'acceptance':'pending competent human review and adjudication',
                'zero_error_upper_bound_if_complete':1-config['review_alpha_per_stratum']**(1/n),
                'census':n==population})
        summary={'scope':'phase3_expansion_proposal_only','rows':sum(status.values()),'status_counts':status,
            'approved_by_label':dict(conn.execute("SELECT nhan,count(*) FROM mau WHERE trang_thai='approved' GROUP BY nhan")),
            'approved_by_language':dict(conn.execute("SELECT ngon_ngu_goi_y,count(*) FROM mau WHERE trang_thai='approved' GROUP BY ngon_ngu_goi_y")),
            'remaining_to_target':max(0,config['target_rows']-status['approved']),
            'unique_normalized':conn.execute('SELECT count(*) FROM van_ban').fetchone()[0],
            'similarity_stats':json.loads(complete[0]),
            'signature_skips':dict(conn.execute("SELECT ly_do,count(*) FROM van_ban WHERE ly_do!='' GROUP BY ly_do")),
            'constraint_groups':conn.execute('SELECT count(DISTINCT id_nhom) FROM nhom').fetchone()[0],
            'largest_constraint_group':conn.execute('SELECT max(so_thanh_vien) FROM nhom').fetchone()[0],
            'quarantined_fingerprints':conn.execute('SELECT count(*) FROM nhom WHERE can_cach_ly=1').fetchone()[0],
            'candidate_conflict_rows':conn.execute("SELECT count(*) FROM mau WHERE ly_do LIKE '%candidate_label_conflict%'").fetchone()[0],
            'human_decisions':conn.execute('SELECT count(*) FROM quyet_dinh').fetchone()[0],
            'strata':strata,'distribution':sources,'sampling_policy_sha256':digest(root/'config/v22/expansion.json'),
            'detector_diagnostic':detector_check(config),'dataset_finalized':False,'phase4_ready':False,
            'training_performed':False,'old_final_test_opened':False,'publication_approved':False}
        for label in config['target_labels']:summary['approved_by_label'].setdefault(label,0)
        for language in config['target_languages']:summary['approved_by_language'].setdefault(language,0)
        summary['source_approval_ready_count']=0
        for label,target in config['target_labels'].items():summary.setdefault('label_gaps',{})[label]=target-summary['approved_by_label'].get(label,0)
        for lang,target in config['target_languages'].items():summary.setdefault('language_gaps',{})[lang]=target-summary['approved_by_language'].get(lang,0)
        # Source-level proposals contain no activation or invented approval.
        proposal=[]
        for source,raw,candidate,count in conn.execute("SELECT nguon,nhan_goc,nhan_ung_vien,count(*) FROM mau WHERE trang_thai NOT IN ('duplicate','excluded') GROUP BY 1,2,3"):
            proposal.append({'source':source,'raw_label':raw,'candidate_label':candidate,'canonical_rows':count,
                'approved':False,'state':'conditional_proposal' if candidate in ('normal_candidate','ham') else 'individual_taxonomy_review_required',
                'ready_for_bulk_approval':False,
                'prerequisites':['source/version/use-rights evidence','competent-language independent review','zero unresolved conflicts','protected-test guard coverage','explicit owner approval']})
        summary['source_proposals']=proposal
    summary['inherited_bootstrap']=inspect_bootstrap(root)
    provenance=root/'reports/v22/phase3_source_evidence.json'
    if provenance.exists():summary['source_evidence']=json.loads(provenance.read_text(encoding='utf-8'))
    resources=root/'reports/v22/phase3_resources.json'
    if resources.exists():summary['measured_resources']=json.loads(resources.read_text(encoding='utf-8'))
    (report/'DATASET_EXPANSION_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'source_approval_proposals.json').write_text(json.dumps(proposal,indent=2),encoding='utf-8')
    _source_report(root,config,summary)
    _quality_report(root,config,summary)
    _main_report(root,config,summary)
    return {k:summary[k] for k in ('rows','status_counts','unique_normalized','constraint_groups','phase4_ready')}


def _source_report(root: Path, config: dict, summary: dict) -> None:
    lines=['# Source Label Verification — V2.2 Phase 3','','Quy tắc lấy mẫu được khóa trong config/v22/expansion.json trước khi có quyết định người thật. Không kích hoạt phê duyệt hàng loạt.',
        '', '| Nguồn | Phương pháp/nhãn gốc | Quyền sử dụng và phân phối | Đề xuất |','|---|---|---|---|',
        '| V2 bootstrap | Nhãn ba lớp đã curation, provenance/parent/group kiểm tra ở V2.1 | Theo từng nguồn; không có quyền phân phối blanket | 38.013 hàng development có bằng chứng kế thừa; đề xuất riêng, chưa approved V2.2 |',
        '| TREC06C | Gold binary Ham/Spam; không tách Phishing | NIST ngừng phân phối 2006/2007 vì PII; quyền mở rộng/redistribute NOT VERIFIED | Ham chỉ đề xuất Normal sau review; Spam cần phân biệt ba lớp |',
        '| Merged | 0 safe; 1 gộp spam/scam/phishing từ 9 corpus | Cần license cấp component, version/checksum local → upstream | Raw 0 là ứng viên, raw 1 không map Phishing |',
        '| Balanced | Dẫn xuất downsample Merged | Cùng chuỗi quyền; không độc lập | Không cộng thêm quota; giữ duplicate/conflict provenance |',
        '| MeAJOR | Binary hợp nhất; local 108.685, paper v2 135.894 | License bài báo không cấp quyền email; cần version/trec5/6/7 provenance | Raw 0 xét theo component; raw 1 cần curation ba lớp |',
        '| Enron Original | Không có nhãn Normal ở cấp email | CMU mô tả nguồn nghiên cứu và cảnh báo privacy | Chỉ duyệt từng mẫu/nhóm có bằng chứng; không map toàn corpus Normal |',
        '| Synthetic Chinese / PhishVN / challenge | Synthetic / URL / evaluation challenge | Xem audit riêng; không tự cấp quyền | Không nhập email thực 300k hoặc final benchmark |',
        '', '## Bằng chứng nguồn đã đối chiếu', '',
        '- [TREC overview](https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf): binary human-adjudicated spam/ham. [NIST hiện tại](https://trec.nist.gov/data/spam.html): không còn phân phối TREC 2006/2007 vì PII.',
        '- [CMU Enron](https://www.cs.cmu.edu/~enron/): nguồn gốc CALO/FERC, redaction, cảnh báo privacy; không cung cấp taxonomy an toàn cấp email. Không suy rằng mọi thư doanh nghiệp là Normal.',
        '- [README tác giả Merged/Balanced](https://github.com/Manar-ibr/Phishing-Email-Detection-Dataset/blob/main/README.md): positive trộn spam/scam/phishing; Balanced downsample từ Merged. Không coi hai bộ là độc lập.',
        '- [Zenodo 17314806 version v2](https://zenodo.org/records/17314806): MD5 Merged a692ca69ffd7dbc9df991292035d29a9 và Balanced c18127659164566291c18803cc7d4c5f đều khớp file local trong lần kiểm tra Phase 3. Bằng chứng này nâng độ tin cậy phiên bản file, không xác minh từng nhãn hay chuỗi quyền của 9 corpus. Metadata license không đủ rõ để cấp quyền blanket.',
        '- [MeAJOR paper v2](https://arxiv.org/html/2507.17978v2): binary đa nguồn và khác version/quy mô file local. [Audit local](../v21/csv_meajor_details.json) giữ source/language; chưa được coi là xác minh ba lớp.',
        '- [Nazario README](https://monkey.org/~jose/phishing/README.txt): hand-classified, CC BY 4.0, có thể có lỗi và dữ liệu chưa ẩn danh ở bản sau. Chỉ cân nhắc các bản đã nằm trong bootstrap hợp lệ, không mở raw/V2 Test để mở rộng.',
        '', '## Protocol kiểm định đã xác định trước', '',
        f'- Seed {config["seed"]}; population là constraint group trong từng source + component + raw_label + language_signal. Chọn ngẫu nhiên theo SHA-256 có seed, tối đa {config["review_sample_per_stratum"]} nhóm/tầng; tầng nhỏ lấy toàn bộ. Mỗi nhóm chọn một canonical đại diện ngẫu nhiên. Inclusion probability được lưu. Một nhóm có thể xuất hiện ở tầng khác: không coi các tầng độc lập để cộng cỡ mẫu.',
        '- Ngôn ngữ signal chỉ là heuristic. Người có năng lực ngôn ngữ gốc phải xác minh EN/VI/ZH/mixed, nguồn native/translated/synthetic, nhãn và chất lượng. Nhóm lớn/collision/translation phải kiểm tra thành viên trước admission; xem một đại diện không phê duyệt cả nhóm.',
        '- Với nhóm nguồn có mapping rõ: yêu cầu không có sai nhãn hoặc unresolved trong mẫu, upper bound một phía 95% dưới 1% theo từng tầng (299 mẫu độc lập, 0 lỗi). Đây là kế hoạch chấp nhận có điều kiện, không bảo đảm toàn nguồn; không tuyên bố confidence đồng thời toàn bộ tầng. Tầng census nhỏ chỉ kết luận cho chính các nhóm được kiểm duyệt.',
        '- Binary positive hỗn hợp và Enron không có mapping chuẩn để đo discordance ba lớp: cần gán nhãn độc lập cấp email, ít nhất hai reviewer cho Phishing/nhãn mơ hồ, adjudication bất đồng. Không dùng keyword/ML/LLM làm chuẩn.',
        '- Chỉ đồng ý xem AI/dịch không được ghi human-verified. Reviewer/evidence/competence/UTC là dữ liệu khai báo cần xác thực; hệ thống không tự chứng nhận năng lực hay chữ ký.',
        '- Quyền dùng local nghiên cứu và quyền phân phối được xem riêng; một quyết định chủ repo không thay thế quyền từ chủ thể corpus. Không phát hành email ở task này.',
        '', f'Hiện có {summary["human_decisions"]} quyết định người thật; observed error rate vẫn null. Không nguồn mới nào đủ cơ sở kích hoạt bulk approval.',
        '', '## Nguồn bổ sung đề xuất, chưa tải', '',
        '- Native VI: thu thập có consent và ẩn danh từ tổ chức/người tham gia, hai người duyệt ba lớp, quyền nghiên cứu được ghi rõ; giữ benchmark tương lai độc lập. Chưa có số mẫu được cam kết.',
        '- Native ZH Phishing: liên hệ [DataCon challenge](https://github.com/yaoyue123/datacon2023-spoof-email) về nhãn cấp email, bản quyền và provenance; public sample hiện chưa đủ điều kiện.',
        '- [Phishing Pot card](https://github.com/rf-peixoto/phishing_pot/blob/main/DATASET_CARD.md) và [license](https://github.com/rf-peixoto/phishing_pot/blob/main/LICENSE): nguồn email thực nghiên cứu phi thương mại theo CC BY-NC 4.0; cần kiểm kê ngôn ngữ, kiểm duyệt nhãn/PII và phiên bản trước khi xin tải. Không hứa có 75k Phishing hoặc có Chinese.',
        '- [STING9](https://sting9.org/dataset) là đầu mối theo audit cũ; phải xác minh điều khoản/license mâu thuẫn, nguồn và số email native trước khi sử dụng. Không tính nguồn này vào quota.',
        '', 'Danh sách quyết định đề xuất có số lượng canonical thực tế: data/v22/phase3/source_approval_proposals.json. Tất cả approved=false; chủ dự án xác nhận quy trình/phạm vi sau khi điều kiện bằng chứng đạt.', '']
    lines+=['','## Phân bố raw label thực tế trước canonical dedup','',
        '| Nguồn | Raw label | Hàng |','|---|---|---:|']
    raw=Counter()
    for row in summary['distribution']:raw[(row['source'],row['raw_label'])]+=row['rows']
    lines+=['| '+source+' | '+(label or '(không nhãn)')+' | '+f'{count:,}'+' |' for (source,label),count in sorted(raw.items())]
    lines+=['','Không gọi bảng raw label này là phân bố Normal/Spam/Phishing đã được xác minh. Approved theo cả ba lớp hiện vẫn 0.']
    (root/'reports/v22/SOURCE_LABEL_VERIFICATION.md').write_text('\n'.join(lines),encoding='utf-8')


def _quality_report(root: Path, config: dict, s: dict) -> None:
    d=s['similarity_stats']; diag=s['detector_diagnostic']
    lines=['# Dataset Quality Report — Phase 3','','## Kết quả đo thực tế','',
        f'- Đọc lại {s["rows"]:,} hàng; {s["unique_normalized"]:,} normalized fingerprint. Không chỉnh raw, V2/V2.1 hoặc Phase 2.',
        f'- Legacy overflow được phủ đủ {d["phase2_overflow_fingerprints_covered"]:,} fingerprint; {d["large_legacy_buckets"]:,} bucket lớn được giữ toàn bộ thành viên trong quarantine. Số fingerprint overflow chưa cách ly: {d["legacy_overflow_unquarantined"]}. Không dùng LIMIT 100 để bỏ qua thành viên.',
        f'- Kiểm tra lại cạnh prefix Phase 2 bằng chữ ký toàn văn: {d["legacy_edges_fulltext_confirmed"]:,} cạnh đạt ngưỡng estimate được giữ ràng buộc nhóm; {d["legacy_edges_deferred"]:,} cạnh chưa đủ bằng chứng chuyển cách ly hai đầu, không ép chung nhóm.',
        f'- MinHash 32 permutations, 8 bands × 4; char 5-gram toàn văn với NFC/NFKC/plain-text/placeholders phục vụ so trùng. {d["minhash_candidate_pairs"]:,} pair visits, {d["minhash_estimated_edges"]:,} cạnh đạt ngưỡng estimate {config["minhash_similarity"]}; {d["secondary_length_rejections"]:,} pair visits bị secondary length blocking. Pair visits có thể lặp qua các band.',
        f'- {d["large_minhash_buckets"]:,} bucket MinHash quá giới hạn pairwise {config["pairwise_bucket_limit"]} được stream toàn bộ thành viên vào quarantine; không ép chung nhóm chỉ vì collision, không mở rộng O(n²). Không tuyên bố đã xác minh mọi cặp trong các bucket này.',
        f'- {s["constraint_groups"]:,} constraint groups; nhóm lớn nhất {s["largest_constraint_group"]:,} fingerprint. {s["quarantined_fingerprints"]:,} fingerprint trong nhóm cách ly. Đây là ràng buộc split bảo thủ, KHÔNG phải số email độc lập hay duplicate ground truth. Nhóm lớn làm giảm khả năng cân bằng split; không tự tách để đẹp số liệu.',
        f'- Skips fuzzy có lý do: {s["signature_skips"]}. Email vượt 50.000 ký tự không dùng prefix thay thế: giữ pending full comparison. Email ngắn chỉ có exact/template/provenance guard, không được chứng nhận fuzzy đầy đủ.',
        f'- {s["candidate_conflict_rows"]:,} hàng có xung đột nhãn ứng viên (bao gồm kế thừa). Nhãn ứng viên khác nhau không tự xác định nhãn đúng; giữ pending/duplicate cùng lý do.',
        '', '## Diagnostic khả năng phát hiện','',
        f'Controlled synthetic pairs EN/VI/ZH: {diag["retrieved_positive_pairs"]}/{diag["positive_pairs"]} cặp có full-char-Jaccard ≥0,85 được detector thu hồi; {diag["accepted_negative_pairs"]}/{diag["negative_pairs"]} cặp dưới ngưỡng được chấp nhận. Đây là test truy hồi synthetic, không phải recall trên corpus thực hoặc metric classifier. Real corpus recall = null; cần người duyệt cặp và mẫu ngoài candidate để đo false negatives.',
        'MinHash là xác suất và estimate hữu hạn, không phải so trùng hoàn chỉnh. Template/provenance hoặc các chuỗi cạnh near vẫn có thể tạo nhóm rộng. LSH collision đơn thuần không đủ để gộp nhóm. Không xóa email khác nhau chỉ vì giống câu chữ. Không dùng similarity để quyết định Normal/Spam/Phishing.',
        '', '## Leakage','',
        'Giữ exact/normalized/template/provenance/known-parent và các cạnh near đạt ngưỡng trong nhóm constraint. Cạnh cũ chưa xác minh/bucket lớn cách ly mọi thành viên; không được đưa các thành viên cách ly vào split trước khi giải quyết quan hệ. Không tạo Train/Validation/Test. Guard Phase 2 về V2 reference chỉ là metadata; không mở Final Test cũ. Chưa có bằng chứng phủ đầy đủ protected-test/translation lineage của mọi thí nghiệm cũ; admission và Phase 4 vẫn bị chặn cho đến khi fingerprint coverage được xác minh. Không gọi dataset hiện tại leakage-free.',
        '', '## Tài nguyên / integrity','',
        'Một worker, SQLite disk indexes/64 MiB cache, batch 1.000 signature khi lập chỉ mục, giao dịch từng CSV part. MinHash tối đa 50.000 ký tự mỗi email; không nạp corpus vào RAM. Grouping dùng mảng int32/bool O(unique fingerprints), không dense TF-IDF hay GPU. Group stage atomic: nếu ngắt thì rollback và chạy lại riêng stage, phần sketch và index đã checkpoint được giữ.',
        'Khóa SHA-256 Phase 2/audit/config; mỗi part đối chiếu SHA-256 checkpoint cũ. Nguồn thay đổi chặn resume. Manifest báo cáo ghi hash code/config/output metadata; file quyết định, queue và email review packs chỉ local trong vùng bị ignore.', '']
    verification=root/'reports/v22/phase3_verification.json'
    if verification.exists():
        lines+=['','## Test và bảo toàn','',*['- '+item for item in json.loads(verification.read_text(encoding='utf-8'))['checks']]]
    (root/'reports/v22/DATASET_QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def _main_report(root: Path, config: dict, s: dict) -> None:
    lines=['# V2.2 — Phase 3 Dataset Expansion','','Nhánh v22/large-scale-dataset. Chỉ expansion proposal; dừng trước dataset freeze, split và training.','',
        '| Trạng thái | Số hàng |','|---|---:|',*['| '+name+' | '+f'{n:,}'+' |' for name,n in sorted(s['status_counts'].items())],
        '', f'Tổng {s["rows"]:,}; chuẩn hóa duy nhất {s["unique_normalized"]:,}. Duplicate là bản ghi tham chiếu canonical, không xóa nguồn, không tính thêm quota. Approved = {s["status_counts"]["approved"]}; còn thiếu {s["remaining_to_target"]:,} mẫu approved so với 300.000.',
        'Pending Review khác Weak Label: pending có chất lượng/overlap/nhóm cần giải quyết hoặc chưa có nhãn; weak còn nhãn nguồn sơ bộ nhưng chưa đạt chính sách ba lớp. Các lý do quyền/ngôn ngữ/nhãn giữ đầy đủ kể cả ở Duplicate và Excluded.',
        '', '## Tính khả thi / thiếu hụt','',
        f'Approved theo lớp: {s["approved_by_label"]}; theo ngôn ngữ: {s["approved_by_language"]}. Approved label gaps: {s["label_gaps"]}. Approved language gaps: {s["language_gaps"]}. Không dùng candidate/raw distribution thay cho số mẫu đạt chuẩn. Chi tiết source/raw label/language/origin/status có trong DATASET_EXPANSION_SUMMARY.json.',
        '38.013 bootstrap development có nhãn kế thừa (Normal 34.734, Spam 1.546, Phishing 1.733; EN 35.587, VI dịch 2.426, ZH 0) là tập con đề xuất khả thi sớm nhất. Chưa tính là approved V2.2, không coi Validation cũ là Final Test mới; phải giữ assignment/provenance cũ và kiểm tra quyền/group trước tái sử dụng.',
        'Nguồn mới có 588.790 fingerprint normalized nhưng còn overlap/nhóm lớn/nhãn hỗn hợp; số fingerprint không chứng minh 300.000 email độc lập hợp lệ. VI metadata chỉ có 2 hàng chưa xác minh, bootstrap VI đều translated. ZH metadata 34.740 normalized unique theo Phase 1, còn thiếu Phishing ba lớp. Không có cơ sở cam kết cân bằng mục tiêu hiện tại.',
        '', '## Đề xuất xác nhận','',
        '1. Chấp nhận protocol review 299 nhóm/tầng, ngưỡng 1% và kiểm duyệt ngôn ngữ/nhãn độc lập đã định trước; chỉ áp dụng source-level approval sau khi có kết quả và bằng chứng quyền đủ điều kiện.',
        '2. Ưu tiên kiểm tra quyền tái sử dụng bootstrap development, TREC Ham, các component raw 0 của Merged/MeAJOR; Balanced chỉ giữ lineage. Không có phê duyệt hàng loạt cho binary positives hoặc Enron.',
        '3. Bổ sung native VI có consent và yêu cầu nguồn native ZH Phishing có nhãn/quyền; danh sách đề xuất trong SOURCE_LABEL_VERIFICATION.md. Chưa tải nguồn mới.',
        '', '## Chạy tiếp kiểm duyệt local','', '```powershell',
        '.\\.venv\\Scripts\\python.exe -u scripts/v22/expand_dataset.py --run',
        '.\\.venv\\Scripts\\python.exe scripts/v22/expand_dataset.py --report',
        '.\\.venv\\Scripts\\python.exe scripts/v22/audit_quality.py --review-pack "SOURCE|ID"',
        '.\\.venv\\Scripts\\python.exe scripts/v22/expand_dataset.py --review-id "SOURCE|ID" --label review --reviewer "NAME" --language-competence "QUALIFICATION" --evidence "EVIDENCE"',
        '```','',
        'Review pack chứa dữ liệu email không đáng tin cậy, chỉ đọc local, không mở URL. Queue và decision manifest metadata trong data/v22/phase3 bị gitignore; người review cần bảo vệ email/PII và CLI history. Ghi quyết định không tự cấp approved. --run bị khóa sau khi có quyết định để không thay sample hay mất lịch sử.',
        '', '## File mới','',
        '- config/v22/expansion.json', '- src/vietmailguard/v22/expansion.py', '- src/vietmailguard/v22/expansion_report.py',
        '- scripts/v22/expand_dataset.py', '- tests/test_v22_expansion.py',
        '- reports/v22/PHASE3_DATASET_EXPANSION_REPORT.md, SOURCE_LABEL_VERIFICATION.md, DATASET_QUALITY_REPORT.md, DATASET_EXPANSION_SUMMARY.json',
        '', '**DATASET_FINALIZED=false; PHASE4_READY=false; TRAINING_PERFORMED=false.** Dừng chờ xác nhận; không thay runtime/model, không gọi Gemini, không commit/push.', '']
    (root/'reports/v22/PHASE3_DATASET_EXPANSION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    paths=[root/'config/v22/expansion.json',root/'src/vietmailguard/v22/expansion.py',root/'src/vietmailguard/v22/expansion_report.py',root/'scripts/v22/expand_dataset.py',root/'tests/test_v22_expansion.py']
    paths+=list((root/'reports/v22').glob('*REPORT.md'))+[root/'reports/v22/SOURCE_LABEL_VERIFICATION.md',root/'reports/v22/DATASET_EXPANSION_SUMMARY.json']
    paths+=list((root/'data/v22/phase3').glob('*.csv'))+[root/'data/v22/phase3/source_approval_proposals.json']
    paths+=[root/'reports/v22/phase3_source_evidence.json',root/'reports/v22/phase3_resources.json',root/'reports/v22/phase3_verification.json']
    manifest={'sha256':{p.relative_to(root).as_posix():digest(p) for p in sorted(set(paths)) if p.is_file()},
        'publication_approved':False,'scope':'phase3_only','contains_email_text':False}
    (root/'reports/v22/phase3_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
