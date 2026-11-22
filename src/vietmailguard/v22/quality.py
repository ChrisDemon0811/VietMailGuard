"""Streaming, disk-backed quality quarantine; source labels never become truth.

Near matches are bounded prefix/sketch candidates, never deletion or approval.
Old audit metadata is read-only, including fingerprints used as conservative V2
overlap guards. No old held-out record file is opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vietmailguard.v21.dedup_audit import DATASETS, dedup_hashes
from vietmailguard.eml_parser import _html_to_text


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def normalize(text: str) -> str:
    """Derived comparison text only; the raw email remains immutable."""
    text = unicodedata.normalize('NFC', text)
    if re.search(r'<(?:html|body|div|p|script)\b', text, re.I):
        text = _html_to_text(text)
    # Unicode Cc is exactly C0/C1. Preserve LF/TAB; use a C-level regex to
    # avoid a Python character loop over millions of long messages.
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def language_signal(text: str) -> str:
    """Conservative script/lexical signal, NOT verified language ground truth."""
    sample = text[:4096].casefold()
    if len(sample) < 20:
        return 'unknown'
    han = len(re.findall(r'[\u3400-\u9fff]', sample))
    if re.search(r'[\u3040-\u30ff\uac00-\ud7af]', sample):
        return 'other_or_mixed'
    vi = len(re.findall(r'\b(?:không|người|được|thông|tiếng|ngày|vui|lòng|chúng|bạn)\b', sample))
    en = len(re.findall(r'\b(?:the|and|this|please|your|you|from|with|meeting)\b', sample))
    signals = [code for code, ok in [('zh', han >= 10), ('vi', vi >= 3), ('en', en >= 3)] if ok]
    return signals[0] if len(signals) == 1 else 'mixed' if signals else 'unknown'


def sketch(text: str, config: dict) -> tuple[str, bytes]:
    prefix = unicodedata.normalize('NFKC', text).casefold()[:config['near_prefix_characters']]
    # Short generic texts get exact/normalized checks, not fuzzy linkage.
    if len(prefix) < 80:
        return '', b''
    values = sorted({(zlib.crc32(prefix[i:i+5].encode()) << 32) |
                     zlib.crc32(prefix[i:i+5][::-1].encode()) for i in range(len(prefix)-4)})
    values = values[:config['near_sketch_size']]
    array = np.array(values, dtype='<u8')
    bits = np.unpackbits(array.view(np.uint8)).reshape(-1, 64).sum(axis=0) * 2 >= len(values)
    signature = np.packbits(bits).tobytes().hex()
    return signature, array.tobytes()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path.resolve().as_uri(), uri=True)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=-65536')
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS cau_hinh (ma TEXT PRIMARY KEY, gia_tri TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tien_do (duong_dan TEXT PRIMARY KEY, ma_bam TEXT NOT NULL, so_dong INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS mau (
        id_mau TEXT PRIMARY KEY, nguon TEXT NOT NULL, nhan_goc TEXT, nhan_ung_vien TEXT,
        ngon_ngu_goc TEXT, ngon_ngu_goi_y TEXT, nguon_goc TEXT,
        ma_bam_chinh_xac TEXT, ma_bam_chuan TEXT, ma_bam_mau TEXT,
        nhom_nguon TEXT, id_cha TEXT, trang_thai TEXT NOT NULL CHECK(trang_thai IN ('review','exclude')),
        ly_do TEXT, canh_bao TEXT);
    CREATE TABLE IF NOT EXISTS van_ban (ma_bam TEXT PRIMARY KEY, chu_ky TEXT, phac_thao BLOB);
    CREATE TABLE IF NOT EXISTS dai (vi_tri INTEGER, gia_tri TEXT, ma_bam TEXT,
        chu_ky TEXT,
        PRIMARY KEY(vi_tri,gia_tri,ma_bam));
    CREATE TABLE IF NOT EXISTS gan_trung (ma_truoc TEXT, ma_sau TEXT, khoang_cach INTEGER, tuong_dong REAL,
        PRIMARY KEY(ma_truoc,ma_sau));
    CREATE TABLE IF NOT EXISTS qua_tai (ma_bam TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS chan_lich_su (id_mau TEXT PRIMARY KEY, ly_do TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS quyet_dinh (id_mau TEXT PRIMARY KEY REFERENCES mau(id_mau),
        nhan TEXT NOT NULL CHECK(nhan IN ('normal','spam','phishing','review','exclude')),
        nguoi_duyet TEXT NOT NULL, bang_chung TEXT NOT NULL, nang_luc TEXT NOT NULL,
        thoi_gian TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS chi_muc_bam ON mau(ma_bam_chuan);
    CREATE INDEX IF NOT EXISTS chi_muc_mau ON mau(ma_bam_mau);
    ''')
    if 'chu_ky' not in {row[1] for row in conn.execute('PRAGMA table_info(dai)')}:
        conn.execute('ALTER TABLE dai ADD COLUMN chu_ky TEXT')
        conn.execute('UPDATE dai SET chu_ky=(SELECT chu_ky FROM van_ban WHERE van_ban.ma_bam=dai.ma_bam)')
        conn.commit()
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def ingest(root: Path, config: dict) -> dict:
    if any(config[key] for key in ('automatic_label_approval', 'training_allowed', 'final_test_access_allowed')):
        raise ValueError('Phase 1/2 cannot approve labels, train or access held-out Test')
    source = root / 'data/v21/cache/dedup_audit.sqlite3'
    output = root / 'data/v22/cache/quality.sqlite3'
    with connect(output) as conn, sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as old:
        old.execute('PRAGMA cache_size=-196608')
        old.row_factory = sqlite3.Row
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        signature += ':' + digest(source)
        previous = conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='cau_hinh'").fetchone()
        if previous and previous[0] != signature:
            raise ValueError('Configuration or source audit changed; use a new run directory')
        conn.execute("INSERT OR IGNORE INTO cau_hinh VALUES ('cau_hinh',?)", (signature,))
        conn.commit()
        csv.field_size_limit(100_000_000)
        for dataset, relative in DATASETS:
            parts = sorted((root/'data/v21/processed'/relative).glob('part-*.csv'))
            if not parts:
                raise FileNotFoundError('Missing standardized source: '+dataset)
            for part in parts:
                if part.is_symlink():
                    raise ValueError('Symlink source is not permitted')
                expected = old.execute('SELECT size_bytes,mtime_ns FROM checkpoints WHERE path=?', (str(part.resolve()),)).fetchone()
                if expected is None or tuple(expected) != (part.stat().st_size, part.stat().st_mtime_ns):
                    raise ValueError('Part differs from historical audit checkpoint')
                key = part.relative_to(root).as_posix()
                sha = digest(part)
                cached = conn.execute('SELECT ma_bam FROM tien_do WHERE duong_dan=?', (key,)).fetchone()
                if cached:
                    if cached[0] != sha:
                        raise ValueError('Changed source part after checkpoint')
                    continue
                count = 0
                # Part transaction: interrupted part rolls back; resume never duplicates rows.
                with conn, part.open(encoding='utf-8-sig', newline='') as handle:
                    for raw in csv.DictReader(handle):
                        identifier = dataset+'|'+raw['id']
                        prior = old.execute('SELECT * FROM rows WHERE audit_id=?', (identifier,)).fetchone()
                        if prior is None:
                            raise ValueError('Source row missing from historical audit')
                        exact, normalized, template = dedup_hashes(raw.get('subject',''), raw.get('body',''))
                        if (exact,normalized,template) != (prior['exact_hash'],prior['normalized_hash'],prior['template_hash']):
                            raise ValueError('Source text fingerprint differs from historical audit')
                        text = normalize(raw.get('subject','')+'\n'+raw.get('body',''))
                        language = language_signal(text)
                        warnings = []
                        for flag in ('decode_error','decode_warning','mime_defects','mime_recovered','missing_content','missing_body'):
                            if raw.get(flag,'').casefold() not in ('','false','0','[]','none'):
                                warnings.append(flag)
                        if '\ufffd' in text:
                            warnings.append('replacement_character')
                        if language not in ('en','vi','zh') or language != prior['language']:
                            warnings.append('language_needs_review')
                        if len(text) < 40:
                            warnings.append('short_text')
                        status = 'exclude' if not normalized or prior['label_status']=='exclude' else 'review'
                        reason = prior['audit_reason'] or 'label_evidence_pending'
                        conn.execute('INSERT INTO mau VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                            identifier,dataset,prior['raw_label'],prior['candidate_label'],prior['language'],language,
                            prior['data_origin'],exact,normalized,template,prior['provenance_group_id'] or prior['v2_group_id'],
                            prior['parent_id'],status,reason,','.join(warnings)))
                        if normalized and not conn.execute('SELECT 1 FROM van_ban WHERE ma_bam=?',(normalized,)).fetchone():
                            sig, blob = sketch(text,config)
                            conn.execute('INSERT INTO van_ban VALUES (?,?,?)',(normalized,sig,blob))
                        count += 1
                    if digest(part) != sha:
                        raise ValueError('Input changed while reading')
                    conn.execute('INSERT INTO tien_do VALUES (?,?,?)',(key,sha,count))
                print(json.dumps({'source':dataset,'part':part.name,'rows':count}),flush=True)
        expected_count = old.execute("SELECT count(*) FROM rows WHERE dataset!='v2_reference'").fetchone()[0]
        actual = conn.execute('SELECT count(*) FROM mau').fetchone()[0]
        if actual != expected_count:
            raise ValueError('Full source reconciliation failed')
        return {'rows': actual, 'parts':conn.execute('SELECT count(*) FROM tien_do').fetchone()[0]}


def near_duplicates(root: Path, config: dict) -> None:
    with connect(root/'data/v22/cache/quality.sqlite3') as conn:
        previous = conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='gan_trung'").fetchone()
        last = previous[0] if previous else ''
        while True:
            batch = conn.execute('SELECT ma_bam,chu_ky,phac_thao FROM van_ban WHERE ma_bam>? ORDER BY ma_bam LIMIT 1000',(last,)).fetchall()
            if not batch:
                break
            with conn:
                for hashed, sig, blob in batch:
                    if sig:
                        found = {}
                        for band in range(4):
                            matches = conn.execute('SELECT ma_bam,chu_ky FROM dai WHERE vi_tri=? AND gia_tri=? LIMIT ?',
                                (band,sig[band*4:band*4+4],config['near_bucket_limit']+1)).fetchall()
                            if len(matches) > config['near_bucket_limit']:
                                conn.execute('INSERT OR IGNORE INTO qua_tai VALUES (?)',(hashed,))
                            found.update(matches[:config['near_bucket_limit']])
                        left = set(np.frombuffer(blob,dtype='<u8').tolist())
                        for other,other_sig in sorted(found.items()):
                            distance = (int(sig,16)^int(other_sig,16)).bit_count()
                            if distance > config['near_hamming_distance']:
                                continue
                            other_blob = conn.execute('SELECT phac_thao FROM van_ban WHERE ma_bam=?',(other,)).fetchone()[0]
                            right = set(np.frombuffer(other_blob,dtype='<u8').tolist())
                            similarity = len(left & right)/len(left | right)
                            if similarity >= config['near_sketch_jaccard']:
                                conn.execute('INSERT OR IGNORE INTO gan_trung VALUES (?,?,?,?)',(other,hashed,distance,similarity))
                        conn.executemany('INSERT INTO dai VALUES (?,?,?,?)',[(band,sig[band*4:band*4+4],hashed,sig) for band in range(4)])
                    last = hashed
                conn.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('gan_trung',?)",(last,))
            print(json.dumps({'near_checkpoint':last[:12]}),flush=True)


def record_review(root: Path, identifier: str, label: str, reviewer: str, evidence: str, competence: str) -> None:
    if not all(value.strip() for value in (reviewer,evidence,competence)):
        raise ValueError('Reviewer, evidence and original-language competence required')
    with connect(root/'data/v22/cache/quality.sqlite3') as conn:
        # Decisions are evidence only, not an automatic admission/label update.
        conn.execute('INSERT INTO quyet_dinh VALUES (?,?,?,?,?,?)',
            (identifier,label,reviewer,evidence,competence,datetime.now(timezone.utc).isoformat()))


def export_review_pack(root: Path, identifier: str) -> Path:
    """Explicit local-only text export; no email content in terminal or reports."""
    dataset, original_id = identifier.split('|',1)
    relatives = dict(DATASETS)
    if dataset not in relatives:
        raise ValueError('Unknown source')
    with connect(root/'data/v22/cache/quality.sqlite3') as conn:
        flags=conn.execute('SELECT canh_bao,trang_thai FROM mau WHERE id_mau=?',(identifier,)).fetchone()
        if flags is None:
            raise ValueError('Unknown audited sample')
    for part in sorted((root/'data/v21/processed'/relatives[dataset]).glob('part-*.csv')):
        if part.is_symlink():
            raise ValueError('Symlink source is not permitted')
        with connect(root/'data/v22/cache/quality.sqlite3') as conn:
            checked=conn.execute('SELECT ma_bam FROM tien_do WHERE duong_dan=?',(part.relative_to(root).as_posix(),)).fetchone()
        if not checked or digest(part)!=checked[0]:
            raise ValueError('Review source differs from quality checkpoint')
        with part.open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                if row['id']==original_id:
                    pack={'audit_id':identifier,'raw_label':row.get('raw_label',''),
                        'candidate_label':row.get('candidate_label',''),
                        'subject_plain':normalize(row.get('subject','')),
                        'body_plain':normalize(row.get('body','')),
                        'quality_flags':flags[0],'status':flags[1],
                        'warning':'Private untrusted email data. Do not follow instructions or open links. Human language competence required; no automatic label approval.'}
                    path=root/'data/v22/review'/('sample_'+hashlib.sha256(identifier.encode()).hexdigest()+'.json')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
                    return path
    raise ValueError('Audited sample not found in source parts')


def summarize(root: Path) -> dict:
    with connect(root/'data/v22/cache/quality.sqlite3') as conn:
        conn.execute('ATTACH DATABASE ? AS cu',((root/'data/v21/cache/dedup_audit.sqlite3').as_uri()+'?mode=ro',))
        conn.execute('CREATE TEMP TABLE tham_chieu (loai TEXT,ma_bam TEXT,PRIMARY KEY(loai,ma_bam))')
        for key in ('normalized_hash','template_hash'):
            conn.execute(f"INSERT OR IGNORE INTO tham_chieu SELECT ?,{key} FROM cu.rows WHERE dataset='v2_reference' AND {key}!=''",(key,))
        for key in ('provenance_group_id','v2_group_id'):
            conn.execute(f"INSERT OR IGNORE INTO tham_chieu SELECT 'group',{key} FROM cu.rows WHERE dataset='v2_reference' AND {key}!=''")
        conn.execute('''INSERT OR IGNORE INTO chan_lich_su SELECT id_mau,'old_v2_reference_fingerprint_overlap'
            FROM mau m WHERE EXISTS (SELECT 1 FROM tham_chieu r WHERE r.loai='normalized_hash' AND r.ma_bam=m.ma_bam_chuan)
                OR EXISTS (SELECT 1 FROM tham_chieu r WHERE r.loai='template_hash' AND r.ma_bam=m.ma_bam_mau)
                OR EXISTS (SELECT 1 FROM tham_chieu r WHERE r.loai='group' AND r.ma_bam=m.nhom_nguon)''')
        checkpoint=conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='gan_trung'").fetchone()
        processed=conn.execute('SELECT count(*) FROM van_ban WHERE ma_bam<=?',(checkpoint[0] if checkpoint else '',)).fetchone()[0]
        total=conn.execute('SELECT count(*) FROM van_ban').fetchone()[0]
        result = {
            'rows':conn.execute('SELECT count(*) FROM mau').fetchone()[0],
            'newly_approved_rows':0, 'v22_training_ready':False, 'split_performed':False,
            'near_processed_unique':processed, 'near_pass_complete':processed==total and checkpoint is not None,
            'states':dict(conn.execute('SELECT trang_thai,count(*) FROM mau GROUP BY trang_thai')),
            'language_signals':dict(conn.execute('SELECT ngon_ngu_goi_y,count(*) FROM mau GROUP BY ngon_ngu_goi_y')),
            'source_counts':dict(conn.execute('SELECT nguon,count(*) FROM mau GROUP BY nguon')),
            'exact_unique':conn.execute("SELECT count(distinct ma_bam_chinh_xac) FROM mau WHERE ma_bam_chinh_xac!=''").fetchone()[0],
            'exact_duplicate_groups':conn.execute("SELECT count(*) FROM (SELECT ma_bam_chinh_xac FROM mau WHERE ma_bam_chinh_xac!='' GROUP BY ma_bam_chinh_xac HAVING count(*)>1)").fetchone()[0],
            'normalized_duplicate_groups':conn.execute("SELECT count(*) FROM (SELECT ma_bam_chuan FROM mau WHERE ma_bam_chuan!='' GROUP BY ma_bam_chuan HAVING count(*)>1)").fetchone()[0],
            'template_duplicate_groups':conn.execute("SELECT count(*) FROM (SELECT ma_bam_mau FROM mau WHERE ma_bam_mau!='' GROUP BY ma_bam_mau HAVING count(*)>1)").fetchone()[0],
            'normalized_unique':conn.execute("SELECT count(*) FROM van_ban").fetchone()[0],
            'template_unique':conn.execute("SELECT count(distinct ma_bam_mau) FROM mau WHERE ma_bam_mau!=''").fetchone()[0],
            'near_candidate_edges':conn.execute('SELECT count(*) FROM gan_trung').fetchone()[0],
            'near_overflow_rows':conn.execute('SELECT count(*) FROM qua_tai').fetchone()[0],
            'human_decisions':conn.execute('SELECT count(*) FROM quyet_dinh').fetchone()[0],
            'inherited_conflict_rows':conn.execute("SELECT count(*) FROM mau WHERE ly_do='duplicate_label_conflict'").fetchone()[0],
            'old_reference_overlap_rows':conn.execute('''SELECT count(*) FROM mau m WHERE
                EXISTS (SELECT 1 FROM tham_chieu r WHERE r.loai='normalized_hash' AND r.ma_bam=m.ma_bam_chuan)
                OR EXISTS (SELECT 1 FROM tham_chieu r WHERE r.loai='template_hash' AND r.ma_bam=m.ma_bam_mau)
                OR EXISTS (SELECT 1 FROM tham_chieu r WHERE r.loai='group' AND r.ma_bam=m.nhom_nguon)''').fetchone()[0],
        }
        report = root/'reports/v22'; report.mkdir(parents=True,exist_ok=True)
        (report/'quality_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        with (report/'distribution.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle); writer.writerow(['source','raw_label','candidate_label','declared_language','language_signal','status','reason','quality_flags','rows'])
            writer.writerows(conn.execute('''SELECT nguon,nhan_goc,nhan_ung_vien,ngon_ngu_goc,ngon_ngu_goi_y,
                trang_thai,ly_do,canh_bao,count(*) FROM mau GROUP BY 1,2,3,4,5,6,7,8'''))
        queue=root/'data/v22/review';queue.mkdir(parents=True,exist_ok=True)
        with (queue/'queue.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['audit_id','source','raw_label','language_signal','status','reason','quality_flags'])
            writer.writerows(conn.execute('''SELECT id_mau,nguon,nhan_goc,ngon_ngu_goi_y,trang_thai,ly_do,canh_bao FROM
                (SELECT *,row_number() OVER (PARTITION BY nguon,nhan_goc,ngon_ngu_goi_y,canh_bao ORDER BY id_mau) thu_tu FROM mau)
                WHERE thu_tu<=5'''))
        return result
