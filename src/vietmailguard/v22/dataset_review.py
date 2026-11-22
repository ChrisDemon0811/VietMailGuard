"""Local source review over the frozen sample; actual people supply decisions."""
from __future__ import annotations
import csv
import hashlib
import json
import re
import sqlite3
import zlib
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path

from .quality import DATASETS,digest,normalize


class DatasetReview:
    def __init__(self,root: Path,*,collection: Path | None=None):
        self.root=root
        if collection is None:
            queue=root/'data/v22/phase3/review_queue.csv'
            self.path=root/'data/v22/phase32/review.sqlite3'
        else:
            collection=Path(collection)
            if not collection.is_absolute():collection=root/collection
            base=(root/'data/v22').resolve()
            resolved=collection.resolve(strict=True)
            if collection.is_symlink() or not resolved.is_relative_to(base) or resolved==base:
                raise ValueError('Review collection must be a local directory inside data/v22')
            queue=resolved/'review_queue.csv'
            self.path=resolved/'review.sqlite3'
            if not queue.is_file() or queue.is_symlink() or self.path.is_symlink():
                raise ValueError('Missing or unsafe review collection')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as conn:
            conn.executescript('''
            CREATE TABLE IF NOT EXISTS cau_hinh(ma TEXT PRIMARY KEY,gia_tri TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS mau(id_mau TEXT PRIMARY KEY,tang TEXT,nguon TEXT,nhan_goc TEXT,
                nhan_ung_vien TEXT,ngon_ngu_goi_y TEXT,id_nhom TEXT,noi_dung_nen BLOB);
            CREATE TABLE IF NOT EXISTS tien_do(duong_dan TEXT PRIMARY KEY,ma_bam TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS quyet_dinh(id_quyet_dinh INTEGER PRIMARY KEY,id_mau TEXT NOT NULL REFERENCES mau,
                nhan TEXT NOT NULL CHECK(nhan IN ('normal','spam','phishing','review','exclude')),
                ngon_ngu TEXT NOT NULL CHECK(ngon_ngu IN ('en','vi','zh','mixed','unknown','other')),
                nguoi_duyet TEXT NOT NULL,nang_luc TEXT NOT NULL,xac_nhan_ngon_ngu INTEGER NOT NULL CHECK(xac_nhan_ngon_ngu IN (0,1)),
                bang_chung TEXT NOT NULL,thoi_gian TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS chi_muc_quyet_dinh ON quyet_dinh(id_mau,nguoi_duyet,id_quyet_dinh);
            ''')
            policy=root/'config/v22/expansion.json'
            lock=json.dumps({'queue':digest(queue),'policy':digest(policy)},sort_keys=True)
            old=conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='sample_lock'").fetchone()
            if old and old[0]!=lock:raise ValueError('Frozen source review sample or policy changed')
            if not old:
                with queue.open(encoding='utf-8',newline='') as f:
                    for row in csv.DictReader(f):
                        conn.execute('INSERT INTO mau VALUES (?,?,?,?,?,?,?,NULL)',(row['audit_id'],row['stratum'],row['source'],
                            row['raw_label'],row['candidate_label'],row['language_signal'],row['constraint_group_id']))
                conn.execute("INSERT INTO cau_hinh VALUES ('sample_lock',?)",(lock,))

    @contextmanager
    def connection(self):
        conn=sqlite3.connect(self.path.resolve().as_uri(),uri=True);conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON');conn.execute('PRAGMA journal_mode=WAL')
        try:
            with conn:yield conn
        finally:conn.close()

    def prepare_content(self) -> dict:
        """Cache original selected emails once; preserve case, URL and headers."""
        csv.field_size_limit(100_000_000)
        with self.connection() as conn:
            pending={row[0] for row in conn.execute('SELECT id_mau FROM mau WHERE noi_dung_nen IS NULL')}
            if not pending:return self.progress()
            reference=self.root/'data/v22/phase3/expansion.sqlite3'
            conn.execute('ATTACH DATABASE ? AS ba',(reference.resolve().as_uri()+'?mode=ro',))
            for source,relative in DATASETS:
                for part in sorted((self.root/'data/v21/processed'/relative).glob('part-*.csv')):
                    if not pending:break
                    if part.is_symlink():raise ValueError('Unsafe source symlink')
                    key=part.relative_to(self.root).as_posix();sha=digest(part)
                    expected=conn.execute('SELECT ma_bam FROM ba.tien_do WHERE duong_dan=?',(key,)).fetchone()
                    if not expected or sha!=expected[0]:raise ValueError('Review source differs from audit')
                    with conn,part.open(encoding='utf-8-sig',newline='') as f:
                        for raw in csv.DictReader(f):
                            identifier=source+'|'+raw['id']
                            if identifier not in pending:continue
                            pack={key:normalize(raw.get(key,'')) for key in ('subject','body','sender','receiver','date')}
                            pack['original_path']=raw.get('original_path','') or raw.get('source_path','')
                            conn.execute('UPDATE mau SET noi_dung_nen=? WHERE id_mau=?',
                                (zlib.compress(json.dumps(pack,ensure_ascii=False).encode('utf-8')),identifier))
                            pending.remove(identifier)
                        if digest(part)!=sha:raise ValueError('Review source changed during lookup')
                        conn.execute('INSERT OR REPLACE INTO tien_do VALUES (?,?)',(key,sha))
            if pending:raise ValueError('Some selected original emails unavailable')
        return self.progress()

    def list_samples(self,source: str='',only_pending: bool=True,reviewer: str='') -> list[dict]:
        with self.connection() as conn:
            query='SELECT id_mau,tang,nguon,nhan_goc,nhan_ung_vien,ngon_ngu_goi_y,id_nhom FROM mau WHERE (?=\'\' OR nguon=?)'
            params=[source,source]
            if only_pending:
                query+=' AND NOT EXISTS (SELECT 1 FROM quyet_dinh q WHERE q.id_mau=mau.id_mau AND (?=\'\' OR q.nguoi_duyet=?))'
                params.extend([reviewer,reviewer])
            return [dict(row) for row in conn.execute(query+' ORDER BY tang,id_mau',params)]

    def open_sample(self,identifier: str) -> dict:
        with self.connection() as conn:
            row=conn.execute('SELECT * FROM mau WHERE id_mau=?',(identifier,)).fetchone()
            if row is None:raise ValueError('Unknown source sample')
            if row['noi_dung_nen'] is None:raise ValueError('Original content cache missing; run preparation command')
            result={key:row[key] for key in row.keys() if key!='noi_dung_nen'}
            result.update(json.loads(zlib.decompress(row['noi_dung_nen']).decode('utf-8')))
            result['decisions']=[dict(x) for x in conn.execute('SELECT * FROM quyet_dinh WHERE id_mau=? ORDER BY id_quyet_dinh',(identifier,))]
            return result

    def _session_key(self,reviewer: str,kind: str,identifier: str='') -> str:
        token=hashlib.sha256(reviewer.strip().encode('utf-8')).hexdigest()
        return kind+':'+token+(':'+hashlib.sha256(identifier.encode()).hexdigest() if identifier else '')

    def _state(self,key: str) -> dict:
        with self.connection() as conn:
            row=conn.execute('SELECT gia_tri FROM cau_hinh WHERE ma=?',(key,)).fetchone()
        return json.loads(row[0]) if row else {}

    def _put_state(self,conn,key: str,value: dict) -> None:
        payload=json.dumps(value,ensure_ascii=False)
        if len(payload.encode('utf-8'))>65536:raise ValueError('Review checkpoint is too large')
        conn.execute('INSERT INTO cau_hinh VALUES (?,?) ON CONFLICT(ma) DO UPDATE SET gia_tri=excluded.gia_tri',(key,payload))

    def checkpoint(self,reviewer: str) -> dict:
        return self._state(self._session_key(reviewer,'phien_lam_viec'))

    def _save_checkpoint(self,conn,reviewer: str,state: dict) -> None:
        identifier=state.get('selected_id')
        if identifier and conn.execute('SELECT 1 FROM mau WHERE id_mau=?',(identifier,)).fetchone() is None:
            raise ValueError('Unknown checkpoint email')
        self._put_state(conn,self._session_key(reviewer,'phien_lam_viec'),{**state,'updated_at':datetime.now(timezone.utc).isoformat()})

    def save_checkpoint(self,reviewer: str,state: dict) -> None:
        with self.connection() as conn:self._save_checkpoint(conn,reviewer,state)

    def draft(self,reviewer: str,identifier: str) -> dict:
        return self._state(self._session_key(reviewer,'ban_nhap',identifier))

    def save_draft(self,reviewer: str,identifier: str,draft: dict) -> None:
        with self.connection() as conn:
            if conn.execute('SELECT 1 FROM mau WHERE id_mau=?',(identifier,)).fetchone() is None:
                raise ValueError('Unknown draft email')
            key=self._session_key(reviewer,'ban_nhap',identifier)
            old=conn.execute('SELECT gia_tri FROM cau_hinh WHERE ma=?',(key,)).fetchone()
            previous=json.loads(old[0]) if old else {}
            # Hiding suggestions again must not erase a previous AI exposure.
            self._put_state(conn,key,{**draft,'ai_exposed':bool(draft.get('ai_exposed') or previous.get('ai_exposed')),
                'updated_at':datetime.now(timezone.utc).isoformat()})

    def reviewed_ids(self,reviewer: str='') -> set[str]:
        with self.connection() as conn:
            return {row[0] for row in conn.execute('SELECT DISTINCT id_mau FROM quyet_dinh WHERE (?=\'\' OR nguoi_duyet=?)',(reviewer,reviewer))}

    def _ai_seen(self,conn,reviewer: str,identifier: str) -> bool:
        draft=conn.execute('SELECT gia_tri FROM cau_hinh WHERE ma=?',(self._session_key(reviewer,'ban_nhap',identifier),)).fetchone()
        if draft and json.loads(draft[0]).get('ai_exposed'):return True
        history=conn.execute('''SELECT q.bang_chung,c.gia_tri FROM quyet_dinh q
            LEFT JOIN cau_hinh c ON c.ma='kiem_tra_quyet_dinh:' || q.id_quyet_dinh
            WHERE q.id_mau=? AND q.nguoi_duyet=?''',(identifier,reviewer.strip()))
        return any('review_method=local_model_assisted;' in evidence or (payload and json.loads(payload).get('ai_exposed'))
                   for evidence,payload in history)

    def ai_seen(self,reviewer: str,identifier: str) -> bool:
        with self.connection() as conn:return self._ai_seen(conn,reviewer,identifier)

    def independent_review_ids(self) -> set[str]:
        """Unresolved requests remain visible; reviewer codes do not certify independence."""
        with self.connection() as conn:
            rows=conn.execute('''SELECT q.id_mau,q.nhan,c.gia_tri FROM quyet_dinh q
                LEFT JOIN cau_hinh c ON c.ma='kiem_tra_quyet_dinh:' || q.id_quyet_dinh''')
            return {identifier for identifier,label,payload in rows
                    if label in ('review','phishing') or (payload and json.loads(payload).get('independent_required'))}

    def save(self,identifier: str,label: str,language: str,reviewer: str,competence: str,attested: bool,evidence: str,
             *,independent_required: bool=False,independent_reason: str='',ai_exposed: bool=False,checkpoint: dict | None=None) -> None:
        if not all(x.strip() for x in (reviewer,competence,evidence)):raise ValueError('Reviewer, competence and evidence are required')
        if attested and language=='unknown':raise ValueError('Confirmed original language must be specified')
        with self.connection() as conn:
            ai_exposed=bool(ai_exposed or 'review_method=local_model_assisted;' in evidence or self._ai_seen(conn,reviewer,identifier))
            cursor=conn.execute('INSERT INTO quyet_dinh(id_mau,nhan,ngon_ngu,nguoi_duyet,nang_luc,xac_nhan_ngon_ngu,bang_chung,thoi_gian) VALUES (?,?,?,?,?,?,?,?)',
                (identifier,label,language,reviewer.strip(),competence.strip(),int(attested),evidence.strip(),datetime.now(timezone.utc).isoformat()))
            self._put_state(conn,'kiem_tra_quyet_dinh:'+str(cursor.lastrowid),{
                'independent_required':bool(independent_required or label in ('review','phishing')),
                'independent_reason':independent_reason or ('uncertain_label' if label=='review' else 'phishing_requires_second_review' if label=='phishing' else ''),
                'ai_exposed':bool(ai_exposed),'approved':False})
            if checkpoint is not None:self._save_checkpoint(conn,reviewer,checkpoint)
        # This is an append-only observation, never an eligible label update.

    def progress(self) -> dict:
        with self.connection() as conn:
            return {'samples':conn.execute('SELECT count(*) FROM mau').fetchone()[0],
                'strata':conn.execute('SELECT count(DISTINCT tang) FROM mau').fetchone()[0],
                'content_ready':conn.execute('SELECT count(*) FROM mau WHERE noi_dung_nen IS NOT NULL').fetchone()[0],
                'reviewed_samples':conn.execute('SELECT count(DISTINCT id_mau) FROM quyet_dinh').fetchone()[0],
                'decision_events':conn.execute('SELECT count(*) FROM quyet_dinh').fetchone()[0],
                'qualified_attested_samples':conn.execute('''SELECT count(DISTINCT id_mau) FROM quyet_dinh
                    WHERE id_quyet_dinh IN (SELECT max(id_quyet_dinh) FROM quyet_dinh GROUP BY id_mau,nguoi_duyet)
                    AND xac_nhan_ngon_ngu=1''').fetchone()[0],
                'automatically_approved':0}

    def export(self) -> Path:
        path=self.path.parent/'human_review_decisions.csv'
        temporary=path.with_suffix('.csv.partial')
        with self.connection() as conn,temporary.open('w',encoding='utf-8',newline='') as f:
            cursor=conn.execute('''SELECT q.*,c.gia_tri FROM quyet_dinh q
                LEFT JOIN cau_hinh c ON c.ma='kiem_tra_quyet_dinh:' || q.id_quyet_dinh ORDER BY q.id_quyet_dinh''')
            writer=csv.writer(f)
            writer.writerow([x[0] for x in cursor.description[:-1]]+['can_kiem_tra_doc_lap','ly_do_kiem_tra_doc_lap','da_xem_goi_y_ai','da_phe_duyet'])
            for row in cursor:
                meta=json.loads(row[-1]) if row[-1] else {}
                writer.writerow([*row[:-1],int(meta.get('independent_required',row['nhan'] in ('review','phishing'))),
                    meta.get('independent_reason',''),int(bool(meta.get('ai_exposed') or 'review_method=local_model_assisted;' in row['bang_chung'])),0])
        temporary.replace(path)
        return path


def redacted_preview(text: str) -> str:
    """Display helper only; no claim of exhaustive anonymization."""
    text=re.sub(r'(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b','[EMAIL]',text)
    return re.sub(r'(?<!\w)(?:\+?\d[\d .()\-]{8,}\d)(?!\w)','[PHONE/ID]',text)
