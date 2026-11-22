"""Phase 3: disk-backed MinHash and conservative split constraints, no admission.

Candidate linkage is not proof of duplicate identity. All oversized bucket
members are quarantined, never silently sampled or forced into one huge group.
All existing artifacts are read-only; new outputs contain metadata, not email.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import sqlite3
import unicodedata
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .quality import DATASETS, digest, normalize
from vietmailguard.v21.dedup_audit import URL_RE, EMAIL_RE


def readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file() or path.is_symlink():
        raise ValueError('Missing or unsafe input')
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    conn.execute('PRAGMA cache_size=-32768')
    return conn


def connect(root: Path) -> sqlite3.Connection:
    path = root / 'data/v22/phase3/expansion.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path.resolve().as_uri(), uri=True)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=-65536')
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS cau_hinh (ma TEXT PRIMARY KEY, gia_tri TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tien_do (duong_dan TEXT PRIMARY KEY, ma_bam TEXT NOT NULL, so_dong INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS van_ban (id_van_ban INTEGER PRIMARY KEY, ma_bam TEXT NOT NULL UNIQUE,
        chu_ky BLOB NOT NULL, so_ky_tu INTEGER NOT NULL, ly_do TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS dai (vi_tri INTEGER NOT NULL, ma_dai BLOB NOT NULL, id_van_ban INTEGER NOT NULL REFERENCES van_ban,
        PRIMARY KEY (vi_tri,ma_dai,id_van_ban)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS canh (id_truoc INTEGER NOT NULL, id_sau INTEGER NOT NULL,
        tuong_dong REAL NOT NULL, PRIMARY KEY(id_truoc,id_sau));
    CREATE TABLE IF NOT EXISTS khoi_lon (loai TEXT NOT NULL, ma_khoi TEXT NOT NULL, so_mau INTEGER NOT NULL,
        PRIMARY KEY(loai,ma_khoi));
    CREATE TABLE IF NOT EXISTS nhom (id_van_ban INTEGER PRIMARY KEY REFERENCES van_ban,
        id_nhom TEXT NOT NULL, so_thanh_vien INTEGER NOT NULL, can_cach_ly INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS mau (id_mau TEXT PRIMARY KEY, nguon TEXT NOT NULL, nhan_goc TEXT,
        nhan_ung_vien TEXT, ngon_ngu_khai_bao TEXT, ngon_ngu_goi_y TEXT, nguon_goc TEXT,
        ma_bam TEXT, id_nhom TEXT, id_dai_dien TEXT, trang_thai TEXT NOT NULL CHECK
        (trang_thai IN ('approved','pending_review','weak_label','duplicate','excluded')),
        ly_do TEXT NOT NULL, bang_chung TEXT NOT NULL, nhan TEXT NOT NULL DEFAULT '',
        quy_tac TEXT NOT NULL DEFAULT '', thanh_phan TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS lay_mau (id_mau TEXT PRIMARY KEY REFERENCES mau, tang TEXT NOT NULL,
        dan_so INTEGER NOT NULL, so_lay INTEGER NOT NULL, xac_suat REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS quyet_dinh (id_mau TEXT NOT NULL REFERENCES lay_mau,
        nguoi_duyet TEXT NOT NULL, nhan TEXT NOT NULL CHECK(nhan IN ('normal','spam','phishing','review','exclude')),
        nang_luc TEXT NOT NULL, bang_chung TEXT NOT NULL, thoi_gian TEXT NOT NULL,
        PRIMARY KEY(id_mau,nguoi_duyet));
    CREATE INDEX IF NOT EXISTS chi_muc_nhom ON mau(id_nhom);
    ''')
    return conn


def validate_config(config: dict) -> None:
    if any(config[k] for k in ('bulk_approval_activated', 'training_allowed', 'final_test_access_allowed')):
        raise ValueError('Phase 3 proposes approvals only; no activation, training or Test')
    p, r = config['minhash_permutations'], config['minhash_rows_per_band']
    if not (p >= 16 and r >= 1 and p % r == 0 and config['workers'] == 1):
        raise ValueError('Invalid bounded MinHash configuration')
    if not (0 < config['review_alpha_per_stratum'] < 1 and 0 < config['review_max_error_rate'] < 1):
        raise ValueError('Invalid review acceptance policy')
    needed = math.ceil(math.log(config['review_alpha_per_stratum']) / math.log(1-config['review_max_error_rate']))
    if config['review_sample_per_stratum'] < needed or config['review_max_unresolved'] != 0:
        raise ValueError('Sampling insufficient for the declared zero-error acceptance rule')


def minhash(text: str, config: dict) -> tuple[bytes, str, int]:
    """32 seeded affine permutations of full-text character shingles.

    Long inputs get no prefix approximation: quarantine them for a separate
    full-text pass. Temporary arrays are bounded by the configured text size.
    """
    text = unicodedata.normalize('NFKC', normalize(text)).casefold()
    text = ' '.join(EMAIL_RE.sub(' <email> ', URL_RE.sub(' <url> ', text)).split())
    length = len(text)
    if length > config['minhash_max_characters']:
        return b'', 'long_text_full_comparison_pending', length
    if length < config['minhash_min_characters']:
        return b'', 'short_text_fuzzy_unreliable', length
    n = config['minhash_shingle_characters']
    values = np.fromiter((zlib.crc32(text[i:i+n].encode('utf-8')) for i in range(length-n+1)), dtype=np.uint64)
    values = np.unique(values)
    # 31-bit coefficients avoid overflow in uint64 multiplication with CRC32.
    rng = np.random.default_rng(config['seed'])
    a = rng.integers(1, 2**31, config['minhash_permutations'], dtype=np.uint64)
    b = rng.integers(0, 2**32, config['minhash_permutations'], dtype=np.uint64)
    signature = np.array([int(((values*x+y) % 4294967291).min()) for x,y in zip(a,b)], dtype='<u4')
    return signature.tobytes(), '', length


def _attach(conn: sqlite3.Connection, root: Path) -> None:
    for alias, relative in [('cu','data/v22/cache/quality.sqlite3'), ('kiem','data/v21/cache/dedup_audit.sqlite3')]:
        conn.execute(f'ATTACH DATABASE ? AS {alias}', ((root/relative).resolve().as_uri()+'?mode=ro',))


def prepare(root: Path, config: dict) -> dict:
    validate_config(config)
    with connect(root) as conn, readonly(root/'data/v22/cache/quality.sqlite3') as old:
        signature = {'config':config, 'phase2_sha256':digest(root/'data/v22/cache/quality.sqlite3'),
                     'audit_sha256':digest(root/'data/v21/cache/dedup_audit.sqlite3')}
        locked = json.dumps(signature, sort_keys=True)
        previous = conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='input_lock'").fetchone()
        if previous and previous[0] != locked:
            raise ValueError('Phase 3 inputs/config changed; resume blocked')
        conn.execute("INSERT OR IGNORE INTO cau_hinh VALUES ('input_lock',?)",(locked,)); conn.commit()
        csv.field_size_limit(100_000_000)
        for source, relative in DATASETS:
            parts=sorted((root/'data/v21/processed'/relative).glob('part-*.csv'))
            if not parts: raise ValueError('Missing standardized source: '+source)
            for part in parts:
                if part.is_symlink(): raise ValueError('Unsafe source symlink')
                key=part.relative_to(root).as_posix()
                sha=digest(part)
                expected=old.execute('SELECT ma_bam FROM tien_do WHERE duong_dan=?',(key,)).fetchone()
                if not expected or expected[0]!=sha: raise ValueError('Input differs from Phase 2 checkpoint')
                previous=conn.execute('SELECT ma_bam FROM tien_do WHERE duong_dan=?',(key,)).fetchone()
                if previous:
                    if previous[0]!=sha: raise ValueError('Changed source after checkpoint')
                    continue
                count=0
                with conn, part.open(encoding='utf-8-sig',newline='') as handle:
                    for row in csv.DictReader(handle):
                        found=old.execute('SELECT ma_bam_chuan FROM mau WHERE id_mau=?',(source+'|'+row['id'],)).fetchone()
                        if found is None: raise ValueError('Missing Phase 2 sample')
                        hashed=found[0]
                        if hashed and not conn.execute('SELECT 1 FROM van_ban WHERE ma_bam=?',(hashed,)).fetchone():
                            blob,reason,length=minhash(row.get('subject','')+'\n'+row.get('body',''),config)
                            conn.execute('INSERT INTO van_ban(ma_bam,chu_ky,so_ky_tu,ly_do) VALUES (?,?,?,?)',(hashed,blob,length,reason))
                        count+=1
                    if digest(part)!=sha: raise ValueError('Input changed while reading')
                    conn.execute('INSERT INTO tien_do VALUES (?,?,?)',(key,sha,count))
                print(json.dumps({'stage':'full_text_minhash','source':source,'part':part.name,'rows':count}),flush=True)
        expected=old.execute('SELECT count(*) FROM mau').fetchone()[0]
        rows=conn.execute('SELECT coalesce(sum(so_dong),0) FROM tien_do').fetchone()[0]
        unique=conn.execute('SELECT count(*) FROM van_ban').fetchone()[0]
        if rows!=expected or unique!=old.execute('SELECT count(*) FROM van_ban').fetchone()[0]:
            raise ValueError('Phase 3 input reconciliation failed')
        return {'rows':rows,'unique':unique}


def index_bands(root: Path, config: dict) -> None:
    with connect(root) as conn:
        last=int((conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='bands'").fetchone() or ['0'])[0])
        while True:
            batch=conn.execute('SELECT id_van_ban,chu_ky FROM van_ban WHERE id_van_ban>? ORDER BY id_van_ban LIMIT 1000',(last,)).fetchall()
            if not batch: break
            with conn:
                for index,blob in batch:
                    if blob:
                        width=config['minhash_rows_per_band']*4
                        conn.executemany('INSERT INTO dai VALUES (?,?,?)',[(i//width,blob[i:i+width],index) for i in range(0,len(blob),width)])
                    last=index
                conn.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('bands',?)",(str(last),))


class Groups:
    """Bounded O(unique fingerprints) arrays; no corpus texts in RAM."""
    def __init__(self, size: int):
        self.parent=np.arange(size+1,dtype=np.int32)
        self.warn=np.zeros(size+1,dtype=np.bool_)

    def find(self, value: int) -> int:
        while self.parent[value]!=value:
            self.parent[value]=self.parent[self.parent[value]]
            value=int(self.parent[value])
        return value

    def union(self, a: int, b: int, quarantine: bool=False) -> None:
        a,b=self.find(a),self.find(b)
        lo,hi=min(a,b),max(a,b)
        self.parent[hi]=lo
        self.warn[lo]=self.warn[a] or self.warn[b] or quarantine


def group_candidates(root: Path, config: dict) -> dict:
    """Restartable grouping, persisted only after a complete transaction.

    Every bucket member participates in a constraint or quarantine.
    Small buckets receive pairwise signature checks with secondary length
    blocking. Unverified collisions remain quarantined constraints.
    """
    with connect(root) as conn:
        _attach(conn,root)
        version=conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='group_algorithm'").fetchone()
        if version and version[0]=='2' and conn.execute("SELECT 1 FROM cau_hinh WHERE ma='groups_complete'").fetchone():
            return json.loads(conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='groups_complete'").fetchone()[0])
        n=conn.execute('SELECT max(id_van_ban) FROM van_ban').fetchone()[0] or 0
        groups=Groups(n)
        stats={'minhash_candidate_pairs':0,'minhash_estimated_edges':0,'secondary_length_rejections':0,
               'phase2_overflow_fingerprints_covered':0,'large_minhash_buckets':0,'large_legacy_buckets':0,
               'legacy_edges_fulltext_confirmed':0,'legacy_edges_deferred':0}
        with conn:
            conn.execute('DELETE FROM canh'); conn.execute('DELETE FROM khoi_lon');conn.execute('DELETE FROM nhom')
            # Conservative grouping of template and known provenance; similarity
            # alone does not deduplicate or approve rows.
            for field in ('ma_bam_mau','nhom_nguon'):
                query=f'''SELECT m.{field},v.id_van_ban FROM cu.mau m JOIN van_ban v ON v.ma_bam=m.ma_bam_chuan
                    WHERE m.{field}!='' ORDER BY m.{field},v.id_van_ban'''
                for _,members in itertools.groupby(conn.execute(query),lambda row:row[0]):
                    first=next(members)[1]
                    for _,other in members: groups.union(first,other)
            # Parent aliases require exact known IDs; unknown parent links are
            # never guessed and remain a pre-admission verification requirement.
            for a,b in conn.execute('''SELECT v.id_van_ban,w.id_van_ban FROM cu.mau m JOIN cu.mau p
                ON p.id_mau=m.id_cha OR p.id_mau=m.nguon||'|'||m.id_cha
                JOIN van_ban v ON v.ma_bam=m.ma_bam_chuan JOIN van_ban w ON w.ma_bam=p.ma_bam_chuan
                WHERE m.id_cha!='' '''): groups.union(a,b)
            for a,b,left,right,nleft,nright in conn.execute('''SELECT v.id_van_ban,w.id_van_ban,v.chu_ky,w.chu_ky,
                v.so_ky_tu,w.so_ky_tu FROM cu.gan_trung g
                JOIN van_ban v ON v.ma_bam=g.ma_truoc JOIN van_ban w ON w.ma_bam=g.ma_sau'''):
                if left and right and min(nleft,nright)/max(nleft,nright)>=.75 and float(np.mean(
                        np.frombuffer(left,dtype='<u4')==np.frombuffer(right,dtype='<u4')))>=config['minhash_similarity']:
                    groups.union(a,b,True);stats['legacy_edges_fulltext_confirmed']+=1
                else:
                    # Unresolved prefix-only evidence is isolated, never
                    # discarded or used to force unrelated texts together.
                    groups.warn[groups.find(a)]=True;groups.warn[groups.find(b)]=True
                    stats['legacy_edges_deferred']+=1
            # Resolve ALL members of every legacy overloaded bucket, without
            # the old LIMIT 100. No pairwise O(n^2) expansion for large buckets.
            conn.execute('''CREATE TEMP TABLE vuot_nguong AS SELECT vi_tri,gia_tri,count(*) so_mau
                FROM cu.dai GROUP BY vi_tri,gia_tri HAVING count(*)>100''')
            conn.execute('CREATE UNIQUE INDEX tam_vuot ON vuot_nguong(vi_tri,gia_tri)')
            legacy=conn.execute('''SELECT d.vi_tri,d.gia_tri,v.id_van_ban,l.so_mau FROM cu.dai d
                JOIN vuot_nguong l ON l.vi_tri=d.vi_tri AND l.gia_tri=d.gia_tri
                JOIN van_ban v ON v.ma_bam=d.ma_bam ORDER BY d.vi_tri,d.gia_tri''')
            for key,members in itertools.groupby(legacy,lambda row:(row[0],row[1])):
                first=next(members);groups.warn[groups.find(first[2])]=True
                for row in members:groups.warn[groups.find(row[2])]=True
                conn.execute('INSERT INTO khoi_lon VALUES (?,?,?)',('legacy',str(key),first[3]))
                stats['large_legacy_buckets']+=1
            stats['phase2_overflow_fingerprints_covered']=conn.execute('''SELECT count(*) FROM cu.qua_tai q
                JOIN van_ban v ON v.ma_bam=q.ma_bam''').fetchone()[0]
            # SQLite index orders 8 bands on disk. One bucket at a time in RAM;
            # large buckets stream all members into quarantine independently.
            stream=conn.execute('''SELECT d.vi_tri,d.ma_dai,v.id_van_ban,v.chu_ky,v.so_ky_tu FROM dai d
                JOIN van_ban v USING(id_van_ban) ORDER BY d.vi_tri,d.ma_dai,d.id_van_ban''')
            for key,items in itertools.groupby(stream,lambda row:(row[0],row[1])):
                bucket=list(itertools.islice(items,config['pairwise_bucket_limit']+1))
                if len(bucket)<2: continue
                anchor=bucket[0][2]
                if len(bucket)>config['pairwise_bucket_limit']:
                    count=len(bucket)
                    for row in bucket:groups.warn[groups.find(row[2])]=True
                    for row in items:
                        groups.warn[groups.find(row[2])]=True;count+=1
                    conn.execute('INSERT INTO khoi_lon VALUES (?,?,?)',('minhash',str(key[0])+':'+key[1].hex(),count))
                    stats['large_minhash_buckets']+=1
                    continue
                signatures={row[2]:np.frombuffer(row[3],dtype='<u4') for row in bucket}
                for left,right in itertools.combinations(bucket,2):
                    stats['minhash_candidate_pairs']+=1
                    if min(left[4],right[4])/max(left[4],right[4]) < 0.75:
                        stats['secondary_length_rejections']+=1;continue
                    score=float(np.mean(signatures[left[2]]==signatures[right[2]]))
                    if score>=config['minhash_similarity']:
                        groups.union(left[2],right[2],True)
                        conn.execute('INSERT OR IGNORE INTO canh VALUES (?,?,?)',(min(left[2],right[2]),max(left[2],right[2]),score))
            stats['minhash_estimated_edges']=conn.execute('SELECT count(*) FROM canh').fetchone()[0]
            roots=np.array([groups.find(i) for i in range(n+1)],dtype=np.int32)
            counts=np.bincount(roots,minlength=n+1)
            conn.executemany('INSERT INTO nhom VALUES (?,?,?,?)',
                ((i,'v22_constraint_'+str(roots[i]),int(counts[roots[i]]),int(groups.warn[roots[i]])) for i in range(1,n+1)))
            stats['legacy_overflow_unquarantined']=conn.execute('''SELECT count(*) FROM cu.qua_tai q
                LEFT JOIN van_ban v ON v.ma_bam=q.ma_bam LEFT JOIN nhom n USING(id_van_ban)
                WHERE n.can_cach_ly IS NULL OR n.can_cach_ly!=1''').fetchone()[0]
            if stats['legacy_overflow_unquarantined']:
                raise ValueError('Legacy overflow coverage incomplete; no grouping approval')
            conn.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('groups_complete',?)",(json.dumps(stats),))
            conn.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('group_algorithm','2')")
        return stats


def classify(root: Path, config: dict) -> dict:
    """Disjoint statuses with all reasons; no source-level approval activated."""
    with connect(root) as conn:
        _attach(conn,root)
        with conn:
            conn.execute('DELETE FROM lay_mau') if not conn.execute('SELECT count(*) FROM quyet_dinh').fetchone()[0] else None
            # Never replace decisions or silently rerun classification after review.
            if conn.execute('SELECT count(*) FROM quyet_dinh').fetchone()[0]:
                raise ValueError('Review decisions exist; classification is locked')
            conn.execute('DELETE FROM mau')
            conn.execute('''CREATE TEMP TABLE dai_dien AS SELECT ma_bam_chuan ma_bam,id_mau FROM (
                SELECT ma_bam_chuan,id_mau,row_number() OVER (PARTITION BY ma_bam_chuan ORDER BY
                CASE nguon WHEN 'balanced_dataset' THEN 9 WHEN 'enron_original' THEN 5 ELSE 1 END,id_mau) thu_tu
                FROM cu.mau WHERE ma_bam_chuan!='' AND trang_thai!='exclude') WHERE thu_tu=1''')
            conn.execute('CREATE UNIQUE INDEX tam_dai_dien ON dai_dien(ma_bam)')
            conn.execute('''CREATE TEMP TABLE xung_dot AS SELECT ma_bam_chuan FROM cu.mau
                WHERE nhan_ung_vien IN ('normal_candidate','spam_or_scam_or_phishing_candidate','ham','spam')
                GROUP BY ma_bam_chuan HAVING count(DISTINCT CASE WHEN nhan_ung_vien IN ('normal_candidate','ham') THEN 0 ELSE 1 END)>1''')
            conn.execute('CREATE UNIQUE INDEX tam_xung_dot ON xung_dot(ma_bam_chuan)')
            cursor=conn.execute('''SELECT m.*,v.ly_do,n.id_nhom,n.can_cach_ly,d.id_mau,
                h.id_mau,x.ma_bam_chuan,k.source
                FROM cu.mau m LEFT JOIN van_ban v ON v.ma_bam=m.ma_bam_chuan
                LEFT JOIN nhom n USING(id_van_ban) LEFT JOIN dai_dien d ON d.ma_bam=m.ma_bam_chuan
                LEFT JOIN cu.chan_lich_su h ON h.id_mau=m.id_mau LEFT JOIN xung_dot x USING(ma_bam_chuan)
                JOIN kiem.rows k ON k.audit_id=m.id_mau ORDER BY m.id_mau''')
            for row in cursor:
                (identifier,source,raw,candidate,declared,signal,origin,exact,hashed,template,provenance,parent,
                 prior,reason,flags,sketch_reason,group,quarantine,representative,overlap,conflict,component)=row
                reasons=['source_label_policy_pending','use_rights_pending','language_not_human_verified']
                if flags:reasons.append('quality:'+flags)
                if sketch_reason:reasons.append(sketch_reason)
                if quarantine:reasons.append('unresolved_similarity_or_overflow_component')
                if overlap:reasons.append('old_reference_overlap_guard')
                if conflict or reason=='duplicate_label_conflict':reasons.append('candidate_label_conflict')
                if parent:reasons.append('parent_lineage_verify_before_admission')
                if prior=='exclude':status='excluded';reasons.append('inherited_exclude:'+reason)
                elif representative and representative!=identifier:status='duplicate';reasons.append('normalized_duplicate_keep_provenance')
                elif flags or sketch_reason or quarantine or overlap or conflict or source=='enron_original':status='pending_review'
                else:status='weak_label'
                conn.execute('INSERT INTO mau VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (identifier,source,raw,candidate,declared,signal,origin,hashed,group or '',representative or identifier,
                     status,';'.join(reasons),'phase2_audit:'+identifier,'','',component or source))
            return {'status_counts':dict(conn.execute('SELECT trang_thai,count(*) FROM mau GROUP BY trang_thai')),
                    'candidate_conflict_fingerprints':conn.execute('SELECT count(*) FROM xung_dot').fetchone()[0]}


def sample_review(root: Path, config: dict) -> dict:
    """Seeded random-priority sample of constraint groups within each stratum.

    The policy and seed are locked before any human decision. All-small-strata
    census; no source purity estimate is manufactured when decisions are absent.
    """
    with connect(root) as conn:
        if conn.execute('SELECT count(*) FROM quyet_dinh').fetchone()[0]:
            raise ValueError('Review has started; sampling cannot be replaced')
        conn.create_function('ngau_nhien',1,lambda key:hashlib.sha256((str(config['seed'])+'|'+key).encode()).hexdigest(),deterministic=True)
        with conn:
            conn.execute('DELETE FROM lay_mau')
            conn.execute('''CREATE TEMP TABLE ung_vien AS SELECT * FROM (
                SELECT id_mau,id_nhom,nguon||'|'||thanh_phan||'|'||nhan_goc||'|'||ngon_ngu_goi_y tang,
                row_number() OVER (PARTITION BY nguon,thanh_phan,nhan_goc,ngon_ngu_goi_y,id_nhom ORDER BY ngau_nhien(id_mau)) thu_tu
                FROM mau WHERE trang_thai NOT IN ('excluded','duplicate') AND id_nhom!='') WHERE thu_tu=1''')
            conn.execute('''INSERT INTO lay_mau SELECT id_mau,tang,dan_so,min(dan_so,?),1.0*min(dan_so,?)/dan_so
                FROM (SELECT *,count(*) OVER (PARTITION BY tang) dan_so,
                row_number() OVER (PARTITION BY tang ORDER BY ngau_nhien(id_nhom)) chon FROM ung_vien)
                WHERE chon<=?''',(config['review_sample_per_stratum'],)*3)
        return {'sample_rows':conn.execute('SELECT count(*) FROM lay_mau').fetchone()[0],
                'strata':conn.execute('SELECT count(DISTINCT tang) FROM lay_mau').fetchone()[0]}


def record_decision(root: Path, identifier: str, label: str, reviewer: str, competence: str, evidence: str) -> None:
    if not all(x.strip() for x in (reviewer,competence,evidence)):
        raise ValueError('Reviewer, original-language competence and evidence required')
    with connect(root) as conn:
        conn.execute('INSERT INTO quyet_dinh VALUES (?,?,?,?,?,?)',
            (identifier,reviewer,label,competence,evidence,datetime.now(timezone.utc).isoformat()))
