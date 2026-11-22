"""Targeted Phase 3.1 recovery over immutable Phase 1-3 caches.

Technical grouping clearance never supplies a class, human decision or license.
The source-level probability sample and its acceptance policy remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sqlite3
import unicodedata
import zlib
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .expansion import Groups, readonly
from .quality import DATASETS, digest, normalize
from vietmailguard.v21.dedup_audit import URL_RE, EMAIL_RE


def comparison_text(text: str) -> str:
    value=unicodedata.normalize('NFKC',normalize(text)).casefold()
    return ' '.join(EMAIL_RE.sub(' <email> ',URL_RE.sub(' <url> ',value)).split())


def full_signature(text: str, config: dict) -> bytes:
    """Same seeded MinHash as Phase 3, with bounded chunks and no length cap."""
    n=config['minhash_shingle_characters']
    if len(text)<n:return b''
    rng=np.random.default_rng(config['seed']);size=config['minhash_permutations']
    a=rng.integers(1,2**31,size,dtype=np.uint64)
    b=rng.integers(0,2**32,size,dtype=np.uint64)
    result=np.full(size,4294967291,dtype=np.uint64)
    # Positions, not disjoint text chunks: all boundary shingles are retained.
    for start in range(0,len(text)-n+1,4096):
        stop=min(start+4096,len(text)-n+1)
        values=np.fromiter((zlib.crc32(text[i:i+n].encode('utf-8')) for i in range(start,stop)),dtype=np.uint64)
        for index,(x,y) in enumerate(zip(a,b)):
            result[index]=min(result[index],((values*x+y)%4294967291).min())
    return result.astype('<u4').tobytes()


def exact_jaccard(a: str,b: str,n: int=5) -> float:
    left={a[i:i+n] for i in range(max(0,len(a)-n+1))}
    right={b[i:i+n] for i in range(max(0,len(b)-n+1))}
    return len(left&right)/len(left|right) if left or right else float(a==b)


def short_pairs(texts: dict[int,str],threshold: float):
    """Exact prefix-filtered Jaccard join within the specified short cohort.

    Global token-frequency order and the standard |S|-ceil(t|S|)+1 prefix
    retain every qualifying pair. No bucket cap or sampled candidate removal.
    Generic texts below 20 characters require human context and are not joined.
    """
    features={i:{s[k:k+3] for k in range(len(s)-2)} for i,s in texts.items() if len(s)>=20}
    yield from set_join(features,threshold)


def set_join(features: dict[int,set[str]],threshold: float):
    """Exact Jaccard prefix join; every qualifying pair in this block is kept."""
    import math
    counts=Counter(token for items in features.values() for token in items)
    postings=defaultdict(list)
    for identifier,items in sorted(features.items(),key=lambda row:(len(row[1]),row[0])):
        ordered=sorted(items,key=lambda token:(counts[token],token))
        prefix=ordered[:len(items)-math.ceil(threshold*len(items))+1]
        candidates=set(other for token in prefix for other in postings[token])
        for other in sorted(candidates):
            right=features[other]
            if len(right)<threshold*len(items):continue
            score=len(items&right)/len(items|right)
            if score>=threshold:yield other,identifier,score
        for token in prefix:postings[token].append(identifier)


def connect(root: Path) -> sqlite3.Connection:
    path=root/'data/v22/phase31/recovery.sqlite3';path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path.resolve().as_uri(),uri=True)
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA cache_size=-65536')
    conn.execute('PRAGMA foreign_keys=ON');conn.execute('PRAGMA temp_store=FILE')
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS cau_hinh(ma TEXT PRIMARY KEY,gia_tri TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS yeu_cau(ma_bam TEXT PRIMARY KEY,id_mau TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS van_ban(ma_bam TEXT PRIMARY KEY REFERENCES yeu_cau,
        noi_dung_nen BLOB NOT NULL,chu_ky BLOB NOT NULL,so_ky_tu INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS tien_do(duong_dan TEXT PRIMARY KEY,ma_bam TEXT NOT NULL,so_mau INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS cap_kiem_dinh(id_truoc INTEGER,id_sau INTEGER,tang TEXT,
        PRIMARY KEY(id_truoc,id_sau,tang));
    CREATE TABLE IF NOT EXISTS ket_qua_cap(id_truoc INTEGER,id_sau INTEGER,tang TEXT,
        jaccard REAL,truy_hoi INTEGER,cach_ly INTEGER,PRIMARY KEY(id_truoc,id_sau,tang));
    CREATE TABLE IF NOT EXISTS canh_bo_sung(id_truoc INTEGER,id_sau INTEGER,loai TEXT,tuong_dong REAL,
        PRIMARY KEY(id_truoc,id_sau,loai));
    CREATE TABLE IF NOT EXISTS phuc_hoi(id_van_ban INTEGER PRIMARY KEY,ma_bam TEXT,id_nhom TEXT,
        cach_ly_truoc INTEGER,ma_ly_do INTEGER,trang_thai TEXT NOT NULL,
        nhan_duoc_duyet TEXT NOT NULL DEFAULT '');
    CREATE TABLE IF NOT EXISTS khoi_kiem_dinh(loai TEXT,ma_khoi TEXT,so_mau INTEGER,trang_thai TEXT,
        PRIMARY KEY(loai,ma_khoi));
    CREATE TABLE IF NOT EXISTS thanh_vien_khoi(loai TEXT,ma_khoi TEXT,id_van_ban INTEGER,
        PRIMARY KEY(loai,ma_khoi,id_van_ban));
    ''')
    for alias,relative in [('ba','data/v22/phase3/expansion.sqlite3'),('hai','data/v22/cache/quality.sqlite3')]:
        conn.execute(f'ATTACH DATABASE ? AS {alias}',((root/relative).resolve().as_uri()+'?mode=ro',))
    conn.create_function('chon',1,lambda key:hashlib.sha256(('phase31_seed42|'+str(key)).encode()).hexdigest(),deterministic=True)
    return conn


def prepare(root: Path,config: dict) -> None:
    """Lock existing samples/inputs, then select only recovery and QA content."""
    with connect(root) as conn:
        lock=json.dumps({'phase3':digest(root/'data/v22/phase3/expansion.sqlite3'),
            'phase2':digest(root/'data/v22/cache/quality.sqlite3'),
            'queue':digest(root/'data/v22/phase3/review_queue.csv'),
            'policy':digest(root/'config/v22/expansion.json')},sort_keys=True)
        prior=conn.execute("SELECT gia_tri FROM cau_hinh WHERE ma='input_lock'").fetchone()
        if prior:
            if prior[0]!=lock:raise ValueError('Recovery inputs or locked review sample changed')
            return
        with conn:
            conn.execute("INSERT INTO cau_hinh VALUES ('input_lock',?)",(lock,))
            # Original sample stays exactly 7,585 / 38; this is technical QA,
            # never a surrogate human label approval.
            conn.execute('''INSERT OR IGNORE INTO yeu_cau SELECT m.ma_bam,m.id_mau FROM ba.lay_mau s
                JOIN ba.mau m USING(id_mau) WHERE m.ma_bam!='' ''')
            conn.execute('''INSERT OR IGNORE INTO yeu_cau SELECT m.ma_bam,m.id_mau FROM ba.mau m
                JOIN ba.van_ban v USING(ma_bam) WHERE v.ly_do!='' AND m.id_mau=m.id_dai_dien''')
            conn.execute('''INSERT INTO cap_kiem_dinh SELECT v.id_van_ban,w.id_van_ban,'legacy_prefix_sample'
                FROM hai.gan_trung g JOIN ba.van_ban v ON v.ma_bam=g.ma_truoc
                JOIN ba.van_ban w ON w.ma_bam=g.ma_sau ORDER BY chon(g.ma_truoc||g.ma_sau) LIMIT 1000''')
            controls=[row[0] for row in conn.execute('SELECT id_van_ban FROM ba.van_ban ORDER BY chon(ma_bam) LIMIT 2000')]
            conn.executemany("INSERT INTO cap_kiem_dinh VALUES (?,?,'random_pair_control')",zip(controls[::2],controls[1::2]))
            conn.execute('''INSERT OR IGNORE INTO yeu_cau SELECT m.ma_bam,m.id_mau FROM ba.mau m WHERE
                m.id_mau=m.id_dai_dien AND m.ma_bam IN (SELECT v.ma_bam FROM ba.van_ban v WHERE v.id_van_ban IN
                (SELECT id_truoc FROM cap_kiem_dinh UNION SELECT id_sau FROM cap_kiem_dinh))''')


def prepare_buckets(root: Path) -> None:
    """A fixed budget of complete small overflow blocks, not sampled members."""
    with connect(root) as conn:
        if conn.execute("SELECT 1 FROM cau_hinh WHERE ma='bucket_plan'").fetchone():return
        with conn:
            for kind,query in [('legacy', '''SELECT vi_tri,gia_tri,count(*) FROM hai.dai
                    GROUP BY 1,2 HAVING count(*) BETWEEN 101 AND 112 ORDER BY chon(vi_tri||gia_tri) LIMIT 8'''),
                ('minhash', '''SELECT vi_tri,ma_dai,count(*) FROM ba.dai
                    GROUP BY 1,2 HAVING count(*) BETWEEN 65 AND 96 ORDER BY chon(vi_tri||hex(ma_dai)) LIMIT 8''')]:
                for band,value,size in conn.execute(query).fetchall():
                    key=str(band)+':'+(value.hex() if isinstance(value,bytes) else value)
                    conn.execute('INSERT INTO khoi_kiem_dinh VALUES (?,?,?,?)',(kind,key,size,'pending'))
                    if kind=='legacy':
                        members=conn.execute('''SELECT v.id_van_ban FROM hai.dai d JOIN ba.van_ban v USING(ma_bam)
                            WHERE d.vi_tri=? AND d.gia_tri=?''',(band,value)).fetchall()
                    else:members=conn.execute('SELECT id_van_ban FROM ba.dai WHERE vi_tri=? AND ma_dai=?',(band,value)).fetchall()
                    conn.executemany('INSERT INTO thanh_vien_khoi VALUES (?,?,?)',((kind,key,i) for (i,) in members))
            conn.execute('''INSERT OR IGNORE INTO yeu_cau SELECT m.ma_bam,m.id_mau FROM ba.mau m
                WHERE m.id_mau=m.id_dai_dien AND m.ma_bam IN (SELECT v.ma_bam FROM ba.van_ban v
                WHERE v.id_van_ban IN (SELECT id_van_ban FROM thanh_vien_khoi))''')
            conn.execute("INSERT INTO cau_hinh VALUES ('bucket_plan','fixed_seed42_complete_blocks_v1')")


def load_selected_texts(root: Path,config: dict) -> dict:
    """One pointer-lookup pass, not a rerun of the 1.1M-row data audit."""
    csv.field_size_limit(100_000_000)
    with connect(root) as conn:
        needed={identifier:hashed for hashed,identifier in conn.execute('''SELECT y.ma_bam,y.id_mau FROM yeu_cau y
            LEFT JOIN van_ban v USING(ma_bam) WHERE v.ma_bam IS NULL''')}
        for source,relative in DATASETS:
            for part in sorted((root/'data/v21/processed'/relative).glob('part-*.csv')):
                if part.is_symlink():raise ValueError('Unsafe input symlink')
                key=part.relative_to(root).as_posix()
                expected=conn.execute('SELECT ma_bam FROM ba.tien_do WHERE duong_dan=?',(key,)).fetchone()
                sha=digest(part)
                if not expected or sha!=expected[0]:raise ValueError('Part differs from Phase 3')
                prior=conn.execute('SELECT ma_bam,so_mau FROM tien_do WHERE duong_dan=?',(key,)).fetchone()
                if prior and prior[0]!=sha:raise ValueError('Changed recovery checkpoint')
                if not needed:break
                selected=0
                with conn,part.open(encoding='utf-8-sig',newline='') as handle:
                    for row in csv.DictReader(handle):
                        hashed=needed.pop(source+'|'+row['id'],None)
                        if not hashed:continue
                        text=comparison_text(row.get('subject','')+'\n'+row.get('body',''))
                        signature=full_signature(text,config)
                        conn.execute('INSERT OR IGNORE INTO van_ban VALUES (?,?,?,?)',
                            (hashed,zlib.compress(text.encode('utf-8')),signature,len(text)))
                        selected+=1
                    if digest(part)!=sha:raise ValueError('Source changed during lookup')
                    conn.execute('INSERT OR REPLACE INTO tien_do VALUES (?,?,?)',(key,sha,selected+(prior[1] if prior else 0)))
                if selected:print(json.dumps({'selected_source':source,'part':part.name,'selected':selected}),flush=True)
        want=conn.execute('SELECT count(*) FROM yeu_cau').fetchone()[0]
        got=conn.execute('SELECT count(*) FROM van_ban').fetchone()[0]
        if want!=got:raise ValueError('Selected content lookup incomplete')
        return {'requested_content':want,'loaded_content':got}


def recover(root: Path,config: dict) -> dict:
    """Resolve grouping causes; leave uncertain blocks and all labels pending."""
    with connect(root) as conn:
        n=conn.execute('SELECT max(id_van_ban) FROM ba.van_ban').fetchone()[0] or 0
        groups=Groups(n);mask=np.zeros(n+1,dtype=np.uint16)
        legacy_counts=np.zeros(n+1,dtype=np.int32);minhash_counts=np.zeros(n+1,dtype=np.int32)
        previous=np.zeros(n+1,dtype=np.bool_)
        ids={hashed:identifier for identifier,hashed in conn.execute('SELECT id_van_ban,ma_bam FROM ba.van_ban')}
        known_groups={}
        for identifier,group,quarantined in conn.execute('SELECT id_van_ban,id_nhom,can_cach_ly FROM ba.nhom'):
            groups.union(known_groups.setdefault(group,identifier),identifier);previous[identifier]=quarantined
        # Causal masks are independent of source labels: 1 legacy overflow,
        # 2 MinHash overflow, 4 unverified old prefix edges, 8 protected overlap,
        # 16 candidate-label conflict, 32 short context, 64 MIME/decode quality.
        with conn:
            conn.execute('DELETE FROM phuc_hoi');conn.execute('DELETE FROM canh_bo_sung');conn.execute('DELETE FROM ket_qua_cap')
            legacy=conn.execute('''SELECT d.ma_bam FROM hai.dai d JOIN
                (SELECT vi_tri,gia_tri FROM hai.dai GROUP BY 1,2 HAVING count(*)>100) b
                ON d.vi_tri=b.vi_tri AND d.gia_tri=b.gia_tri''')
            for (hashed,) in legacy:legacy_counts[ids[hashed]]+=1
            overflow=conn.execute('''SELECT d.id_van_ban FROM ba.dai d JOIN
                (SELECT vi_tri,ma_dai FROM ba.dai GROUP BY 1,2 HAVING count(*)>?) b
                ON d.vi_tri=b.vi_tri AND d.ma_dai=b.ma_dai''',(config['pairwise_bucket_limit'],))
            for (identifier,) in overflow:minhash_counts[identifier]+=1
            for hashed,reason in conn.execute('SELECT ma_bam,ly_do FROM ba.mau'):
                if hashed not in ids:continue
                if 'old_reference_overlap_guard' in reason:mask[ids[hashed]]|=8
                if 'candidate_label_conflict' in reason:mask[ids[hashed]]|=16
                if any(flag in reason for flag in ('decode_error','decode_warning','replacement_character','missing_content','missing_body','mime_defects')):
                    mask[ids[hashed]]|=64
            # Cached text only: bounded LRU, no logging or external API.
            @lru_cache(maxsize=8)
            def text_for(identifier):
                row=conn.execute('''SELECT v.noi_dung_nen FROM van_ban v JOIN ba.van_ban b USING(ma_bam)
                    WHERE b.id_van_ban=?''',(identifier,)).fetchone()
                return zlib.decompress(row[0]).decode('utf-8') if row else None
            @lru_cache(maxsize=256)
            def signature_for(identifier):
                row=conn.execute('''SELECT coalesce(v.chu_ky,b.chu_ky),b.so_ky_tu FROM ba.van_ban b
                    LEFT JOIN van_ban v USING(ma_bam) WHERE b.id_van_ban=?''',(identifier,)).fetchone()
                return row
            blocks_checked=blocks_pending=block_edges=0
            for kind,key,size in conn.execute('SELECT loai,ma_khoi,so_mau FROM khoi_kiem_dinh').fetchall():
                members=[row[0] for row in conn.execute('SELECT id_van_ban FROM thanh_vien_khoi WHERE loai=? AND ma_khoi=?',(kind,key))]
                texts={identifier:text_for(identifier) for identifier in members}
                if len(texts)!=size or any(value is None or len(value)>50000 for value in texts.values()):
                    blocks_pending+=1;continue
                features={identifier:{value[i:i+5] for i in range(len(value)-4)} for identifier,value in texts.items()}
                for left,right,score in set_join(features,config['minhash_similarity']):
                    groups.union(left,right);block_edges+=1
                    conn.execute('INSERT OR IGNORE INTO canh_bo_sung VALUES (?,?,?,?)',(min(left,right),max(left,right),'complete_bucket_exact_5gram',score))
                target=legacy_counts if kind=='legacy' else minhash_counts
                for identifier in members:target[identifier]-=1
                conn.execute("UPDATE khoi_kiem_dinh SET trang_thai='complete_lexical_comparison' WHERE loai=? AND ma_khoi=?",(kind,key))
                blocks_checked+=1
            mask[legacy_counts>0]|=1;mask[minhash_counts>0]|=2
            checked=positive=0
            # Old-prefix edges outside the text cache remain unresolved. No
            # sample decision is generalized to other edges or entire sources.
            for left,right,ls,rs,nl,nr in conn.execute('''SELECT v.id_van_ban,w.id_van_ban,v.chu_ky,w.chu_ky,v.so_ky_tu,w.so_ky_tu
                FROM hai.gan_trung g JOIN ba.van_ban v ON v.ma_bam=g.ma_truoc JOIN ba.van_ban w ON w.ma_bam=g.ma_sau'''):
                matched=bool(ls and rs and min(nl,nr)/max(nl,nr)>=.75 and
                    np.mean(np.frombuffer(ls,dtype='<u4')==np.frombuffer(rs,dtype='<u4'))>=config['minhash_similarity'])
                if matched:continue
                lt,rt=text_for(left),text_for(right)
                if lt is not None and rt is not None and max(len(lt),len(rt))<=50000:
                    score=exact_jaccard(lt,rt);checked+=1
                    if score>=config['minhash_similarity']:
                        groups.union(left,right);positive+=1
                        conn.execute('INSERT OR IGNORE INTO canh_bo_sung VALUES (?,?,?,?)',(left,right,'verified_lexical_5gram',score))
                else:mask[left]|=4;mask[right]|=4
            # Complete the missing signatures against the existing global LSH
            # index. Saturated neighborhoods remain isolated, not silently cut.
            extras=conn.execute('''SELECT b.id_van_ban,b.ma_bam,v.chu_ky,v.so_ky_tu FROM ba.van_ban b
                JOIN van_ban v USING(ma_bam) WHERE b.ly_do!='' ''').fetchall()
            conn.execute('CREATE TEMP TABLE dai_moi(vi_tri INTEGER,ma_dai BLOB,id_van_ban INTEGER,PRIMARY KEY(vi_tri,ma_dai,id_van_ban)) WITHOUT ROWID')
            for identifier,hashed,sig,length in extras:
                if length<20:mask[identifier]|=32;continue
                if sig:
                    conn.executemany('INSERT INTO dai_moi VALUES (?,?,?)',[(j//16,sig[j:j+16],identifier) for j in range(0,len(sig),16)])
            supplemental_matches=0
            for identifier,hashed,sig,length in extras:
                if length<20 or not sig:continue
                candidates=set()
                for j in range(0,len(sig),16):
                    rows=conn.execute('''SELECT id_van_ban FROM ba.dai WHERE vi_tri=? AND ma_dai=? UNION
                        SELECT id_van_ban FROM dai_moi WHERE vi_tri=? AND ma_dai=?''',(j//16,sig[j:j+16],j//16,sig[j:j+16])).fetchall()
                    if len(rows)>config['pairwise_bucket_limit']:
                        for (other,) in rows:mask[other]|=2
                        mask[identifier]|=2
                    else:candidates.update(other for (other,) in rows if other!=identifier)
                for other in candidates:
                    other_sig,other_length=signature_for(other)
                    if not other_sig or min(length,other_length)/max(length,other_length)<.75:continue
                    score=float(np.mean(np.frombuffer(sig,dtype='<u4')==np.frombuffer(other_sig,dtype='<u4')))
                    if score>=config['minhash_similarity']:
                        groups.union(identifier,other);supplemental_matches+=1
                        conn.execute('INSERT OR IGNORE INTO canh_bo_sung VALUES (?,?,?,?)',
                            (min(identifier,other),max(identifier,other),'full_signature_recovery',score))
            short={identifier:text_for(identifier) for identifier,hashed,sig,length in extras if 20<=length<80}
            short_edges=0
            for left,right,score in short_pairs(short,config['minhash_similarity']):
                groups.union(left,right);short_edges+=1
                conn.execute('INSERT OR IGNORE INTO canh_bo_sung VALUES (?,?,?,?)',(left,right,'short_exact_3gram',score))
            # Real-corpus lexical retrieval check, separate cohorts; this is NOT
            # a human semantic-duplicate benchmark or a classifier evaluation.
            for left,right,stratum in conn.execute('SELECT * FROM cap_kiem_dinh'):
                lt,rt=text_for(left),text_for(right)
                if lt is None or rt is None or max(len(lt),len(rt))>50000:continue
                score=exact_jaccard(lt,rt)
                ls,nl=signature_for(left);rs,nr=signature_for(right)
                collision=bool(ls and rs and any(ls[i:i+16]==rs[i:i+16] for i in range(0,len(ls),16)))
                retrieved=bool(collision and min(nl,nr)/max(nl,nr)>=.75 and
                    np.mean(np.frombuffer(ls,dtype='<u4')==np.frombuffer(rs,dtype='<u4'))>=config['minhash_similarity'])
                conn.execute('INSERT INTO ket_qua_cap VALUES (?,?,?,?,?,?)',(left,right,stratum,score,int(retrieved),int(previous[left] or previous[right])))
            roots=np.array([groups.find(i) for i in range(n+1)],dtype=np.int32)
            combined=np.zeros(n+1,dtype=np.uint16)
            for i in range(1,n+1):combined[roots[i]]|=mask[i]
            for hashed,identifier in ids.items():
                reason=int(combined[roots[identifier]])
                state='technical_recovered_label_pending' if previous[identifier] and reason==0 else 'quarantined' if reason else 'technical_clear_label_pending'
                conn.execute('INSERT INTO phuc_hoi VALUES (?,?,?,?,?,?,?)',
                    (identifier,hashed,'v22_recovery_'+str(roots[identifier]),int(previous[identifier]),reason,state,''))
            stats={'exact_prefix_edges_checked':checked,'exact_prefix_positive_edges':positive,
                'overflow_blocks_completely_checked':blocks_checked,'overflow_blocks_still_pending':blocks_pending,
                'overflow_exact_positive_edges':block_edges,
                'supplemental_signature_matches':supplemental_matches,'short_exact_edges':short_edges,
                'missing_signatures_completed':sum(bool(sig) for _,_,sig,_ in extras),
                'long_signatures_completed':sum(bool(sig) and length>50000 for _,_,sig,length in extras),
                'short_signatures_completed':sum(bool(sig) and length<80 for _,_,sig,length in extras),
                'status_counts':dict(conn.execute('SELECT trang_thai,count(*) FROM phuc_hoi GROUP BY trang_thai'))}
            conn.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('recovery_complete',?)",(json.dumps(stats),))
        return stats
