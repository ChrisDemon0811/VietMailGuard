"""Source-only downloads and streaming quarantine of acquired real emails.

No fetched source label is silently promoted to project ground truth. Raw
archives are never extracted or executed; remote images/URLs remain untouched.
"""
from __future__ import annotations
import csv
import hashlib
import json
import mailbox
import sqlite3
import tarfile
import time
import urllib.request
import numpy as np
from pathlib import Path, PurePosixPath

from vietmailguard.eml_parser import parse_eml_bytes
from vietmailguard.v21.dedup_audit import dedup_hashes
from .quality import digest,language_signal,normalize


def connect(root: Path):
    path=root/'data/v22/phase32/acquisition.sqlite3';path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path.resolve().as_uri(),uri=True)
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA cache_size=-65536')
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS tep(duong_dan TEXT PRIMARY KEY,nguon TEXT,url TEXT,ma_bam TEXT,
        so_byte INTEGER,giay_phep TEXT,phien_ban TEXT,trang_thai TEXT);
    CREATE TABLE IF NOT EXISTS tien_do(duong_dan TEXT PRIMARY KEY,ma_bam TEXT,so_dong INTEGER);
    CREATE TABLE IF NOT EXISTS mau(id_mau TEXT PRIMARY KEY,nguon TEXT,duong_dan_goc TEXT,nhan_goc TEXT,
        nhan_ung_vien TEXT,ngon_ngu_goi_y TEXT,ma_bam_chuan TEXT,ma_bam_mau TEXT,
        trang_thai TEXT,ly_do TEXT,du_lieu_json TEXT);
    CREATE INDEX IF NOT EXISTS chi_muc_bam ON mau(ma_bam_chuan);
    CREATE TABLE IF NOT EXISTS chu_ky_moi(id_mau TEXT PRIMARY KEY,chu_ky BLOB,so_ky_tu INTEGER);
    CREATE TABLE IF NOT EXISTS dai_moi(vi_tri INTEGER,ma_dai BLOB,id_mau TEXT,PRIMARY KEY(vi_tri,ma_dai,id_mau)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS gan_trung(id_mau TEXT,id_lien_quan TEXT,loai TEXT,tuong_dong REAL,
        PRIMARY KEY(id_mau,id_lien_quan,loai));
    ''')
    return conn


def fetch(url: str,path: Path,limit: int) -> dict:
    """Bounded HTTPS transfer to .part, atomic publication, no automatic retry."""
    if not url.startswith(('https://monkey.org/~jose/phishing/','https://codeload.github.com/rf-peixoto/phishing_pot/')):
        raise ValueError('URL is outside reviewed source allowlist')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.part')
    req=urllib.request.Request(url,headers={'User-Agent':'VietMailGuard-local-research-dataset-acquisition'})
    size=0;hashed=hashlib.sha256()
    with urllib.request.urlopen(req,timeout=90) as response,temporary.open('wb') as out:
        final=response.geturl()
        if not final.startswith(('https://monkey.org/~jose/phishing/','https://codeload.github.com/rf-peixoto/phishing_pot/')):
            raise ValueError('Unexpected dataset redirect')
        declared=response.headers.get('Content-Length')
        if declared and int(declared)>limit:raise ValueError('Download exceeds configured budget')
        while chunk:=response.read(1024*1024):
            size+=len(chunk)
            if size>limit:raise ValueError('Download exceeds configured budget')
            hashed.update(chunk);out.write(chunk)
        if declared and size!=int(declared):raise ValueError('Incomplete dataset transfer')
    temporary.replace(path)
    return {'sha256':hashed.hexdigest(),'size_bytes':size}


def download_sources(root: Path,config: dict) -> None:
    with connect(root) as conn:
        for source in config['sources']:
            for filename in source['files']:
                name=filename+('.tar.gz' if source['format']=='tar_eml' else '.mbox')
                path=root/'data/v22/phase32/raw'/source['source']/name
                relative=path.relative_to(root).as_posix();url=source['base_url']+filename
                old=conn.execute('SELECT ma_bam,so_byte FROM tep WHERE duong_dan=?',(relative,)).fetchone()
                if old:
                    if not path.is_file() or digest(path)!=old[0]:raise ValueError('Downloaded archive changed')
                    continue
                try:
                    info=fetch(url,path,config['max_download_bytes'])
                    with conn:conn.execute('INSERT INTO tep VALUES (?,?,?,?,?,?,?,?)',
                        (relative,source['source'],url,info['sha256'],info['size_bytes'],source['license'],source.get('revision',filename),'downloaded'))
                    print(json.dumps({'downloaded_source':source['source'],'artifact':name,**info}),flush=True)
                except Exception as exc:
                    # No response body, URL query, environment, email or secret.
                    print(json.dumps({'download_failed_source':source['source'],'artifact':name,'error_type':type(exc).__name__,
                        'http_status':getattr(exc,'code',None)}),flush=True)


def messages(path: Path,format_: str,limit: int):
    if format_=='mbox':
        box=mailbox.mbox(path,create=False)
        try:
            for key in box.iterkeys():
                with box.get_file(key) as f:data=f.read(limit+1)
                yield str(key),data if len(data)<=limit else None
        finally:box.close()
    else:
        with tarfile.open(path,'r|gz') as archive:
            for entry in archive:
                parts=PurePosixPath(entry.name).parts
                if (not entry.isfile() or entry.issym() or entry.islnk() or '..' in parts
                    or len(parts)<3 or parts[1]!='email' or not entry.name.endswith('.eml')):continue
                if entry.size>limit:yield '/'.join(parts[1:]),None;continue
                stream=archive.extractfile(entry)
                if stream is not None:
                    with stream:data=stream.read(limit+1)
                    yield '/'.join(parts[1:]),data if len(data)<=limit else None


def ingest_sources(root: Path,config: dict) -> None:
    sources={x['source']:x for x in config['sources']}
    with connect(root) as conn:
        reference=root/'data/v21/cache/dedup_audit.sqlite3'
        conn.execute('ATTACH DATABASE ? AS cu',(reference.resolve().as_uri()+'?mode=ro',))
        for relative,source,sha in conn.execute('SELECT duong_dan,nguon,ma_bam FROM tep').fetchall():
            path=root/relative;settings=sources[source]
            if path.is_symlink() or digest(path)!=sha:raise ValueError('Raw archive integrity failed')
            old=conn.execute('SELECT ma_bam FROM tien_do WHERE duong_dan=?',(relative,)).fetchone()
            if old:
                if old[0]!=sha:raise ValueError('Acquisition checkpoint changed')
                continue
            count=0
            with conn:
                for original,data in messages(path,settings['format'],config['max_email_bytes']):
                    count+=1;identifier=source+':'+sha[:16]+':'+original
                    status='weak_label';reason='source_label_pending_project_approval';email={};normalized=template='';signal='unknown'
                    if data is None:status='excluded';reason='email_size_budget_exceeded'
                    else:
                        try:
                            email=parse_eml_bytes(data)
                        except Exception:
                            status='excluded';reason='mime_parse_failed'
                        else:
                            exact,normalized,template=dedup_hashes(email.get('subject',''),email.get('body',''))
                            signal=language_signal(normalize(email.get('subject','')+' '+email.get('body','')))
                            if not normalized:status='excluded';reason='empty_email'
                            elif "DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA" in email.get('body',''):
                                status='excluded';reason='mbox_administrative_record'
                            elif conn.execute('SELECT 1 FROM mau WHERE ma_bam_chuan=?',(normalized,)).fetchone():
                                status='duplicate';reason='normalized_duplicate_in_acquisition'
                            elif conn.execute('SELECT 1 FROM cu.rows WHERE normalized_hash=? LIMIT 1',(normalized,)).fetchone():
                                status='duplicate';reason='existing_pre_split_corpus_fingerprint_overlap'
                            elif template and conn.execute('SELECT 1 FROM cu.rows WHERE template_hash=? LIMIT 1',(template,)).fetchone():
                                status='review';reason='existing_template_overlap'
                            elif '\ufffd' in email.get('body','') or signal not in ('en','vi','zh'):
                                status='review';reason='decode_or_language_verification_pending'
                    record={**email,'id':identifier,'source':source,'source_version':settings.get('revision',relative),
                        'original_path':relative+'#'+original,'raw_label':settings['raw_label'],
                        'candidate_label':settings['candidate_label'],'label':'','label_status':status,
                        'label_provenance':settings['label_provenance'],'language':signal,'data_origin':'original',
                        'content_hash':normalized,'template_hash':template,'parent_id':'','final_group_id':'',
                        'training_eligible':False,'split':'unassigned','license':settings['license'],
                        'attribution':settings['attribution'],'approval_blocker':reason+';near_duplicate_and_source_quality_review_pending'}
                    conn.execute('INSERT INTO mau VALUES (?,?,?,?,?,?,?,?,?,?,?)',(identifier,source,record['original_path'],
                        settings['raw_label'],settings['candidate_label'],signal,normalized,template,status,reason,json.dumps(record,ensure_ascii=False)))
                if digest(path)!=sha:raise ValueError('Raw archive changed during ingestion')
                conn.execute('INSERT INTO tien_do VALUES (?,?,?)',(relative,sha,count))
            print(json.dumps({'parsed_source':source,'artifact':path.name,'records':count}),flush=True)


def report(root: Path) -> dict:
    output=root/'data/v22/phase32/weak_labeled';output.mkdir(parents=True,exist_ok=True)
    with connect(root) as conn:
        result={'downloaded_files':conn.execute('SELECT count(*) FROM tep').fetchone()[0],
            'downloaded_bytes':conn.execute('SELECT coalesce(sum(so_byte),0) FROM tep').fetchone()[0],
            'rows_received':conn.execute('SELECT count(*) FROM mau').fetchone()[0],
            'status_counts':dict(conn.execute('SELECT trang_thai,count(*) FROM mau GROUP BY trang_thai')),
            'language_signal_counts':dict(conn.execute('SELECT ngon_ngu_goi_y,count(*) FROM mau GROUP BY ngon_ngu_goi_y')),
            'sources':[list(row) for row in conn.execute('SELECT nguon,trang_thai,count(*) FROM mau GROUP BY 1,2')],
            'training_approved_new':0,'pseudo_labeled_rows':0,'validation_or_test_rows':0}
        result['near_checked_rows']=conn.execute('SELECT count(*) FROM chu_ky_moi').fetchone()[0]
        result['near_matched_rows']=conn.execute('SELECT count(DISTINCT id_mau) FROM gan_trung').fetchone()[0]
        result['near_overflow_rows']=conn.execute("SELECT count(*) FROM mau WHERE ly_do LIKE '%near_bucket_overflow%'").fetchone()[0]
        fields=None;writer=None;handle=None;written=set()
        try:
            for index,(payload,) in enumerate(conn.execute("SELECT du_lieu_json FROM mau WHERE trang_thai='weak_label' ORDER BY id_mau")):
                row=json.loads(payload)
                if index%5000==0:
                    if handle:handle.close()
                    part=output/f'part-{index//5000+1:05d}.csv';written.add(part.name)
                    handle=part.open('w',encoding='utf-8',newline='')
                    fields=list(row);writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                writer.writerow(row)
        finally:
            if handle:handle.close()
        # Remove only obsolete chunks produced by this exporter in its own
        # derived directory; never raw data, arbitrary paths or symlink targets.
        for part in output.glob('part-*.csv'):
            if part.name not in written:
                if part.is_symlink() or part.resolve().parent!=output.resolve():raise ValueError('Unsafe stale export path')
                part.unlink()
        path=root/'reports/v22/phase32_acquisition_summary.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result


def screen_near(root: Path) -> None:
    """Reuse the frozen MinHash implementation/index on newly acquired rows."""
    from .expansion import minhash
    settings=json.loads((root/'config/v22/expansion.json').read_text(encoding='utf-8'))
    with connect(root) as conn:
        conn.execute('ATTACH DATABASE ? AS ba',((root/'data/v22/phase3/expansion.sqlite3').resolve().as_uri()+'?mode=ro',))
        pending=conn.execute('''SELECT id_mau FROM mau WHERE trang_thai NOT IN ('excluded','duplicate')
            AND id_mau NOT IN (SELECT id_mau FROM chu_ky_moi) ORDER BY id_mau''').fetchall()
        for start in range(0,len(pending),250):
            with conn:
                for (identifier,) in pending[start:start+250]:
                    payload,status,reason=conn.execute('SELECT du_lieu_json,trang_thai,ly_do FROM mau WHERE id_mau=?',(identifier,)).fetchone()
                    row=json.loads(payload);sig,skipped,length=minhash(row.get('subject','')+'\n'+row.get('body',''),settings)
                    conn.execute('INSERT INTO chu_ky_moi VALUES (?,?,?)',(identifier,sig,length))
                    flags=set();matches={}
                    if not sig:flags.add('fuzzy_short_or_long_pending')
                    else:
                        for band in range(8):
                            value=sig[band*16:band*16+16]
                            old=conn.execute('''SELECT v.id_van_ban,v.chu_ky,v.so_ky_tu FROM ba.dai d
                                JOIN ba.van_ban v USING(id_van_ban) WHERE d.vi_tri=? AND d.ma_dai=? LIMIT 65''',(band,value)).fetchall()
                            new=conn.execute('''SELECT s.id_mau,s.chu_ky,s.so_ky_tu FROM dai_moi d
                                JOIN chu_ky_moi s USING(id_mau) WHERE d.vi_tri=? AND d.ma_dai=? LIMIT 65''',(band,value)).fetchall()
                            for kind,items in [('preexisting',old),('acquired',new)]:
                                if len(items)>64:flags.add('near_bucket_overflow')
                                else:
                                    for other,blob,nchars in items:matches[(kind,str(other))]=(blob,nchars)
                        for (kind,other),(blob,nchars) in matches.items():
                            if not blob or min(length,nchars)/max(length,nchars)<.75:continue
                            score=float(np.mean(np.frombuffer(sig,dtype='<u4')==np.frombuffer(blob,dtype='<u4')))
                            if score>=settings['minhash_similarity']:
                                flags.add('near_overlap_review')
                                conn.execute('INSERT OR IGNORE INTO gan_trung VALUES (?,?,?,?)',(identifier,other,kind,score))
                        conn.executemany('INSERT INTO dai_moi VALUES (?,?,?)',[(band,sig[band*16:band*16+16],identifier) for band in range(8)])
                    if flags:
                        status='review';reason=reason+';'+','.join(sorted(flags));row['label_status']=status
                        row['approval_blocker']=row['approval_blocker']+';'+','.join(sorted(flags))
                        conn.execute('UPDATE mau SET trang_thai=?,ly_do=?,du_lieu_json=? WHERE id_mau=?',
                            (status,reason,json.dumps(row,ensure_ascii=False),identifier))
            print(json.dumps({'near_checked_new_rows':min(start+250,len(pending)),'total_new_rows':len(pending)}),flush=True)


def approve_bootstrap(root: Path,authorization: dict) -> dict:
    """Apply the owner's explicit local reuse decision, retaining original IDs."""
    if (authorization['authorization']!='owner_user_request_phase3_2' or authorization['publication_approved']
        or authorization['scope']!='local_development_only' or not authorization['preserve_inherited_splits']):
        raise ValueError('Missing the explicit restricted-local owner authorization')
    proposal=root/'data/v22/phase31/bootstrap_reuse_manifest.csv'
    proof=json.loads((root/'reports/v22/phase31_manifest.json').read_text(encoding='utf-8'))
    if digest(proposal)!=proof['sha256'][proposal.relative_to(root).as_posix()]:raise ValueError('Bootstrap evidence changed')
    with proposal.open(encoding='utf-8',newline='') as f:
        selected={row['id']:row for row in csv.DictReader(f) if row['local_use_evidence']=='cc_by_4_0_explicit'}
    if len(selected)!=authorization['expected_rows']:raise ValueError('Approved subset scope changed')
    for row in selected.values():
        if row['label']!='phishing' or row['split'] not in ('train','validation'):raise ValueError('Unexpected inherited label/split')
        if row['source']=='controlled_translation':
            parent=selected.get(row['parent_id'])
            if not parent or (parent['split'],parent['final_group_id'],parent['label'])!=(row['split'],row['final_group_id'],row['label']):
                raise ValueError('Translation parent/group/split mismatch')
    original=root/'data/v21/curated/bootstrap_v2_development'
    manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    if manifest['leakage']['passed'] is not True:raise ValueError('Inherited leakage guard missing')
    output=root/'data/v22/phase32/verified';output.mkdir(parents=True,exist_ok=True)
    result={'approved_reused_rows':0,'new_independent_rows':0,'by_language':{},'by_label':{},'split_rows':{},
        'authorization':authorization,'privacy_handling':'restricted local; not anonymized; no API/public export',
        'project_human_review_fabricated':False,'attribution':'Jose Nazario, Phishing Corpus, CC BY 4.0; controlled Vietnamese translations are adaptations',
        'source_url':'https://monkey.org/~jose/phishing/README.txt','output_sha256':{}}
    csv.field_size_limit(100_000_000)
    found=set()
    for split in ('train','validation'):
        path=original/(split+'.csv')
        if digest(path)!=manifest['output_sha256'][split]:raise ValueError('Frozen development data changed')
        target=output/('bootstrap_'+split+'.csv');temporary=target.with_suffix('.csv.part')
        count=0
        with path.open(encoding='utf-8-sig',newline='') as f,temporary.open('w',encoding='utf-8',newline='') as out:
            reader=csv.DictReader(f)
            fields=list(reader.fieldnames)+['license','attribution','local_approval_reference','privacy_handling']
            writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
            for row in reader:
                if row['id'] not in selected:continue
                proof_row=selected[row['id']]
                if any(row[key]!=proof_row[key] for key in ('source','label','label_provenance','parent_id','final_group_id','split','language','data_origin')):
                    raise ValueError('Approved metadata differs from source row')
                writer.writerow({**row,'license':'CC-BY-4.0','attribution':proof_row['attribution'],
                    'local_approval_reference':authorization['authorization'],'privacy_handling':result['privacy_handling']})
                found.add(row['id']);count+=1
                for field,group in [('language','by_language'),('label','by_label')]:result[group][row[field]]=result[group].get(row[field],0)+1
        temporary.replace(target);result['split_rows'][split]=count;result['output_sha256'][split]=digest(target)
        result['approved_reused_rows']+=count
    if found!=set(selected):raise ValueError('Approved subset incomplete')
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Empty pseudo-label partition is intentional: no classifier/LLM runs here.
    pseudo=root/'data/v22/phase32/pseudo_labeled';pseudo.mkdir(parents=True,exist_ok=True)
    (pseudo/'manifest.json').write_text(json.dumps({'rows':0,'status':'not_generated','validation_allowed':False,'test_allowed':False}),encoding='utf-8')
    return result
