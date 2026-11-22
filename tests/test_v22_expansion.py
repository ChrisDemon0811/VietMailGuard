"""Synthetic fixtures only; no held-out files, downloads or real email text."""
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from vietmailguard.v22 import expansion as e, quality as q
from test_v22_quality import fixture


def config():
    return json.loads((Path(__file__).parents[1]/'config/v22/expansion.json').read_text())


def test_full_text_signature_differs_after_shared_prefix_and_is_repeatable():
    c=config()
    prefix='A common synthetic header for an ordinary office message. '*15
    a=prefix+'Please review the financial spreadsheet and meeting agenda. '*30
    b=prefix+'A botanical garden displays orchids and ferns on the trail. '*30
    sig,why,_=e.minhash(a,c)
    assert not why and sig==e.minhash(a,c)[0]
    assert np.mean(np.frombuffer(sig,dtype='<u4')==np.frombuffer(e.minhash(b,c)[0],dtype='<u4'))<.8
    assert e.minhash('x'*50001,c)[1]=='long_text_full_comparison_pending'
    assert e.minhash('short',c)[1]=='short_text_fuzzy_unreliable'


def build(tmp_path,monkeypatch):
    c,part=fixture(tmp_path,monkeypatch)
    q.ingest(tmp_path,c);q.near_duplicates(tmp_path,c);q.summarize(tmp_path)
    # Expansion consumes the same fake standardized corpus, never real sources.
    monkeypatch.setattr(e,'DATASETS',q.DATASETS)
    ec=config()
    e.prepare(tmp_path,ec);e.index_bands(tmp_path,ec)
    return ec,part


def test_pipeline_resume_status_sampling_and_decisions(tmp_path,monkeypatch):
    c,part=build(tmp_path,monkeypatch)
    protected=q.digest(part)
    phase2=q.digest(tmp_path/'data/v22/cache/quality.sqlite3')
    first=e.group_candidates(tmp_path,c)
    assert first==e.group_candidates(tmp_path,c)
    assert e.classify(tmp_path,c)['status_counts']=={'excluded':1,'pending_review':2}
    assert e.sample_review(tmp_path,c)['sample_rows']==2 # different raw-label strata
    with e.connect(tmp_path) as conn:
        before=conn.execute('SELECT * FROM lay_mau ORDER BY id_mau').fetchall()
        assert len(set(row[0] for row in conn.execute('SELECT id_nhom FROM nhom')))==2
        assert first['legacy_edges_deferred']==1
    e.prepare(tmp_path,c);e.index_bands(tmp_path,c);e.sample_review(tmp_path,c)
    with e.connect(tmp_path) as conn:
        assert before==conn.execute('SELECT * FROM lay_mau ORDER BY id_mau').fetchall()
        tables=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert all(t.isascii() and t==t.lower() for t in tables)
        assert all(r[1].isascii() and r[1]==r[1].lower() for t in tables for r in conn.execute(f'PRAGMA table_info({t})'))
    with pytest.raises(ValueError):e.record_decision(tmp_path,'trec06c|a','normal','tester','','evidence')
    with pytest.raises(sqlite3.IntegrityError):e.record_decision(tmp_path,'unknown','normal','tester','competent','evidence')
    e.record_decision(tmp_path,'trec06c|a','normal','tester','competent','synthetic decision')
    with pytest.raises(sqlite3.IntegrityError):e.record_decision(tmp_path,'trec06c|a','normal','tester','competent','overwrite')
    with pytest.raises(ValueError):e.classify(tmp_path,c)
    with e.connect(tmp_path) as conn:
        assert conn.execute("SELECT count(*) FROM mau WHERE trang_thai='approved'").fetchone()[0]==0
    assert q.digest(part)==protected and q.digest(tmp_path/'data/v22/cache/quality.sqlite3')==phase2


def test_overflow_all_members_quarantined_without_forcing_same_group(tmp_path,monkeypatch):
    c,_=build(tmp_path,monkeypatch)
    c['pairwise_bucket_limit']=1
    stats=e.group_candidates(tmp_path,c)
    assert stats['large_minhash_buckets']>0
    with e.connect(tmp_path) as conn:
        assert conn.execute('SELECT count(DISTINCT id_nhom),min(can_cach_ly) FROM nhom').fetchone()==(2,1)
    assert e.classify(tmp_path,c)['status_counts'].get('approved',0)==0


def test_legacy_overflow_is_not_dropped(tmp_path,monkeypatch):
    c,_=build(tmp_path,monkeypatch)
    # Two real sample signatures plus 100 synthetic unique hashes in one legacy
    # bucket. No LIMIT is permitted to drop any of the 102 members.
    with q.connect(tmp_path/'data/v22/cache/quality.sqlite3') as conn:
        conn.execute('DELETE FROM dai');conn.execute('DELETE FROM qua_tai')
        existing=[row[0] for row in conn.execute('SELECT ma_bam FROM van_ban')]
        all_hashes=existing+['synthetic_hash_'+str(i) for i in range(100)]
        conn.executemany('INSERT INTO dai VALUES (0,?,?,?)',(('shared',h,'0000') for h in all_hashes))
        conn.executemany('INSERT INTO qua_tai VALUES (?)',((h,) for h in all_hashes))
    with e.connect(tmp_path) as conn:
        conn.executemany("INSERT INTO van_ban(ma_bam,chu_ky,so_ky_tu,ly_do) VALUES (?,X'',100,'fixture')",((h,) for h in all_hashes[2:]))
    stats=e.group_candidates(tmp_path,c)
    assert stats['phase2_overflow_fingerprints_covered']==102 and stats['large_legacy_buckets']==1
    assert stats['legacy_overflow_unquarantined']==0
    with e.connect(tmp_path) as conn:
        assert conn.execute('SELECT count(*),min(can_cach_ly) FROM nhom').fetchone()==(102,1)


@pytest.mark.parametrize('flag',['bulk_approval_activated','training_allowed','final_test_access_allowed'])
def test_phase3_cannot_approve_train_or_open_test(flag):
    c=config();c[flag]=True
    with pytest.raises(ValueError):e.validate_config(c)


def test_changed_inputs_or_sampling_policy_cannot_resume(tmp_path,monkeypatch):
    c,part=build(tmp_path,monkeypatch)
    changed=dict(c,seed=43)
    with pytest.raises(ValueError,match='inputs/config changed'):e.prepare(tmp_path,changed)
    part.write_text('changed fixture',encoding='utf-8')
    with pytest.raises(ValueError,match='checkpoint'):e.prepare(tmp_path,c)


def test_duplicate_keeps_source_label_and_provenance(tmp_path,monkeypatch):
    c,_=build(tmp_path,monkeypatch)
    with q.connect(tmp_path/'data/v22/cache/quality.sqlite3') as conn:
        a=list(conn.execute("SELECT * FROM mau WHERE id_mau='trec06c|a'").fetchone())
        a[0]='trec06c|duplicate';conn.execute('INSERT INTO mau VALUES ('+','.join('?'*len(a))+')',a)
    with sqlite3.connect(tmp_path/'data/v21/cache/dedup_audit.sqlite3') as conn:
        row=list(conn.execute("SELECT * FROM rows WHERE audit_id='trec06c|a'").fetchone())
        row[0]='trec06c|duplicate';conn.execute('INSERT INTO rows VALUES ('+','.join('?'*len(row))+')',row)
    e.group_candidates(tmp_path,c);counts=e.classify(tmp_path,c)['status_counts']
    assert counts['duplicate']==1 and sum(counts.values())==4
    with e.connect(tmp_path) as conn:
        row=conn.execute("SELECT nhan_goc,nhan,trang_thai,bang_chung FROM mau WHERE id_mau='trec06c|duplicate'").fetchone()
        assert row==('ham','','duplicate','phase2_audit:trec06c|duplicate')


def test_report_has_actual_counts_private_exports_and_no_approval(tmp_path,monkeypatch):
    from vietmailguard.v22.expansion_report import generate
    c,_=build(tmp_path,monkeypatch)
    e.group_candidates(tmp_path,c);e.classify(tmp_path,c);e.sample_review(tmp_path,c)
    policy=tmp_path/'config/v22/expansion.json';policy.parent.mkdir(parents=True);policy.write_text(json.dumps(c))
    bootstrap=tmp_path/'reports/v21/bootstrap_development_summary.json'
    bootstrap.parent.mkdir(parents=True,exist_ok=True);bootstrap.write_text('{}')
    # The production report must use actual bootstrap statistics, not fixture
    # expectations from historical documentation.
    result=generate(tmp_path,c)
    assert result['rows']==3 and not result['phase4_ready']
    summary=json.loads((tmp_path/'reports/v22/DATASET_EXPANSION_SUMMARY.json').read_text(encoding='utf-8'))
    assert summary['status_counts']['approved']==0 and summary['approved_by_label']=={'normal':0,'spam':0,'phishing':0}
    assert all(row['observed_label_error_rate'] is None for row in summary['strata'])
    assert not summary['dataset_finalized'] and summary['detector_diagnostic']['real_corpus_recall'] is None
    # No body/subject is embedded in the metadata decision export.
    with (tmp_path/'data/v22/phase3/decisions_manifest.csv').open(encoding='utf-8') as f:
        reader=csv.DictReader(f);assert 'body' not in reader.fieldnames and len(list(reader))==3
    manifest=json.loads((tmp_path/'reports/v22/phase3_manifest.json').read_text(encoding='utf-8'))
    assert all(q.digest(tmp_path/p)==sha for p,sha in manifest['sha256'].items())


def test_bootstrap_uses_actual_files_verified_hashes_and_provenance(tmp_path):
    from vietmailguard.v22.expansion_report import inspect_bootstrap
    area=tmp_path/'data/v21/curated/bootstrap_v2_development';area.mkdir(parents=True)
    (tmp_path/'data/v22/phase3').mkdir(parents=True)
    policy=tmp_path/'config/v21/bootstrap_development_policy.json';policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({'allowed_label_status':['eligible','evidence_confirmed'],
        'allowed_labels_by_provenance':{'verified_fixture':'normal'}}))
    fields=['id','source','label','language','data_origin','parent_id','final_group_id','content_hash',
        'label_provenance','curation_rule_id','split','label_status','training_eligible']
    for split in ('train','validation'):
        row=dict(zip(fields,[split,'synthetic_fixture','normal','en','synthetic','',split,'hash',
            'verified_fixture','fixture_rule',split,'evidence_confirmed','True']))
        with (area/(split+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
    manifest={'leakage':{'passed':True},'output_rows':{'train':1,'validation':1},
        'output_sha256':{s:q.digest(area/(s+'.csv')) for s in ('train','validation')}}
    (area/'manifest.json').write_text(json.dumps(manifest))
    result=inspect_bootstrap(tmp_path)
    assert result['rows']==2 and result['groups']==2 and result['by_label']=={'normal':2}
    assert result['by_language_origin']=={'en|synthetic':2} and not result['old_test_opened']
    with (area/'train.csv').open('a') as f:f.write('changed')
    with pytest.raises(ValueError,match='manifest'):inspect_bootstrap(tmp_path)


def test_candidate_conflicts_quarantined_without_changing_label(tmp_path,monkeypatch):
    c,_=build(tmp_path,monkeypatch)
    with q.connect(tmp_path/'data/v22/cache/quality.sqlite3') as conn:
        a=list(conn.execute("SELECT * FROM mau WHERE id_mau='trec06c|a'").fetchone())
        a[0]='trec06c|conflict';a[2]='spam';a[3]='spam'
        conn.execute('INSERT INTO mau VALUES ('+','.join('?'*len(a))+')',a)
    with sqlite3.connect(tmp_path/'data/v21/cache/dedup_audit.sqlite3') as conn:
        row=list(conn.execute("SELECT * FROM rows WHERE audit_id='trec06c|a'").fetchone())
        row[0]='trec06c|conflict';conn.execute('INSERT INTO rows VALUES ('+','.join('?'*len(row))+')',row)
    e.group_candidates(tmp_path,c)
    assert e.classify(tmp_path,c)['candidate_conflict_fingerprints']==1
    with e.connect(tmp_path) as conn:
        rows=conn.execute("SELECT nhan,trang_thai,ly_do FROM mau WHERE id_mau IN ('trec06c|a','trec06c|conflict')").fetchall()
        assert all(not label and status!='approved' and 'candidate_label_conflict' in reason for label,status,reason in rows)


def test_secondary_signature_check_does_not_force_collision_into_same_group(tmp_path,monkeypatch):
    c,_=build(tmp_path,monkeypatch)
    with e.connect(tmp_path) as conn:
        conn.execute('UPDATE van_ban SET chu_ky=? WHERE id_van_ban=2',
            (np.full(c['minhash_permutations'],4294967290,dtype='<u4').tobytes(),))
    result=e.group_candidates(tmp_path,c)
    assert result['legacy_edges_deferred']==1
    with e.connect(tmp_path) as conn:
        assert conn.execute('SELECT count(DISTINCT id_nhom),min(can_cach_ly) FROM nhom').fetchone()==(2,1)


def test_high_similarity_signatures_retain_split_constraint_and_quarantine(tmp_path,monkeypatch):
    c,_=build(tmp_path,monkeypatch)
    with e.connect(tmp_path) as conn:
        sig=conn.execute('SELECT chu_ky FROM van_ban WHERE id_van_ban=1').fetchone()[0]
        conn.execute('UPDATE van_ban SET chu_ky=? WHERE id_van_ban=2',(sig,))
    stats=e.group_candidates(tmp_path,c)
    assert stats['legacy_edges_fulltext_confirmed']==1
    with e.connect(tmp_path) as conn:
        assert conn.execute('SELECT count(DISTINCT id_nhom),min(can_cach_ly) FROM nhom').fetchone()==(1,1)
