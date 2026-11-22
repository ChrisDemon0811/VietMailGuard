"""Focused recovery invariants using synthetic fixtures only."""
import itertools

import numpy as np
import pytest

from vietmailguard.v22 import recovery as r
from vietmailguard.v22.expansion import minhash
from test_v22_expansion import config, build
from vietmailguard.v22 import expansion as e


def test_chunked_signature_matches_original_and_keeps_boundary_shingles():
    c=config()
    text=' '.join(f'synthetic message token {i}' for i in range(1000))
    compared=r.comparison_text(text)
    assert r.full_signature(compared,c)==minhash(text,c)[0]
    long=text*4
    assert minhash(long,c)[0]==b''
    assert len(r.full_signature(r.comparison_text(long),c))==128
    assert r.full_signature('a',c)==b''


def test_short_join_matches_exhaustive_jaccard_without_sampling():
    texts={1:'meeting today in the office room twenty',2:'meeting today in the office room twenty one',
        3:'an independent synthetic garden announcement',4:'a',5:'meeting today in the office room twenty'}
    actual={(a,b) for a,b,_ in r.short_pairs(texts,.8125)}
    expected={(a,b) for a,b in itertools.combinations(texts,2) if min(len(texts[a]),len(texts[b]))>=20 and r.exact_jaccard(texts[a],texts[b],3)>=.8125}
    assert {tuple(sorted(pair)) for pair in actual}==expected


def test_missing_signature_recovery_preserves_input_and_never_approves(tmp_path,monkeypatch):
    from vietmailguard.v22.quality import digest
    c,part=build(tmp_path,monkeypatch)
    e.group_candidates(tmp_path,c);e.classify(tmp_path,c);e.sample_review(tmp_path,c)
    # Tiny test config policy and queue are real files for input locking.
    import json
    policy=tmp_path/'config/v22/expansion.json';policy.parent.mkdir(parents=True);policy.write_text(json.dumps(c))
    queue=tmp_path/'data/v22/phase3/review_queue.csv';queue.write_text('fixture queue')
    monkeypatch.setattr(r,'DATASETS',e.DATASETS)
    before=digest(tmp_path/'data/v22/phase3/expansion.sqlite3');raw=digest(part)
    r.prepare(tmp_path,c)
    got=r.load_selected_texts(tmp_path,c)
    assert got['loaded_content']==got['requested_content']
    result=r.recover(tmp_path,c)
    with r.connect(tmp_path) as conn:
        assert conn.execute("SELECT count(*) FROM phuc_hoi WHERE nhan_duoc_duyet!=''").fetchone()[0]==0
        assert conn.execute('SELECT count(*) FROM phuc_hoi').fetchone()[0]==2
    assert digest(tmp_path/'data/v22/phase3/expansion.sqlite3')==before and digest(part)==raw
    assert r.recover(tmp_path,c)==result
    queue.write_text('changed sample')
    with pytest.raises(ValueError,match='sample changed'):r.prepare(tmp_path,c)


def test_exact_prefix_join_recovers_all_qualifying_pairs():
    import random
    rng=random.Random(42)
    features={i:set(rng.sample(list('abcdefghijklmnop'),rng.randint(5,14))) for i in range(100)}
    expected={(a,b) for a,b in itertools.combinations(features,2)
        if len(features[a]&features[b])/len(features[a]|features[b])>=.8125}
    actual={tuple(sorted((a,b))) for a,b,_ in r.set_join(features,.8125)}
    assert actual==expected


def test_bootstrap_licensed_subset_keeps_assignment_and_never_adopts(tmp_path):
    import csv,json
    from vietmailguard.v22.quality import digest
    from vietmailguard.v22.recovery_report import bootstrap_evidence
    area=tmp_path/'data/v21/curated/bootstrap_v2_development';area.mkdir(parents=True)
    for split in ('train','validation'):(area/(split+'.csv')).write_text('synthetic fixture '+split)
    manifest={'leakage':{'passed':True},'output_sha256':{s:digest(area/(s+'.csv')) for s in ('train','validation')},'output_rows':{'train':2,'validation':0}}
    (area/'manifest.json').write_text(json.dumps(manifest))
    policy=tmp_path/'config/v21/bootstrap_development_policy.json';policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({'allowed_labels_by_provenance':{'nazario_phishing':'phishing','controlled_translation_from_nazario_phishing':'phishing'}}))
    p=tmp_path/'data/v22/phase3/bootstrap_proposal.csv';p.parent.mkdir(parents=True)
    fields=['id','source','label','language','data_origin','parent_id','final_group_id','label_provenance','curation_rule_id','split','v22_approved']
    rows=[dict(zip(fields,['p','Nazario','phishing','en','synthetic','','g','nazario_phishing','fixture','train','false'])),
        dict(zip(fields,['t','controlled_translation','phishing','vi','translated','p','g','controlled_translation_from_nazario_phishing','fixture','train','false']))]
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=tmp_path/'reports/v22';report.mkdir(parents=True)
    (report/'phase3_manifest.json').write_text(json.dumps({'sha256':{p.relative_to(tmp_path).as_posix():digest(p)}}))
    (tmp_path/'data/v22/phase31').mkdir(parents=True)
    result=bootstrap_evidence(tmp_path)
    assert result['explicitly_licensed_inherited_rows']==2 and not result['owner_adoption_confirmed']
    assert result['split_rows']=={'train':2}
    assert not result['independent_benchmark'] and not result['old_test_opened']
