"""Synthetic source-only quality tests; no real email or held-out access."""
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from vietmailguard.v22 import quality as q
from vietmailguard.v21 import dedup_audit as old


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(q,'DATASETS',[('trec06c','fixture')])
    config=json.loads((Path(__file__).resolve().parents[1]/'config/v22/quality.json').read_text())
    part=tmp_path/'data/v21/processed/fixture/part-000.csv'
    part.parent.mkdir(parents=True)
    prefix='This is the meeting agenda and please review the project with your team. '*10
    rows=[dict(id='a',subject='Meeting',body=prefix+' Monday',raw_label='ham',label='review',candidate_label='ham'),
          dict(id='b',subject='Meeting',body=prefix+' Tuesday',raw_label='spam',label='review',candidate_label='spam'),
          dict(id='c',subject='',body='',raw_label='ham',label='review',candidate_label='ham')]
    with part.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    conn=old._connect(tmp_path/'data/v21/cache/dedup_audit.sqlite3')
    for raw in rows:
        conn.execute('INSERT INTO rows VALUES ('+','.join('?' for _ in old.FIELDS)+')',old._row('trec06c',raw,part))
    conn.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?)',(str(part.resolve()),part.stat().st_size,part.stat().st_mtime_ns,3,old.AUDIT_VERSION))
    conn.commit();conn.close()
    return config,part


def test_unicode_html_script_and_uncertain_languages():
    assert q.normalize('e\u0301\x00  mail')=='é mail'
    safe=q.normalize('<html><script>alert(1)</script><p>Hello</p><img src="https://example.invalid/pixel"></html>')
    assert 'alert' not in safe and '<img' not in safe
    assert q.language_signal('x')=='unknown'
    assert q.language_signal('这是一个中文电子邮件内容测试，请检查今天的会议安排。')=='zh'
    assert q.language_signal('Xin vui lòng kiểm tra thông tin của bạn ngày mai.')=='vi'


def test_full_pipeline_resume_near_candidates_do_not_approve(tmp_path, monkeypatch):
    config,part=fixture(tmp_path,monkeypatch)
    original=part.read_bytes()
    assert q.ingest(tmp_path,config)=={'rows':3,'parts':1}
    q.near_duplicates(tmp_path,config)
    first=q.summarize(tmp_path)
    assert first['rows']==3 and first['states']=={'exclude':1,'review':2}
    assert first['near_candidate_edges']==1 and first['near_pass_complete']
    assert first['newly_approved_rows']==0 and not first['v22_training_ready']
    pack=q.export_review_pack(tmp_path,'trec06c|a')
    assert pack.is_relative_to(tmp_path/'data/v22/review')
    assert json.loads(pack.read_text())['raw_label']=='ham'
    q.ingest(tmp_path,config);q.near_duplicates(tmp_path,config)
    assert q.summarize(tmp_path)==first and part.read_bytes()==original
    q.record_review(tmp_path,'trec06c|a','normal','synthetic_reviewer','synthetic evidence','Chinese competent')
    assert q.summarize(tmp_path)['human_decisions']==1
    assert q.summarize(tmp_path)['newly_approved_rows']==0
    with pytest.raises(sqlite3.IntegrityError):
        q.record_review(tmp_path,'unknown','normal','tester','evidence','competent')
    with pytest.raises(sqlite3.IntegrityError):
        q.record_review(tmp_path,'trec06c|a','normal','tester','replacement','competent')
    with pytest.raises(ValueError):
        q.record_review(tmp_path,'trec06c|b','phishing','tester','','competent')
    with q.connect(tmp_path/'data/v22/cache/quality.sqlite3') as conn:
        tables=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert all(name.isascii() and name==name.lower() for name in tables)
        assert all(r[1].isascii() and r[1]==r[1].lower() for table in tables for r in conn.execute(f'PRAGMA table_info({table})'))
    part.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError):
        q.ingest(tmp_path,config)


def test_contract_prevents_training_approval_and_test_access(tmp_path):
    for key in ('automatic_label_approval','training_allowed','final_test_access_allowed'):
        config={name:False for name in ('automatic_label_approval','training_allowed','final_test_access_allowed')}
        config[key]=True
        with pytest.raises(ValueError,match='Phase 1/2'):
            q.ingest(tmp_path,config)


def test_bucket_limit_is_visible_and_never_silent(tmp_path,monkeypatch):
    config,_=fixture(tmp_path,monkeypatch)
    q.ingest(tmp_path,config)
    config['near_bucket_limit']=0
    q.near_duplicates(tmp_path,config)
    result=q.summarize(tmp_path)
    assert result['near_overflow_rows']==1 and result['near_candidate_edges']==0
    assert not result['v22_training_ready']


def test_failed_part_rolls_back_rows_and_checkpoint(tmp_path,monkeypatch):
    config,part=fixture(tmp_path,monkeypatch)
    text=part.read_text(encoding='utf-8').replace('b,Meeting','missing,Meeting')
    part.write_text(text,encoding='utf-8')
    with sqlite3.connect(tmp_path/'data/v21/cache/dedup_audit.sqlite3') as conn:
        conn.execute('UPDATE checkpoints SET size_bytes=?,mtime_ns=?',(part.stat().st_size,part.stat().st_mtime_ns))
    with pytest.raises(ValueError,match='missing from historical audit'):
        q.ingest(tmp_path,config)
    with q.connect(tmp_path/'data/v22/cache/quality.sqlite3') as conn:
        assert conn.execute('SELECT count(*) FROM mau').fetchone()[0]==0
        assert conn.execute('SELECT count(*) FROM tien_do').fetchone()[0]==0


@pytest.mark.parametrize('link',['fingerprint','provenance'])
def test_old_reference_overlap_guard_without_reading_test_records(tmp_path,monkeypatch,link):
    config,part=fixture(tmp_path,monkeypatch)
    with part.open(encoding='utf-8',newline='') as handle:
        raw=next(csv.DictReader(handle))
    raw.update(label='normal',training_eligible='true')
    if link=='provenance':
        raw.update(body='Completely different historical fixture content.',final_group_id='shared_fixture')
    audit=tmp_path/'data/v21/cache/dedup_audit.sqlite3'
    with sqlite3.connect(audit) as conn:
        if link=='provenance':
            conn.execute("UPDATE rows SET provenance_group_id='v2_shared_fixture' WHERE audit_id='trec06c|a'")
        conn.execute('INSERT INTO rows VALUES ('+','.join('?' for _ in old.FIELDS)+')',old._row('v2_reference',raw,part))
    before=q.digest(audit)
    q.ingest(tmp_path,config);q.near_duplicates(tmp_path,config)
    result=q.summarize(tmp_path)
    assert result['rows']==3 and result['old_reference_overlap_rows']==1
    assert result['newly_approved_rows']==0 and q.digest(audit)==before
    with q.connect(tmp_path/'data/v22/cache/quality.sqlite3') as conn:
        assert conn.execute('SELECT id_mau FROM chan_lich_su').fetchall()==[('trec06c|a',)]
