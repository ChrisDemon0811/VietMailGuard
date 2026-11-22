"""Temporary synthetic fixtures only; none count toward acquired email data."""
import csv
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from vietmailguard.v22 import acquisition as a,dataset_review as d
from vietmailguard.v22.quality import digest


def review_fixture(root):
    queue=root/'data/v22/phase3/review_queue.csv';queue.parent.mkdir(parents=True)
    fields=['audit_id','stratum','source','raw_label','candidate_label','language_signal','constraint_group_id']
    with queue.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for i in range(2):w.writerow([f'fixture|{i}','fixture_normal_en','fixture','0','normal_candidate','en',f'g{i}'])
    policy=root/'config/v22/expansion.json';policy.parent.mkdir(parents=True);policy.write_text('{}')
    store=d.DatasetReview(root)
    import zlib
    with store.connection() as conn:
        for i in range(2):
            body={'subject':'Synthetic fixture '+str(i),'body':'Please review this synthetic meeting message. https://example.invalid/form',
                'sender':'sender@example.com','receiver':'recipient@example.org','date':'','original_path':'fixture'}
            conn.execute('UPDATE mau SET noi_dung_nen=? WHERE id_mau=?',(zlib.compress(json.dumps(body).encode()),f'fixture|{i}'))
    return store


def test_review_requires_real_input_preserves_history_and_never_approves(tmp_path):
    store=review_fixture(tmp_path)
    assert store.progress()['reviewed_samples']==0
    with pytest.raises(ValueError):store.save('fixture|0','normal','en','','English',True,'evidence')
    with pytest.raises(sqlite3.IntegrityError):store.save('unknown','normal','en','tester','English',True,'evidence')
    store.save('fixture|0','review','unknown','tester','needs expert',False,'insufficient language competence')
    store.save('fixture|0','normal','en','tester','English reader',True,'synthetic evidence')
    assert len(store.open_sample('fixture|0')['decisions'])==2
    assert store.progress()['automatically_approved']==0
    assert len(store.list_samples(only_pending=True,reviewer='tester'))==1
    assert len(store.list_samples(only_pending=True,reviewer='another reviewer'))==2
    assert len(store.list_samples(source="' OR 1=1 --"))==0
    assert d.DatasetReview(tmp_path).progress()['decision_events']==2
    assert store.export().is_file()


def test_redaction_is_display_only_and_policy_change_is_rejected(tmp_path):
    store=review_fixture(tmp_path)
    before=store.open_sample('fixture|0')['sender']
    assert d.redacted_preview(before)=='[EMAIL]'
    assert store.open_sample('fixture|0')['sender']==before
    (tmp_path/'config/v22/expansion.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='sample or policy'):d.DatasetReview(tmp_path)


def test_separate_review_collection_preserves_default_queue_and_requires_real_decision(tmp_path):
    import shutil
    old=review_fixture(tmp_path)
    default_hash=digest(old.path)
    collection=tmp_path/'data/v22/spam_review'
    collection.mkdir()
    shutil.copyfile(tmp_path/'data/v22/phase3/review_queue.csv',collection/'review_queue.csv')
    store=d.DatasetReview(tmp_path,collection=collection.relative_to(tmp_path))
    assert store.path!=old.path and len(store.list_samples())==2
    store.save('fixture|0','review','unknown','fixture reviewer','needs language expert',False,'synthetic unresolved evidence')
    assert store.progress()['decision_events']==1 and store.progress()['automatically_approved']==0
    assert old.progress()['decision_events']==0 and digest(old.path)==default_hash
    assert d.DatasetReview(tmp_path,collection=collection).progress()['decision_events']==1
    (collection/'review_queue.csv').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='sample or policy'):d.DatasetReview(tmp_path,collection=collection)


def test_review_collection_rejects_paths_outside_v22_and_missing_queue(tmp_path):
    review_fixture(tmp_path)
    outside=tmp_path/'outside';outside.mkdir()
    with pytest.raises(ValueError,match='inside data/v22'):d.DatasetReview(tmp_path,collection=outside)
    with pytest.raises(ValueError,match='inside data/v22'):d.DatasetReview(tmp_path,collection=tmp_path/'data/v22')
    incomplete=tmp_path/'data/v22/incomplete';incomplete.mkdir()
    with pytest.raises(ValueError,match='Missing or unsafe'):d.DatasetReview(tmp_path,collection=incomplete)


def test_stream_archives_do_not_extract_links_or_unbounded_email(tmp_path):
    path=tmp_path/'corpus.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        for name,content in [('revision/email/a.eml',b'Subject: Fixture\n\nSynthetic body'),('revision/email/large.eml',b'x'*100),('revision/../../escape.eml',b'x')]:
            info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
        link=tarfile.TarInfo('revision/email/link.eml');link.type=tarfile.SYMTYPE;link.linkname='/private';tar.addfile(link)
    before=digest(path);rows=list(a.messages(path,'tar_eml',50))
    assert rows[0][0]=='email/a.eml' and rows[1]==('email/large.eml',None) and len(rows)==2
    assert digest(path)==before and not (tmp_path/'escape.eml').exists()


def test_acquisition_download_allowlist_blocks_email_urls(tmp_path):
    with pytest.raises(ValueError,match='allowlist'):a.fetch('https://example.invalid/email-link',tmp_path/'a',100)


def test_new_mbox_labels_are_weak_and_resume_does_not_duplicate(tmp_path):
    raw=tmp_path/'data/v22/phase32/raw/fixture.mbox';raw.parent.mkdir(parents=True)
    message=b'From sender@example.com Sat Jan 01 00:00:00 2022\nSubject: Project meeting\nFrom: sender@example.com\n\nPlease review the meeting for your team and the agenda.\n'
    raw.write_bytes(message+message)
    reference=tmp_path/'data/v21/cache/dedup_audit.sqlite3';reference.parent.mkdir(parents=True)
    with sqlite3.connect(reference) as conn:conn.execute('CREATE TABLE rows(normalized_hash TEXT,template_hash TEXT)')
    with a.connect(tmp_path) as conn:conn.execute('INSERT INTO tep VALUES (?,?,?,?,?,?,?,?)',
        (raw.relative_to(tmp_path).as_posix(),'fixture','https://example.invalid',digest(raw),raw.stat().st_size,'CC-BY-4.0','fixture','downloaded'))
    config={'max_email_bytes':10000,'sources':[{'source':'fixture','format':'mbox','raw_label':'publisher_phishing','candidate_label':'phishing',
        'label_provenance':'fixture_source_not_groundtruth','license':'CC-BY-4.0','attribution':'Synthetic fixture only'}]}
    before=digest(raw);a.ingest_sources(tmp_path,config);first=a.report(tmp_path)
    a.ingest_sources(tmp_path,config);assert a.report(tmp_path)==first
    assert first['rows_received']==2 and first['status_counts']=={'duplicate':1,'weak_label':1}
    assert first['training_approved_new']==first['validation_or_test_rows']==0 and digest(raw)==before
    with a.connect(tmp_path) as conn:conn.execute("UPDATE mau SET trang_thai='review' WHERE trang_thai='weak_label'")
    a.report(tmp_path)
    assert not list((tmp_path/'data/v22/phase32/weak_labeled').glob('part-*.csv'))


def test_review_app_navigation_and_no_startup_decisions(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    store=review_fixture(tmp_path);factory=d.DatasetReview
    monkeypatch.setattr(d,'DatasetReview',lambda root:factory(tmp_path))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'scripts/v22/dataset_review_app.py')).run(timeout=30)
    assert not app.exception and store.progress()['decision_events']==0
    next(x for x in app.button if x.label=='Mẫu sau →').click().run()
    assert not app.exception and app.session_state['dataset_selected']=='fixture|1'
    next(x for x in app.button if x.label=='← Mẫu trước').click().run()
    assert app.session_state['dataset_selected']=='fixture|0'
    app.text_input[0].input('tester').run();app.text_input[1].input('English fixture reader').run()
    next(x for x in app.text_area if x.label.startswith('Lý do')).input('synthetic fixture evidence').run()
    next(x for x in app.button if x.label=='Lưu quyết định').click().run()
    assert not app.exception and store.progress()['decision_events']==1 and store.progress()['automatically_approved']==0


def test_review_app_uses_requested_collection_and_dynamic_source(tmp_path,monkeypatch,streamlit_loopback):
    import sys,zlib
    from streamlit.testing.v1 import AppTest
    original=review_fixture(tmp_path);factory=d.DatasetReview
    collection=tmp_path/'data/v22/spam_review';collection.mkdir()
    queue=(tmp_path/'data/v22/phase3/review_queue.csv').read_text(encoding='utf-8')
    (collection/'review_queue.csv').write_text(queue.replace(',fixture,',',spamassassin_original,'),encoding='utf-8')
    store=factory(tmp_path,collection=collection)
    with store.connection() as conn:
        for identifier in ('fixture|0','fixture|1'):
            content={k:v for k,v in original.open_sample(identifier).items() if k in ('subject','body','sender','receiver','date','original_path')}
            conn.execute('UPDATE mau SET noi_dung_nen=? WHERE id_mau=?',(zlib.compress(json.dumps(content).encode()),identifier))
    monkeypatch.setattr(sys,'argv',['dataset_review_app.py','--collection','data/v22/spam_review'])
    monkeypatch.setattr(d,'DatasetReview',lambda root,**kwargs:factory(tmp_path,**kwargs))
    app=AppTest.from_file(str(Path(__file__).parents[1]/'scripts/v22/dataset_review_app.py')).run(timeout=30)
    assert not app.exception and store.progress()['decision_events']==0
    source=next(x for x in app.selectbox if x.label=='Nguồn')
    assert source.options==['Tất cả','spamassassin_original']
    source.select('spamassassin_original').run()
    assert not app.exception
    app.text_input[0].input('synthetic reviewer').run();app.text_input[1].input('English synthetic fixture').run()
    next(x for x in app.text_area if x.label.startswith('Lý do')).input('synthetic pending decision').run()
    next(x for x in app.button if x.label=='Lưu quyết định').click().run()
    assert not app.exception and store.progress()['decision_events']==1
    assert original.progress()['decision_events']==0 and store.progress()['automatically_approved']==0
