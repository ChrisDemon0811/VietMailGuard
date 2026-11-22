"""Synthetic UI checks; no real review decisions, models, or external requests."""
import csv,json,sys,zlib
from pathlib import Path

from vietmailguard.v22.dataset_review import DatasetReview


def test_existing_review_app_shows_full_source_headers_and_provenance_without_ai_leak(tmp_path,monkeypatch,streamlit_loopback):
    from streamlit.testing.v1 import AppTest
    from vietmailguard.v22 import dataset_review
    collection=tmp_path/'data/v22/fixture';collection.mkdir(parents=True)
    policy=tmp_path/'config/v22/expansion.json';policy.parent.mkdir(parents=True);policy.write_text('{}')
    with (collection/'review_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['audit_id','stratum','source','raw_label','candidate_label','language_signal','constraint_group_id'])
        w.writerow(['fixture:0','opaque','fixture','unlabeled','','en','g0'])
    store=DatasetReview(tmp_path,collection=collection)
    body='Synthetic full content. '*300
    header='From: sender@example.com\nTo: reader@example.org\nDate: Fri, 1 Jan 2021 10:00:00 +0000\nX-Fixture-Header: full-source-value'
    content={'subject':'Synthetic subject','body':body,'sender':'sender@example.com','receiver':'reader@example.org',
        'date':'Fri, 1 Jan 2021 10:00:00 +0000','original_path':'synthetic/0.eml','raw_header_block':header,
        'raw_header_availability':'full_original_mime_header_block',
        'source_metadata':{'license':'Synthetic fixture only','raw_sha256':'a'*64,'ai_suggested_label':'DO_NOT_DISPLAY_TEACHER_LABEL'},
        'raw_source_verification':{'subject_body_reparse_matches_cached_original':True}}
    with store.connection() as c:c.execute('update mau set noi_dung_nen=?',(zlib.compress(json.dumps(content).encode()),))
    (collection/'review_guide_vi.md').write_text('Independent review guide: use Uncertain when evidence is missing.',encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['dataset_review_app.py'])
    monkeypatch.setattr(dataset_review,'DatasetReview',lambda root:store)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'scripts/v22/dataset_review_app.py')).run(timeout=30)
    next(x for x in app.checkbox if x.label=='Ẩn địa chỉ email và số liên hệ').uncheck().run()
    assert not app.exception and store.progress()['decision_events']==0
    assert next(x for x in app.text_area if x.label.startswith('Nội dung gốc')).value==body
    assert any(x.value==header for x in app.code)
    text='\n'.join(x.value for x in app.text)
    assert 'From: sender@example.com' in text and 'To: reader@example.org' in text and 'Date: Fri, 1 Jan 2021' in text
    assert '"raw_sha256"' in text and 'a'*64 in text
    assert 'DO_NOT_DISPLAY_TEACHER_LABEL' not in text
    assert any('Independent review guide' in x.value for x in app.markdown)
    assert not any(x.label.startswith('Hiện gợi ý') for x in app.checkbox)


def test_missing_raw_headers_are_reported_without_fabricating_transport_evidence(tmp_path,monkeypatch,streamlit_loopback):
    from streamlit.testing.v1 import AppTest
    from vietmailguard.v22 import dataset_review
    collection=tmp_path/'data/v22/fixture';collection.mkdir(parents=True)
    policy=tmp_path/'config/v22/expansion.json';policy.parent.mkdir(parents=True);policy.write_text('{}')
    with (collection/'review_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['audit_id','stratum','source','raw_label','candidate_label','language_signal','constraint_group_id'])
        w.writerow(['fixture:0','opaque','fixture','','','en','g0'])
    store=DatasetReview(tmp_path,collection=collection)
    pack={'subject':'Fixture','body':'Synthetic content','sender':'','receiver':'','date':'','original_path':'fixture'}
    with store.connection() as c:c.execute('update mau set noi_dung_nen=?',(zlib.compress(json.dumps(pack).encode()),))
    monkeypatch.setattr(sys,'argv',['dataset_review_app.py']);monkeypatch.setattr(dataset_review,'DatasetReview',lambda root:store)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'scripts/v22/dataset_review_app.py')).run(timeout=30)
    assert not app.exception
    assert any('không có full raw headers' in x.value for x in app.caption)
    assert store.progress()['decision_events']==0
