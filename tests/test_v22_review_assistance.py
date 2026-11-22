"""Synthetic offline review suggestions: no inference, ground truth or training."""
import csv
import json
import sys
import zlib
from pathlib import Path

import pytest

from vietmailguard.v22 import review_assistance as a
from vietmailguard.v22.dataset_review import DatasetReview
from vietmailguard.v22.quality import digest


@pytest.mark.parametrize('tfidf',sorted(a.LABELS))
@pytest.mark.parametrize('minilm',sorted(a.LABELS))
def test_all_teacher_pairs_remain_unverified_and_source_family_is_not_three_class(tfidf,minilm):
    row=a.suggestion(identifier='synthetic',content_sha256='a'*64,group='g',tfidf=tfidf,minilm=minilm,
                     provenance_verified=True,research_use_verified=True)
    assert not row['training_eligible'] and row['human_verified_label'] is None
    assert row['human_check_required'] and row['confidence'] is None and row['calibrated_probability'] is None
    assert ('models_disagree' in row['flags'])==(tfidf!=minilm)
    assert ('outside_source_positive_family' in row['flags'])==('normal' in (tfidf,minilm))
    assert ('phishing_model_signal' in row['flags'])==('phishing' in (tfidf,minilm))
    assert row['source_approval_consideration_only']==(tfidf==minilm=='spam')
    assert ('three_class_label_unverified' in row['flags'])
    assert a.summarize([row])['automatically_approved']==0


def test_missing_model_or_evidence_cannot_qualify_for_source_consideration():
    missing=a.suggestion(identifier='x',content_sha256='a'*64,group='g',tfidf='spam',minilm=None,
                         provenance_verified=False,research_use_verified=False)
    assert missing['priority']=='urgent' and not missing['source_approval_consideration_only']
    assert {'model_unavailable','provenance_missing','local_research_rights_missing'}<=set(missing['flags'])
    with pytest.raises(ValueError):
        a.suggestion(identifier='x',content_sha256='a'*64,group='g',tfidf='ham',minilm='spam',
                     provenance_verified=True,research_use_verified=True)


def fixture(root):
    collection=root/'data/v22/fixture_review';collection.mkdir(parents=True)
    policy=root/'config/v22/expansion.json';policy.parent.mkdir(parents=True);policy.write_text('{}')
    with (collection/'review_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['audit_id','stratum','source','raw_label','candidate_label','language_signal','constraint_group_id'])
        for i in range(2):w.writerow([f'synthetic:{i}','s','spamassassin_original','spam','spam_or_phishing_candidate','en',f'g{i}'])
    store=DatasetReview(root,collection=collection)
    content=[];suggestions=[]
    with store.connection() as conn:
        for i in range(2):
            identifier=f'synthetic:{i}'
            text={'subject':'Synthetic review fixture','body':'An independently written synthetic message. https://example.invalid/',
                  'sender':'fixture@example.com','receiver':'reader@example.org','date':'','original_path':f'fixture/{i}'}
            conn.execute('UPDATE mau SET noi_dung_nen=? WHERE id_mau=?',(zlib.compress(json.dumps(text).encode()),identifier))
            content.append({'id':identifier,**text})
            suggestions.append(a.suggestion(identifier=identifier,content_sha256=a.input_digest(text['subject'],text['body']),
                group=f'g{i}',tfidf='spam',minilm='normal' if i else 'spam',provenance_verified=True,research_use_verified=True))
    (collection/'content.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in content),encoding='utf-8')
    folder=collection/'assistance';folder.mkdir()
    (folder/'suggestions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in suggestions),encoding='utf-8')
    manifest={'policy_version':a.POLICY_VERSION,'record_kind':'review_assistance_only','rows':2,
        'queue_sha256':digest(collection/'review_queue.csv'),'content_sha256':digest(collection/'content.jsonl'),
        'suggestions_sha256':digest(folder/'suggestions.jsonl'),'summary':a.summarize(suggestions)}
    (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return store,manifest,suggestions


def test_sidecar_reload_priority_and_human_progress_are_independent(tmp_path):
    store,_,_=fixture(tmp_path)
    before=digest(store.path)
    manifest,records=a.load_assistance(store.path.parent)
    assert manifest['summary']['priority_counts']=={'standard':1,'urgent':1}
    assert sorted(records,key=lambda identifier:a.priority_order(identifier,records[identifier]))[0]=='synthetic:1'
    assert a.load_assistance(store.path.parent)==(manifest,records)
    assert store.progress()['decision_events']==0 and digest(store.path)==before


@pytest.mark.parametrize('tamper',['content','queue','confidence','approve','summary','foreign_id'])
def test_stale_or_invalid_suggestions_fail_closed(tmp_path,tamper):
    store,manifest,suggestions=fixture(tmp_path);folder=store.path.parent/'assistance'
    if tamper in ('content','queue'):
        path=store.path.parent/('content.jsonl' if tamper=='content' else 'review_queue.csv')
        path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    else:
        if tamper=='confidence':suggestions[0]['confidence']=.99
        elif tamper=='approve':suggestions[0]['training_eligible']=True
        elif tamper=='foreign_id':suggestions[0]['id']='not-in-collection'
        else:manifest['summary']['automatically_approved']=2
        (folder/'suggestions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in suggestions),encoding='utf-8')
        manifest['suggestions_sha256']=digest(folder/'suggestions.jsonl')
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):a.load_assistance(store.path.parent)
    assert store.progress()['decision_events']==0


def test_assisted_review_ui_uses_saved_outputs_and_records_exposure(tmp_path,monkeypatch,streamlit_loopback):
    from streamlit.testing.v1 import AppTest
    from vietmailguard.v22 import dataset_review
    store,_,_=fixture(tmp_path)
    monkeypatch.setattr(sys,'argv',['dataset_review_app.py'])
    monkeypatch.setattr(dataset_review,'DatasetReview',lambda root:store)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'scripts/v22/dataset_review_app.py')).run(timeout=30)
    assert not app.exception and app.session_state['dataset_selected']=='synthetic:1'
    assert next(x for x in app.radio if x.label=='Quyết định').value=='review'
    ai=next(x for x in app.checkbox if x.label.startswith('Hiện gợi ý'))
    assert not ai.value and ai.disabled
    assert next(x for x in app.selectbox if x.label=='Nhóm cần xem').value=='Ưu tiên cao'
    app.text_input[0].input('synthetic reviewer').run();app.text_input[1].input('English synthetic fixture').run()
    next(x for x in app.button if x.label=='Lưu quyết định').click().run()
    assert store.progress()['decision_events']==0  # a prefix cannot replace missing human evidence
    next(x for x in app.text_area if x.label.startswith('Lý do')).input('Synthetic unresolved evidence').run()
    next(x for x in app.button if x.label=='Lưu quyết định').click().run()
    decision=store.open_sample('synthetic:1')['decisions'][0]
    assert decision['nhan']=='review' and decision['bang_chung'].startswith('review_method=models_hidden_assistance_available;')
    next(x for x in app.checkbox if x.label.startswith('Hiện gợi ý')).check().run()
    assert not app.exception and any(x.value=='TF-IDF + LinearSVC' for x in app.markdown)
    next(x for x in app.checkbox if x.label.startswith('Hiện gợi ý')).uncheck().run()
    next(x for x in app.button if x.label=='Lưu quyết định').click().run()
    assert store.open_sample('synthetic:1')['decisions'][-1]['bang_chung'].startswith('review_method=local_model_assisted;')
    assert store.progress()['automatically_approved']==0
    next(x for x in app.selectbox if x.label=='Nhóm cần xem').select('Đồng thuận Spam — chưa duyệt').run()
    assert not app.exception and app.session_state['dataset_selected']=='synthetic:0'


def test_checkpoint_and_draft_resume_without_creating_decisions(tmp_path):
    store,_,_=fixture(tmp_path)
    store.save_checkpoint('reader',{'selected_id':'synthetic:1','subset':'Ưu tiên cao','pending':True})
    store.save_draft('reader','synthetic:1',{'label':'spam','language':'en','evidence':'Synthetic unfinished note','ai_exposed':True})
    store.save_draft('reader','synthetic:1',{'label':'review','evidence':'Synthetic revision','ai_exposed':False})
    restarted=DatasetReview(tmp_path,collection=store.path.parent)
    assert restarted.checkpoint('reader')['selected_id']=='synthetic:1'
    assert restarted.draft('reader','synthetic:1')['ai_exposed'] is True
    assert restarted.draft('reader','synthetic:1')['evidence']=='Synthetic revision'
    assert restarted.checkpoint('other-reader')=={} and restarted.draft('other-reader','synthetic:1')=={}
    assert restarted.progress()['decision_events']==0 and restarted.progress()['automatically_approved']==0


def test_save_and_checkpoint_are_atomic_and_independent_request_persists(tmp_path):
    import sqlite3
    store,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='checkpoint'):
        store.save('synthetic:0','spam','en','reader','English fixture',True,'synthetic evidence',
                   checkpoint={'selected_id':'does-not-exist'})
    assert store.progress()['decision_events']==0
    store.save_draft('reader','synthetic:0',{'ai_exposed':True})
    store.save('synthetic:0','spam','en','reader','English fixture',True,'synthetic ambiguity',
               independent_required=True,independent_reason='ambiguous_spam_phishing',checkpoint={'selected_id':'synthetic:1'})
    assert store.checkpoint('reader')['selected_id']=='synthetic:1'
    assert store.independent_review_ids()=={'synthetic:0'}
    store.save('synthetic:0','normal','en','reader','English fixture',True,'synthetic self correction')
    assert store.independent_review_ids()=={'synthetic:0'}  # self-correction is not independent adjudication
    with store.export().open(encoding='utf-8',newline='') as f:exported=list(csv.DictReader(f))
    assert exported[0]['can_kiem_tra_doc_lap']=='1' and exported[0]['da_xem_goi_y_ai']=='1'
    assert all(row['da_phe_duyet']=='0' for row in exported)
    with pytest.raises(sqlite3.IntegrityError):
        store.save('synthetic:1','approved','en','reader','English fixture',True,'cannot approve by label')
    assert store.progress()['automatically_approved']==0 and store.progress()['decision_events']==2


def test_previous_assisted_decision_is_not_later_mislabeled_as_unexposed(tmp_path):
    store,_,_=fixture(tmp_path)
    # Legacy evidence has no separate exposure checkpoint, but remains auditable.
    store.save('synthetic:0','spam','en','reader','English fixture',True,
               'review_method=local_model_assisted; synthetic legacy evidence')
    assert store.ai_seen('reader','synthetic:0') and not store.ai_seen('other-reader','synthetic:0')
    store.save('synthetic:0','spam','en','reader','English fixture',True,'Synthetic later correction')
    with store.export().open(encoding='utf-8',newline='') as f:exported=list(csv.DictReader(f))
    assert all(row['da_xem_goi_y_ai']=='1' for row in exported)


def test_save_next_and_browser_session_restart_restore_position_and_unsaved_draft(tmp_path,monkeypatch,streamlit_loopback):
    from streamlit.testing.v1 import AppTest
    from vietmailguard.v22 import dataset_review
    store,_,_=fixture(tmp_path)
    monkeypatch.setattr(sys,'argv',['dataset_review_app.py'])
    monkeypatch.setattr(dataset_review,'DatasetReview',lambda root:store)
    path=str(Path(__file__).parents[1]/'scripts/v22/dataset_review_app.py')
    app=AppTest.from_file(path).run(timeout=30)
    app.text_input[0].input('reader').run();app.text_input[1].input('English synthetic fixture').run()
    next(x for x in app.selectbox if x.label=='Nhóm cần xem').select('Tất cả').run()
    first=app.session_state['dataset_selected']
    next(x for x in app.button if x.label=='Lưu và chuyển email tiếp theo').click().run()
    assert not app.exception and app.session_state['dataset_selected']==first and store.progress()['decision_events']==0
    next(x for x in app.radio if x.label=='Quyết định').set_value('spam').run()
    next(x for x in app.selectbox if x.label=='Ngôn ngữ đã xác nhận').select('en').run()
    next(x for x in app.checkbox if x.label.startswith('Tôi có năng lực')).check().run()
    next(x for x in app.checkbox if x.label.startswith('Spam/Phishing mơ hồ')).check().run()
    next(x for x in app.text_area if x.label.startswith('Lý do')).input('Synthetic ambiguity needs independent check').run()
    assert store.progress()['decision_events']==0  # autosave is a draft, never an opinion
    next(x for x in app.button if x.label=='Lưu và chuyển email tiếp theo').click().run()
    assert not app.exception and store.progress()['decision_events']==1
    second=app.session_state['dataset_selected'];assert second!=first
    assert store.checkpoint('reader')['selected_id']==second and first in store.independent_review_ids()
    next(x for x in app.text_area if x.label.startswith('Lý do')).input('Synthetic unfinished second-email note').run()
    restarted=AppTest.from_file(path).run(timeout=30)
    restarted.text_input[0].input('reader').run()
    assert not restarted.exception and restarted.session_state['dataset_selected']==second
    assert next(x for x in restarted.selectbox if x.label=='Nhóm cần xem').value=='Tất cả'
    assert next(x for x in restarted.text_area if x.label.startswith('Lý do')).value=='Synthetic unfinished second-email note'
    assert store.progress()['decision_events']==1 and store.progress()['automatically_approved']==0
    next(x for x in restarted.selectbox if x.label=='Nhóm cần xem').select('Cần kiểm tra độc lập').run()
    assert not restarted.exception and restarted.session_state['dataset_selected']==first
