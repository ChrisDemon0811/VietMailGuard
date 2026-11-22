"""Existing offline review UI: human-first decisions, resumable local checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import streamlit as st
from vietmailguard.v22.dataset_review import DatasetReview,redacted_preview
from vietmailguard.v22.review_assistance import load_assistance,priority_order,input_digest,FLAG_TEXT,source_evidence

CONTROLS={'source':'dataset_source','pending':'dataset_pending','redact':'dataset_redact',
          'prioritize':'dataset_prioritize','subset':'dataset_subset','competence':'dataset_competence'}


def _session(selected=None):
    return {**{name:st.session_state.get(key) for name,key in CONTROLS.items()},
            'selected_id':selected if selected is not None else st.session_state.get('dataset_selected'),
            'pinned_id':st.session_state.get('dataset_review_pin')}


def _checkpoint(store,reviewer):
    try:store.save_checkpoint(reviewer,_session())
    except Exception:st.session_state['dataset_flash']=('warning','Chưa lưu được checkpoint local; kiểm tra quyền ghi database.')


def _select(store,reviewer,identifier=None):
    identifier=identifier or st.session_state['dataset_sample_choice']
    st.session_state['dataset_sample_choice']=identifier
    st.session_state['dataset_selected']=identifier
    st.session_state['dataset_review_pin']=None
    _checkpoint(store,reviewer)


def _draft_changed(store,reviewer,identifier,keys):
    if reviewer!=st.session_state.get('dataset_reviewer','').strip():return
    try:
        store.save_draft(reviewer,identifier,{name:st.session_state.get(key) for name,key in keys.items()})
        _checkpoint(store,reviewer)
    except Exception:st.session_state['dataset_flash']=('warning','Chưa tự lưu được bản nháp; kiểm tra quyền ghi database.')


def _save_opinion(store,reviewer,identifier,ids,keys,advance,policy):
    if reviewer!=st.session_state.get('dataset_reviewer','').strip():
        st.session_state['dataset_flash']=('error','Mã người duyệt đã đổi; kiểm tra lại nhận định rồi lưu.');return
    draft={name:st.session_state.get(key) for name,key in keys.items()}
    exposed=bool(draft.get('ai_exposed') or store.ai_seen(reviewer,identifier))
    evidence=draft.get('evidence','')
    if evidence.strip() and policy:
        method='local_model_assisted' if exposed else 'models_hidden_assistance_available'
        evidence=f'review_method={method}; assistance_policy={policy}; '+evidence
    next_id=identifier
    if advance:
        index=ids.index(identifier)
        choices=ids[index+1:]+ids[:index]
        reviewed=store.reviewed_ids(reviewer) if st.session_state.get('dataset_pending') else set()
        next_id=next((value for value in choices if value not in reviewed),None)
    checkpoint={**_session(),'selected_id':next_id,'pinned_id':None if advance else identifier}
    try:
        store.save(identifier,draft['label'],draft['language'],reviewer,st.session_state.get('dataset_competence',''),
            bool(draft['attested']),evidence,independent_required=bool(draft['ambiguous']),
            independent_reason='ambiguous_spam_phishing' if draft['ambiguous'] else '',
            ai_exposed=exposed,checkpoint=checkpoint)
    except Exception:
        st.session_state['dataset_flash']=('error','Chưa lưu được. Điền mã người duyệt, năng lực, nhận định và bằng chứng hợp lệ.');return
    st.session_state['dataset_review_pin']=checkpoint['pinned_id']
    st.session_state['dataset_selected']=next_id
    st.session_state['dataset_sample_choice']=next_id
    try:store.export()
    except Exception:
        st.session_state['dataset_flash']=('warning','Đã lưu SQLite và checkpoint; chưa xuất được CSV. Không cần lưu lại quyết định.');return
    st.session_state['dataset_flash']=('success','Đã lưu nhận định và checkpoint. Mẫu chưa được Approved hoặc đưa vào Training.')


def main():
    st.set_page_config(page_title='VietMailGuard · Kiểm duyệt dataset',layout='wide')
    st.title('Kiểm duyệt dataset V2.2')
    st.caption('Công cụ local · Đọc thư và ghi nhận định trước khi xem gợi ý AI')
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--collection',type=Path)
    options,_=parser.parse_known_args()
    try:store=DatasetReview(ROOT,collection=options.collection) if options.collection is not None else DatasetReview(ROOT)
    except Exception:
        st.error('Không mở được hàng đợi. Kiểm tra dữ liệu local và snapshot review.');return
    guide=store.path.parent/'review_guide_vi.md'
    if guide.is_file() and not guide.is_symlink() and guide.stat().st_size<=65536:
        with st.expander('Hướng dẫn gán nhãn và kiểm tra độc lập'):
            st.markdown(guide.read_text(encoding='utf-8'),unsafe_allow_html=False)
    assistance=None
    try:assistance=load_assistance(store.path.parent)
    except (OSError,ValueError,KeyError,TypeError):st.warning('Gợi ý local không khớp collection; tiếp tục kiểm duyệt thủ công.')
    manifest,suggestions=assistance if assistance else ({},{})
    reviewer=st.sidebar.text_input('Tên hoặc mã người kiểm duyệt',key='dataset_reviewer').strip()
    scope=(str(store.path),reviewer)
    if st.session_state.get('dataset_review_scope')!=scope:
        saved=store.checkpoint(reviewer)
        defaults={'source':'','pending':True,'redact':True,'prioritize':True,
                  'subset':'Ưu tiên cao' if assistance else 'Tất cả','competence':''}
        for name,key in CONTROLS.items():st.session_state[key]=saved.get(name,defaults[name])
        st.session_state['dataset_sample_choice']=saved.get('selected_id')
        st.session_state['dataset_selected']=saved.get('selected_id')
        st.session_state['dataset_review_pin']=saved.get('pinned_id')
        st.session_state['dataset_review_scope']=scope
    st.sidebar.text_input('Năng lực ngôn ngữ / vai trò',key='dataset_competence',on_change=_checkpoint,args=(store,reviewer))
    st.sidebar.caption('Dùng cùng mã người duyệt để khôi phục checkpoint. Năng lực tự khai không phải chứng nhận.')
    all_rows=store.list_samples(only_pending=False)
    sources=sorted({row['nguon'] for row in all_rows})
    if st.session_state['dataset_source'] not in ['',*sources]:st.session_state['dataset_source']=''
    st.sidebar.selectbox('Nguồn',['',*sources],format_func=lambda x:x or 'Tất cả',key='dataset_source',on_change=_checkpoint,args=(store,reviewer))
    st.sidebar.checkbox('Chỉ mẫu chưa được tôi kiểm duyệt',key='dataset_pending',on_change=_checkpoint,args=(store,reviewer))
    redact=st.sidebar.checkbox('Ẩn địa chỉ email và số liên hệ',key='dataset_redact',on_change=_checkpoint,args=(store,reviewer))
    st.sidebar.caption('Ẩn tự động có giới hạn; tắt tùy chọn để xem nội dung nguyên văn tại máy local.')
    subset='Tất cả';prioritize=False
    choices=['Tất cả','Ưu tiên cao','Hai mô hình bất đồng','Có đề xuất Phishing','Khác họ nhãn nguồn',
             'Thiếu bằng chứng nguồn/quyền','Đồng thuận Spam — chưa duyệt','Cần kiểm tra độc lập']
    if assistance:
        if st.session_state['dataset_subset'] not in choices:st.session_state['dataset_subset']='Ưu tiên cao'
        prioritize=st.sidebar.checkbox('Ưu tiên trường hợp bất đồng / đáng ngờ',key='dataset_prioritize',on_change=_checkpoint,args=(store,reviewer))
        subset=st.sidebar.selectbox('Nhóm cần xem',choices,key='dataset_subset',on_change=_checkpoint,args=(store,reviewer))
    independent=store.independent_review_ids()
    rows=[row for row in all_rows if not st.session_state['dataset_source'] or row['nguon']==st.session_state['dataset_source']]
    if subset=='Cần kiểm tra độc lập':rows=[row for row in rows if row['id_mau'] in independent]
    elif assistance:
        if subset=='Ưu tiên cao':rows=[row for row in rows if suggestions[row['id_mau']]['priority']=='urgent']
        elif subset=='Đồng thuận Spam — chưa duyệt':rows=[row for row in rows if suggestions[row['id_mau']]['source_approval_consideration_only']]
        else:
            flags={'Hai mô hình bất đồng':{'models_disagree'},'Có đề xuất Phishing':{'phishing_model_signal'},
                   'Khác họ nhãn nguồn':{'outside_source_positive_family'},
                   'Thiếu bằng chứng nguồn/quyền':{'provenance_missing','local_research_rights_missing'}}.get(subset)
            if flags:rows=[row for row in rows if flags & set(suggestions[row['id_mau']]['flags'])]
        if prioritize:rows.sort(key=lambda row:priority_order(row['id_mau'],suggestions[row['id_mau']]))
    reviewed=store.reviewed_ids(reviewer);global_reviewed=store.reviewed_ids()
    group_done=sum(row['id_mau'] in reviewed for row in rows)
    st.caption(f'Toàn bộ: {len(global_reviewed)}/{len(all_rows)} thư có nhận định · Còn {len(all_rows)-len(global_reviewed)} · Cần kiểm tra độc lập: {len(independent)} · Approved: 0')
    st.caption(f'Nhóm hiện tại: đã duyệt {group_done}/{len(rows)} · Còn {len(rows)-group_done}')
    st.progress(group_done/len(rows) if rows else 0)
    flash=st.session_state.pop('dataset_flash',None)
    if flash:getattr(st,flash[0])(flash[1])
    if st.session_state['dataset_pending'] and subset!='Cần kiểm tra độc lập':
        rows=[row for row in rows if row['id_mau'] not in reviewed or row['id_mau']==st.session_state.get('dataset_review_pin')]
    if not rows:
        st.info('Không còn mẫu phù hợp bộ lọc. Chọn Tất cả hoặc Cần kiểm tra độc lập để xem tiếp.');return
    ids=[row['id_mau'] for row in rows]
    if st.session_state.get('dataset_sample_choice') not in ids:st.session_state['dataset_sample_choice']=ids[0]
    current=st.session_state['dataset_sample_choice'];index=ids.index(current)
    previous,next_=st.columns(2)
    previous.button('← Mẫu trước',disabled=index==0,on_click=_select,args=(store,reviewer,ids[max(index-1,0)]))
    next_.button('Mẫu sau →',disabled=index==len(ids)-1,on_click=_select,args=(store,reviewer,ids[min(index+1,len(ids)-1)]))
    identifier=st.selectbox('Mẫu',ids,key='dataset_sample_choice',on_change=_select,args=(store,reviewer))
    st.session_state['dataset_selected']=identifier
    _checkpoint(store,reviewer)
    try:email=store.open_sample(identifier)
    except Exception:
        st.error('Chưa có nội dung gốc trong cache local.');return
    show=redacted_preview if redact else lambda value:value
    st.subheader('Email gốc')
    st.text(show(email['subject']) or '(Không tiêu đề)')
    st.text('From: '+show(email.get('sender','')))
    st.text('To: '+show(email.get('receiver','')))
    st.text('Date: '+show(email.get('date','')))
    st.text_area('Nội dung gốc dạng plain text — liên kết không được tự mở',show(email['body']),height=370,disabled=True,key='dataset_body_'+identifier)
    with st.expander('Headers đầy đủ theo dữ liệu nguồn'):
        raw_headers=email.get('raw_header_block','')
        if raw_headers:
            st.code(show(raw_headers),language=None)
        else:
            st.caption('Nguồn hiện chỉ lưu From/To/Date đã parse; không có full raw headers. Không suy ra xác thực người gửi.')
        if email.get('raw_header_availability')=='publisher_header_subset':
            st.caption('Chỉ có các trường do nhà công bố giữ lại; không có bản SMTP/MIME headers đầy đủ để kiểm chứng xác thực.')
    with st.expander('Nhãn gốc và bằng chứng nguồn'):
        st.text(f'Nguồn: {email["nguon"]} · Raw label: {email["nhan_goc"] or "không có"} · Ngôn ngữ gợi ý: {email["ngon_ngu_goi_y"]}')
        st.text('Source path: '+email.get('original_path',''))
        source_metadata=email.get('source_metadata',{})
        if isinstance(source_metadata,dict) and source_metadata:
            allowed={'original_path','source_path','source_part','source_archive','source_archive_sha256',
                'raw_sha256','raw_content_sha256','source_version','license','attribution','data_origin',
                'label_provenance','approval_blocker','use_basis','source_text_unmodified'}
            visible={key:value for key,value in source_metadata.items() if key in allowed}
            st.text(show(json.dumps(visible,ensure_ascii=False,indent=2)))
        if email.get('raw_source_verification'):
            st.text(show(json.dumps(email['raw_source_verification'],ensure_ascii=False,indent=2)))
        try:origin=source_evidence(ROOT,identifier,email['subject'],email['body'])
        except (OSError,ValueError,KeyError,TypeError):origin=None
        if origin:
            for field in ('source_url','archive_sha256','raw_email_sha256','research_evidence_url','research_evidence_sha256'):
                st.text(f'{field}: {origin[field]}')
            st.caption('Raw Spam là nhãn binary, chưa phân biệt Spam/Phishing ba lớp. Quyền tái phân phối NOT VERIFIED.')
        else:st.caption('Chưa đủ bằng chứng truy vết để phê duyệt mẫu này.')
    if identifier in independent:st.warning('Mẫu cần kiểm tra độc lập; không tự giải quyết chỉ bằng một nhãn mới hoặc đồng thuận AI.')
    st.caption('Nội dung không bị rút gọn. HTML/script, ảnh remote và attachment không được thực thi.')
    token=hashlib.sha256((str(scope)+identifier).encode()).hexdigest()[:20]
    keys={name:'review_'+name+'_'+token for name in ('label','language','attested','ambiguous','evidence','ai_exposed')}
    draft=store.draft(reviewer,identifier)
    for name,key in keys.items():
        if key not in st.session_state:st.session_state[key]=draft.get(name,{'label':'review','language':'unknown','attested':False,'ambiguous':False,'evidence':'','ai_exposed':False}[name])
    own=[row for row in email['decisions'] if row['nguoi_duyet']==reviewer]
    st.subheader('Nhận định của tôi')
    st.radio('Quyết định',['normal','spam','phishing','review','exclude'],format_func=lambda value:{'normal':'Normal','spam':'Spam','phishing':'Phishing','review':'Chưa chắc chắn','exclude':'Loại'}[value],
             horizontal=True,key=keys['label'],on_change=_draft_changed,args=(store,reviewer,identifier,keys))
    st.selectbox('Ngôn ngữ đã xác nhận',['unknown','en','vi','zh','mixed','other'],key=keys['language'],on_change=_draft_changed,args=(store,reviewer,identifier,keys))
    st.checkbox('Tôi có năng lực ngôn ngữ tương ứng và đã kiểm tra nội dung gốc',key=keys['attested'],on_change=_draft_changed,args=(store,reviewer,identifier,keys))
    st.checkbox('Spam/Phishing mơ hồ — cần kiểm tra độc lập',key=keys['ambiguous'],on_change=_draft_changed,args=(store,reviewer,identifier,keys))
    st.text_area('Lý do và bằng chứng (tránh chép thông tin cá nhân)',key=keys['evidence'],max_chars=8000,on_change=_draft_changed,args=(store,reviewer,identifier,keys))
    suggestion=suggestions.get(identifier)
    valid_suggestion=suggestion and suggestion['input_sha256']==input_digest(email['subject'],email['body'])
    if valid_suggestion:
        view_key='assist_visible_'+token
        reveal=st.checkbox('Hiện gợi ý hai mô hình local (tham khảo, có thể sai)',disabled=not bool(own),key=view_key)
        if not own:st.caption('Lưu nhận định đầu tiên trước khi mở gợi ý AI. Có thể chọn Chưa chắc chắn nếu thiếu căn cứ.')
        if own and reveal:
            st.session_state[keys['ai_exposed']]=True
            _draft_changed(store,reviewer,identifier,keys)
            left,right=st.columns(2)
            labels={'normal':'Normal','spam':'Spam','phishing':'Phishing',None:'Chưa có kết quả'}
            left.write('TF-IDF + LinearSVC');left.text(labels[suggestion['tfidf_suggested_label']])
            right.write('MiniLM + MLP');right.text(labels[suggestion['minilm_suggested_label']])
            for flag in suggestion['flags']:st.text(FLAG_TEXT[flag])
            st.caption('Đề xuất không phải ground truth hoặc xác suất đã hiệu chuẩn. Đồng thuận không phê duyệt nhãn.')
    save,next_save=st.columns(2)
    save.button('Lưu quyết định',on_click=_save_opinion,args=(store,reviewer,identifier,ids,keys,False,manifest.get('policy_version')))
    next_save.button('Lưu và chuyển email tiếp theo',type='primary',on_click=_save_opinion,args=(store,reviewer,identifier,ids,keys,True,manifest.get('policy_version')))
    st.caption('Bản nháp tự lưu khi thay đổi rời khỏi ô nhập; nút Lưu ghi nhận quyết định. Approved còn cần xác minh nhãn, quyền sử dụng, provenance và duplicate/leakage theo quy trình curation. Mẫu đồng thuận không được tự động duyệt.')
    with st.expander('Điều kiện trước khi Approved'):
        st.text('Nhãn: cần kiểm chứng quyết định người duyệt, taxonomy ba lớp và các trường hợp mơ hồ.')
        st.text('Quyền sử dụng / provenance: cần bằng chứng nguồn phù hợp với phạm vi nghiên cứu local.')
        st.text('Duplicate/leakage: phải hoàn tất group và protected-test guards trước admission.')
        st.text('Kiểm tra độc lập / chính sách nguồn: cần biên bản xác minh khi được yêu cầu.')
        st.caption('Công cụ này chỉ lưu nhận định và hàng đợi kiểm tra; không cấp Approved từ checkbox hoặc nhãn AI.')
    with st.expander('Lịch sử quyết định của mẫu'):
        if own:st.dataframe([{key:value for key,value in row.items() if key!='id_mau'} for row in email['decisions']],hide_index=True)
        else:st.caption('Lưu nhận định của bạn trước khi mở lịch sử để hạn chế ảnh hưởng từ lượt đánh giá trước.')


if __name__=='__main__':main()
