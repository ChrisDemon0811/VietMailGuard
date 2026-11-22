"""Bounded full-text triage over the existing 299 emails; no ML fit/predict or API."""
from pathlib import Path
import argparse
import csv
import json
import re
import sys
from collections import Counter,defaultdict
from contextlib import ExitStack
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.content_analyzer import analyze_content,load_security_config
from vietmailguard.sender_analyzer import analyze_sender
from vietmailguard.url_analyzer import analyze_urls
from vietmailguard.dataset_standardizer import normalize_text
from vietmailguard.v21.dedup_audit import dedup_hashes
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
from vietmailguard.v22.dataset_review import DatasetReview
from vietmailguard.v22.expansion import readonly
from vietmailguard.v22.quality import digest
from vietmailguard.v22.review_assistance import input_digest,load_assistance,source_evidence

POLICY='automated-review-299-v1'
CONTENT_ALERTS={'credential_request','account_suspension','financial_bait','prize_winner','threat'}
URL_ALERTS={'obfuscated_url','ip_host','missing_hostname','suspicious_token','excessive_subdomains','punycode','shortener','domain_pattern'}


def classify(*,tfidf,minilm,content_codes,url_codes,sender_codes,provenance_verified,research_use_verified,
             quality_flags=(),duplicate_flags=()):
    """A predeclared triage policy, not an independently validated label classifier."""
    codes=set(content_codes);url_risk=set(url_codes)&URL_ALERTS
    model_labels={value for value in (tfidf,minilm) if value}
    flags=set(quality_flags)|set(duplicate_flags)
    if tfidf is None or minilm is None:flags.add('model_unavailable')
    if tfidf and minilm and tfidf!=minilm:flags.add('models_disagree')
    if 'normal' in model_labels:flags.add('model_outside_source_binary_positive_family')
    if 'phishing' in model_labels:flags.add('model_phishing_proposal_unverified')
    if not provenance_verified:flags.add('source_provenance_missing')
    if not research_use_verified:flags.add('local_research_use_evidence_missing')
    # HTTP alone, an external domain, promotion, or a single word cannot imply phishing.
    credential_chain=('credential_request' in codes and bool(codes&{'account_suspension','urgency','threat','call_to_action'})
                      and bool(url_risk or sender_codes))
    financial_chain=({'financial_bait','prize_winner','call_to_action'}<=codes and bool(url_risk or sender_codes))
    phishing_chain=credential_chain or financial_chain
    commercial='commercial_promotion' in codes
    if phishing_chain:
        proposed='phishing';flags.add('multi_indicator_phishing_hypothesis_requires_context_review')
    elif 'phishing' in model_labels:
        proposed='uncertain';flags.add('phishing_proposal_not_corroborated_by_rule_chain')
    elif commercial:
        proposed='spam'
    elif tfidf==minilm=='normal' and not codes&CONTENT_ALERTS:
        proposed='normal';flags.add('normal_has_model_support_only_no_verified_normal_rule')
    elif tfidf==minilm=='spam':
        proposed='spam';flags.add('spam_agreement_without_independent_content_support')
    else:
        proposed='uncertain';flags.add('insufficient_three_class_evidence')
    if codes&CONTENT_ALERTS:flags.add('sensitive_content_requires_context_review')
    if url_risk:flags.add('suspicious_url_indicators_not_reputation_verdict')
    if sender_codes:flags.add('sender_consistency_indicators_require_review')
    hard_failure=bool(set(quality_flags)&{'empty_body','input_hash_mismatch','source_content_mismatch'})
    if hard_failure:status='Excluded Candidate'
    elif (not flags and commercial and tfidf==minilm=='spam' and provenance_verified and research_use_verified):
        status='Auto-reviewed'
    else:status='Needs Human Review'
    return {'ai_suggested_label':proposed,'status':status,'flags':sorted(flags),
        'evidence_strength':'moderate_for_triage_only' if status=='Auto-reviewed' else 'limited_requires_verification',
        'confidence_probability':None,'human_verified':False,'approved':False,'training_eligible':False,
        'ai_label_status':'weak_label' if proposed in {'normal','spam','phishing'} else 'review',
        'source_policy_consideration_only':status=='Auto-reviewed' and proposed=='spam'}


def full_text_findings(email,rules):
    """Scan every subject/body character; export references, never raw quotations or URLs."""
    text=email['subject']+'\n'+email['body']
    content=analyze_content(text,rules['content_rules'],detected_language=email.get('ngon_ngu_goi_y','unknown'))
    urls=analyze_urls(text,rules['url_rules'])
    sender=analyze_sender(email.get('sender',''),rules['sender_rules'])
    observed={rule for finding in content['findings'] for rule in finding['rule_ids']}
    normalized=normalize_text(text,'NFKC')
    spans=[]
    for language,language_rules in rules['content_rules']['languages'].items():
        for category,patterns in language_rules['patterns'].items():
            for pattern in patterns:
                if pattern['id'] not in observed:continue
                for match in re.finditer(pattern['regex'],normalized,re.I):
                    spans.append({'rule_id':pattern['id'],'category':category,'field':'normalized_subject_and_body',
                                  'coordinate_space':'NFKC_whitespace_normalized_full_text',
                                  'start':match.start(),'end':match.end(),'language':language})
    return {'content_codes':sorted({x['code'] for x in content['findings']}),
        'url_codes':sorted({finding['code'] for url in urls for finding in url['findings']}),
        'sender_codes':sorted({x['code'] for x in sender['findings']}),
        'rule_ids':sorted(observed),'span_references':spans,'url_count':len(urls),
        'subject_characters_scanned':len(email['subject']),'body_characters_scanned':len(email['body']),
        'analysis_truncated':False}


def write_csv(path,rows):
    if not rows:raise ValueError('CSV schema requires at least one template record')
    fields=list(rows[0])
    temporary=path.with_suffix('.csv.partial')
    with temporary.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,ensure_ascii=False,sort_keys=True) if isinstance(value,(list,dict)) else value
                             for key,value in row.items()})
    temporary.replace(path)


def run(collection,output):
    allowed=(ROOT/'data/v22/spam_evidence_resolution/automated_review_299').resolve()
    output=output.resolve()
    if not allowed.is_relative_to((ROOT/'data/v22').resolve()) or (output!=allowed and not output.is_relative_to(allowed)):
        raise ValueError('Keep review outputs in the separate V2.2 automated-review area')
    output.mkdir(parents=True,exist_ok=True)
    with offline_boundary(ROOT),ExitStack() as stack:
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
        from sklearn.neural_network import MLPClassifier
        def forbidden(*args,**kwargs):raise AssertionError('Model calls are forbidden; cached suggestions only')
        stack.enter_context(patch.object(joblib,'load',forbidden))
        for cls in (Pipeline,LinearSVC,MLPClassifier):
            stack.enter_context(patch.object(cls,'predict',forbidden));stack.enter_context(patch.object(cls,'fit',forbidden))
        store=DatasetReview(ROOT,collection=collection)
        db_before=digest(store.path);progress_before=store.progress()
        loaded=load_assistance(store.path.parent)
        if loaded is None:raise ValueError('Existing verified model suggestions are required')
        assistance,model_rows=loaded
        if len(model_rows)!=299:raise ValueError('Use exactly the existing 299-email collection')
        rules_path=ROOT/'config/security_rules_v2.json';rules=load_security_config(rules_path)
        screening=ROOT/'data/v22/spam_evidence_resolution/screening_summary.json'
        reservations=json.loads((collection/'reservation.json').read_text(encoding='utf-8'))
        reserved=set(reservations['selected_groups_reserved_from_training'])
        groups={};fingerprints=defaultdict(list);rows=[]
        source=ROOT/'data/v22/large_scale_expansion_01'
        receipts=json.loads((source/'download_receipts.json').read_text(encoding='utf-8'))
        for archive in ('20030228_spam.tar.bz2','20050311_spam_2.tar.bz2'):
            if digest(source/'raw'/archive)!=receipts[archive]['sha256']:raise ValueError('Archive integrity mismatch')
        with readonly(ROOT/'data/v21/cache/dedup_audit.sqlite3') as guard:
            for metadata in store.list_samples(only_pending=False):
                identifier=metadata['id_mau'];email=store.open_sample(identifier);cached=model_rows[identifier]
                quality=[];duplicates=[]
                if not email['body'].strip():quality.append('empty_body')
                if '\ufffd' in email['subject']+email['body']:quality.append('decode_replacement_requires_review')
                if cached['input_sha256']!=input_digest(email['subject'],email['body']):quality.append('input_hash_mismatch')
                try:evidence=source_evidence(ROOT,identifier,email['subject'],email['body'])
                except ValueError:evidence=None;quality.append('source_content_mismatch')
                provenance=bool(evidence and evidence['original_path'] and evidence['archive_sha256'] and evidence['raw_email_sha256'])
                rights=bool(evidence and evidence['research_use_evidence_verified'])
                if metadata['id_nhom'] not in reserved:duplicates.append('group_reservation_missing')
                if metadata['id_nhom'] in groups:duplicates.append('shared_selected_group_requires_review')
                groups[metadata['id_nhom']]=identifier
                exact,normal,template=dedup_hashes(email['subject'],email['body'])
                for kind,value,column in (('exact',exact,'exact_hash'),('normalized',normal,'normalized_hash'),('template',template,'template_hash')):
                    if not value:continue
                    fingerprints[(kind,value)].append(identifier)
                    # Fingerprints/dataset membership only, never held-out text or labels.
                    if guard.execute(f"SELECT 1 FROM rows WHERE {column}=? AND dataset='v2_reference' LIMIT 1",(value,)).fetchone():
                        duplicates.append('old_v2_reference_'+kind+'_overlap')
                finding=full_text_findings(email,rules)
                result=classify(tfidf=cached['tfidf_suggested_label'],minilm=cached['minilm_suggested_label'],
                    content_codes=finding['content_codes'],url_codes=finding['url_codes'],sender_codes=finding['sender_codes'],
                    provenance_verified=provenance,research_use_verified=rights,quality_flags=quality,duplicate_flags=duplicates)
                rows.append({'id':identifier,'source_label':metadata['nhan_goc'],'source_label_semantics':'binary_positive_not_verified_three_class',
                    'tfidf_prediction':cached['tfidf_suggested_label'],'minilm_prediction':cached['minilm_suggested_label'],
                    **result,'supporting_evidence':finding,'confidence_reason':
                        'Source binary label + full-text commercial indicator + local-model support; screening only, not verified correctness'
                        if result['status']=='Auto-reviewed' else 'Unresolved source/model/content/quality evidence; inspect flags and full-text references',
                    'model_input_was_truncated':('input_truncated' in cached['flags']),
                    'source_literal_prediction_difference':any(value and value!='spam' for value in (cached['tfidf_suggested_label'],cached['minilm_suggested_label'])),
                    'source_provenance_verified':provenance,'local_research_use_evidence_verified':rights,
                    'redistribution_permission':'NOT VERIFIED','constraint_group_id':metadata['id_nhom'],
                    'input_sha256':input_digest(email['subject'],email['body']),
                    'archive_sha256':evidence['archive_sha256'] if evidence else '',
                    'raw_email_sha256':evidence['raw_email_sha256'] if evidence else '',
                    'original_path':evidence['original_path'] if evidence else '',
                    'missing_fields':[field for field in ('subject','body','sender','receiver','date') if not str(email.get(field,'')).strip()],
                    'confidence_semantics':'qualitative_triage_support_not_calibrated_probability',
                    'admission_blockers':['three_class_ground_truth_unverified','source_rule_not_approved','complete_near_and_protected_group_guards_pending'],
                    'split':'unassigned','review_scope':'selected_canonical_only_not_all_group_members'})
        overlaps=[{'kind':kind,'ids':ids} for (kind,_),ids in fingerprints.items() if len(ids)>1]
        for overlap in overlaps:
            for row in rows:
                if row['id'] in overlap['ids']:
                    row['flags'].append('selected_'+overlap['kind']+'_duplicate_group')
                    row['status']='Needs Human Review';row['source_policy_consideration_only']=False
                    row['evidence_strength']='limited_requires_verification'
        for row in rows:
            flags=set(row['flags'])
            row['needs_human_review']=row['status']=='Needs Human Review'
            row['review_priority']=('P1' if flags & {'model_phishing_proposal_unverified','multi_indicator_phishing_hypothesis_requires_context_review'} else
                'P2' if flags & {'models_disagree','model_outside_source_binary_positive_family','source_provenance_missing','local_research_use_evidence_missing'} else
                'P3' if flags & {'sensitive_content_requires_context_review','suspicious_url_indicators_not_reputation_verdict','sender_consistency_indicators_require_review'} else 'P4')
            if flags:row['confidence_reason']+='; '+ '; '.join(sorted(flags))
        assert len(rows)==len({row['id'] for row in rows})==299
        write_csv(output/'AUTOMATED_REVIEW_299_RESULTS.csv',rows)
        human=sorted((row for row in rows if row['status']=='Needs Human Review'),key=lambda row:(row['review_priority'],row['id']))
        if human:write_csv(output/'HUMAN_REVIEW_REQUIRED.csv',human)
        else:
            with (output/'HUMAN_REVIEW_REQUIRED.csv').open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerow(list(rows[0]))
        assert store.progress()==progress_before and digest(store.path)==db_before
        summary={'policy':POLICY,'date':'2026-11-20','rows_automatically_assessed':299,
            'proposed_labels':{label:sum(row['ai_suggested_label']==label for row in rows) for label in ('normal','spam','phishing','uncertain')},
            'statuses':{status:sum(row['status']==status for row in rows) for status in ('Auto-reviewed','Needs Human Review','Excluded Candidate')},'human_review_required':len(human),
            'source_policy_consideration_only':sum(row['source_policy_consideration_only'] for row in rows),
            'actually_approved_by_valid_process_in_this_task':0,'new_human_decisions':0,
            'flag_counts':dict(Counter(flag for row in rows for flag in row['flags'])),
            'human_priority_counts':dict(Counter(row['review_priority'] for row in human)),
            'missing_optional_metadata_fields':dict(Counter(field for row in rows for field in row['missing_fields'])),
            'full_subject_characters_scanned':sum(row['supporting_evidence']['subject_characters_scanned'] for row in rows),
            'full_body_characters_scanned':sum(row['supporting_evidence']['body_characters_scanned'] for row in rows),
            'existing_truncated_model_inputs':sum(row['model_input_was_truncated'] for row in rows),
            'internal_fingerprint_duplicate_groups':overlaps,'near_recall_independently_verified':False,
            'model_inference_calls':0,'network_access_allowed':False,'final_test_opened':False,
            'new_train_pseudo_labels':0,'validation_or_test_admissions':0,'human_decisions_at_start':progress_before['decision_events'],
            'source_cache_manifest_sha256':digest(collection/'assistance/manifest.json'),
            'security_rules_sha256':digest(rules_path),'existing_group_screening_sha256':digest(screening),
            'queue_sha256':digest(collection/'review_queue.csv'),'content_sha256':digest(collection/'content.jsonl'),
            'output_sha256':{p.name:digest(p) for p in output.glob('*.csv')}}
        (output/'AUTOMATED_REVIEW_299_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',type=Path,default=ROOT/'data/v22/spam_evidence_resolution/review_collection')
    parser.add_argument('--output',type=Path,default=ROOT/'data/v22/spam_evidence_resolution/automated_review_299')
    args=parser.parse_args()
    run(args.collection.resolve(),args.output.resolve())
