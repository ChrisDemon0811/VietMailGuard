"""Prepare one locked group split, then explicitly train the verified-only pilot."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import itertools
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from vietmailguard.v21.dedup_audit import dedup_hashes
from vietmailguard.v21.multilingual_dl_experiment import (
    offline_boundary, local_encoder_snapshot, make_classifier, evaluate, _validate_matrix,
)
from vietmailguard.v21.development_training import make_tfidf_pipeline, resource_snapshot
from vietmailguard.v22.group_pilot import Groups, normalized, pair_evidence, split_groups
from vietmailguard.v22.quality import digest
from vietmailguard.v22.recovery import comparison_text, exact_jaccard

NAME = 'group_aware_supervised_pilot_01'
CONFIG = ROOT/'config/v22'/f'{NAME}.json'
AREA = ROOT/'data/v22'/NAME
RESULTS = ROOT/'results/v22_multilingual/experiments'/NAME
MODELS = ROOT/'models/v22_multilingual/experiments'/NAME


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text=json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if immutable and path.exists():
        if path.read_text(encoding='utf-8')!=text: raise ValueError('Locked evidence differs: '+path.name)
        return
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def csv_write(path, records, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ValueError('Refusing to overwrite experiment CSV: '+path.name)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)


def preserve_check():
    before=read(ROOT/'.local-test-deps/v22/group-aware-pilot-01/before.json')
    for path,expected in before.items():
        if digest(ROOT/path)!=expected: raise ValueError('Protected input changed: '+path)
    return len(before)


def load_original(config):
    source=ROOT/config['input'];manifest=read(source/'cohort/manifest.json');rows=[]
    for split in ('train','validation'):
        path=source/'cohort'/f'{split}.csv'
        assert digest(path)==manifest['output_sha256'][split]
        with path.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                assert row['split']==split and row['training_eligible'].casefold()=='true'
                assert row['label_status'] in ('eligible','evidence_confirmed')
                assert all(row[k] for k in ('label_provenance','curation_rule_id','v22_owner_decision','v22_use_permission_basis'))
                rows.append(row)
    assert len(rows)==9686 and len({r['id'] for r in rows})==9686
    return rows


def prepare(config):
    if (AREA/'cohort/manifest.json').exists():
        verify_locked(config)
        print(json.dumps({'stage':'already_locked','manifest_sha256':digest(AREA/'cohort/manifest.json')}),flush=True)
        return
    protected=preserve_check();rows=load_original(config);source=ROOT/config['input']
    prior=read(source/'run_manifest.json');by_id={r['id']:i for i,r in enumerate(rows)}
    groups=Groups(len(rows));hard=[];hash_counts={};label_conflicts=[]

    def link(a,b,kind):
        groups.union(a,b);hard.append((a,b,kind))

    def equivalent(kind, keys, full_input=False):
        table=defaultdict(list)
        for i,key in enumerate(keys):
            if key:table[key].append(i)
        linked=[]
        for key,indices in table.items():
            if len(indices)<2:continue
            if full_input and len({rows[i]['label'] for i in indices})>1:
                label_conflicts.append({'kind':kind,'ids':[rows[i]['id'] for i in indices]})
            for i in indices[1:]:link(indices[0],i,kind)
            linked.append(indices)
        hash_counts[kind]={'groups':len(linked),'rows':sum(map(len,linked))}
        return linked

    equivalent('original_group',[r['final_group_id'] for r in rows])
    parent_links=0
    for i,row in enumerate(rows):
        if not row['parent_id']:continue
        parent=by_id[row['parent_id']]
        assert rows[parent]['label']==row['label'] and rows[parent]['final_group_id']==row['final_group_id']
        link(i,parent,'parent_translation');parent_links+=1
    for cap in (None,4096):
        values=[]
        for r in rows:
            s,b=r['subject'],r['body']
            if cap:s,b=s[:cap],b[:cap]
            values.append((*dedup_hashes(s,b)[:2],*dedup_hashes('',b)[:2]))
        for col,key in enumerate(('full_exact','full_normalized','body_exact','body_normalized')):
            equivalent(('whole_' if cap is None else 'effective4096_')+key,[v[col] for v in values],full_input=col<2)

    # Reuse the already exhaustive whole-content audit with all input/output hashes.
    candidates=defaultdict(dict)
    for view in ('body_normalized','full_normalized','body_template','full_template'):
        cache=source/f'{view}_jaccard_pairs.jsonl';meta=read(cache.with_suffix('.meta.json'))
        assert meta['signature']['cohort_sha256']==digest(source/'cohort/manifest.json')
        assert meta['signature']['threshold']==config['relations']['jaccard_cutoff']
        assert digest(cache)==meta['pairs_sha256']==prior['output_sha256'][cache.relative_to(ROOT).as_posix()]
        assert meta['unordered_pairs_covered']==len(rows)*(len(rows)-1)//2
        with cache.open(encoding='utf-8') as f:
            for line in f:
                a,b,score=json.loads(line);candidates[tuple(sorted((a,b)))][view]=score
    from transformers import AutoTokenizer
    encoder_cfg=read(ROOT/config['mlp_config'])['encoder']
    tokenizer=AutoTokenizer.from_pretrained(local_encoder_snapshot(encoder_cfg),local_files_only=True,trust_remote_code=False)
    token_keys=[]
    for start in range(0,len(rows),32):
        texts=[(r['subject'][:4096]+'\n'+r['body'][:4096]).strip() for r in rows[start:start+32]]
        encoded=tokenizer(texts,padding=False,truncation=True,max_length=128)['input_ids']
        token_keys.extend(hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in encoded)
    token_groups=equivalent('token128_input_collision',token_keys)
    token_pairs={tuple(sorted(pair)) for members in token_groups for pair in itertools.combinations(members,2)}
    for pair in token_pairs:candidates[pair]
    old707=set()
    with (source/'cross_split_pair_evidence.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            pair=tuple(sorted((by_id[r['train_id']],by_id[r['validation_id']])))
            old707.add(pair);candidates[pair]
    assert len(old707)==707
    texts={}
    for view in ('body_normalized','full_normalized','body_template','full_template'):
        values=[]
        for r in rows:
            value=r['body'] if view.startswith('body') else r['subject']+'\n'+r['body']
            values.append(comparison_text(value) if view.endswith('template') else normalized(value))
        texts[view]=values
    relations=[]
    for (a,b),scores in sorted(candidates.items()):
        for view,values in texts.items():
            if view not in scores:scores[view]=exact_jaccard(values[a],values[b])
        evidence=pair_evidence(rows[a],rows[b],scores,config['relations'])
        if evidence['must_group_for_content']:link(a,b,evidence['category'])
        relations.append({'left_id':rows[a]['id'],'right_id':rows[b]['id'],
            'in_original_707':(a,b) in old707,'tokenizer_equal':(a,b) in token_pairs,**evidence})
    del texts,tokenizer;gc.collect()
    group_ids=groups.identifiers([r['id'] for r in rows])
    assignment=split_groups(rows,group_ids,config['splitting'])
    high_cross=[{'left_id':rows[a]['id'],'right_id':rows[b]['id'],'kind':kind}
                for a,b,kind in hard if assignment[rows[a]['id']]!=assignment[rows[b]['id']]]
    assert not high_cross
    if label_conflicts:
        write(AREA/'label_conflicts.json',{'status':'BLOCKED','conflicts':label_conflicts})
        raise ValueError('Identical complete input has conflicting verified labels; training blocked')
    all_group_roles=defaultdict(set)
    for i,r in enumerate(rows):all_group_roles[group_ids[i]].add(assignment[r['id']])
    assert all(len(x)==1 for x in all_group_roles.values())
    for relation in relations:
        relation['new_cross_split']=assignment[relation['left_id']]!=assignment[relation['right_id']]
    warnings_cross=[r for r in relations if r['new_cross_split'] and not r['must_group_for_content']]
    counts={s:dict(Counter(r['label'] for r in rows if assignment[r['id']]==s)) for s in ('train','validation')}
    group_counts={s:len({group_ids[i] for i,r in enumerate(rows) if assignment[r['id']]==s}) for s in ('train','validation')}
    group_rows=[]
    fields=list(rows[0])+['experiment_group_id','experiment_split']
    for s in ('train','validation'):
        selected=[]
        for i,r in enumerate(rows):
            if assignment[r['id']]!=s:continue
            selected.append({**r,'experiment_group_id':group_ids[i],'experiment_split':s})
        selected.sort(key=lambda r:r['id'])
        csv_write(AREA/'cohort'/f'{s}.csv',selected,fields)
    for i,r in enumerate(rows):
        group_rows.append({k:r[k] for k in ('id','split','label','language','data_origin','parent_id','final_group_id','label_provenance','curation_rule_id','v22_owner_decision','v22_use_permission_basis')}
            |{'experiment_group_id':group_ids[i],'experiment_split':assignment[r['id']]})
    csv_write(AREA/'group_assignment.csv',group_rows,list(group_rows[0]))
    csv_write(AREA/'pair_classifications.csv',relations,list(relations[0]))
    old_counts=dict(Counter(r['category'] for r in relations if r['in_original_707']))
    warning_counts=dict(Counter(r['category'] for r in warnings_cross))
    audit={'status':'PASS','high_risk_cross_split_links':0,'exact_normalized_input_label_conflicts':0,
        'parent_links_checked':parent_links,'equivalence_checks':hash_counts,
        'whole_content_jaccard_views':4,'all_pairs_covered_per_view':len(rows)*(len(rows)-1)//2,
        'prior_707_pair_classifications':old_counts,'all_candidate_pair_classifications':dict(Counter(r['category'] for r in relations)),
        'tokenizer_feature_equal_pairs_grouped':len(token_pairs),
        'remaining_nonblocking_template_or_insufficient_evidence_pairs':len(warnings_cross),
        'remaining_warning_categories':warning_counts,'new_cross_split_warnings_are_human_verified_duplicates':False,
        'labels_approved_or_changed':0,'rows_deleted':0,'models_fitted_before_gate':0,
        'thresholds_selected_using_validation_metrics':False}
    write(AREA/'split_audit.json',audit,True)
    manifest={'schema_version':1,'experiment':NAME,'scope':config['scope'],'rows':len(rows),
        'split_rows':{s:sum(counts[s].values()) for s in counts},'split_class_rows':counts,
        'split_group_counts':group_counts,'groups':len(set(group_ids)),
        'split_language_rows':{s:dict(Counter(r['language'] for r in rows if assignment[r['id']]==s)) for s in counts},
        'rows_with_changed_experimental_assignment':sum(r['split']!=assignment[r['id']] for r in rows),
        'original_split_and_provenance_fields_unchanged':True,'original_rows_retained':9686,
        'source_manifest_sha256':digest(source/'cohort/manifest.json'),'config_sha256':digest(CONFIG),
        'assignment_sha256':digest(AREA/'group_assignment.csv'),'relations_sha256':digest(AREA/'pair_classifications.csv'),
        'audit_sha256':digest(AREA/'split_audit.json'),
        'output_sha256':{s:digest(AREA/'cohort'/f'{s}.csv') for s in counts},
        'supervised_development_training_ready':True,'final_evaluation_ready':False,'publication_approved':False,
        'weak_unlabeled_pseudo_rows':0}
    write(AREA/'cohort/manifest.json',manifest,True)
    versions={p:importlib.metadata.version(p) for p in ('scikit-learn','numpy','pandas','scipy','joblib','torch','sentence-transformers','transformers')}
    protocol={'experiment':NAME,'config':config,'manifest_sha256':digest(AREA/'cohort/manifest.json'),
        'split_audit_sha256':digest(AREA/'split_audit.json'),'versions':versions,
        'frozen_parameter_sha256':{key:digest(ROOT/config[key]) for key in ('tfidf_config','mlp_config')},
        'code_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in (Path(__file__),ROOT/'src/vietmailguard/v22/group_pilot.py')},
        'locked_before_training_and_validation_predictions':True,'protected_prior_files':protected,
        'encoder_fitted':False,'mlp_weights':'balanced_train_only','tuning':False,
        'prior_validation_reuse_disclosed':True,'final_evaluation_ready':False}
    write(RESULTS/'protocol.json',protocol,True)
    print(json.dumps({'stage':'PREPARED','manifest':manifest,'audit':audit}),flush=True)


def verify_locked(config):
    manifest=read(AREA/'cohort/manifest.json');protocol=read(RESULTS/'protocol.json')
    assert digest(CONFIG)==manifest['config_sha256'] and config==protocol['config']
    assert digest(AREA/'cohort/manifest.json')==protocol['manifest_sha256']
    assert digest(AREA/'split_audit.json')==manifest['audit_sha256']==protocol['split_audit_sha256']
    assert read(AREA/'split_audit.json')['status']=='PASS'
    assert digest(AREA/'group_assignment.csv')==manifest['assignment_sha256']
    assert digest(AREA/'pair_classifications.csv')==manifest['relations_sha256']
    for name,expected in protocol['code_sha256'].items():assert digest(ROOT/name)==expected
    for key,expected in protocol['frozen_parameter_sha256'].items():assert digest(ROOT/config[key])==expected
    for s,expected in manifest['output_sha256'].items():assert digest(AREA/'cohort'/f'{s}.csv')==expected
    return manifest,protocol


def train(config):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight
    from threadpoolctl import threadpool_limits
    from vietmailguard.embedding_modeling import EmbeddingClassifierArtifact, compose_email_texts
    from vietmailguard.v21.multilingual_dl_experiment import load_encoder

    manifest,protocol=verify_locked(config);preserve_check()
    frames={s:pd.read_csv(AREA/'cohort'/f'{s}.csv',keep_default_na=False) for s in ('train','validation')}
    original_frames={s:frame.copy() for s,frame in frames.items()}
    for frame in frames.values():
        for field in ('subject','body'):frame[field]=frame[field].astype(str).str.slice(0,config['max_input_characters_per_field'])
    resources=resource_snapshot()
    if resources.get('free_ram_gib',0)<.75:raise RuntimeError('Insufficient free RAM for the locked classifier pilot')
    state_path=RESULTS/'metrics.json'
    state=read(state_path) if state_path.exists() else {
        'experiment':NAME,'status':'RUNNING','protocol_sha256':digest(RESULTS/'protocol.json'),
        'manifest_sha256':digest(AREA/'cohort/manifest.json'),'resources':resources,
        'scope':config['scope'],'candidates':[],'final_test_opened':False,
        'weak_unlabeled_pseudo_rows_used':0,'publication_approved':False}
    assert state['protocol_sha256']==digest(RESULTS/'protocol.json')
    for candidate in state['candidates']:
        assert digest(ROOT/candidate['artifact'])==candidate['artifact_sha256']
    completed={c['candidate'] for c in state['candidates']}

    def metrics(prediction):
        result=evaluate(frames['validation'],np.asarray(prediction))
        result['weighted_f1']=float(f1_score(frames['validation'].label,prediction,average='weighted'))
        result['spam_recall']=result['per_class']['spam']['recall']
        return result

    def save(candidate,estimator,payload,prediction,fit_seconds,predict_seconds,extra):
        path=MODELS/f'{candidate}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('Refusing to overwrite a fitted candidate')
        joblib.dump(payload,path)
        write(path.with_suffix('.json'),{'experiment':NAME,'artifact_sha256':digest(path),
            'protocol_sha256':digest(RESULTS/'protocol.json'),'classes':config['classes'],
            'max_input_characters_per_field':4096,'input_composition':'subject + newline + body',
            'calibrated':False,'production':False,'final_evaluation_ready':False})
        prediction=np.asarray(prediction)
        record={'candidate':candidate,'artifact':path.relative_to(ROOT).as_posix(),
            'artifact_sha256':digest(path),'artifact_bytes':path.stat().st_size,
            'metrics':metrics(prediction),'fit_seconds':fit_seconds,
            'prediction_seconds':predict_seconds,'prediction_ms_per_email':1000*predict_seconds/len(prediction),
            'calibrated_probability_available':False,'training_rows':len(frames['train']),
            'validation_rows':len(frames['validation']),**extra}
        state['candidates'].append(record);write(state_path,state)
        records=[{'id':row.id,'experiment_group_id':row.experiment_group_id,
                  'experiment_split':'validation','language':row.language,'data_origin':row.data_origin,
                  'truth':row.label,'prediction':str(prediction[i])}
                 for i,row in frames['validation'].iterrows()]
        csv_write(RESULTS/f'{candidate}_predictions.csv',records,list(records[0]))
        labels=record['metrics']['confusion_matrix_order']
        csv_write(RESULTS/f'{candidate}_confusion_matrix.csv',
            [{'actual':label,**dict(zip(labels,row))} for label,row in zip(labels,record['metrics']['confusion_matrix'])],
            ['actual',*labels])
        print(json.dumps({'stage':'TRAINED','candidate':candidate,'fit_seconds':fit_seconds,
            'macro_f1':record['metrics']['macro_f1'],'spam_recall':record['metrics']['spam_recall'],
            'phishing_recall':record['metrics']['phishing_recall']}),flush=True)

    with threadpool_limits(limits=config['cpu_threads']):
        if 'tfidf_linearsvc' not in completed:
            estimator=make_tfidf_pipeline(read(ROOT/config['tfidf_config']))
            begin=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');estimator.fit(frames['train'],frames['train'].label.to_numpy())
            elapsed=time.perf_counter()-begin
            # The raw frame must be bounded by the same training preprocessing contract.
            begin=time.perf_counter()
            bounded=original_frames['validation'].copy()
            for field in ('subject','body'):bounded[field]=bounded[field].astype(str).str.slice(0,4096)
            prediction=estimator.predict(bounded);predict_seconds=time.perf_counter()-begin
            save('tfidf_linearsvc',estimator,estimator,prediction,elapsed,predict_seconds,
                {'inference_scope':'composer_and_sparse_vectorizer_and_classifier_on_bounded_input',
                 'device':'cpu','fit_warning_types':[type(w.message).__name__ for w in caught],
                 'feature_dimensions':sum(len(v.vocabulary_) for _,v in estimator.named_steps['tfidf'].transformer_list),
                 'encoder_included_in_timing':False})
            del estimator,bounded;gc.collect()

        if 'minilm_mlp' not in completed:
            dl=read(ROOT/config['mlp_config']);encoder_config=dl['encoder']
            now=resource_snapshot()
            if now.get('free_ram_gib',0)<encoder_config['min_free_ram_gib']:
                raise RuntimeError('Insufficient free RAM for the pinned local encoder')
            if now['cuda'] and now.get('free_vram_gib',0)<encoder_config['min_free_vram_gib']:
                raise RuntimeError('Insufficient free VRAM; no device fallback or model download')
            import torch
            torch.set_num_threads(config['cpu_threads'])
            device='cuda' if now['cuda'] else 'cpu'
            begin=time.perf_counter()
            matrices,cache_receipt=selected_cache(frames)
            cache_seconds=time.perf_counter()-begin
            begin=time.perf_counter();encoder=load_encoder(local_encoder_snapshot(encoder_config),encoder_config,device)
            encoder_load_seconds=time.perf_counter()-begin

            def encode(frame):
                return np.asarray(encoder.encode(compose_email_texts(frame).tolist(),
                    batch_size=encoder_config['batch_size_cuda'] if device=='cuda' else encoder_config['batch_size_cpu'],
                    normalize_embeddings=True,convert_to_numpy=True,device=device,show_progress_bar=False),dtype=np.float32)

            # Verify cached Train vectors against the actual pinned encoder, not labels.
            check_indices=np.linspace(0,len(frames['train'])-1,16,dtype=int)
            fresh=encode(frames['train'].iloc[check_indices]);cached=matrices['train'][check_indices]
            error=float(np.max(np.abs(fresh-cached)))
            cache_receipt['fresh_train_sample_rows']=16;cache_receipt['fresh_max_absolute_error']=error
            if not np.allclose(fresh,cached,atol=2e-4,rtol=2e-4):
                begin=time.perf_counter()
                matrices={s:encode(frame) for s,frame in frames.items()}
                cache_receipt['recomputed_with_pinned_encoder_seconds']=time.perf_counter()-begin
                cache_receipt['cache_used_for_classifier_fit']=False
            else:cache_receipt['cache_used_for_classifier_fit']=True
            for s,array in matrices.items():_validate_matrix(array,len(frames[s]),384)
            write(RESULTS/'embedding_cache_verification.json',cache_receipt)
            estimator=make_classifier('C_minilm_mlp',dl)
            weights=compute_sample_weight('balanced',frames['train'].label.to_numpy())
            begin=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                estimator.fit(matrices['train'],frames['train'].label.to_numpy(),classifier__sample_weight=weights)
            elapsed=time.perf_counter()-begin
            begin=time.perf_counter();cached_prediction=estimator.predict(matrices['validation']);head_seconds=time.perf_counter()-begin
            torch.cuda.synchronize() if device=='cuda' else None
            begin=time.perf_counter()
            bounded=original_frames['validation'].copy()
            for field in ('subject','body'):bounded[field]=bounded[field].astype(str).str.slice(0,4096)
            actual=encode(bounded);prediction=estimator.predict(actual)
            torch.cuda.synchronize() if device=='cuda' else None
            predict_seconds=time.perf_counter()-begin
            artifact=EmbeddingClassifierArtifact(experiment_id=NAME,classifier=estimator,
                model_id=encoder_config['model_id'],model_revision=encoder_config['revision'],
                max_sequence_length=128,embedding_dimension=384,label_order=config['classes'],
                training_scope=config['scope'])
            save('minilm_mlp',estimator,artifact,prediction,elapsed,predict_seconds,
                {'inference_scope':'bounded_preprocessing_and_frozen_encoder_and_scaler_and_mlp',
                 'device':device,'head_only_prediction_seconds':head_seconds,'encoder_included_in_timing':True,
                 'encoder_load_seconds_excluded_from_inference':encoder_load_seconds,
                 'embedding_cache_selection_seconds':cache_seconds,'cache_train_used':cache_receipt['cache_used_for_classifier_fit'],
                 'cached_and_live_validation_prediction_differences':int(np.sum(cached_prediction!=prediction)),
                 'fit_warning_types':[type(w.message).__name__ for w in caught],
                 'mlp_iterations':int(estimator.named_steps['classifier'].n_iter_),
                 'encoder_fit_seconds':0.,'feature_dimensions':384})
            del estimator,encoder,matrices;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
    state['status']='COMPLETE';state['protected_files_verified']=preserve_check()
    write(state_path,state)
    csv_write(RESULTS/'model_comparison.csv',[
        {'candidate':r['candidate'],'accuracy':r['metrics']['accuracy'],'macro_f1':r['metrics']['macro_f1'],
         'weighted_f1':r['metrics']['weighted_f1'],'spam_recall':r['metrics']['spam_recall'],
         'phishing_recall':r['metrics']['phishing_recall'],'fit_seconds':r['fit_seconds'],
         'inference_seconds':r['prediction_seconds'],'artifact_bytes':r['artifact_bytes']}
        for r in state['candidates']],['candidate','accuracy','macro_f1','weighted_f1','spam_recall',
            'phishing_recall','fit_seconds','inference_seconds','artifact_bytes'])


def selected_cache(frames):
    """A new group split may select rows from BOTH old frozen encoder caches."""
    import numpy as np
    receipt=read(ROOT/'results/v21_multilingual/experiments/multilingual_dl_01/embedding_verification.json')
    matrices={s:np.empty((len(f),384),dtype=np.float32) for s,f in frames.items()}
    wanted={row.id:(s,int(i),row) for s,frame in frames.items() for i,row in frame.iterrows()}
    found=set();sources={}
    for old_split in ('train','validation'):
        cache=receipt['cache'][old_split];source=ROOT/'data/v21/splits/development_bootstrap'/f'{old_split}.csv'
        path=ROOT/cache['matrix']
        assert digest(source)==receipt['fingerprint'][old_split+'_sha256']
        assert digest(path)==cache['sha256'] and digest(path.with_suffix('.json'))==cache['manifest_sha256']
        matrix=np.load(path,mmap_mode='r',allow_pickle=False);_validate_matrix(matrix,cache['rows'],384)
        selected=0
        with source.open(encoding='utf-8',newline='') as f:
            for position,r in enumerate(csv.DictReader(f)):
                if r['id'] not in wanted:continue
                s,i,approved=wanted[r['id']]
                assert r['id'] not in found
                assert r['content_hash']==approved.content_hash
                assert r['subject'][:4096]==approved.subject and r['body'][:4096]==approved.body
                matrices[s][i]=matrix[position];found.add(r['id']);selected+=1
        sources[old_split]={'source_sha256':digest(source),'matrix_sha256':cache['sha256'],
            'sidecar_sha256':cache['manifest_sha256'],'selected_rows':selected,'frozen_encoder':True}
    assert found==set(wanted)
    return matrices,{'sources':sources,'all_requested_ids_verified':len(found),
        'selection_uses_original_id_and_content_and_composed_text':True,'encoder_fit':False}


if __name__=='__main__':
    csv.field_size_limit(100_000_000)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true',help='Train only after an immutable verified split exists')
    args=parser.parse_args();cfg=read(CONFIG)
    with offline_boundary(ROOT):
        if args.train:train(cfg)
        else:prepare(cfg)
