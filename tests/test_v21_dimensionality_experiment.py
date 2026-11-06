"""Synthetic-only reduction tests: train-only fitting, sparse safety, integrity."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from vietmailguard.v21 import dimensionality_experiment as dr
from vietmailguard.v21.development_training import make_tfidf_pipeline
from vietmailguard.v21.paths import resolve_v21_paths

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path):
    paths = resolve_v21_paths(project_root=tmp_path,environ={})
    config = json.loads((ROOT/'config/v21/experiments/dimensionality_reduction_02.json').read_text())
    paths.config.joinpath('experiments').mkdir(parents=True)
    (paths.config/'experiments/dimensionality_reduction_02.json').write_text(json.dumps(config))
    return paths,config


def test_declared_grid_and_contract(tmp_path):
    paths,config = configuration(tmp_path)
    assert dr.load_config(paths)['svd_components']==[100,200,300]
    assert config['pca_components']==[50,100,200]
    for key in ('download_allowed','final_test_allowed','runtime_modified'):
        bad = {**config,key:True}
        (paths.config/'experiments/dimensionality_reduction_02.json').write_text(json.dumps(bad))
        with pytest.raises(dr.ExperimentBlocked):
            dr.load_config(paths)
    with pytest.raises(dr.ExperimentBlocked,match='classes minus one'):
        dr.make_reducer('lda',3,config)


def test_pca_mean_uses_train_only(tmp_path):
    _,config = configuration(tmp_path)
    generator = np.random.RandomState(42)
    train = generator.normal(size=(24,6)).astype('float32')
    validation = generator.normal(size=(8,6)).astype('float32')+100
    pca = dr.make_reducer('pca',2,config)
    before = validation.copy()
    transformed_train,transformed_validation,*_ = dr.fit_projection(pca,train,validation,np.array(['normal']*24))
    np.testing.assert_allclose(pca.mean_,train.mean(axis=0),atol=1e-7)
    np.testing.assert_array_equal(validation,before)
    np.testing.assert_allclose(transformed_validation,pca.transform(validation))
    assert transformed_train.shape==(24,2)


def test_lda_receives_only_training_labels(tmp_path):
    _,config = configuration(tmp_path)
    generator = np.random.RandomState(42)
    train,validation = generator.normal(size=(30,6)),generator.normal(size=(12,6))+10
    labels = np.tile(dr.CLASSES,10)
    class ObservedLDA(LinearDiscriminantAnalysis):
        def fit(self,x,y):
            assert x is train
            np.testing.assert_array_equal(y,labels)
            return super().fit(x,y)
    lda = ObservedLDA(n_components=2)
    z_train,z_validation,*_ = dr.fit_projection(lda,train,validation,labels)
    assert z_train.shape==(30,2) and z_validation.shape==(12,2)
    assert lda.n_features_in_==6


def test_svd_never_densifies_original_matrix(tmp_path,monkeypatch):
    _,config = configuration(tmp_path)
    generator = np.random.RandomState(42)
    train = sparse.random(40,60,density=.2,random_state=generator,dtype=np.float32,format='csr')
    validation = train[:8].copy()
    def forbidden(*args,**kwargs):
        pytest.fail('Full sparse TF-IDF must never be converted with toarray')
    monkeypatch.setattr(sparse.csr_matrix,'toarray',forbidden)
    z_train,z_validation,*_ = dr.fit_projection(dr.make_reducer('svd',3,config),train,validation,None)
    assert z_train.shape==(40,3) and z_validation.shape==(8,3)
    loss = dr.class_energy(validation,z_validation,np.array(['normal']*8))
    assert loss['spam'] is None and 0<=loss['normal']<=1


def test_resource_gate_blocks_without_reducing_components(tmp_path,monkeypatch):
    _,config = configuration(tmp_path)
    monkeypatch.setattr(dr,'memory_snapshot',lambda:{'free_ram_gib':.05})
    with pytest.raises(dr.ExperimentBlocked,match='RAM gate'):
        dr.require_ram(dr.workspace_bytes('svd',31304,140000,300,6709),config)
    assert config['svd_components']==[100,200,300]


def test_save_refuses_overwrite_and_reload_matches(tmp_path):
    from sklearn.svm import LinearSVC
    model = LinearSVC().fit([[0,0],[1,1],[2,0]],dr.CLASSES)
    path = tmp_path/'candidate.joblib'
    loaded = dr.save_artifact(path,model)
    np.testing.assert_array_equal(loaded.predict([[1,1]]),model.predict([[1,1]]))
    with pytest.raises(dr.ExperimentBlocked,match='overwrite'):
        dr.save_artifact(path,model)


def synthetic_development(tmp_path):
    paths,config = configuration(tmp_path)
    config.update(svd_components=[2],pca_components=[2],transform_batch_rows=4,plot_max_rows_per_class=5)
    config['encoder']['dimension']=4
    (paths.config/'experiments/dimensionality_reduction_02.json').write_text(json.dumps(config))
    folder = paths.splits/'development_bootstrap'
    folder.mkdir(parents=True)
    rows = {}
    for split,count in (('train',8),('validation',5)):
        rows[split] = []
        for label in dr.CLASSES:
            for index in range(count):
                identifier = f'{split}_{label}_{index}'
                rows[split].append({'id':identifier,'subject':f'{label} update {index}',
                    'body':f'{label} synthetic message {index} {split}', 'source':'synthetic_fixture',
                    'language':'en','data_origin':'synthetic','label':label,'label_status':'eligible',
                    'label_provenance':'independent_test_fixture','training_eligible':'True','curation_rule_id':'fixture',
                    'content_hash':f'content_{identifier}','normalized_content_hash':f'norm_{identifier}',
                    'template_hash':f'template_{identifier}','final_group_id':f'group_{identifier}','parent_id':'','split':split})
        with (folder/f'{split}.csv').open('w',encoding='utf-8',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[split][0]))
            writer.writeheader()
            writer.writerows(rows[split])
    paths.processed.mkdir(parents=True)
    master = paths.processed/'Development_Master_Dataset.csv'
    with master.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows['train'][0]))
        writer.writeheader()
        writer.writerows(rows['train']+rows['validation'])
    manifest = {'mode':'development_bootstrap','development_training_ready':True,'final_evaluation_ready':False,
        'final_test_path':None,'v2_test_rows_exported':0,'leakage':{'passed':True},
        'output_rows':{'train':24,'validation':15,'master':39},
        'output_sha256':{**{split:dr.sha256_file(folder/f'{split}.csv') for split in rows},'master':dr.sha256_file(master)}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    _,train,validation = dr.verify_and_load_development(paths,4096)
    params = {'seed':42,'tfidf':{
        'word':{'ngram_range':[1,2],'min_df':1,'max_features':100,'sublinear_tf':True,'dtype':'float32'},
        'character':{'analyzer':'char_wb','ngram_range':[3,5],'min_df':1,'max_features':100,'sublinear_tf':True,'dtype':'float32'},
        'svm':{'C':1,'class_weight':'balanced','max_iter':1000}}}
    baseline = make_tfidf_pipeline(params).fit(train[['subject','body']],train.label)
    paths.model.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    artifact = paths.model/'frozen_baseline.joblib'
    joblib.dump(baseline,artifact)
    metric_path = paths.results/'frozen_validation.json'
    metric_path.write_text(json.dumps({'candidates':[{'candidate':'frozen','metrics':dr.evaluate(validation,baseline.predict(validation))}]}))
    training_config = paths.config/'frozen_training.json'
    training_config.write_text('{}')
    runtime = {'artifact':artifact.relative_to(tmp_path).as_posix(),'artifact_sha256':dr.sha256_file(artifact),
        'validation_metrics':metric_path.relative_to(tmp_path).as_posix(),'validation_metrics_sha256':dr.sha256_file(metric_path),
        'training_config':training_config.relative_to(tmp_path).as_posix(),'training_config_sha256':dr.sha256_file(training_config),
        'candidate_id':'frozen'}
    (paths.config/'runtime_model.json').write_text(json.dumps(runtime))
    return paths,config,artifact


def test_synthetic_end_to_end_reductions_cache_integrity_and_resume(tmp_path,monkeypatch):
    from test_v21_multilingual_dl_experiment import Encoder
    from vietmailguard.v21.multilingual_dl_experiment import embedding_matrix
    from sklearn.svm import LinearSVC
    from vietmailguard.embedding_modeling import EmbeddingClassifierArtifact

    paths,config,baseline = synthetic_development(tmp_path)
    original = baseline.read_bytes()
    monkeypatch.setattr(dr,'memory_snapshot',lambda:{'free_ram_gib':20,'rss_gib':.1,'process_peak_rss_gib':.1})
    prepared = dr.prepare(paths)
    assert prepared['tfidf']['status']=='PASS'
    _,train,validation,signature,_ = dr.load_development(paths,config)
    cache = {}
    matrices = {}
    for split,frame in (('train',train),('validation',validation)):
        matrices[split],cache[split] = embedding_matrix(frame,split,paths,{**config['encoder'],'batch_size_cpu':4},Encoder(),'cpu')
    old_results = paths.results/'experiments/multilingual_dl_01'
    old_results.mkdir(parents=True)
    (old_results/'embedding_verification.json').write_text(json.dumps({'fingerprint':signature,'cache':cache}))
    control = LinearSVC(random_state=42).fit(matrices['train'],train.label)
    wrapper = EmbeddingClassifierArtifact('control',control,config['encoder']['model_id'],config['encoder']['revision'],128,4,list(control.classes_),'synthetic_fixture')
    control_path = paths.model/'frozen_control.joblib'
    joblib.dump(wrapper,control_path)
    (old_results/'validation_metrics.json').write_text(json.dumps({'candidates':[{'candidate':'B_minilm_svc','status':'PASS',
        'artifact':control_path.relative_to(tmp_path).as_posix(),'artifact_sha256':dr.sha256_file(control_path),
        'metrics':dr.evaluate(validation,control.predict(matrices['validation']))}]}))
    # Summary source hashes refer only to code contracts, never to raw email.
    for name in ('src/vietmailguard/v21/dimensionality_experiment.py','scripts/v21/dimensionality_reduction_02.py',
                 'src/vietmailguard/v21/development_training.py','src/vietmailguard/v21/multilingual_dl_experiment.py',
                 'src/vietmailguard/modeling.py','src/vietmailguard/embedding_modeling.py'):
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# synthetic source identity fixture')
    names = dr.candidate_names(config)
    for name in names:
        assert dr.candidate(name,paths)['status']=='PASS'
    output = dr.summarize(paths)
    assert len(output['candidates'])==1+len(names)
    assert not output['runtime_changed'] and not output['heldout_opened']
    assert (tmp_path/'DIMENSIONALITY_REDUCTION_EXPERIMENT_02.md').exists()
    _,results = dr.directories(paths)
    assert (results/'lda_projection.png').exists()
    assert (results/'artifact_manifest.json').exists()
    assert baseline.read_bytes()==original
    monkeypatch.setattr(dr,'_fit_candidate',lambda *args:pytest.fail('Completed candidate must not refit'))
    assert dr.candidate('C_pca_2',paths)['reused_checkpoint']
    target = paths.cache/'experiments/dimensionality_reduction_02/train.npz'
    target.write_bytes(b'corrupt')
    with pytest.raises(dr.ExperimentBlocked,match='checksum'):
        dr.prepare_tfidf(paths,config,train,validation,joblib.load(baseline),signature)
