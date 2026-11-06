"""Synthetic-only tests for the isolated frozen-encoder development experiment."""
import builtins
import hashlib
import json
from pathlib import Path
import socket

import joblib
import numpy as np
import pandas as pd
import pytest

from vietmailguard.v21 import multilingual_dl_experiment as dl
from vietmailguard.v21.paths import resolve_v21_paths

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    directory = paths.config / 'experiments'
    directory.mkdir(parents=True)
    config = json.loads((ROOT/'config/v21/experiments/multilingual_dl_01.json').read_text(encoding='utf-8'))
    (directory/'multilingual_dl_01.json').write_text(json.dumps(config), encoding='utf-8')
    return paths, config


class Encoder:
    def __init__(self, dimension=4, fail_call=None):
        self.calls = 0
        self.dimension = dimension
        self.fail_call = fail_call

    def encode(self, texts, **kwargs):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError('synthetic interruption')
        values = np.array([list(hashlib.sha256(text.encode()).digest()[:self.dimension]) for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def frame():
    return pd.DataFrame({'subject':['Meeting','Notice','Verify','Lịch họp','Ưu đãi'],
        'body':['Schedule tomorrow','Offer today','Password request','Ngày mai','Giảm giá'],
        'content_hash':['h1','h2','h3','h4','h5'],
        'label':['normal','spam','phishing','normal','spam'],
        'language':['en','en','en','vi','vi'],
        'data_origin':['not_declared']*3+['translated']*2})


def test_config_forbids_download_test_and_validation_early_stopping(tmp_path):
    paths, config = configuration(tmp_path)
    assert dl.load_config(paths)['encoder']['frozen']
    for key in ('download_allowed','final_test_allowed'):
        bad = {**config, key:True}
        (paths.config/'experiments/multilingual_dl_01.json').write_text(json.dumps(bad))
        with pytest.raises(dl.ExperimentBlocked):
            dl.load_config(paths)
    config['mlp']['early_stopping'] = True
    (paths.config/'experiments/multilingual_dl_01.json').write_text(json.dumps(config))
    with pytest.raises(dl.ExperimentBlocked):
        dl.load_config(paths)


def test_network_and_heldout_boundary(tmp_path):
    blocked = tmp_path/'data/splits/v2/test.csv'
    blocked.parent.mkdir(parents=True)
    blocked.write_text('synthetic sentinel')
    safe = tmp_path/'validation.csv'
    safe.write_text('allowed')
    with dl.offline_boundary(tmp_path):
        assert safe.read_text() == 'allowed'
        for opener in (builtins.open, lambda path:Path(path).open()):
            with pytest.raises(dl.ExperimentBlocked, match='Held-out'):
                opener(blocked)
        with pytest.raises(dl.ExperimentBlocked, match='Network'):
            socket.create_connection(('example.com',443))
        with pytest.raises(dl.ExperimentBlocked, match='Network'):
            socket.getaddrinfo('example.com',443)


@pytest.mark.parametrize('resources,allowed', [
    ({'free_ram_gib':0.5,'cuda':False},False),
    ({'free_ram_gib':3,'cuda':True,'free_vram_gib':0.5},False),
    ({'free_ram_gib':3,'cuda':True,'free_vram_gib':2},True),
    ({'free_ram_gib':3,'cuda':False},True),
])
def test_resource_gate_no_silent_substitution(tmp_path,resources,allowed):
    _, config = configuration(tmp_path)
    assert dl.encoder_gate(resources,config['encoder'])[0] is allowed


def test_absent_language_classes_are_na_not_fabricated():
    data = frame()
    result = dl.evaluate(data,data.label.to_numpy())
    assert result['macro_f1'] == 1
    assert result['by_language']['vi']['phishing_recall'] is None
    assert result['by_language']['vi']['per_class']['phishing']['support'] == 0
    assert result['by_language']['zh']['accuracy'] is None
    assert result['by_language']['vi']['macro_f1'] is None


def test_cache_reuse_order_and_readonly_existing_cache(tmp_path):
    paths, config = configuration(tmp_path)
    config = {**config['encoder'],'dimension':4,'batch_size_cpu':2}
    encoder = Encoder()
    matrix, meta = dl.embedding_matrix(frame(),'train',paths,config,encoder,'cpu')
    assert not meta['cache_hit'] and matrix.shape == (5,4)
    del matrix
    original = tmp_path/meta['matrix']
    shared = paths.cache/'e'/'old'
    shared.mkdir(parents=True)
    target = shared/original.name
    original.replace(target)
    original.with_suffix('.json').replace(target.with_suffix('.json'))
    before = target.read_bytes(), target.with_suffix('.json').read_bytes()
    matrix, second = dl.embedding_matrix(frame(),'train',paths,config,Encoder(),'cpu')
    assert second['cache_hit']
    assert (target.read_bytes(),target.with_suffix('.json').read_bytes()) == before
    _,_,key = dl._embedding_contract(frame(),config)
    _,_,reversed_key = dl._embedding_contract(frame().iloc[::-1],config)
    assert key != reversed_key
    del matrix


def test_interrupted_embedding_generation_resumes(tmp_path):
    paths, config = configuration(tmp_path)
    config = {**config['encoder'],'dimension':4,'batch_size_cpu':2}
    with pytest.raises(RuntimeError,match='interruption'):
        dl.embedding_matrix(frame(),'train',paths,config,Encoder(fail_call=2),'cpu')
    progress = next((paths.cache/'experiments').rglob('*.progress.json'))
    assert json.loads(progress.read_text())['next_row'] == 2
    matrix, _ = dl.embedding_matrix(frame(),'train',paths,config,Encoder(),'cpu')
    np.testing.assert_allclose(matrix,Encoder().encode(dl.compose_email_texts(frame()).tolist()))
    del matrix


def test_cache_corruption_is_rejected(tmp_path):
    paths, config = configuration(tmp_path)
    config = {**config['encoder'],'dimension':4}
    matrix, meta = dl.embedding_matrix(frame(),'train',paths,config,Encoder(),'cpu')
    del matrix
    path = tmp_path/meta['matrix']
    bad = np.load(path).copy()
    bad[0] = np.nan
    np.save(path,bad)
    with pytest.raises(dl.ExperimentBlocked,match='finite'):
        dl.embedding_matrix(frame(),'train',paths,config,Encoder(),'cpu')


def test_cached_frozen_vectors_need_no_encoder_but_missing_cache_blocks(tmp_path):
    paths, config = configuration(tmp_path)
    config = {**config['encoder'],'dimension':4}
    with pytest.raises(dl.ExperimentBlocked,match='No matching'):
        dl.embedding_matrix(frame(),'train',paths,config,None,'cpu')
    matrix, _ = dl.embedding_matrix(frame(),'train',paths,config,Encoder(),'cpu')
    del matrix
    matrix, meta = dl.embedding_matrix(frame(),'train',paths,config,None,'cpu')
    assert meta['cache_hit'] and not meta['fresh_encoder_sample_verified']
    del matrix


@pytest.mark.parametrize('candidate',['B_minilm_svc','C_minilm_mlp'])
def test_heads_fit_train_only_and_reload(tmp_path,candidate):
    _, config = configuration(tmp_path)
    config['encoder']['dimension'] = 4
    config['mlp'].update(hidden_layer_sizes=[4],max_iter=5,batch_size=4)
    train = Encoder().encode(dl.compose_email_texts(frame()).tolist())
    model = dl.make_classifier(candidate,config)
    model.fit(train,frame().label.to_numpy())
    if candidate == 'C_minilm_mlp':
        np.testing.assert_allclose(model.named_steps['scale_train_only'].mean_,train.mean(axis=0),atol=1e-7)
        assert not model.named_steps['classifier'].early_stopping
    path, loaded = dl._save_classifier(candidate,model,tmp_path,config)
    assert joblib.load(path).training_scope == 'dl_01_development_bootstrap_only'
    assert loaded.label_order == list(model.classes_)
    np.testing.assert_array_equal(loaded.predict_embeddings(train),model.predict(train))
    with pytest.raises(dl.ExperimentBlocked,match='overwrite'):
        dl._save_classifier(candidate,model,tmp_path,config)


def test_full_synthetic_development_experiment_and_resume(tmp_path,monkeypatch):
    from test_v21_development_training import _fixture
    from vietmailguard.v21.development_training import make_tfidf_pipeline

    paths, _, _ = _fixture(tmp_path)
    _, config = configuration(tmp_path)
    config['encoder']['dimension'] = 4
    config['mlp'].update(hidden_layer_sizes=[4],max_iter=5,batch_size=3)
    (paths.config/'experiments/multilingual_dl_01.json').write_text(json.dumps(config))
    source_config = json.loads((ROOT/'config/v21/development_training.json').read_text())
    (paths.config/'development_training.json').write_text(json.dumps(source_config))
    _, train, validation = dl.verify_and_load_development(paths,4096)
    tiny = {'seed':42,'tfidf':{
        'word':{'ngram_range':[1,2],'min_df':1,'max_features':100,'sublinear_tf':True,'dtype':'float32'},
        'character':{'analyzer':'char_wb','ngram_range':[3,5],'min_df':1,'max_features':100,'sublinear_tf':True,'dtype':'float32'},
        'svm':{'C':1,'class_weight':'balanced','max_iter':1000}}}
    baseline = make_tfidf_pipeline(tiny).fit(train[['subject','body']],train.label)
    paths.model.mkdir(parents=True)
    model = paths.model/'baseline.joblib'
    joblib.dump(baseline,model)
    paths.results.mkdir(parents=True)
    saved = paths.results/'saved_development.json'
    saved.write_text(json.dumps({'candidates':[{'candidate':'baseline','metrics':
        dl.evaluate(validation,baseline.predict(validation[['subject','body']]))}]}))
    runtime = {'artifact':model.relative_to(tmp_path).as_posix(),'artifact_sha256':dl.sha256_file(model),
        'validation_metrics':saved.relative_to(tmp_path).as_posix(),'candidate_id':'baseline'}
    (paths.config/'runtime_model.json').write_text(json.dumps(runtime))
    resources = {'free_ram_gib':3,'cuda':False,'total_vram_gib':0}
    monkeypatch.setattr(dl,'measured_resources',lambda:resources)
    monkeypatch.setattr(dl,'preflight',lambda *args:{'resources':resources,'encoder_allowed':True,
        'encoder_reason':'synthetic fixture','snapshot':tmp_path,'mlp_sample_weight_supported':True})
    monkeypatch.setattr(dl,'load_encoder',lambda *args:Encoder())
    original = model.read_bytes()
    result = dl.run_experiment(paths)
    assert [item['status'] for item in result['candidates']] == ['PASS']*3
    assert not result['final_test_opened'] and not result['runtime_modified']
    assert result['leakage']['passed']
    assert model.read_bytes() == original
    assert (tmp_path/'MULTILINGUAL_DL_EXPERIMENT_01.md').exists()
    monkeypatch.setattr(dl,'make_classifier',lambda *args:pytest.fail('Completed head must not refit'))
    resumed = dl.run_experiment(paths)
    assert [item['artifact_sha256'] for item in result['candidates']] == [
        item['artifact_sha256'] for item in resumed['candidates']]
    measured = dl.measure_saved_candidates(paths)
    assert not measured['fit_performed'] and measured['device'] == 'cpu'
    assert len(measured['candidates']) == 2
    for item in measured['candidates'].values():
        assert item['latency']['includes_encoder'] and item['latency']['sample_rows'] == 3
    checkpoint = json.loads((paths.results/'experiments/multilingual_dl_01/checkpoint.json').read_text())
    assert all(item['latency'].get('status') != 'BLOCKED' for item in checkpoint['candidates'])
