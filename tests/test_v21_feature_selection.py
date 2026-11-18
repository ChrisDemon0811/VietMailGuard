"""Train-only sparse feature selection and group-aware fold isolation."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from vietmailguard.v21 import feature_selection_experiment as exp


@pytest.mark.parametrize("k", [1, 2, 3])
def test_chi2_stays_sparse_and_fits_only_supplied_train(k):
    train = sparse.csr_matrix([[3., 0, 0], [2, 0, 1], [0, 4, 0], [0, 3, 1]])
    selector, reduced = exp.select_sparse(train, ['normal','normal','spam','spam'], k)
    scores = selector.scores_.copy()
    assert sparse.issparse(reduced) and reduced.shape == (4, k)
    selector.transform(sparse.csr_matrix([[0.,0,999]]))
    np.testing.assert_array_equal(scores, selector.scores_)


@pytest.mark.parametrize("matrix,k", [(np.eye(3), 2), (sparse.csr_matrix([[-1., 0]]), 1),
                                      (sparse.csr_matrix([[np.nan, 0]]), 1), (sparse.eye(3).tocsr(), 4)])
def test_invalid_sparse_inputs_rejected(matrix, k):
    with pytest.raises(exp.ExperimentBlocked):
        exp.select_sparse(matrix, ['normal'] * matrix.shape[0], k)


def test_union_connects_template_normalized_and_parent_transitively():
    rows = [{'id':'a','content_hash':'x'}, {'id':'b','content_hash':'x','template_hash':'y'},
            {'id':'c','template_hash':'y'}, {'id':'d','parent_id':'c'}, {'id':'e'}]
    groups = exp.related_groups(rows)
    assert len(set(groups[:4])) == 1 and groups[-1] != groups[0]
    with pytest.raises(exp.ExperimentBlocked):
        exp.related_groups([{'id':'a','parent_id':'missing'}])


def test_cv_resource_gate_never_reads_data_or_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, 'memory_snapshot', lambda: {'free_ram_gib': 1.0})
    result = exp.run_cv(None, None, {'cv_min_free_ram_gib': 6.0}, None)
    assert result['status'] == 'BLOCKED' and 'selected' not in result


def test_cv_refits_vectorizers_in_every_fold_and_groups_do_not_cross(tmp_path, monkeypatch):
    frame = pd.DataFrame([{'id':str(i), 'subject':f'unique{i}', 'body':label+' message', 'label':label}
                          for i in range(36) for label in [['normal','spam','phishing'][i%3]]])
    directory = tmp_path/'development_bootstrap'
    directory.mkdir()
    rows = [{'id':str(i), 'final_group_id':str(i), 'parent_id':'', 'content_hash':str(i),
             'normalized_content_hash':str(i), 'template_hash':str(i)} for i in range(36)]
    with (directory/'train.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=exp.related_groups(rows)
    folds=exp.group_folds(frame,groups,3,42)
    assert all(not(set(groups[a]) & set(groups[b])) for a,b in folds)
    calls=[]
    original=TfidfVectorizer.fit_transform
    def counted(self, texts, y=None):
        calls.append(set(texts))
        return original(self,texts,y)
    monkeypatch.setattr(TfidfVectorizer,'fit_transform',counted)
    monkeypatch.setattr(exp,'memory_snapshot',lambda:{'free_ram_gib':10})
    config={'cv_min_free_ram_gib':6,'cv_folds':3,'seed':42,'k':[8],'C':[.1,1]}
    training={'seed':42,'tfidf':{'word':{'ngram_range':[1,1],'min_df':1,'max_features':100},
              'character':{'analyzer':'char_wb','ngram_range':[3,3],'min_df':1,'max_features':100},
              'svm':{'C':1,'class_weight':'balanced','max_iter':5000}}}
    result=exp.run_cv(frame,SimpleNamespace(splits=tmp_path),config,training)
    assert result['status']=='PASS' and result['selection_data']=='Train CV only'
    assert len(calls)==12  # two vectorizers * three folds * two C values
    expected=[set((frame.iloc[a].subject+'\n'+frame.iloc[a].body).tolist()) for a,b in folds]
    assert calls==[texts for _ in range(2) for texts in expected for _ in range(2)]


def test_synthetic_candidate_reload_resume_integrity_and_preservation(tmp_path, monkeypatch):
    """Exercise real sklearn/persistence with synthetic Development files only."""
    from test_v21_dimensionality_experiment import synthetic_development

    paths, _, baseline_path = synthetic_development(tmp_path)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'config/v21/experiments/feature_selection_03.json').read_text())
    config.update(k=[15], transform_batch_rows=4, latency_repeats=2)
    (paths.config / 'experiments/feature_selection_03.json').write_text(json.dumps(config))
    training_path = paths.config / 'frozen_training.json'
    training_path.write_text(json.dumps({'tfidf': {'svm': {'C': 1.0}}}))
    runtime_path = paths.config / 'runtime_model.json'
    runtime = json.loads(runtime_path.read_text())
    runtime['training_config_sha256'] = exp.sha256_file(training_path)
    runtime_path.write_text(json.dumps(runtime))
    protected = [baseline_path, runtime_path, training_path,
                 paths.splits / 'development_bootstrap/train.csv',
                 paths.splits / 'development_bootstrap/validation.csv']
    before = {str(path): exp.sha256_file(path) for path in protected}
    monkeypatch.setattr(exp, 'memory_snapshot', lambda: {'free_ram_gib': 10.0})
    result = exp.candidate('A_chi2_15', paths)
    baseline = exp.candidate('baseline', paths)
    assert result['status'] == baseline['status'] == 'PASS'
    assert result['train_rows'] == 24 and result['validation_rows'] == 15
    assert result['output_dimensions'] == 15 and result['C'] == 1.0
    assert result['metrics']['by_language']['zh']['phishing_recall'] is None
    artifact = tmp_path / result['artifact']
    model = joblib.load(artifact)
    _, _, train, validation, _ = exp.context(paths)
    expected = exp.SelectKBest(exp.chi2, k=15).fit(model[:-2].transform(train), train.label)
    np.testing.assert_allclose(expected.scores_, model.named_steps['select'].scores_)
    np.testing.assert_array_equal(model.predict(validation), joblib.load(artifact).predict(validation))
    assert {str(path): exp.sha256_file(path) for path in protected} == before
    diagnostics = exp.feature_diagnostics(paths, [baseline, result])['A_chi2_15']
    assert diagnostics['word_features'] + diagnostics['character_features'] == 15
    assert set(diagnostics['baseline_absolute_coefficient_mass_retained']) == {'normal', 'spam', 'phishing'}
    assert all(0 <= value <= 1 for value in diagnostics['baseline_absolute_coefficient_mass_retained'].values())
    monkeypatch.setattr(exp, 'select_sparse', lambda *args: pytest.fail('Resume must not refit'))
    assert exp.candidate('A_chi2_15', paths) == result
    artifact.write_bytes(b'corrupted synthetic artifact')
    with pytest.raises(exp.ExperimentBlocked, match='checksum'):
        exp.candidate('A_chi2_15', paths)
    with pytest.raises(exp.ExperimentBlocked, match='checksum'):
        exp.feature_diagnostics(paths, [baseline, result])
    assert {str(path): exp.sha256_file(path) for path in protected} == before


def test_context_rejects_changed_baseline_before_loading_development(tmp_path, monkeypatch):
    from vietmailguard.v21.paths import resolve_v21_paths

    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    (paths.config / 'experiments').mkdir(parents=True)
    (paths.config / 'experiments/feature_selection_03.json').write_text(json.dumps({
        'download_allowed': False, 'final_test_allowed': False}))
    artifact = tmp_path / 'baseline.joblib'
    artifact.write_bytes(b'synthetic corruption')
    (paths.config / 'runtime_model.json').write_text(json.dumps({
        'artifact': 'baseline.joblib', 'artifact_sha256': 'wrong'}))
    monkeypatch.setattr(exp, 'verify_and_load_development', lambda *args: pytest.fail('Must fail before data access'))
    with pytest.raises(exp.ExperimentBlocked, match='checksum'):
        exp.context(paths)
