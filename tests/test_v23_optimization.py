"""Synthetic tests of fold isolation, gradient bounds and Train-only selection."""
import numpy as np
import torch

from vietmailguard.v23.optimization import (
    assign_guarded_gradients, grouped_folds, metric_values,
    offset_predictions, choose_on_oof, train_recipe)


def test_absent_template_does_not_link_unrelated_emails_to_every_fold():
    from vietmailguard.v23.optimization import fingerprint_fold_map
    result = fingerprint_fold_map(['', '', '', 'real_template', 'real_template'], [0, 1, 2, 0, 2])
    assert '' not in result
    assert result.get('', 0) == 0
    assert result['real_template'] == 5


def test_jaccard_guard_preserves_all_matching_fold_memberships():
    from vietmailguard.v23.train_guard import ReferenceJoin
    from vietmailguard.v23.optimization import reference_fold_bits
    texts = [' '.join(f'word{i}' for i in range(300)), ' '.join(f'word{i}' for i in range(305)),
             'a completely independent reference with enough distinct words to form a query']
    join = ReferenceJoin(texts)
    folds = np.array([0, 2, 1])
    queries = [texts[0], texts[2], 'short']
    actual = reference_fold_bits(join, queries, folds)
    for index, query in enumerate(queries):
        expected = 0
        for reference, _ in join.matches(query):expected |= 1 << int(folds[reference])
        assert int(actual[index]) == expected


def test_grouped_folds_cover_every_row_without_splitting_relatives():
    labels = np.repeat([0, 1, 2], 18)
    groups = np.repeat(np.arange(27), 2)
    folds = grouped_folds(labels, groups)
    used = []
    for fit, validation in folds:
        assert not (set(groups[fit]) & set(groups[validation]))
        assert set(labels[fit]) == set(labels[validation]) == {0, 1, 2}
        used.extend(validation)
    assert sorted(used) == list(range(len(labels)))


def test_conflicting_auxiliary_gradient_is_projected_and_bounded():
    weight = torch.nn.Parameter(torch.tensor([1., 2.]))
    supervised = weight[0]
    auxiliary = -100*weight[0]+20*weight[1]
    result = assign_guarded_gradients([weight], supervised, auxiliary, .2)
    assert result['conflict'] and result['source_reached_optimizer']
    torch.testing.assert_close(weight.grad, torch.tensor([1., .2]))
    assert result['aux_ratio'] <= .200001


def test_aligned_auxiliary_cannot_dominate_verified_gradient():
    weight = torch.nn.Parameter(torch.tensor([1., 2.]))
    result = assign_guarded_gradients([weight], weight.sum(), 1000*weight.sum(), .2)
    torch.testing.assert_close(weight.grad, torch.tensor([1.2, 1.2]))
    assert result['aux_ratio'] <= .200001


def test_oof_selection_uses_declared_gate_and_not_largest_offset_automatically():
    truth = np.array([0]*200+[1]*20+[2]*20)
    perfect = np.eye(3)[truth]*4
    unreliable = perfect.copy();unreliable[:20] = [0, 4, 0]
    oof = {'verified_control': perfect, 'denoise_recovery': perfect, 'guarded_weak_recovery': unreliable}
    chosen, records = choose_on_oof(oof, truth, [0., .5, 1.])
    assert chosen['recipe'] == 'denoise_recovery' and chosen['normal_logit_offset'] == 0
    assert chosen['normal_fpr_gate_passed']
    assert len(records) == 6
    assert np.array_equal(offset_predictions(perfect, 0), truth)


def test_metrics_count_normal_errors_as_normal_false_positives():
    metrics = metric_values([0, 0, 1, 2], [1, 2, 0, 0])
    assert metrics['normal_false_positives'] == 2
    assert metrics['per_class']['normal']['recall'] == 0


def test_training_counts_real_unlabeled_gradients_and_saves_preprocessing(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    train = rng.normal(size=(18, 4)).astype(np.float32)
    source = rng.normal(size=(12, 4)).astype(np.float32)
    labels = np.array([0, 1, 2]*6)
    config = {'seed': 42, 'hidden': [8, 5], 'batch_rows': 6, 'warmup_epochs': 2,
              'warmup_learning_rate': .001, 'weight_decay': .0001, 'source_learning_rate': .0001,
              'source_epochs': 1, 'recovery_epochs': 2, 'recovery_learning_rate': .0003,
              'embedding_dropout': .1, 'ema_decay': .99,
              'gradient_guard': {'max_prediction_path_aux_to_verified_norm': .2}}
    recipe = {'name': 'denoise_recovery', 'weak': 0., 'consistency': 0., 'reconstruction': .01}
    _, receipt = train_recipe(train, labels, source, np.arange(12), np.array([0, 1, 2]*4),
        recipe, config, tmp_path/'new_model.pt', 'locked',
        preprocessing={'scaler_mean': np.zeros(4), 'scaler_scale': np.ones(4)})
    assert receipt['source_parameter_update_rows'] == 12
    assert receipt['stages'][-2]['gradient_stats']['max_aux_ratio'] <= .200001
    assert receipt['final_trunk_change_from_warmup'] > 0
    saved = torch.load(tmp_path/'new_model.pt', map_location='cpu', weights_only=False)
    assert 'scaler_mean' in saved and saved['unique_verified_rows'] == 18
