"""Synthetic-only checks of the new uncertain-label training objectives."""
import copy
import numpy as np
import pytest
import torch

from vietmailguard.v23.classifier_expansion import (
    ExpansionHead, partial_label_loss, consistency_loss, ordered_indices,
    fit_head, update_teacher, memory_bytes)


def test_partial_label_does_not_choose_spam_instead_of_phishing():
    first = torch.tensor([[0., 3., 1.]], requires_grad=True)
    second = torch.tensor([[0., 1., 3.]], requires_grad=True)
    torch.testing.assert_close(partial_label_loss(first), partial_label_loss(second))
    loss = partial_label_loss(first);loss.backward()
    assert first.grad[0, 0] > 0
    assert first.grad[0, 1] < 0 and first.grad[0, 2] < 0


def test_all_candidate_classes_have_no_forced_reference_label():
    value = torch.tensor([[1., 2., 3.]], requires_grad=True)
    loss = partial_label_loss(value, (0, 1, 2))
    loss.backward()
    assert float(loss.detach()) == 0
    torch.testing.assert_close(value.grad, torch.zeros_like(value))


def test_consistency_targets_detached_and_not_calibrated_probability():
    teacher = torch.tensor([[1., 2., 3.]], requires_grad=True)
    student = torch.zeros((1, 3), requires_grad=True)
    loss, weight = consistency_loss(student, teacher)
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0
    assert .05 <= weight <= 1


def test_source_order_is_nested_and_independent_of_input_order():
    identifiers = np.array([9, 2, 71, 4])
    first = identifiers[ordered_indices(identifiers, 42)]
    reverse = identifiers[::-1]
    second = reverse[ordered_indices(reverse, 42)]
    np.testing.assert_array_equal(first, second)
    assert len(set(first[:2]) & set(first[:3])) == 2


def test_teacher_is_parameter_ema():
    student = ExpansionHead(4, (5, 3))
    teacher = copy.deepcopy(student)
    initial = {key: value.clone() for key, value in teacher.state_dict().items()}
    with torch.no_grad():
        for p in student.parameters():p.add_(2.)
    update_teacher(teacher, student, .75)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, initial[key]+.5)


def test_unlabeled_objectives_reach_classifier_and_representation_parameters():
    torch.manual_seed(42)
    model = ExpansionHead(4, (5, 3))
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    original = torch.randn(8, 4)
    corrupted = original * torch.tensor([[0., 1., 1., 0.]])
    predicted, reconstructed = model(corrupted)
    targets, _ = teacher(original)
    consistency, _ = consistency_loss(predicted, targets)
    loss = consistency + .1*torch.nn.functional.mse_loss(reconstructed, original)
    loss.backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert model.trunk[0].weight.grad.abs().sum() > 0
    assert model.decoder.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("method,expected", [("verified_replay_control", 0), ("weak_partial", 4), ("weak_partial_consistency_denoise", 8)])
def test_actual_update_counts_and_resume_keep_uncertain_labels_separate(tmp_path, method, expected):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train = rng.normal(size=(12, 4)).astype(np.float32)
    labels = np.array([0, 1, 2]*4, dtype=np.int64)
    expansion = rng.normal(size=(8, 4)).astype(np.float32)
    weak = np.array([True]*4+[False]*4)
    config = {"seed": 42, "hidden": [5, 3], "expansion_learning_rate": .001,
              "weight_decay": .0001, "batch_rows": 4, "expansion_epochs": 1,
              "weak_candidate_indices": [1, 2], "weak_loss_weight": .25,
              "consistency_loss_weight": .05, "reconstruction_loss_weight": .1,
              "embedding_dropout": .1, "ema_decay": .99}
    initial = ExpansionHead(4, (5, 3)).state_dict()
    model, result = fit_head(train, labels, expansion, weak, np.arange(8), config,
        method, initial, tmp_path/'checkpoint.pt', 'locked')
    assert result['expansion_unique_parameter_update_rows'] == expected
    assert result['hard_pseudo_label_rows'] == 0
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    resumed, receipt = fit_head(train, labels, expansion, weak, np.arange(8), config,
        method, initial, tmp_path/'checkpoint.pt', 'locked')
    assert receipt == result
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key])
    np.testing.assert_array_equal(weak, [True]*4+[False]*4)


def test_os_resource_monitor_works_without_new_dependency():
    rss, available = memory_bytes()
    assert rss > 0 and available > 0


def test_epoch_checkpoint_resume_matches_uninterrupted_fit(tmp_path, monkeypatch):
    from pathlib import Path
    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    train = rng.normal(size=(12, 4)).astype(np.float32)
    labels = np.array([0, 1, 2]*4, dtype=np.int64)
    source = rng.normal(size=(8, 4)).astype(np.float32)
    weak = np.array([True]*4+[False]*4)
    config = {"seed": 42, "hidden": [5, 3], "expansion_learning_rate": .001,
              "weight_decay": .0001, "batch_rows": 4, "expansion_epochs": 2,
              "weak_candidate_indices": [1, 2], "weak_loss_weight": .25,
              "consistency_loss_weight": .05, "reconstruction_loss_weight": .1,
              "embedding_dropout": .1, "ema_decay": .99}
    initial = ExpansionHead(4, (5, 3)).state_dict()
    method = "weak_partial_consistency_denoise"
    uninterrupted, receipt = fit_head(train, labels, source, weak, np.arange(8), config,
        method, initial, tmp_path/'complete.pt', 'protocol')
    original_replace = Path.replace
    checkpoint = tmp_path/'interrupted.pt'
    def crash_after_checkpoint(path, target):
        result = original_replace(path, target)
        if target == checkpoint:
            raise RuntimeError('synthetic interruption after durable epoch checkpoint')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', crash_after_checkpoint)
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            fit_head(train, labels, source, weak, np.arange(8), config,
                     method, initial, checkpoint, 'protocol')
    resumed, resumed_receipt = fit_head(train, labels, source, weak, np.arange(8), config,
        method, initial, checkpoint, 'protocol')
    assert receipt == resumed_receipt
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
