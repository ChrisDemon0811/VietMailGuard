"""Report integrity checks with synthetic numbers only; no model evaluation."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v23_report", ROOT/"scripts/v23/consolidate_research.py")
report = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


def test_saved_matrix_and_metric_disagreement_is_rejected():
    value = {'confusion_matrix': [[1565, 0, 0], [0, 33, 0], [0, 0, 340]],
             'accuracy': 1., 'macro_f1': 1., 'weighted_f1': 1.,
             'per_class': {name: {'support': n, 'recall': 1.} for name, n in zip(report.LABELS, [1565, 33, 340])}}
    report.verify_saved_metrics(value)
    value['per_class']['spam']['precision'] = .5
    with pytest.raises(ValueError, match='class metric inconsistent'):report.verify_saved_metrics(value)
    value['per_class']['spam']['precision'] = 1.
    value['macro_f1'] = .99
    with pytest.raises(ValueError, match='inconsistent'):report.verify_saved_metrics(value)


def test_different_benchmarks_cannot_enter_common_ranking():
    report.require_common_reference([{'train': 'a', 'validation': 'b'}]*3)
    with pytest.raises(ValueError, match='Different benchmark'):
        report.require_common_reference([{'validation': 'b'}, {'validation': 'other'}])


def test_report_file_access_blocks_escape_test_and_secret(tmp_path):
    with pytest.raises(ValueError):report.checked_path(tmp_path, '../outside.json')
    with pytest.raises(ValueError):report.checked_path(tmp_path, 'data/splits/v2/test.csv')
    with pytest.raises(ValueError):report.checked_path(tmp_path, '.env')
    assert report.checked_path(tmp_path, 'results/final_test_metrics.json').name == 'final_test_metrics.json'
