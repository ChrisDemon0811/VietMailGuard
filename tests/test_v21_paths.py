"""Source-only checks for Version 2.1 path and inventory configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vietmailguard.v21.paths import DATA_DIR_ENV, resolve_v21_paths


def test_default_paths_are_isolated_and_have_no_creation_side_effect(
    tmp_path: Path,
) -> None:
    root = tmp_path / "VietMailGuard project"
    paths = resolve_v21_paths(project_root=root, environ={})

    assert paths.dataset_dir == root / "data" / "v21" / "raw"
    assert paths.processed == root / "data" / "v21" / "processed"
    assert paths.curated == root / "data" / "v21" / "curated"
    assert paths.splits == root / "data" / "v21" / "splits"
    assert paths.model == root / "models" / "v21_multilingual"
    assert paths.results == root / "results" / "v21_multilingual"
    assert not root.exists()


def test_external_unicode_path_with_spaces_is_preserved(tmp_path: Path) -> None:
    root = tmp_path / "Project với khoảng trắng"
    external = tmp_path / "Nguồn dữ liệu" / "中文 邮件"
    paths = resolve_v21_paths(
        project_root=root,
        environ={DATA_DIR_ENV: str(external)},
    )

    assert paths.dataset_dir == external.resolve()
    assert paths.config == root / "config" / "v21"
    assert paths.cache == root / "data" / "v21" / "cache"
    assert not external.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Path semantics require Windows")
def test_windows_absolute_path_with_spaces_and_unicode(tmp_path: Path) -> None:
    windows_dataset = tmp_path / "Thư mục dữ liệu" / "中文 và Tiếng Việt"
    paths = resolve_v21_paths(
        project_root=tmp_path / "Project folder",
        environ={DATA_DIR_ENV: f'"{windows_dataset}"'},
    )

    assert paths.dataset_dir == windows_dataset.resolve()
    assert paths.dataset_dir.is_absolute()
    assert "Thư mục dữ liệu" in str(paths.dataset_dir)
    assert "中文 và Tiếng Việt" in str(paths.dataset_dir)


def test_relative_external_path_is_relative_to_project_not_current_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project root"
    paths = resolve_v21_paths(
        project_root=root,
        environ={DATA_DIR_ENV: "local data/Thư tiếng Việt"},
    )

    assert paths.dataset_dir == root / "local data" / "Thư tiếng Việt"


def test_empty_external_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=DATA_DIR_ENV):
        resolve_v21_paths(project_root=tmp_path, environ={DATA_DIR_ENV: "  "})


def test_example_inventory_has_no_assumed_sources_or_label_mappings() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "config" / "v21"
    example = json.loads((config_dir / "datasets.example.json").read_text(encoding="utf-8"))
    schema = json.loads((config_dir / "schema.json").read_text(encoding="utf-8"))

    assert example["data_directory_env"] == DATA_DIR_ENV
    assert example["datasets"] == []
    assert set(schema["required"]) <= set(example)
    assert schema["properties"]["datasets"]["items"]["properties"]["label_policy"]
