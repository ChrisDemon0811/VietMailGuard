"""Resolve isolated Version 2.1 paths without creating or reading datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping


DATA_DIR_ENV = "VIETMAILGUARD_V21_DATA_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class V21Paths:
    """Locations for V2.1 inputs and outputs; construction has no I/O side effects."""

    project_root: Path
    dataset_dir: Path
    config: Path
    scripts: Path
    source: Path
    raw: Path
    processed: Path
    curated: Path
    splits: Path
    cache: Path
    evaluation: Path
    model: Path
    results: Path
    reports: Path


def resolve_v21_paths(
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> V21Paths:
    """Resolve paths from the project root and optional external dataset directory.

    An unset environment variable uses ``data/v21/raw``. A relative value is
    relative to the project root, not the caller's current directory. No path
    is created or required to exist. Windows absolute paths are rejected on
    non-Windows hosts instead of being mistaken for relative paths.
    """

    root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    variables = os.environ if environ is None else environ
    configured = variables.get(DATA_DIR_ENV)
    if configured is None:
        dataset_dir = root / "data" / "v21" / "raw"
    else:
        value = configured.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise ValueError(f"{DATA_DIR_ENV} must not be empty")
        if os.name != "nt" and PureWindowsPath(value).is_absolute():
            raise ValueError("A Windows absolute dataset path requires Windows")
        dataset_dir = Path(value).expanduser()
        if not dataset_dir.is_absolute():
            dataset_dir = root / dataset_dir

    data_root = root / "data" / "v21"
    return V21Paths(
        project_root=root,
        dataset_dir=dataset_dir.resolve(),
        config=root / "config" / "v21",
        scripts=root / "scripts" / "v21",
        source=root / "src" / "vietmailguard" / "v21",
        raw=data_root / "raw",
        processed=data_root / "processed",
        curated=data_root / "curated",
        splits=data_root / "splits",
        cache=data_root / "cache",
        evaluation=data_root / "evaluation",
        model=root / "models" / "v21_multilingual",
        results=root / "results" / "v21_multilingual",
        reports=root / "reports" / "v21",
    )

