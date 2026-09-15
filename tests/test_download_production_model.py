from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_production_model import (
    download_verified_artifact,
    expected_artifact_sha256,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_expected_hash_comes_from_metadata(tmp_path: Path) -> None:
    expected = "a" * 64
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"artifact_sha256": expected}), encoding="utf-8")
    assert expected_artifact_sha256(metadata) == expected


def test_local_fixture_download_is_atomic_and_hash_verified(tmp_path: Path) -> None:
    content = b"fixture model artifact"
    source = tmp_path / "source.joblib"
    destination = tmp_path / "models" / "production_pipeline.joblib"
    source.write_bytes(content)

    actual = download_verified_artifact(
        source.as_uri(),
        destination,
        _sha256(content),
        allowed_schemes=("file",),
    )

    assert actual == _sha256(content)
    assert destination.read_bytes() == content
    assert list(destination.parent.glob("*.download")) == []


def test_hash_mismatch_does_not_publish_download(tmp_path: Path) -> None:
    source = tmp_path / "source.joblib"
    destination = tmp_path / "production_pipeline.joblib"
    source.write_bytes(b"unexpected content")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_verified_artifact(
            source.as_uri(),
            destination,
            "0" * 64,
            allowed_schemes=("file",),
        )

    assert not destination.exists()


def test_missing_url_fails_with_configuration_guidance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="V2_PRODUCTION_MODEL_URL"):
        download_verified_artifact("", tmp_path / "model.joblib", "0" * 64)
