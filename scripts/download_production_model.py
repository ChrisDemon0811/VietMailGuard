"""Download and verify the frozen production model release artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Collection
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "models" / "v2_bilingual" / "model_metadata.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "v2_bilingual" / "production_pipeline.joblib"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifact_sha256(metadata_path: Path) -> str:
    """Read and validate the expected production artifact digest from metadata."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    expected = str(metadata.get("artifact_sha256", "")).strip().lower()
    if not SHA256_PATTERN.fullmatch(expected):
        raise ValueError(
            f"Missing or invalid artifact_sha256 in {Path(metadata_path).as_posix()}"
        )
    return expected


def download_verified_artifact(
    url: str,
    output_path: Path,
    expected_sha256: str,
    *,
    timeout_seconds: float = 120.0,
    allowed_schemes: Collection[str] = ("https",),
) -> str:
    """Download to a temporary file and publish only after hash verification."""
    clean_url = str(url).strip()
    if not clean_url:
        raise ValueError(
            "No production model URL configured. Set the repository variable "
            "V2_PRODUCTION_MODEL_URL to the official GitHub Release asset URL."
        )
    scheme = urlsplit(clean_url).scheme.casefold()
    normalized_schemes = {value.casefold() for value in allowed_schemes}
    if scheme not in normalized_schemes:
        raise ValueError(
            f"Unsupported artifact URL scheme {scheme!r}; allowed: "
            f"{sorted(normalized_schemes)}"
        )
    expected = str(expected_sha256).strip().lower()
    if not SHA256_PATTERN.fullmatch(expected):
        raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected:
        return expected

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(clean_url, headers={"User-Agent": "VietMailGuard-release-fetch/2"})
            with urlopen(request, timeout=timeout_seconds) as response:
                shutil.copyfileobj(response, temporary)

        actual = sha256_file(temporary_path)
        if actual != expected:
            raise RuntimeError(
                "Production artifact SHA-256 mismatch: "
                f"expected {expected}, downloaded {actual}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return actual
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official frozen V2 model and verify its metadata SHA-256."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("V2_PRODUCTION_MODEL_URL", ""),
        help="Official HTTPS GitHub Release asset URL (or V2_PRODUCTION_MODEL_URL).",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = expected_artifact_sha256(args.metadata)
        actual = download_verified_artifact(
            args.url,
            args.output,
            expected,
            timeout_seconds=args.timeout,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Production model download failed: {exc}") from exc
    print(f"Verified production model: {args.output}")
    print(f"SHA-256: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
