# VietMailGuard V2 environment

Snapshot date: 2026-09-15  
Scope: frozen Version 2 production inference, Streamlit application, data tooling and tests

## Tested platform

| Component | Tested value |
|---|---|
| Operating system | Windows 11, build `10.0.26200` |
| Architecture | AMD64 / 64-bit |
| Python implementation | CPython |
| Python | `3.13.14` (`MSC v.1944 64 bit`) |
| pip | `26.1.2` |
| setuptools | `78.1.0` |
| Project package | `vietmailguard 2.0.0`, src layout |

The project metadata continues to support Python `>=3.11`. The exact lock file records the environment actually tested here; it is not a claim that every allowed Python/OS combination has been tested.

## Production-critical direct dependencies

Every version below was read from the active repository `.venv`; none was inferred from a version range.

| Package | Exact tested version | Role |
|---|---:|---|
| pandas | `3.0.5` | Tabular inputs, reports and dashboard data |
| numpy | `2.5.3` | Numerical arrays and model outputs |
| scipy | `1.18.1` | Sparse TF-IDF matrices and scientific routines |
| scikit-learn | `1.9.1` | TF-IDF, calibrated LinearSVC and metrics |
| joblib | `1.6.0` | Loading the frozen production pipeline |
| streamlit | `1.63.0` | Web application |
| beautifulsoup4 | `4.15.0` | HTML extraction in `.eml` parsing |
| tldextract | `5.3.2` | URL/domain analysis |
| regex | `2026.9.10` | Text processing |
| openpyxl | `3.1.5` | Optional XLSX dataset input |
| matplotlib | `3.11.2` | Static experiment plots |
| plotly | `6.9.0` | Interactive dashboard plots |
| pytest | `9.1.1` | Automated verification |

All active transitive dependencies and their exact versions are recorded in [`requirements-lock.txt`](requirements-lock.txt). `requirements.txt` remains the human-readable supported-range specification.

## Optional research environments

The current `.venv` also contains packages used by prior V2 research/translation work:

| Package | Installed version | Core production runtime? |
|---|---:|---|
| torch | `2.8.0+cu128` | No |
| transformers | `4.56.2` | No |
| sentencepiece | `0.2.1` | No |
| sacremoses | `0.1.1` | No |
| sentence-transformers | `5.7.0` | No |
| huggingface-hub | `0.36.2` | No |

These packages are intentionally excluded from `requirements-lock.txt`. Use:

- `requirements-embeddings.txt` for multilingual embedding experiments;
- `requirements-translation.txt` for controlled translation.

Neither optional stack is required to load the frozen TF-IDF + calibrated LinearSVC production model.

## Reproduction strategy

For the tested Windows environment, run from the repository root:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip==26.1.2
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python -m pip check
.venv\Scripts\python -m pytest -q
```

Use `requirements.txt` instead when installing within supported ranges is more important than recreating the audited environment exactly.

## Serialized model compatibility

The production artifact is serialized with joblib/pickle and contains scikit-learn estimator classes. Such artifacts are not a stable cross-version interchange format. Loading with materially different scikit-learn, NumPy, SciPy or joblib versions may warn, fail, or change behavior.

For reproducible production inference:

1. install the exact core lock on the tested Python/platform family;
2. verify the production artifact SHA-256 against `models/v2_bilingual/model_metadata.json`;
3. load only a trusted model artifact;
4. run the model reload, probability and smoke tests before deployment;
5. do not retrain, refit or recalibrate the frozen V2 artifact merely to resolve an environment warning.

The lock file captures exact package versions, not wheel hashes. A future release may add a platform-specific, hash-locked wheel manifest after building and testing the release bundle on clean machines.

## Verification for this snapshot

Executed with `.venv\Scripts\python.exe` on 2026-09-15:

| Check | Result |
|---|---|
| Editable package metadata | `vietmailguard==2.0.0` |
| Module `__version__` | `2.0.0` |
| `python -m pip check` | PASS — `No broken requirements found.` |
| Lock-to-environment comparison | PASS — 63 pins, 0 missing, 0 version mismatches |
| `python -m pytest -q` | PASS — 178 tests passed in 28.44 seconds |
| Frozen model checksum | PASS — matches `model_metadata.json` |

The production model was loaded by the existing test suite but was not retrained, refit, recalibrated or rewritten.
