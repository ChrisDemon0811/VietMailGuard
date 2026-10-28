# VietMailGuard V2.1 Research Preview — safe Git preflight

Date: 2026-10-22. This is a **local preflight, not a release approval**. No commit, push, history rewrite, model training, final evaluation, or live Gemini call was made. Scans below report paths and aggregate counts only, not message text or secret values.

## Git state and preservation

| Check | Finding |
| --- | --- |
| Public remote | `ChrisDemon0811/VietMailGuard` is public; `origin` points to its HTTPS Git URL. |
| Branch | Created local `v21/research-preview` from `main`, preserving all uncommitted work. Before branching, `main` and `origin/main` both resolved to `55182854b4a8895661a547d5cec32cd29c2a0b05`. No fetch/merge was needed. |
| Working tree before this report | 14 unstaged modified paths, 91 untracked paths, **0 staged** paths; no tracked V2 frozen-path modification. Branch creation did not stage changes. |
| Pre-branch backup | Private sibling directory `../VietMailGuard-local-backups/`, archive `v21-source-preflight-20261022-213038.zip`: 185 source/sample-config/documentation entries, SHA-256 `a47b90fc2df6ac28bdfccb5b2947f27a50015d2bec87e68801334aa56219db16`. Archive-entry audit found 0 `data/`, `models/`, `results/`, `.env`, `.eml`, `.db` or `.joblib` entries. Keep this archive local; it is not a release asset. |
| Frozen V2 | `production_pipeline.joblib` SHA-256 `b2ebecf641f2a5fd3ce9f0d0680a90884850e46b59c48ce0ab5bd4fee64f87fa` agrees with `model_metadata.json`. `git diff --name-only` for frozen model/results/splits/config paths is empty. |

## Proposed source-only commit scope — **not staged**

Review the exact diff before selective staging. No `git add .` or blanket recursive add.

- Root documentation/policy: `AGENTS.md`, `README.md`, `V21_RESEARCH_PREVIEW.md`, `.gitignore`, blank-key `.env.example`, `requirements-gemini.txt`, `pyproject.toml`.
- Existing app integration: modified `app/app.py`, `app/pages/{dataset_dashboard,email_analyzer,mail_client,model_dashboard}.py`, `config/translations.json`, `src/vietmailguard/{mail_database,mail_repository,mail_service}.py`, `tests/test_mail_database.py`. These are application changes only; re-review their diffs before staging.
- V2.1 implementation: 24 `src/vietmailguard/v21/*.py`, 17 `scripts/v21/*.py`, and 21 `tests/test_v21_*.py` currently present. Stage by explicit reviewed path list.
- V2.1 non-local policy/schema/examples: `config/v21/*.json`, `config/v21/*.md`, `config/v21/label_decisions.example.csv`, after per-file inspection. The current config scan found no personal absolute `C:\Users\...` or `D:\Study\...` path. Do not stage future `*.local.*`, `*secret*`, `*credential*`, `*api_key*` or `private/` files.
- Reviewed aggregate-only documents: `reports/v21/research_preview_qa.md` and this report. Other V2.1 generated reports remain ignored pending a separate email-text/privacy inspection. `.gitkeep` markers are optional, not datasets.

**Important:** this is a *candidate path set*, not clearance for an immediate push. Source examples/tests contain artificial email addresses; path-level inspection found no high-confidence live secret pattern, but a human should verify that example content is synthetic before staging. `config/v21/runtime_model.json` points to a local research candidate and does not publish that ignored model binary.

## Must remain local / do not stage

- `.env`, API keys, `.streamlit/secrets.toml`, private config, certificate/private-key files.
- Raw, processed, curated, split, cache and evaluation email datasets under `data/` or the external V2.1 dataset directory; review/decision manifests with row-level content.
- Private `.eml`, `.mbox`, `.msg`, `.pst`, `.ost` mail exports anywhere in the tree; SQLite runtime DB, checkpoints, model binaries and downloads including `*.joblib`.
- V2.1 generated model/results files and any report reproducing raw email content. The backup archive itself must stay outside Git.

`.gitignore` now explicitly covers private mail-export extensions in addition to existing environment, raw data, V2.1 data, runtime, cache and binary rules. Tested representative paths with `git check-ignore -v --no-index`; the V2 `model_metadata.json` exception remains intact. **Ignore rules do not affect files already tracked or Git history.**

## Secret, privacy and license audit

No `gitleaks` executable is installed locally, so this is a bounded pattern scan, **not a comprehensive secret-scanner attestation**. Path-only `rg` inspected proposed source/config/test/docs for Google-style keys, GitHub tokens, private-key headers and non-placeholder long assignments. `git grep -l -I -E` inspected current tracked files and every tree in all **6 local commits**, including dataset paths; `git log -G` checked additions/removals. **0 high-confidence pattern hits** were found. Historical filename inspection found no tracked `.env`, `secrets.toml`, `.pem` or `.key`. `.env.example` currently has blank `GEMINI_API_KEY=`. The earlier 11G QA records a key-shaped value in this *untracked* example before it was blanked; whether it was live cannot be determined here. If it was live, revoke/rotate it. Never copy it into a commit.

Four **already tracked and already public** V2 CSVs contain row-level mail text or derivatives. A streaming read-only schema/count check produced the following; no values were logged:

| Tracked path | Rows | Rows with text | Rows with email-address pattern in text | Other identity risk / rights |
| --- | ---: | ---: | ---: | --- |
| `data/curated/v2/Vietnamese_Curated_Dataset.csv` | 5,565 | 5,565 | 6 | Derived translated mail, names/organizations may occur in text; redistribution **NOT VERIFIED**. |
| `data/curated/v2/vietnamese_positive_review.csv` | 1,357 | 1,357 | 2 | Subject/body excerpts; redistribution **NOT VERIFIED**. |
| `data/evaluation/v2/vietnamese_robustness.csv` | 240 | 240 | 0 | Transformed mail text and metadata; redistribution **NOT VERIFIED**. |
| `data/evaluation/v2/short_form_challenge.csv` | 90 | 90 | 0 | Project-authored synthetic challenge, but separate dataset license **NOT VERIFIED**. |

Email-pattern count is a narrow indicator, **not** full PII clearance: personal names, identifiers, domains, URLs or context may still identify people. `DATASETS.md` and `reports/dataset_license_audit.md` state that third-party corpus and translated-derivative redistribution is not cleared. The [CMU Enron publisher](https://www.cs.cmu.edu/~enron/) expressly notes privacy concerns; the [SpamAssassin corpus README](https://spamassassin.apache.org/old/publiccorpus/readme.html) says message-text copyright remains with senders. A public source or Zenodo aggregate license is not sufficient evidence that every derivative message can be republished. The project MIT software license is unrelated to these email rights.

`git log --all -- data/curated/v2 data/evaluation/v2` places all four CSVs in the **initial commit** (`04b381e`). Therefore adding ignore entries, or even future index-only removal, would not erase their current public history. No removal or history rewrite was authorized or attempted. A separate owner decision is required: verify rights/privacy for each file; if uncleared, authorize targeted `git rm --cached -- <exact path>` to stop future tracking while retaining local copies, then independently decide whether public-history remediation and any notice/rotation are necessary. Do not rewrite shared history without its own approved plan.

## Verification and release gate

| Check | Result |
| --- | --- |
| Full `python -m pytest -q` in local `.venv` | **PASS:** 441 passed, 3 skipped. |
| Source-only `python -m pytest -m "not production_artifact" -q` | **PASS:** 414 passed, 3 skipped, 27 deselected. |
| `python -m pip check` | **PASS:** no broken requirements. |
| `git diff --check` | **PASS:** exit 0; Git printed LF→CRLF working-copy warnings, not whitespace errors. |
| Frozen V2 hash and tracked-path diff | **PASS:** hash match and no frozen-path diff. |
| Current/history high-confidence secret-pattern scan | **PASS within limited pattern coverage**; no dedicated `gitleaks` run. |
| Proposed-commit privacy/licensing and existing public Git history | **BLOCKED:** four tracked CSVs contain mail text/derivatives with unresolved redistribution and privacy review. |
| Gemini live call, V2/V2.1 Final Test, model training | **NOT RUN by design.** |

**SAFE_TO_COMMIT: BLOCKED** for the repository release as a whole. A *selectively staged source-only* commit may become appropriate after human review of the exact staged diff and a dedicated secret scan, but this preflight does not approve one while the tracked row-level artifacts remain unresolved.

**SAFE_TO_PUSH: BLOCKED.** The remote is public and its initial commit already contains the four CSVs; another push would not solve that risk. Resolve rights/privacy and history strategy separately, then rerun preflight against the exact proposed commit. No commit or push was performed.
