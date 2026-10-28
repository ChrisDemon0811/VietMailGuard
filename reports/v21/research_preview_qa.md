# Prompt 11G — VietMailGuard V2.1 Research Preview QA

Date: 2026-10-22. Scope: **local preview and preservation audit only**. No training, Final Test, paid Gemini request, commit, push or release. Test messages were synthetic/non-sensitive; pytest used disposable SQLite databases. No new scientific metric was calculated.

## Git, publication and secret audit

| Check | Result | Evidence / action |
| --- | --- | --- |
| Branch/remote | **PASS** | Local branch `main`; `HEAD` and read-only `git ls-remote origin refs/heads/main` both `55182854b4a8895661a547d5cec32cd29c2a0b05`. Public [GitHub repository](https://github.com/ChrisDemon0811/VietMailGuard) shows the V2-oriented tracked tree. V2.1 source/config/tests and preview docs are local uncommitted work, not yet on GitHub. No fetch/commit/push performed. |
| AGENTS contract | **PASS** | Local `AGENTS.md` section 97 explicitly marks V2 frozen and V2.1 in development, separates EN/VI/ZH email language from EN/VI UI, prohibits LLM label override, Test reuse and sensitive-data publication. Earlier rules remain as historical requirements. `AGENTS.md` itself is modified but uncommitted. |
| Ignore policy | **PASS for new files; WARNING for already tracked files** | `.env`, `.env.*` except blank-key `.env.example`, raw/V2.1 data, runtime DB, cache, checkpoints, binary models and local private config are ignored. `reports/v21/` remains ignored except `.gitkeep` and this metadata-only QA report. `.gitignore` cannot retroactively untrack already committed content. |
| Secret example | **FIXED locally** | Untracked `.env.example` contained a non-placeholder API-key-shaped value. It has been replaced with an **empty** `GEMINI_API_KEY=`. The actual `.env` was not opened, printed, edited or committed. A path-only scan (`rg -l AIza ...`) found no matching value in selected source/config/test/docs after replacement. If the prior value was active, the project owner should revoke/rotate it; do not reuse it. `git log --all -- .env.example` returned no committed history locally. |
| Previously tracked email text | **WARNING — publication review required** | `git ls-files` includes `data/curated/v2/Vietnamese_Curated_Dataset.csv` (35,380,746 bytes, `body` and `original_text` columns), `data/curated/v2/vietnamese_positive_review.csv` (959,943 bytes, `body_excerpt`), and two `data/evaluation/v2/*.csv` files with text columns. These predate 11G and already exist in the tracked tree. They were not opened beyond header rows, edited or removed here. Do **not** assume their content may be redistributed merely because code is MIT-licensed. A separate rights/privacy decision and, if approved, a deliberate `git rm --cached -- <exact path>` plus Git-history review are needed; an ignore rule alone is insufficient. |

**Safe commit proposal, not an automatic staging instruction:** after a diff and privacy scan, review `.gitignore`, `AGENTS.md`, `README.md`, `V21_RESEARCH_PREVIEW.md`, blank-key `.env.example`, `requirements-gemini.txt`, `pyproject.toml`, modified app/mail-service/database code and corresponding tests; the source-only `src/vietmailguard/v21/*.py`, `scripts/v21/*.py`, `tests/test_v21_*.py`; and non-local `config/v21/` policy/schema/example files. This report is explicitly unignored. Review every V2.1 report individually before changing its ignore status. **Never include** `.env`, raw/processed/curated/split/evaluation email files, local review decisions, `data/runtime/*.db`, embedding/cache/checkpoint files, or any `*.joblib`/large model download. Existing tracked V2 CSVs require a separate remediation decision, not another blanket `git add`.

## Application mode and end-to-end status

| Area | Status | What was verified / limit |
| --- | --- | --- |
| V2 frozen default | **PASS** | AppTest starts in `v2`; V2 analyzer remains `vietmailguard.inference`. Frozen model SHA-256 `b2ebecf641f2a5fd3ce9f0d0680a90884850e46b59c48ce0ab5bd4fee64f87fa` matches `model_metadata.json`. No V2 artifact was edited. |
| V2.1 ML gate | **PASS locally** | `model_availability()` returned verified `True` for the configured development candidate. Hash, saved Validation scope, classes, composer, TF-IDF/LinearSVC pipeline and uncalibrated contract are checked before use; invalid/missing artifact hides V2.1 and falls back to V2. A clean GitHub clone lacks the ignored binary/results. |
| V2.1 Hybrid / Gemini | **PASS offline/mock; LIVE NOT RUN** | Offline config check detected installed SDK, a configured key and model identifier without revealing key or calling network. AppTest/mock confirms selection and per-email consent/button before exactly one provider request. Reviewer disagreement yields `REVIEW` separately; it does not alter ML label, risk, routing or stored ML history. Live API availability, quota, translation quality and cost were intentionally not tested. |
| TREC06C binary | **PASS isolation** | The Ham/Spam model and its sealed binary Test live under separate research paths. No TREC binary classifier is imported into the main app/router; it is not a fourth three-class mailbox mode. The separate review UI has a documented local command. |
| Input / `.eml` / edge cases | **PASS automated; browser upload NOT VERIFIED** | Existing AppTest and parser/service suites cover text submit, empty handling, mixed-language/URL paths, `.eml` import and repeat detection, malformed input and fallback. Real browser file-picker/upload interaction was not available in this environment. No real email was sent to Gemini. |
| Mailbox / SQLite / feedback | **PASS automated** | Disposable-DB tests cover Inbox, Spam, Quarantine, Deleted, routing, open without re-inference, persistence after restart, feedback without ML rewrite, DB `integrity_check=ok`, foreign keys and history. |
| Streamlit server | **PASS health; browser manual NOT VERIFIED** | `python -m streamlit run app/app.py --server.headless true --server.port 8508` started and `/_stcore/health` returned HTTP 200; port 8508 was verified closed afterward. This smoke used Streamlit's default listener, which briefly advertised network URLs; future local runs should bind `--server.address 127.0.0.1`. Browser-control inventory returned no browser, so no claim of visual/manual browser QA. |
| Scientific claims | **PASS in preview docs** | V2 one-shot Test, V2.1 inherited Development Validation, TREC binary Validation and synthetic A/B Validation are separate. Chinese synthetic challenge is not real-world performance; no Chinese phishing or native Vietnamese final claim is made. |

### Research evidence and missing benchmarks

- V2 saved held-out result: 6,704 rows, Accuracy 0.997763, Macro F1 0.990900, Phishing recall 0.970588. Vietnamese translated Test: 179 rows, 0 Phishing.
- V2.1 development bootstrap: 31,304 Train + 6,709 inherited Validation; candidate Macro F1 0.985852. This Validation previously participated in V2 selection and is **not** V2.1 Final Test. Candidate confidence is unavailable because LinearSVC is uncalibrated.
- Audit: 1,147,062 rows = 1,102,345 new-source + 44,717 read-only V2 references; new sources remain 0 eligible after approval review. Bootstrap reuses 38,013 V2 Train/Validation rows, not V2 Test.
- TREC06C: 59,079 quality-screened for **binary** research; Validation Macro F1 0.993805; 0 human-reviewed/three-class eligible decisions. Test is sealed.
- Chinese synthetic: 3,030 audited; exact-candidate A 0.985852 vs B 0.984599 Macro F1 on the same EN/VI Development Validation. Synthetic-only diagnostic is not native Chinese evidence.
- Independent Chinese real-mail benchmark: manifest pending, **0 accepted**. Missing verified Normal, Spam and especially Phishing. Native Vietnamese three-class benchmark also absent.

Benchmark completion sequence: obtain permission/provenance for original Chinese emails; have qualified Chinese-language reviewers independently annotate all three classes with adjudication; perform campaign/parent/exact/normalized/template and cross-corpus leakage review against existing development and sealed references without opening held-out labels; document origin/variety/privacy; lock an independent manifest; freeze candidate/reviewer/calibration/risk protocol; seek an explicit one-time Final Evaluation instruction. Until then, no final Chinese or V2.1 three-language metric.

## Commands and verification

Run from repository root in PowerShell. These are the checks executed for 11G, not instructions to retrain:

| Command/check | Result |
| --- | --- |
| `git status --short --untracked-files=all`; `git branch --show-current`; `git remote -v`; `git ls-remote origin refs/heads/main`; `git rev-parse HEAD` | **PASS**; dirty local main, remote HEAD equal, no push |
| `git check-ignore -v` on `.env`, raw/V2.1 data, runtime DB, checkpoint and model path | **PASS**; all ignored by explicit rules |
| `.venv\Scripts\python.exe scripts/v21/check_gemini.py` | **PASS**, offline only, no API call/key output |
| `.venv\Scripts\python.exe -c "from vietmailguard.v21.runtime_inference import model_availability; print(model_availability())"` | **PASS**, local candidate verified |
| SHA-256 of `models/v2_bilingual/production_pipeline.joblib` vs `model_metadata.json` | **PASS**, exact match |
| `.venv\Scripts\python.exe -m pip check` | **PASS**, no broken requirements |
| `.venv\Scripts\python.exe -m pytest -q` | **PASS**, 441 passed, 3 skipped (before documentation-only finalization) |
| `.venv\Scripts\python.exe -m pytest -m "not production_artifact" -q` | **PASS**, 414 passed, 3 skipped, 27 deselected |
| `.venv\Scripts\python.exe -m pytest tests/test_v21_application_qa.py tests/test_v21_runtime_integration.py tests/test_v21_gemini_reviewer.py tests/test_v21_llm_reviewer.py tests/test_mailbox_ui.py tests/test_streamlit_app.py tests/test_mail_service.py tests/test_mail_router.py tests/test_mail_repository.py tests/test_eml_parser.py tests/test_inference_v2.py tests/test_language_detection.py tests/test_url_analyzer.py -q` | **PASS**, 143 passed |
| Streamlit startup and `Invoke-WebRequest -UseBasicParsing http://localhost:8508/_stcore/health` | **PASS**, HTTP 200; server stopped |
| Manual browser click/upload/visual inspection | **NOT VERIFIED**: no browser exposed to the automation environment. AppTest/widget-level checks are not a substitute for a real-browser upload/responsive pass. |
| Gemini live and all held-out Tests | **NOT RUN BY DESIGN**. |
| `git diff --check`; `git diff --name-only -- models/v2_bilingual results/v2_bilingual data/splits/v2 config/v2_production_frozen.json` | **PASS**, no whitespace errors or tracked V2 frozen-path changes |

Read-only smoke on synthetic examples (these are **not evaluation metrics**): `scripts/smoke_inference_v2.py` returned EN Normal→`normal`, EN Spam→`spam`, EN Phishing→`phishing`, VI Normal→`normal`, VI promotional→`normal` with `REVIEW` recommendation, VI phishing-style→`phishing`, and mixed EN/VI→`phishing`. This preserves the known short-promotion limitation and does not count a security recommendation as an ML Spam success. Direct V2.1 candidate smoke returned EN/VI/ZH/ mixed+URL predictions of `normal`/`normal`/`normal`/`phishing`, respectively, with `confidence=None`; ZH and mixed support status were `unvalidated`. These individual demonstrations cannot establish accuracy or Chinese phishing recall.

This preview is **runnable locally** with the configured artifacts, but **not ready for V2.1 production release or final multilingual claims**. The tracked-email-text publication warning and real-browser smoke gap remain open.
