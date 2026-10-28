"""Write a metadata-only TREC06C Vietnamese review progress report; no API calls."""

from __future__ import annotations

import json

from vietmailguard.v21.paths import resolve_v21_paths
from vietmailguard.v21.trec_review_assist import TrecReviewStore, local_model_status


def main() -> None:
    paths = resolve_v21_paths()
    store = TrecReviewStore(paths)
    progress = store.progress()
    plan = json.loads((store.directory / "deep_review_plan.json").read_text(encoding="utf-8"))
    output = paths.reports / "trec06c_translation_review_report.md"
    lines = [
        "# TREC06C Chinese review with Vietnamese assistance — Prompt 11B.2",
        "",
        "This is a local review-workflow snapshot, not a model evaluation or source-level approval.",
        "Original Chinese content remains authoritative. Translation and Gemini suggestions are unverified aids.",
        "",
        "## Current state",
        "",
        f"- Indexed TREC: {sum(plan['source_counts'].values()):,} (Ham {plan['source_counts']['ham']:,}; Spam {plan['source_counts']['spam']:,}).",
        f"- Ham conditional pool proposed, not approved: {sum(plan['ham_frame_by_length'].values()):,}.",
        f"- Local translations cached: {progress['translations_cached']:,} (cache records, not a unique-email estimate).",
        f"- Email IDs viewed in the review UI: {progress['viewed']:,} (includes smoke-test views).",
        f"- AI weak suggestions: {progress['ai_suggestions']:,}.",
        f"- Human decisions in existing manifest: {progress['human_decisions']:,}.",
        f"- Chinese-reader/original-read attestations via this UI: {progress['human_verified']:,} (self-attested, not independently authenticated).",
        f"- Eligible individual decisions: {progress['eligible']:,}; bulk-approved rows: 0.",
        f"- Referred to Chinese expert queue: {progress['expert_queue']:,}.",
        "",
        "## Translation and privacy",
        "",
        f"- Offline NLLB-compatible model configured and present: {'yes' if local_model_status()['available'] else 'no'}.",
        "- The optional local adapter loads only a user-supplied model directory (`VIETMAILGUARD_V21_ZH_VI_MODEL_DIR`) with `local_files_only=True`; no model is downloaded by default.",
        "- Gemini translation and suggestion each require a separate per-email button and explicit consent. The UI discloses possible charges. URL/email masking is partial and other personal data may remain.",
        "- Cache and manifests are under Git-ignored `data/v21/curated/trec06c/`. Do not publish their email text or AI quotes.",
        "- Translation may contain errors or omissions. It cannot certify a three-class label, and a translated-only reviewer cannot write a qualified decision through this UI.",
        "",
        "## Approval boundary",
        "",
        "The 6,802 Ham pool remains a proposal. Completion of the frozen human sample, anomaly adjudication, explicit error tolerance, user approval, and fresh V2/group leakage checks are still required before any batch rule is enabled. TREC Spam is not automatically VietMailGuard Spam; Phishing is not inferred from keywords. No V2 artifacts, V2.1 model, validation, or final test were changed or used for this workflow.",
        "",
        "Run locally on Windows PowerShell from the repository root:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m streamlit run scripts/v21/trec06c_review_app.py --server.address 127.0.0.1",
        ".\\.venv\\Scripts\\python.exe -m scripts.v21.report_trec06c_review",
        "```",
        "",
        "The loopback bind is important because the review UI displays potentially sensitive email content and has no authentication.",
        "",
        "## Verification on 2026-10-21",
        "",
        "- `.\\.venv\\Scripts\\python.exe -m pytest tests/test_v21_trec_curation.py -q` — PASS: 14 passed, 2 optional real-data tests skipped.",
        "- With `VIETMAILGUARD_RUN_TREC06C_CURATION_INTEGRATION=1`, the same command — PASS: 16 passed against the local TREC parts and UI smoke.",
        "- `.\\.venv\\Scripts\\python.exe -m pytest -m \"not production_artifact\" -q` — PASS: 380 passed, 3 skipped, 27 deselected.",
        "- `.\\.venv\\Scripts\\python.exe -m pytest -q` — PASS: 407 passed, 3 skipped.",
        "- `.\\.venv\\Scripts\\python.exe -m streamlit run scripts/v21/trec06c_review_app.py --server.address 127.0.0.1 --server.port 8765 --server.headless true` — PASS: local HTTP health 200; Streamlit AppTest reported zero UI exceptions.",
        "- No paid Gemini call or actual translation was made. The 1 viewed ID above was opened by the UI smoke test, not a human label review.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), **progress}, ensure_ascii=False))


if __name__ == "__main__":
    main()
