# V2.1 label curation policy

`label_curation_policy.json` is the machine-readable source policy. At this
stage **no source/raw-label combination is approved for automatic mapping**.
This is a scientific decision, not a missing default. Existing binary labels
remain candidates until upstream semantics, provenance, rights, representative
content and duplicate conflicts are checked.

## Four states

- `eligible`: a canonical `normal`, `spam` or `phishing` label has direct,
  documented evidence and passed the audit/duplicate gates. Only this state
  can enter supervised V2.1 preparation.
- `review`: no usable label evidence, parse warning, contradictory label or
  other unresolved question. Not trainable.
- `weak_label`: a source/candidate label exists but its meaning is not yet
  established for VietMailGuard's three classes. Not trainable.
- `exclude`: unusable content such as an empty message. Not trainable.

`normal` means legitimate/ordinary email after source and content validation.
`spam` means predominantly unsolicited commercial/bulk promotion without
strong credential theft or financial-fraud evidence. `phishing` requires
social-engineering evidence such as impersonation, deceptive credential or
private-information requests. A binary positive, keyword, LLM prediction or
short message is not sufficient evidence. Mixed spam/scam/phishing stays in
review. Lottery/419 fraud requires an explicit taxonomy decision if it is not
clearly a phishing/social-engineering attempt.

## Reproducible human workflow

1. Read the upstream dataset documentation and verify raw-label semantics,
   collection provenance, origin, language and redistribution rights. Record
   any approved batch mapping in a separately reviewed policy change. This
   policy currently enables none.
2. Inspect `reports/v21/label_curation_distribution.csv` by dataset, original
   source, raw label and language. The local review queue contains a
   deterministic **sample**, not a claim that all rows were read. Inspect
   complete emails and further random samples before any source-wide approval.
   The queue's bounded-memory hash sample is reproducible within each
   dataset/source/language/raw-label/audit-reason group; it does not certify
   class purity or substitute for a statistically designed validation sample.
3. Copy `config/v21/label_decisions.example.csv` to the local, Git-ignored
   `data/v21/curated/label_decisions.csv`. For each genuinely reviewed row,
   enter `audit_id`, canonical decision or `review`/`exclude`, human reviewer,
   specific evidence and UTC review time. Do not fabricate or prefill decisions.
   Human `review` and `exclude` decisions are persisted as non-training states;
   a decision cannot restore an audit-excluded row without a separate audit fix.
4. Run `python -m scripts.v21.curate_labels`. The script checks audit identity,
   unchanged standardized parts, content, language/origin, duplicate/template
   and provenance links, label conflict and frozen-V2 overlap. Current policy
   conservatively blocks eligibility for any multi-row linked group until a
   separate group-level review procedure is approved. Rejected decisions stay
   out of `eligible` and appear by reason in the summary.
5. When evidence-confirmed rows exist, the script writes local
   `data/v21/curated/eligible/part-00001.csv` with original `raw_label`,
   canonical label, `label_provenance=human_review_v21`, curation rule/evidence,
   and `final_group_id`. It refuses to overwrite an existing eligible part.
   Prompt 07 must still approve this provenance in its own split policy and
   re-run leakage validation. Curation does not create splits or train models.

The full row-level status manifest and review sample are local derived data;
they may contain sensitive identifiers or text and must not be published.
The checkpointed audit index and curation sample cache allow resumable,
bounded-memory processing. A `.partial` status manifest is never accepted as
complete.

## Source-specific current decisions

| Source | Current raw-label interpretation | Automatic mapping |
|---|---|---|
| Enron original | No per-message label; corporate mailbox is not proof of normal | Disabled |
| TREC06C | `ham`/`spam` are two-class candidates, not three-class truth | Disabled |
| Merged | `0.0` normal candidate; `1.0` mixed positive candidate | Disabled |
| Balanced | Same underlying corpus family as Merged; not independent | Disabled |
| MeAJOR | `0.0`/`1.0` require subcorpus-specific semantics; preserve `source`, language and URL metadata | Disabled |
| Frozen V2 reference | Previously verified labels are overlap evidence only; contains held-out material | Never a V2.1 curation input |

The Phishing validation CSV remains challenge-only; PhishVN is URL-only, not
an email classification source. Neither is included here.
