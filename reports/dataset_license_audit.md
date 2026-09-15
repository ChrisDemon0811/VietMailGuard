# Dataset license audit

Audit date: 2026-09-15  
Scope: VietMailGuard Version 1/2 corpus sources and corpus-derived artifacts  
Decision basis: repository metadata plus authoritative publisher/corpus pages available on the audit date

This report is a provenance and release-readiness audit, not legal advice.

## Executive conclusion

The VietMailGuard source code now has a root MIT License. That license is scoped to the project's software and does not automatically cover third-party datasets or derived email text.

Only the Nazario corpus has an explicit corpus-level license verified directly from its official README: CC BY 4.0. Zenodo record `8339691` declares CC BY 4.0 for its aggregate release, but this does not resolve every underlying message right or privacy obligation. The remaining English corpora, `data_vi.csv`, controlled translations and compound derivative datasets therefore remain `LICENSE_REVIEW` for redistribution.

No model or dataset was edited, deleted, moved, staged, committed or pushed during this task.
The new lightweight audit report was narrowly allowlisted in `.gitignore` so it can be tracked with the other release documentation; no dataset or model ignore rule was opened.

## Evidence reviewed

| Evidence | Finding |
|---|---|
| `AGENTS.md` | Raw data is immutable; translated/synthetic data must not be presented as native; provenance must be preserved. |
| `config/datasets.json` and existing audit/curation reports | Established label semantics, local source names, training eligibility and translation provenance. |
| [Zenodo record 8339691](https://doi.org/10.5281/zenodo.8339691) and [REST metadata](https://zenodo.org/api/records/8339691) | Record contains the six English CSVs; metadata declares `cc-by-4.0` and open access. |
| [CEAS challenge page](https://ceas.cc/2008/challenge.html?redirect=true) | Confirms challenge context; no explicit corpus redistribution license found on the audited page. |
| [AUEB software/data page](https://nlp.cs.aueb.gr/en/software_data.html) | Identifies Enron-Spam and Ling-Spam composition; no explicit license found on the audited page. |
| [CMU Enron page](https://www.cs.cmu.edu/~enron/) | Describes public/research distribution and privacy concerns; no standard content license found on the audited page. |
| [Nazario README](https://monkey.org/~jose/phishing/README.txt) | Explicitly declares CC BY 4.0 and describes hand classification/personal-inbox provenance. |
| [CLAIR fraud page](https://yale-lily.github.io/downloads/fraud/index.html) | Describes 2,978-message collection and asks users to contact the lab for citation; no explicit license found. |
| [SpamAssassin corpus README](https://spamassassin.apache.org/old/publiccorpus/readme.html) | Describes public/consented sources and explicitly says message-text copyright remains with original senders. |
| [Helsinki-NLP model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi) | Declares Apache-2.0 for the translation model; this is not a license for source emails or translated outputs. |

## Dataset decisions

| Dataset/artifact | Verified license evidence | Redistribution status | Decision |
|---|---|---|---|
| CEAS_08 | Zenodo aggregate: CC BY 4.0; upstream corpus grant not found | License / redistribution permission: NOT VERIFIED | Keep email text local |
| Enron | Zenodo aggregate: CC BY 4.0; AUEB/CMU pages do not state a standard content license | License / redistribution permission: NOT VERIFIED | Keep email text local; privacy review required |
| Ling | Zenodo aggregate: CC BY 4.0; AUEB page has no explicit license | License / redistribution permission: NOT VERIFIED | Keep email text local |
| Nazario | Official README: CC BY 4.0 | Verified with attribution/modification requirements | May redistribute only with license compliance plus privacy/security review |
| Nigerian_Fraud | Zenodo aggregate: CC BY 4.0; CLAIR page has no explicit license | License / redistribution permission: NOT VERIFIED | Keep email text local |
| SpamAssassin corpus | README permits public research availability but reserves message-text copyright to original senders | License / redistribution permission: NOT VERIFIED | Do not apply Apache software license; keep text local |
| `data_vi.csv` | No authoritative source/license metadata found | License / redistribution permission: NOT VERIFIED | Do not publish raw or derivatives |
| Controlled translated augmentation | Translation model is Apache-2.0; parent email rights vary | License / redistribution permission: NOT VERIFIED | Do not publish row-level translations |
| Vietnamese curated/review files | Derivatives of `data_vi.csv` and linked English sources | License / redistribution permission: NOT VERIFIED | Exclude from GitHub Release |
| Bilingual master/splits/robustness data | Compound or transformed derivatives | License / redistribution permission: NOT VERIFIED | Local only/license review |
| Synthetic short-form challenge | Project-authored synthetic data; no separate dataset license declared | License / redistribution permission: NOT VERIFIED | Decide a dataset license separately before release |

## Why the Zenodo record is not treated as blanket clearance

The Zenodo API metadata unambiguously reports `cc-by-4.0` for the aggregate record. This is important evidence and must be preserved in attribution. It is not treated as blanket clearance because:

- the record combines multiple pre-existing corpora;
- the upstream SpamAssassin README expressly retains copyright in message text with original senders;
- the Enron source warns about privacy and does not state a standard content license on the audited page;
- AUEB, CEAS and CLAIR pages audited here do not provide explicit redistribution terms;
- copyright permission does not automatically resolve privacy, publicity, confidentiality or personal-data obligations.

This conservative decision does not assert that redistribution is prohibited. It states that permission has not been verified sufficiently for a public release.

## Large Vietnamese curated files

Both requested files exist and are currently tracked in the Git index:

| Path | Size | SHA-256 | Current decision |
|---|---:|---|---|
| `data/curated/v2/Vietnamese_Curated_Dataset.csv` | 35,380,746 bytes | `0d866aab639c19f2399ba7c634ca78e016e6f78e06e020b91d0de546eef99a96` | Do not include raw email text in GitHub Release; rights not verified |
| `data/curated/v2/vietnamese_positive_review.csv` | 959,943 bytes | `38daa6c809c6ee2a10ba53eafddc4c1d1d80a349267124f80d4f1b211d49216a` | Do not include excerpts in GitHub Release; rights not verified |

Because they are already tracked, `.gitignore` alone cannot remove them from the index. If project policy later authorizes index removal while retaining local files, run these exact commands from the repository root:

```text
git rm --cached -- "data/curated/v2/Vietnamese_Curated_Dataset.csv"
git rm --cached -- "data/curated/v2/vietnamese_positive_review.csv"
```

These commands were **not executed**. They affect the current index only; they do not erase content from previous commits. Before making the repository public, inspect Git history and choose an authorized history-remediation procedure if the files ever appeared in published history.

## Integrity observations

- The six English local files are traceable to Zenodo record `8339691`; four have exact published MD5 matches and Enron/Ling were previously verified as logical matches after newline normalization.
- `data_vi.csv` remains marked `translated`, never `native`.
- Controlled translations retain English parent provenance and remain marked `translated` / `controlled_translation`.
- Translation-model Apache-2.0 and project-code MIT are not propagated to email content.
- Review/exclude semantics do not change any rights conclusion.
- Hashes and local file sizes were read only; no dataset content was rewritten.

## Release recommendation

### Safe for main branch from a licensing-scope perspective

- VietMailGuard source code under MIT;
- configs, tests and documentation authored by the project;
- aggregate metrics/reports that do not reproduce third-party email text;
- `DATASETS.md` and this audit.

This statement does not override the repository's size, secret-scanning or artifact policies.

### Requires separate Release review

- frozen model binaries, because their training sources have mixed/unverified redistribution status;
- any scientific supplement containing row-level messages or excerpts.

### Keep local / license review

- raw, processed, curated and split datasets;
- cross-language overlap/review files containing text;
- translated augmentation and robustness rows;
- the two tracked Vietnamese curated CSVs until rights and history handling are resolved.

## Follow-up needed before public release

1. Ask the original/maintaining publishers of CEAS, Enron-Spam, Ling-Spam and CLAIR/Nigerian Fraud for explicit redistribution terms applicable to the local representations.
2. Resolve the relationship between the Zenodo aggregate license and underlying message copyrights, especially SpamAssassin.
3. Identify the source, translator/creator and license for `data_vi.csv`.
4. Perform privacy/PII review independently of copyright licensing.
5. Decide a separate data license for project-authored synthetic challenge data if it will be published.
6. Remove or remediate tracked corpus-derived content only through an explicitly authorized Git/history policy.
