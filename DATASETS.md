# Dataset provenance and licensing

Ngày kiểm tra: 2026-09-15

Tài liệu này ghi nhận provenance, cách VietMailGuard sử dụng dữ liệu và trạng thái quyền phân phối đã xác minh được. Đây không phải tư vấn pháp lý.

## Software license và dataset license là hai phạm vi khác nhau

Source code do dự án VietMailGuard tạo ra được phát hành theo MIT License trong [`LICENSE`](LICENSE).

MIT License của source code **không** tự động áp dụng cho:

- corpus email của bên thứ ba;
- nội dung email trong raw, processed, curated hoặc split data;
- bản dịch hoặc biến đổi bắt nguồn từ email của bên thứ ba;
- model/dataset bên ngoài chỉ vì chúng được dùng bởi VietMailGuard.

Quyền tác giả, quyền riêng tư, dữ liệu cá nhân và điều kiện phân phối lại của từng nguồn phải được đánh giá riêng. Việc một corpus có thể tải công khai hoặc được dùng cho nghiên cứu không đồng nghĩa với một giấy phép phân phối lại rõ ràng.

## Nguồn phân phối chung của sáu corpus tiếng Anh

Sáu file `CEAS_08.csv`, `Enron.csv`, `Ling.csv`, `Nazario.csv`, `Nigerian_Fraud.csv` và `SpamAssasin.csv` được lấy từ bản phát hành tổng hợp [Phishing Email Curated Datasets](https://doi.org/10.5281/zenodo.8339691), Zenodo record `8339691`, version `v1`, công bố ngày 2023-09-13.

Metadata REST của record khai báo `license.id = cc-by-4.0` và `access_right = open`. Tuy nhiên, record là một tuyển tập từ nhiều corpus có chủ thể quyền khác nhau. Vì vậy tài liệu này ghi riêng:

1. license khai báo ở record tổng hợp; và
2. bằng chứng license/quyền phân phối của corpus gốc.

Khi chuỗi quyền từ nguồn gốc đến file local chưa rõ, quyết định an toàn của dự án là **không phân phối nội dung email**, kể cả khi metadata của record tổng hợp khai báo CC BY 4.0.

## Tóm tắt sử dụng trong Version 2

Các con số dưới đây là row đủ điều kiện sau cleaning/curation và đã đi vào split V2. Row `review`, `exclude`, corrupt hoặc weakly labeled không được tính.

| Source | Train | Validation | Test | Vai trò |
|---|---:|---:|---:|---|
| CEAS_08 | 12,093 | 2,648 | 2,648 | English normal và row-level high-confidence curated spam |
| Enron | 11,159 | 2,555 | 2,553 | English normal và row-level high-confidence curated spam |
| Ling | 1,752 | 358 | 357 | English normal và row-level high-confidence curated spam |
| Nazario | 936 | 306 | 306 | English phishing |
| SpamAssassin | 3,118 | 662 | 661 | English normal và row-level high-confidence curated spam |
| data_vi | 786 | 180 | 179 | Existing translated Vietnamese normal/evidence-confirmed spam |
| controlled_translation | 1,460 | 0 | 0 | Controlled Vietnamese augmentation; train only |

Không có native Vietnamese corpus trong V2. `Nigerian_Fraud` không đi vào train/validation/test V2.

## CEAS 2008

- **Dataset name:** CEAS 2008 Spam Filter Challenge corpus representation.
- **Local filename:** `data/raw/CEAS_08.csv`.
- **Original/distribution sources:** [CEAS 2008 Spam Filter Challenge](https://ceas.cc/2008/challenge.html?redirect=true); [Zenodo curated record](https://doi.org/10.5281/zenodo.8339691).
- **Purpose in project:** audit label semantics; normal source; row-level curation of selected positive examples for spam; V1/V2 modeling.
- **Language/origin:** English; original third-party email corpus, not synthetic and not translated by VietMailGuard.
- **Original label semantics:** raw `0` is ham/legitimate mail and maps to `normal`; raw `1` is the positive spam-filter class but local audit found commercial spam mixed with phishing/scam and boundary noise. Raw `1` remains `review` unless a row has separate high-confidence curation provenance.
- **Use:** eligible rows occur in V2 train/validation/test as shown above; uncurated raw-positive rows are excluded from training.
- **File provenance confidence:** high. Local MD5 `1f0d59191eec892a709995d48cd8decb` matches the Zenodo file listing.
- **License evidence:** the Zenodo aggregate record declares CC BY 4.0. The audited CEAS challenge page describes the challenge but does not state an explicit license for redistribution of the underlying email contents.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** do not include raw or row-level derivative email text in Git or a GitHub Release without further rights/privacy review.

## Enron-Spam representation

- **Dataset name:** Enron-Spam-derived curated CSV.
- **Local filename:** `data/raw/Enron.csv`.
- **Original/distribution sources:** [AUEB Enron-Spam listing](https://nlp.cs.aueb.gr/en/software_data.html); [CMU Enron Email Dataset](https://www.cs.cmu.edu/~enron/); [Zenodo curated record](https://doi.org/10.5281/zenodo.8339691).
- **Purpose in project:** normal source; row-level curation of selected spam positives; V1/V2 modeling.
- **Language/origin:** English; combined third-party corpus. Ham derives from Enron mail and positive examples derive from the Enron-Spam collection.
- **Original label semantics:** raw `0` is legitimate Enron mail and maps to `normal`; raw `1` is an Enron-Spam positive class containing bulk spam plus phishing and financial-fraud examples in the local audit. Raw `1` remains `review` except for separately curated high-confidence spam rows.
- **Use:** eligible rows occur in V2 train/validation/test; uncurated raw-positive rows are excluded.
- **File provenance confidence:** high for logical identity with the Zenodo CSV. The byte-level MD5 differs because the local copy uses different newline representation; prior audit found equivalent DataFrame content after newline normalization.
- **License evidence:** Zenodo aggregate metadata declares CC BY 4.0. AUEB identifies the corpus composition but does not publish an explicit license on the audited page. CMU distributes the underlying Enron data for research and warns about privacy, but the audited page does not grant a standard content license.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** do not distribute raw or derivative row-level email text without resolving both Enron and added-spam rights and privacy obligations.

## Ling-Spam representation

- **Dataset name:** Ling-Spam-derived curated CSV.
- **Local filename:** `data/raw/Ling.csv`.
- **Original/distribution sources:** [AUEB Ling-Spam listing](https://nlp.cs.aueb.gr/en/software_data.html); [Zenodo curated record](https://doi.org/10.5281/zenodo.8339691).
- **Purpose in project:** normal source; row-level spam curation; V1/V2 modeling.
- **Language/origin:** English; third-party mailing-list ham plus spam, not translated or synthetic.
- **Original label semantics:** raw `0` is ham/legitimate mailing-list mail and maps to `normal`; raw `1` is predominantly spam but includes scam/fraud in local samples. Raw `1` remains `review` except for separately curated high-confidence spam rows.
- **Use:** eligible rows occur in V2 train/validation/test; uncertain positives are excluded.
- **File provenance confidence:** high for logical identity with the Zenodo CSV after newline normalization.
- **License evidence:** Zenodo aggregate metadata declares CC BY 4.0. The audited AUEB listing identifies and offers the corpus but does not state an explicit redistribution license.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** keep raw and row-level derivatives local pending verification.

## Nazario phishing corpus

- **Dataset name:** Jose Nazario phishing corpus representation.
- **Local filename:** `data/raw/Nazario.csv`.
- **Original/distribution sources:** [official Nazario corpus README](https://monkey.org/~jose/phishing/README.txt); [Zenodo curated record](https://doi.org/10.5281/zenodo.8339691).
- **Purpose in project:** high-confidence English phishing source for V1/V2.
- **Language/origin:** primarily English; third-party phishing email collected from the maintainer's personal inbox, not translated or synthetic.
- **Original label semantics:** the corpus README describes the messages as hand-classified phishing. Local raw label `1` maps to `phishing`, subject to normal empty/corrupt/duplicate cleaning.
- **Use:** eligible rows occur in V2 train/validation/test.
- **File provenance confidence:** high. Local MD5 `4022d055bb7cf8602f30f768f652e91e` matches the Zenodo file listing, and corpus semantics agree with the official README.
- **License status:** verified as **CC BY 4.0** by the official corpus README. Redistribution and adaptations require attribution and an indication of modifications.
- **Redistribution decision:** license permits redistribution with attribution, but email content still requires privacy, personal-data and security review before public release. Do not send corpus messages through live email systems.

## Nigerian Fraud / CLAIR fraud email representation

- **Dataset name:** Nigerian/419 fraud email CSV derived from the CLAIR collection.
- **Local filename:** `data/raw/Nigerian_Fraud.csv`.
- **Original/distribution sources:** [CLAIR fraud collection page](https://yale-lily.github.io/downloads/fraud/index.html); [Zenodo curated record](https://doi.org/10.5281/zenodo.8339691).
- **Purpose in project:** taxonomy and curation review for 419/advance-fee, inheritance, lottery and related fraud; not used as spam or phishing ground truth in V2.
- **Language/origin:** English; third-party fraud email corpus, not translated or synthetic.
- **Original label semantics:** raw `1` denotes fraud mail. VietMailGuard keeps it as `review` because the current three-class policy does not automatically equate all financial/419 fraud with credential phishing.
- **Use:** review/curation only; no rows in V2 train/validation/test.
- **File provenance confidence:** high for the local Zenodo artifact (MD5 `65fd47ae7cb4ae762e4be42b11905d63` matches), but only medium for its exact transformation from the upstream CLAIR mbox. The upstream page says 2,978 messages, while the curated CSV has 3,332 rows; prior audit found no basis to delete rows merely to match the upstream count.
- **License evidence:** Zenodo aggregate metadata declares CC BY 4.0. The audited CLAIR page credits the collector and asks users to contact the lab for the appropriate citation, but it does not state an explicit redistribution license.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** keep raw and derived row-level content local pending written clarification.

## SpamAssassin public mail corpus representation

- **Dataset name:** SpamAssassin public mail corpus-derived CSV. The local filename contains the historical typo `SpamAssasin`.
- **Local filename:** `data/raw/SpamAssasin.csv`.
- **Original/distribution sources:** [SpamAssassin public corpus README](https://spamassassin.apache.org/old/publiccorpus/readme.html); [Zenodo curated record](https://doi.org/10.5281/zenodo.8339691).
- **Purpose in project:** normal source; row-level spam curation; V1/V2 modeling.
- **Language/origin:** primarily English; third-party public/consented mail, mailing-list material and newsletters, not translated or synthetic.
- **Original label semantics:** `easy_ham`/`hard_ham` correspond to legitimate mail and local raw `0` maps to `normal`; the spam subsets correspond to local raw `1`. Local positive samples also include fraud/social-engineering content, so raw `1` remains `review` except for separately curated high-confidence spam rows.
- **Use:** eligible rows occur in V2 train/validation/test; uncertain positives are excluded.
- **File provenance confidence:** high for the local Zenodo artifact. Local MD5 `d921e0b2333bfaa058bd38193e6b3fbd` matches the record listing.
- **License evidence:** the corpus README says the messages were public, provided with knowledge they might be made public, sent by the corpus maintainer, or public newsletters. It also explicitly states that copyright in message text remains with the original senders. The Apache License for SpamAssassin software is **not** a license for these messages.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** do not publish raw or row-level derivative content without message-level rights/privacy review.

## `data_vi.csv`

- **Dataset name:** existing Vietnamese translated email dataset.
- **Local filename:** `data/raw/data_vi.csv`.
- **Original source:** not documented in repository metadata beyond the local file. No authoritative source URL, translator identity, selection method or license statement was found.
- **Purpose in project:** V2 Vietnamese audit, standardization, cross-language linking and cautious curation.
- **Language/origin:** Vietnamese; `data_origin = translated`. Audit found strong observable evidence of English-origin translations, including Enron identities, English organizations, translated subject prefixes and cross-language matches. It is not native Vietnamese ground truth.
- **Original label semantics:** raw `0` is accepted as translated normal with provenance `translated_dataset_verified_normal`; raw `1` is a mixed positive pool and remains `review` unless supported by high-confidence English-parent evidence or genuine human confirmation.
- **Use:** after cleaning/linking, 786 eligible rows occur in V2 train, 180 in validation and 179 in test. Remaining review/exclude rows do not enter splits.
- **File provenance confidence:** high that it is translated rather than native; low for exact original source, translator and licensing provenance. SHA-256 is `d9fee4aee5d13126852d543758595d9ea78e67d074d5ab24eb0dbeb2f9ea2125`.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** keep raw, standardized, curated and split row-level content local; do not call it native Vietnamese or publish it in a GitHub Release.

## Controlled Vietnamese translated augmentation

- **Dataset name:** VietMailGuard controlled English-to-Vietnamese training augmentation.
- **Local filename:** `data/processed/Vietnamese_Translated_Augmentation.csv`.
- **Original source:** generated locally from 1,500 unique, high-confidence English train parents selected across CEAS_08, Enron, Ling, SpamAssassin and Nazario.
- **Translation implementation:** [Helsinki-NLP/opus-mt-en-vi](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi), pinned revision `989c9fb9ec63987901022baf0182dcec3e149be6`, deterministic local generation. Model card license: Apache-2.0.
- **Purpose in project:** train-only bilingual augmentation with parent/group linkage and protected-token validation.
- **Language/origin:** Vietnamese; `data_origin = translated`, `augmentation_type = controlled_translation`. It is neither native nor real-world Vietnamese data.
- **Label semantics:** labels are inherited only from eligible, high-confidence English parents; no rule-only or review label is promoted.
- **Use:** 1,460 quality-passing rows are included in V2 train only; 40 failed-quality rows are not training eligible. No augmentation row is in validation/test.
- **Provenance confidence:** high for the local transformation method and parent linkage.
- **License evidence:** Apache-2.0 applies to the translation model implementation/weights as declared by its model card. It does not grant rights to the source emails and does not automatically license translations derived from those emails.
- **License / redistribution permission: NOT VERIFIED**
- **Redistribution decision:** do not publish row-level augmentation until every parent-source rights chain and privacy obligation is resolved.

## Derived datasets and evaluation artifacts

| Artifact family | Origin and purpose | License / redistribution status | Public-release decision |
|---|---|---|---|
| `data/processed/English_Master_Dataset.csv` | Standardized, curated compilation of the six English sources | **License / redistribution permission: NOT VERIFIED** | Local only/license review |
| `data/processed/Vietnamese_Translated_Master.csv` | Standardized derivative of `data_vi.csv` | **License / redistribution permission: NOT VERIFIED** | Local only/license review |
| `data/curated/v2/Vietnamese_Curated_Dataset.csv` | Full standardized Vietnamese content plus curated status/provenance | **License / redistribution permission: NOT VERIFIED** | Do not include in GitHub Release |
| `data/curated/v2/vietnamese_positive_review.csv` | Positive review pool containing subject/body excerpts | **License / redistribution permission: NOT VERIFIED** | Do not include in GitHub Release |
| `data/processed/Bilingual_Master_Dataset.csv` and `data/splits/v2/*` | Compound derivatives used for leakage-safe V2 train/validation/test | **License / redistribution permission: NOT VERIFIED** | Local only/license review |
| `data/evaluation/v2/vietnamese_robustness.csv` | Transformations derived from translated Vietnamese corpus text | **License / redistribution permission: NOT VERIFIED** | Local only/license review |
| `data/evaluation/v2/short_form_challenge.csv` | Project-authored synthetic challenge scenarios; not production held-out data | No separate dataset license has been declared; **License / redistribution permission: NOT VERIFIED** | May be licensed separately later; do not infer coverage from third-party corpus licenses |

Compilation, cleaning, deduplication, label curation or translation does not erase upstream rights. A mixed dataset cannot be assigned a blanket license merely because one component or the processing code has a permissive license.

## Required attribution and release policy

- Cite the Zenodo record and its requested papers when using its distributed files.
- For Nazario material, comply with CC BY 4.0 attribution and modification-notice requirements.
- Do not use the Apache software license as a license for SpamAssassin message text.
- Do not treat the Apache-2.0 translation-model license as a license for controlled translations.
- Do not publish email bodies, addresses or headers without privacy/security review even when copyright permission appears adequate.
- Until an item is explicitly cleared, keep raw, processed, curated, split and corpus-derived evaluation datasets outside Git and GitHub Releases.

## Reproducibility without redistributing email text

The public repository can contain source code, configuration, schemas, aggregate counts, hashes and scientific reports that do not reproduce row-level email text. Runtime model binaries can be separate release assets only after their redistribution and privacy implications are reviewed. Users should obtain eligible third-party datasets from their original publishers and run the documented pipeline locally.
