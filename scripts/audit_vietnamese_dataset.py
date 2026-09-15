"""Audit the translated Vietnamese corpus without modifying or standardizing it."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

from vietmailguard.dataset_standardizer import normalized_duplicate_text


PLACEHOLDERS = {"", "empty", "none", "null", "nan", "n/a", "na"}
SPACED_URL_PATTERN = re.compile(
    r"\b(?:https?\s*:\s*/\s*/|www\s*\.)\s*[\w.-]+(?:\s*\.\s*[\w.-]+)+(?:\s*/\s*[^\s,;]*)?",
    re.IGNORECASE,
)
SPACED_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+\s*@\s*[A-Z0-9.-]+(?:\s*\.\s*[A-Z]{2,})\b",
    re.IGNORECASE,
)

TRANSLATION_EVIDENCE_PATTERNS = {
    "translated_subject_header": re.compile(r"^\s*(?:chủ đề|chủ thể)\s*:", re.IGNORECASE),
    "enron_name": re.compile(r"\benron\b", re.IGNORECASE),
    "enron_people": re.compile(
        r"\b(?:vince kaminski|sandeep kohli|mike(?: a)? roberts|stinson gibner|"
        r"vasant shanbhogue|jeff dasovich|sara shackleton)\b",
        re.IGNORECASE,
    ),
    "english_organization_or_product": re.compile(
        r"\b(?:microsoft|adobe|yahoo|norton|corel|powerisk|lasalle|sky bank|"
        r"morgan chase|goldengraphix|windows xp)\b",
        re.IGNORECASE,
    ),
    "translated_mail_routing": re.compile(
        r"(?:chuyển tiếp bởi|\btới\s*:|\bcc\s*:|\bbcc\s*:)", re.IGNORECASE
    ),
    "spaced_or_malformed_url": re.compile(
        r"(?:https?\s*:\s*/\s*/|www\s*\.\s+)", re.IGNORECASE
    ),
    "spaced_or_malformed_email": SPACED_EMAIL_PATTERN,
}

COMMERCIAL_RULES = {
    "advertising_or_marketing": re.compile(
        r"(?:quảng cáo|tiếp thị|chào mời|bản tin thương mại)", re.IGNORECASE
    ),
    "promotion_or_discount": re.compile(
        r"(?:khuyến mãi|giảm giá|ưu đãi|giá đặc biệt|sale|discount)", re.IGNORECASE
    ),
    "purchase_or_product_offer": re.compile(
        r"(?:mua ngay|đặt hàng|sản phẩm|dịch vụ|phần mềm|khoản vay|thế chấp)",
        re.IGNORECASE,
    ),
    "marketing_opt_out": re.compile(
        r"(?:hủy đăng ký|xóa khỏi danh sách|loại khỏi danh sách|không muốn nhận|unsubscribe)",
        re.IGNORECASE,
    ),
    "promotional_call_to_action": re.compile(
        r"(?:bấm vào đây|nhấp vào đây|truy cập trang web|đăng ký ngay)", re.IGNORECASE
    ),
}

PHISHING_SCAM_RULES = {
    "credential_or_login": re.compile(
        r"(?:mật khẩu|đăng nhập|xác minh.{0,50}tài khoản|"
        r"tài khoản.{0,50}xác minh)",
        re.IGNORECASE,
    ),
    "account_suspension": re.compile(
        r"(?:tài khoản.{0,60}(?:bị khóa|bị đóng|vô hiệu hóa|đình chỉ)|"
        r"quyền truy cập.{0,40}(?:bị hạn chế|bị khóa))",
        re.IGNORECASE,
    ),
    "advance_fee_or_inheritance": re.compile(
        r"(?:người thừa kế|thừa kế|nigeria|nigerian|"
        r"quỹ.{0,80}(?:triệu|đô la)|western union)",
        re.IGNORECASE,
    ),
    "lottery_or_prize_fraud": re.compile(
        r"(?:xổ số|trúng thưởng|người chiến thắng|giải thưởng.{0,50}(?:nhận|yêu cầu))",
        re.IGNORECASE,
    ),
    "bank_details_or_transfer": re.compile(
        r"(?:thông tin ngân hàng|chi tiết ngân hàng|tài khoản ngân hàng|"
        r"chuyển khoản|chuyển tiền.{0,50}tài khoản)",
        re.IGNORECASE,
    ),
    "identity_document_request": re.compile(
        r"(?:hộ chiếu|bằng lái xe|giấy tờ tùy thân).{0,100}(?:gửi|cung cấp|yêu cầu)",
        re.IGNORECASE,
    ),
}

CROSS_LANGUAGE_PAIRS = [
    ("trạm thời tiết sacramento", "sacramento weather station"),
    ("powerisk 2001", "powerisk 2001"),
    ("quầy tin tức enron ấn độ", "enron india newsdesk"),
    ("tay súng giao dịch chứng khoán", "stock trading gunslinger"),
    ("những ngôi nhà mới đáng kinh ngạc", "unbelievable new homes made easy"),
    ("in 4 màu", "4 color printing"),
]

SAMPLE_ASSESSMENTS = {
    0: {
        1368: "legitimate_business_correspondence",
        1369: "legitimate_internal_enron",
        1370: "legitimate_internal_news_forward",
        1371: "legitimate_business_invitation",
        1991: "low_information_ambiguous",
        3712: "requested_commercial_information_boundary",
    },
    1: {
        0: "commercial_design_solicitation",
        1: "gibberish_or_obfuscated_spam",
        2: "loan_offer_financial_lure",
        3: "commercial_printing_ad",
        31: "advance_fee_fraud",
        68: "credential_phishing",
        118: "credential_phishing",
        228: "lottery_fraud",
        568: "malformed_or_near_empty",
    },
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_scalar(value: object) -> str:
    """Normalize scalar text for missing and placeholder checks."""
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def relaxed_template_text(value: object) -> str:
    """Normalize translated URL/email spacing before template comparison."""
    text = normalized_duplicate_text("", value, template=True)
    text = SPACED_URL_PATTERN.sub(" <url> ", text)
    text = SPACED_EMAIL_PATTERN.sub(" <email> ", text)
    return " ".join(text.split())


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def duplicate_summary(hashes: pd.Series, labels: pd.Series) -> dict[str, int]:
    counts = hashes.value_counts()
    duplicate_hashes = counts[counts > 1]
    conflicting = (
        pd.DataFrame({"hash": hashes, "label": labels})
        .groupby("hash")["label"]
        .nunique()
    )
    conflicting_hashes = set(conflicting[conflicting > 1].index)
    return {
        "groups": int(len(duplicate_hashes)),
        "participating_rows": int(duplicate_hashes.sum()),
        "redundant_rows_beyond_first": int((duplicate_hashes - 1).sum()),
        "conflicting_label_groups": int(len(conflicting_hashes)),
        "conflicting_label_rows": int(
            sum(int(counts[item]) for item in conflicting_hashes)
        ),
    }


def duplicate_records(
    frame: pd.DataFrame,
    hashes: pd.Series,
    duplicate_type: str,
    label_column: str,
    text_column: str,
) -> list[dict[str, object]]:
    counts = hashes.value_counts()
    records: list[dict[str, object]] = []
    for digest, count in counts[counts > 1].items():
        indexes = hashes[hashes.eq(digest)].index
        labels = sorted({str(frame.at[index, label_column]) for index in indexes})
        conflict = len(labels) > 1
        for index in indexes:
            records.append(
                {
                    "duplicate_type": duplicate_type,
                    "group_id": f"{duplicate_type}_{digest[:20]}",
                    "group_size": int(count),
                    "conflicting_labels": conflict,
                    "labels_in_group": "|".join(labels),
                    "original_row_id": int(index) + 2,
                    "raw_label": frame.at[index, label_column],
                    "text_excerpt": normalize_scalar(frame.at[index, text_column])[:300],
                }
            )
    return records


def length_statistics(series: pd.Series) -> dict[str, float | int]:
    chars = series.map(lambda value: len(str(value)))
    words = series.map(lambda value: len(str(value).split()))
    return {
        "min_chars": int(chars.min()),
        "p25_chars": float(chars.quantile(0.25)),
        "median_chars": float(chars.median()),
        "mean_chars": float(chars.mean()),
        "p75_chars": float(chars.quantile(0.75)),
        "p95_chars": float(chars.quantile(0.95)),
        "max_chars": int(chars.max()),
        "median_words": float(words.median()),
        "mean_words": float(words.mean()),
    }


def pattern_counts(series: pd.Series, patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    return {
        name: int(series.map(lambda value: bool(pattern.search(str(value)))).sum())
        for name, pattern in patterns.items()
    }


def sample_rows(frame: pd.DataFrame, label_column: str, text_column: str) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for label, assessments in SAMPLE_ASSESSMENTS.items():
        for index, assessment in assessments.items():
            if index not in frame.index or frame.at[index, label_column] != label:
                continue
            text = normalize_scalar(frame.at[index, text_column])
            samples.append(
                {
                    "original_row_id": int(index) + 2,
                    "raw_label": label,
                    "observed_assessment": assessment,
                    "excerpt": text[:360],
                }
            )
    return samples


def cross_language_evidence(
    vietnamese: pd.DataFrame,
    english_path: Path,
    text_column: str,
) -> list[dict[str, object]]:
    if not english_path.exists():
        return []
    english = pd.read_csv(english_path, encoding="utf-8-sig")
    english_text = (
        english.get("subject", pd.Series("", index=english.index)).fillna("").astype(str)
        + " "
        + english.get("body", pd.Series("", index=english.index)).fillna("").astype(str)
    ).str.casefold()
    vietnamese_text = vietnamese[text_column].fillna("").astype(str).str.casefold()
    evidence: list[dict[str, object]] = []
    for vietnamese_marker, english_marker in CROSS_LANGUAGE_PAIRS:
        vi_indexes = vietnamese_text[vietnamese_text.str.contains(vietnamese_marker, regex=False)].index
        en_indexes = english_text[english_text.str.contains(english_marker, regex=False)].index
        if len(vi_indexes) == 0 or len(en_indexes) == 0:
            continue
        vi_index = int(vi_indexes[0])
        en_index = int(en_indexes[0])
        evidence.append(
            {
                "vietnamese_original_row_id": vi_index + 2,
                "vietnamese_marker": vietnamese_marker,
                "vietnamese_raw_label": vietnamese.at[vi_index, "spam"],
                "english_original_row_id": en_index + 2,
                "english_marker": english_marker,
                "english_raw_label": english.at[en_index, "label"],
            }
        )
    return evidence


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(
    *,
    source_path: Path,
    source_hash: str,
    frame: pd.DataFrame,
    label_column: str,
    text_column: str,
    missing: dict[str, int],
    empty_text_rows: int,
    fully_empty_rows: int,
    duplicate_summaries: dict[str, dict[str, int]],
    lengths: dict[str, dict[str, float | int]],
    translation_counts: dict[str, int],
    commercial_counts: dict[str, int],
    phishing_counts: dict[str, int],
    commercial_any: int,
    phishing_any: int,
    signal_overlap: int,
    samples: list[dict[str, object]],
    cross_language: list[dict[str, object]],
    unicode_stats: dict[str, int],
) -> str:
    label_counts = frame[label_column].value_counts(dropna=False).sort_index()
    lines = [
        "# Audit dataset tiếng Việt cho VietMailGuard Version 2",
        "",
        "## Phạm vi và kết luận",
        "",
        f"File được audit: `{source_path.as_posix()}`. SHA-256: `{source_hash}`.",
        "",
        "Kết luận an toàn: dataset có bằng chứng quan sát được rất mạnh cho thấy đây là bản dịch tiếng Việt của email tiếng Anh, với nhiều mẫu liên kết trực tiếp tới corpus Enron/Enron-Spam trong repository. Theo chính sách AGENTS.md, nên đặt `language = vi` và `data_origin = translated`. Chưa có metadata đủ để xác định công cụ dịch, người dịch, thời điểm dịch hoặc nguồn phát hành chính xác.",
        "",
        "Có thể dùng có điều kiện cho bilingual training, augmentation và robustness experiments sau khi liên kết bản dịch với email tiếng Anh gốc để chống cross-language leakage. Không được dùng làm benchmark native Vietnamese hoặc làm nguồn duy nhất để tuyên bố độ chính xác tiếng Việt thực tế.",
        "",
        "## Cấu trúc và chất lượng dữ liệu",
        "",
        f"- Số dòng dữ liệu: **{len(frame):,}**",
        f"- Số cột: **{len(frame.columns)}**",
        f"- Schema quan sát được: `{', '.join(frame.columns)}`",
        f"- Dtype: `{', '.join(f'{column}={dtype}' for column, dtype in frame.dtypes.items())}`",
        f"- Số giá trị unique: `{', '.join(f'{column}={frame[column].nunique(dropna=True):,}' for column in frame.columns)}`",
        f"- Dòng có text empty/placeholder: **{empty_text_rows:,}**",
        f"- Dòng trống hoàn toàn theo schema hiện có: **{fully_empty_rows:,}**",
        "- Các trường sender, receiver, date và subject riêng biệt không tồn tại trong schema; `Chủ đề:` đang nằm bên trong cột `text`. Không được coi các trường vắng mặt này là giá trị đã biết.",
        "",
        "### Missing values theo cột",
        "",
        "| Column | Missing | Percent |",
        "| --- | ---: | ---: |",
    ]
    for column, count in missing.items():
        lines.append(f"| {column} | {count:,} | {count / len(frame) * 100:.3f}% |")

    lines.extend(
        [
            "",
            "### Phân bố raw label",
            "",
            "| Raw label | Rows | Percent | Proposed mapping | Confidence |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
    )
    for label, count in label_counts.items():
        mapping = "normal" if label == 0 else "review"
        confidence = "high" if label in {0, 1} else "low"
        lines.append(
            f"| {markdown_escape(label)} | {int(count):,} | {count / len(frame) * 100:.3f}% | {mapping} | {confidence} |"
        )

    lines.extend(
        [
            "",
            "Raw label 0 phù hợp với lớp ham/legitimate của các bản ghi Enron đối chiếu được và có thể đề xuất map `normal`. Raw label 1 không được map toàn bộ sang `spam`: nội dung quan sát được gồm commercial spam, credential phishing, lottery fraud và 419/advance-fee fraud. Vì taxonomy đích có ba lớp, raw label 1 phải giữ `review` cho đến khi curation ở mức row hoàn tất.",
            "",
            "## Thống kê độ dài text",
            "",
            "| Scope | Min chars | P25 | Median | Mean | P75 | P95 | Max | Median words | Mean words |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope, stats in lengths.items():
        lines.append(
            f"| {scope} | {stats['min_chars']:,} | {stats['p25_chars']:.1f} | "
            f"{stats['median_chars']:.1f} | {stats['mean_chars']:.1f} | {stats['p75_chars']:.1f} | "
            f"{stats['p95_chars']:.1f} | {stats['max_chars']:,} | "
            f"{stats['median_words']:.1f} | {stats['mean_words']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Duplicate và label conflict",
            "",
            "| Representation | Duplicate groups | Participating rows | Redundant beyond first | Conflicting-label groups | Conflicting-label rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, summary in duplicate_summaries.items():
        lines.append(
            f"| {name} | {summary['groups']:,} | {summary['participating_rows']:,} | "
            f"{summary['redundant_rows_beyond_first']:,} | {summary['conflicting_label_groups']:,} | "
            f"{summary['conflicting_label_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "`normalized` dùng Unicode NFKC, casefold và whitespace normalization. `template_standard` thay URL/email đúng định dạng theo utility V1. `template_relaxed` bổ sung nhận dạng URL/email bị chèn khoảng trắng do dịch; đây là aid cho audit, không phải lý do tự động xóa row. Chi tiết từng group nằm trong `reports/v2/vietnamese_duplicate_report.csv`.",
            "",
            "## Bằng chứng về nguồn gốc translated",
            "",
            "### Dấu hiệu định lượng",
            "",
            "| Evidence signal | Rows | Percent |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, count in translation_counts.items():
        lines.append(f"| {name} | {count:,} | {count / len(frame) * 100:.3f}% |")
    lines.extend(
        [
            f"| non-NFC rows | {unicode_stats['non_nfc_rows']:,} | {unicode_stats['non_nfc_rows'] / len(frame) * 100:.3f}% |",
            f"| replacement-character rows | {unicode_stats['replacement_character_rows']:,} | {unicode_stats['replacement_character_rows'] / len(frame) * 100:.3f}% |",
            f"| ASCII-only text rows | {unicode_stats['ascii_only_rows']:,} | {unicode_stats['ascii_only_rows'] / len(frame) * 100:.3f}% |",
            "",
            "### Liên kết quan sát được với Enron.csv",
            "",
            "| Vietnamese row | Vietnamese marker | Raw label | English row | English marker | English raw label |",
            "| ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for item in cross_language:
        lines.append(
            f"| {item['vietnamese_original_row_id']} | {markdown_escape(item['vietnamese_marker'])} | "
            f"{item['vietnamese_raw_label']} | {item['english_original_row_id']} | "
            f"{markdown_escape(item['english_marker'])} | {item['english_raw_label']} |"
        )
    lines.extend(
        [
            "",
            "Các cặp trên không dựa vào semantic embedding: marker subject/entity còn nhận diện được xuất hiện ở cả hai corpus và raw label tương ứng trùng nhau. Cùng với tên Enron, địa chỉ/routing kiểu Enron và câu tiếng Việt có trật tự bất thường, đây là bằng chứng mạnh cho `translated`. Tuy nhiên chưa đủ để gán `parent_id` cho toàn bộ 5.728 dòng; bước linkage riêng vẫn bắt buộc trước split.",
            "",
            "Ví dụ artifact quan sát được gồm `Chủ đề:`/`Chủ thể:` được chèn vào body, `http : / /`, domain có khoảng trắng, địa chỉ dạng `name @ enron . com`, và các câu như “hoàn thành biểu mẫu phê duyệt bài đăng dài 1 phút”. Những dấu hiệu này phù hợp với quá trình dịch/tiền xử lý máy, nhưng không xác định được công cụ dịch cụ thể.",
            "",
            "## Audit raw label 1: spam có lẫn phishing/scam",
            "",
            "Các count dưới đây là rule-assisted evidence, không phải ground-truth label và không dùng một keyword đơn lẻ để map class.",
            "",
            "### Commercial/spam signals",
            "",
            "| Rule | Positive-label rows |",
            "| --- | ---: |",
        ]
    )
    for name, count in commercial_counts.items():
        lines.append(f"| {name} | {count:,} |")
    lines.extend(
        [
            "",
            "### Phishing/scam signals",
            "",
            "| Rule | Positive-label rows |",
            "| --- | ---: |",
        ]
    )
    for name, count in phishing_counts.items():
        lines.append(f"| {name} | {count:,} |")
    lines.extend(
        [
            "",
            f"- Raw label 1 có ít nhất một commercial signal: **{commercial_any:,}** dòng.",
            f"- Raw label 1 có ít nhất một phishing/scam signal: **{phishing_any:,}** dòng.",
            f"- Có cả hai nhóm signal: **{signal_overlap:,}** dòng.",
            "",
            "Các ví dụ ngân hàng Lasalle/Sky Bank yêu cầu khôi phục hoặc xác minh tài khoản là credential phishing. Các thư Nigeria, thừa kế và chuyển quỹ là 419/advance-fee fraud. Thư xổ số yêu cầu danh tính là lottery fraud. Do đó positive class là hỗn hợp; không thể dùng raw label 1 làm class `spam` hoặc `phishing` thuần.",
            "",
            "## Sample đại diện đã đọc",
            "",
            "| Original row | Raw label | Observed assessment | Excerpt |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for sample in samples:
        lines.append(
            f"| {sample['original_row_id']} | {sample['raw_label']} | "
            f"{sample['observed_assessment']} | {markdown_escape(sample['excerpt'])} |"
        )
    lines.extend(
        [
            "",
            "Các assessment trên chỉ mô tả sample đã đọc. Chúng không phải manual labels cho toàn corpus.",
            "",
            "## Chính sách sử dụng đề xuất",
            "",
            "| Question | Decision |",
            "| --- | --- |",
            "| Training | Có điều kiện. Raw label 0 có thể chuẩn hóa thành `normal` sau quality filtering và cross-language linkage. Raw label 1 giữ `review`; chỉ dùng các row được curate riêng thành spam/phishing theo taxonomy. |",
            "| Test | Không dùng làm native Vietnamese test. Chỉ có thể tạo `Translated Vietnamese Test` sau khi mọi bản dịch và English parent nằm cùng split hoặc English parent bị loại khỏi training. |",
            "| language | `vi` |",
            "| data_origin | `translated` |",
            "| raw label 0 | Đề xuất `normal`, confidence high từ đối chiếu Enron và sample thực tế. |",
            "| raw label 1 | `review`, confidence high rằng class bị trộn; không auto-map. |",
            "",
            "## Limitations bắt buộc ghi trong báo cáo Version 2",
            "",
            "- Exact provenance, giấy phép, công cụ dịch và quy trình chọn 5.728 row chưa có trong repository.",
            "- Dataset là translated, không đại diện đầy đủ cho email tiếng Việt native hoặc văn phong người dùng Việt Nam.",
            "- Nhiều email giữ entity, tổ chức, địa chỉ và bối cảnh Hoa Kỳ/Enron; model có thể học source artifacts thay vì ngôn ngữ/nguy cơ tổng quát.",
            "- Subject và body bị gộp; sender/receiver/date không có cột riêng.",
            "- URL/email bị chèn khoảng trắng, nên URL extraction chuẩn có thể bỏ sót nếu chưa có quy tắc riêng cho dữ liệu translated.",
            "- Positive class trộn commercial spam, phishing và fraud; raw label 1 cần row-level curation.",
            "- Phải xử lý duplicate/template groups và conflicting labels trước khi sampling.",
            "- Phải liên kết English parent và Vietnamese translation trước split để tránh cross-language semantic leakage.",
            "- Không được báo cáo “Vietnamese real-world accuracy” chỉ từ dataset này.",
            "",
            "Không có thao tác train, merge, standardize hoặc thay đổi artifact Version 1 trong audit này.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_dataset(
    source_path: Path,
    english_path: Path,
    report_path: Path,
    duplicate_path: Path,
    *,
    label_column: str = "spam",
    text_column: str = "text",
) -> dict[str, object]:
    before_hash = sha256_file(source_path)
    frame = pd.read_csv(source_path, encoding="utf-8")
    required = {label_column, text_column}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing configured audit columns: {sorted(missing_columns)}")

    missing = {column: int(frame[column].isna().sum()) for column in frame.columns}
    normalized_fields = frame.apply(
        lambda column: column.map(normalize_scalar)
    )
    empty_text_mask = normalized_fields[text_column].str.casefold().isin(PLACEHOLDERS)
    fully_empty_mask = normalized_fields.apply(
        lambda column: column.str.casefold().isin(PLACEHOLDERS)
    ).all(axis=1)

    raw_hashes = frame[text_column].map(lambda value: digest_text(str(value)))
    normalized_hashes = frame[text_column].map(
        lambda value: digest_text(normalized_duplicate_text("", value))
    )
    standard_template_hashes = frame[text_column].map(
        lambda value: digest_text(normalized_duplicate_text("", value, template=True))
    )
    relaxed_template_hashes = frame[text_column].map(
        lambda value: digest_text(relaxed_template_text(value))
    )
    hash_sets = {
        "exact_raw": raw_hashes,
        "normalized": normalized_hashes,
        "template_standard": standard_template_hashes,
        "template_relaxed": relaxed_template_hashes,
    }
    duplicate_summaries = {
        name: duplicate_summary(hashes, frame[label_column])
        for name, hashes in hash_sets.items()
    }
    records: list[dict[str, object]] = []
    for name, hashes in hash_sets.items():
        records.extend(
            duplicate_records(frame, hashes, name, label_column, text_column)
        )

    positive = frame.loc[frame[label_column].eq(1), text_column]
    commercial_masks = {
        name: positive.map(lambda value: bool(pattern.search(str(value))))
        for name, pattern in COMMERCIAL_RULES.items()
    }
    phishing_masks = {
        name: positive.map(lambda value: bool(pattern.search(str(value))))
        for name, pattern in PHISHING_SCAM_RULES.items()
    }
    commercial_any_mask = pd.DataFrame(commercial_masks).any(axis=1)
    phishing_any_mask = pd.DataFrame(phishing_masks).any(axis=1)

    lengths = {"all": length_statistics(frame[text_column])}
    for label, group in frame.groupby(label_column, dropna=False):
        lengths[f"raw_label_{label}"] = length_statistics(group[text_column])

    translation_counts = pattern_counts(
        frame[text_column], TRANSLATION_EVIDENCE_PATTERNS
    )
    unicode_stats = {
        "non_nfc_rows": int(
            frame[text_column]
            .map(lambda value: str(value) != unicodedata.normalize("NFC", str(value)))
            .sum()
        ),
        "replacement_character_rows": int(
            frame[text_column].map(lambda value: "\ufffd" in str(value)).sum()
        ),
        "ascii_only_rows": int(
            frame[text_column].map(lambda value: str(value).isascii()).sum()
        ),
    }

    cross_language = cross_language_evidence(frame, english_path, text_column)
    samples = sample_rows(frame, label_column, text_column)
    report = render_report(
        source_path=source_path,
        source_hash=before_hash,
        frame=frame,
        label_column=label_column,
        text_column=text_column,
        missing=missing,
        empty_text_rows=int(empty_text_mask.sum()),
        fully_empty_rows=int(fully_empty_mask.sum()),
        duplicate_summaries=duplicate_summaries,
        lengths=lengths,
        translation_counts=translation_counts,
        commercial_counts={name: int(mask.sum()) for name, mask in commercial_masks.items()},
        phishing_counts={name: int(mask.sum()) for name, mask in phishing_masks.items()},
        commercial_any=int(commercial_any_mask.sum()),
        phishing_any=int(phishing_any_mask.sum()),
        signal_overlap=int((commercial_any_mask & phishing_any_mask).sum()),
        samples=samples,
        cross_language=cross_language,
        unicode_stats=unicode_stats,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    duplicate_frame = pd.DataFrame.from_records(records)
    duplicate_frame.to_csv(duplicate_path, index=False, encoding="utf-8-sig")

    after_hash = sha256_file(source_path)
    if before_hash != after_hash:
        raise RuntimeError("Raw dataset hash changed during read-only audit")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "labels": frame[label_column].value_counts().sort_index().to_dict(),
        "duplicates": duplicate_summaries,
        "cross_language_pairs": len(cross_language),
        "raw_sha256_unchanged": True,
        "report": str(report_path),
        "duplicate_report": str(duplicate_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/data_vi.csv"))
    parser.add_argument("--english", type=Path, default=Path("data/raw/Enron.csv"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/v2/vietnamese_dataset_audit.md"),
    )
    parser.add_argument(
        "--duplicates",
        type=Path,
        default=Path("reports/v2/vietnamese_duplicate_report.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_dataset(
        args.input,
        args.english,
        args.report,
        args.duplicates,
    )
    print("Vietnamese dataset audit complete")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Raw labels: {summary['labels']}")
    for name, values in summary["duplicates"].items():
        print(
            f"{name}: groups={values['groups']:,}, "
            f"rows={values['participating_rows']:,}, "
            f"conflicting_groups={values['conflicting_label_groups']:,}"
        )
    print(f"Cross-language evidence pairs: {summary['cross_language_pairs']}")
    print(f"Raw SHA-256 unchanged: {summary['raw_sha256_unchanged']}")
    print(f"Report: {summary['report']}")
    print(f"Duplicate report: {summary['duplicate_report']}")


if __name__ == "__main__":
    main()
