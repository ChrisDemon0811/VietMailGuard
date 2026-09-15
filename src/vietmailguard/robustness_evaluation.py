"""Post-freeze robustness evaluation without model fitting or test-score revision."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from vietmailguard.dataset_standardizer import content_hash, normalize_text, template_hash

VALID_LABELS = ("normal", "spam", "phishing")
TRANSFORMATION_TYPES = (
    "clean_accented",
    "remove_accents",
    "character_substitution",
    "spacing_obfuscation",
    "punctuation_noise",
    "mixed_language",
)
LENGTH_BUCKETS = ("very_short", "short", "normal_length")


def compose_text(subject: object, body: object) -> str:
    """Mirror the production input boundary while preserving subject/body fields."""
    clean_subject = "" if pd.isna(subject) else str(subject)
    clean_body = "" if pd.isna(body) else str(body)
    return f"{clean_subject}\n{clean_body}".strip()


def token_count(subject: object, body: object) -> int:
    return len(re.findall(r"\S+", compose_text(subject, body)))


def length_bucket(count: int) -> str:
    if count < 50:
        return "very_short"
    if count <= 150:
        return "short"
    return "normal_length"


def remove_accents(text: str) -> str:
    """Remove Vietnamese combining marks while handling đ/Đ explicitly."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.replace("đ", "d").replace("Đ", "D")


def _replace_limited(text: str, pattern: str, replacement: str, maximum: int = 2) -> str:
    return re.sub(pattern, replacement, text, count=maximum, flags=re.IGNORECASE)


def character_substitution(text: str) -> str:
    """Apply deterministic, bounded substitutions used by common obfuscation."""
    transformed = text
    phrase_replacements = (
        (r"tài\s+khoản", "t@i kh0an"),
        (r"khuyến\s+mãi", "khuy3n m@i"),
        (r"xác\s+minh", "x@c m1nh"),
        (r"mật\s+khẩu", "m@t kh@u"),
        (r"giảm\s+giá", "gi@m gi@"),
    )
    replacements = 0
    for pattern, replacement in phrase_replacements:
        updated, count = re.subn(pattern, replacement, transformed, count=1, flags=re.IGNORECASE)
        transformed = updated
        replacements += count
        if replacements >= 2:
            break
    if replacements == 0:
        transformed = _replace_limited(transformed, r"(?i)a", "@", maximum=2)
        transformed = _replace_limited(transformed, r"(?i)o", "0", maximum=2)
    return transformed


def spacing_obfuscation(text: str) -> str:
    """Space one meaningful phrase, or a bounded fallback word, without deleting text."""
    targets = (
        "tài khoản", "khuyến mãi", "xác minh", "mật khẩu", "giảm giá",
        "cuộc họp", "thanh toán", "đăng nhập",
    )
    for target in targets:
        match = re.search(re.escape(target), text, flags=re.IGNORECASE)
        if match:
            spaced = "  ".join(" ".join(part) for part in match.group(0).split())
            return text[: match.start()] + spaced + text[match.end() :]
    fallback = re.search(r"(?u)\b[^\W\d_]{5,}\b", text)
    if fallback:
        spaced = " ".join(fallback.group(0))
        return text[: fallback.start()] + spaced + text[fallback.end() :]
    return text


def punctuation_noise(text: str) -> str:
    transformed = re.sub(r"([.!?])\s+", r"\1.. ", text, count=3)
    return f"!!! {transformed} !!!"


def mixed_language(text: str) -> str:
    """Create a deterministic English/Vietnamese code-switched variant."""
    replacements = (
        (r"tài khoản", "account"),
        (r"xác minh", "verify"),
        (r"khuyến mãi", "special offer"),
        (r"giảm giá", "discount"),
        (r"cuộc họp", "meeting"),
        (r"mật khẩu", "password"),
        (r"đăng nhập", "login"),
        (r"cập nhật", "update"),
    )
    transformed = text
    replacement_count = 0
    for pattern, replacement in replacements:
        transformed, count = re.subn(
            pattern, replacement, transformed, count=2, flags=re.IGNORECASE
        )
        replacement_count += count
        if replacement_count >= 3:
            break
    if replacement_count == 0:
        transformed = f"Please review this message. {transformed}"
    return transformed


def apply_transformation(text: str, transformation_type: str) -> str:
    transforms = {
        "clean_accented": lambda value: unicodedata.normalize("NFC", value),
        "remove_accents": remove_accents,
        "character_substitution": character_substitution,
        "spacing_obfuscation": spacing_obfuscation,
        "punctuation_noise": punctuation_noise,
        "mixed_language": mixed_language,
    }
    if transformation_type not in transforms:
        raise ValueError(f"Unsupported transformation: {transformation_type}")
    return transforms[transformation_type](text)


def _stable_order(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _select_across_length_strata(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count >= len(frame):
        return frame.copy()
    ordered = frame.assign(_token_count=[token_count(s, b) for s, b in zip(
        frame["subject"], frame["body"], strict=True
    )]).sort_values(["_token_count", "id"], kind="stable")
    strata = np.array_split(ordered.index.to_numpy(), 4)
    base, remainder = divmod(count, 4)
    selected: list[Any] = []
    for stratum_index, indices in enumerate(strata):
        quota = base + int(stratum_index < remainder)
        candidates = ordered.loc[indices].copy()
        candidates["_stable"] = candidates["id"].map(lambda value: _stable_order(value, seed))
        selected.extend(candidates.sort_values("_stable").head(quota).index.tolist())
    return frame.loc[selected].copy()


def select_vietnamese_seed(
    test: pd.DataFrame,
    *,
    normal_count: int = 32,
    spam_count: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Select test-only translated Vietnamese parents without inspecting predictions."""
    required = {
        "id", "parent_id", "source", "label", "language", "subject", "body",
        "training_eligible", "content_hash", "group_id", "normalized_content_hash",
        "template_hash", "final_group_id",
    }
    missing = required - set(test.columns)
    if missing:
        raise ValueError(f"Test split is missing seed columns: {sorted(missing)}")
    eligible_flag = test["training_eligible"].astype(str).str.casefold().isin({"true", "1"})
    eligible = test[
        test["language"].eq("vi")
        & test["label"].isin({"normal", "spam"})
        & eligible_flag
        & (test["subject"].fillna("").astype(str).str.strip().ne("")
           | test["body"].fillna("").astype(str).str.strip().ne(""))
    ].copy()
    normal = _select_across_length_strata(
        eligible[eligible["label"].eq("normal")], normal_count, seed
    )
    spam = _select_across_length_strata(
        eligible[eligible["label"].eq("spam")], spam_count, seed
    )
    selected = pd.concat([normal, spam], ignore_index=True)
    if len(normal) < min(normal_count, int(eligible["label"].eq("normal").sum())):
        raise RuntimeError("Normal robustness seed selection returned too few rows")
    if len(spam) < min(spam_count, int(eligible["label"].eq("spam").sum())):
        raise RuntimeError("Spam robustness seed selection returned too few rows")
    return selected.sort_values(["label", "id"], kind="stable").reset_index(drop=True)


def contamination_audit(seed: pd.DataFrame, production_training: pd.DataFrame) -> dict[str, int]:
    """Raise if a seed or its known lineage/duplicate group appears in production training."""
    checks: dict[str, tuple[str, str]] = {
        "id_to_id": ("id", "id"),
        "parent_id_to_id": ("parent_id", "id"),
        "id_to_parent_id": ("id", "parent_id"),
        "parent_id_to_parent_id": ("parent_id", "parent_id"),
        "content_hash": ("content_hash", "content_hash"),
        "group_id": ("group_id", "group_id"),
        "normalized_content_hash": ("normalized_content_hash", "normalized_content_hash"),
        "template_hash": ("template_hash", "template_hash"),
        "final_group_id": ("final_group_id", "final_group_id"),
    }
    overlaps: dict[str, int] = {}
    for name, (seed_column, training_column) in checks.items():
        if seed_column not in seed or training_column not in production_training:
            overlaps[name] = 0
            continue
        left = {str(value) for value in seed[seed_column].dropna() if str(value).strip()}
        right = {
            str(value) for value in production_training[training_column].dropna()
            if str(value).strip()
        }
        overlaps[name] = len(left & right)
    contaminated = {name: value for name, value in overlaps.items() if value}
    if contaminated:
        raise RuntimeError(f"Evaluation seed overlaps production training: {contaminated}")
    return overlaps


def build_robustness_variants(seed: pd.DataFrame) -> pd.DataFrame:
    """Expand each clean translated parent into the six frozen transformations."""
    records: list[dict[str, Any]] = []
    for parent in seed.to_dict(orient="records"):
        original_subject = "" if pd.isna(parent.get("subject")) else str(parent.get("subject"))
        original_body = "" if pd.isna(parent.get("body")) else str(parent.get("body"))
        original_text = compose_text(original_subject, original_body)
        for transformation_type in TRANSFORMATION_TYPES:
            transformed_subject = apply_transformation(original_subject, transformation_type)
            transformed_body = apply_transformation(original_body, transformation_type)
            transformed_text = compose_text(transformed_subject, transformed_body)
            digest = hashlib.sha256(
                f"{parent['id']}|{transformation_type}".encode("utf-8")
            ).hexdigest()[:20]
            records.append(
                {
                    "robustness_id": f"robust_{digest}",
                    "parent_sample_id": str(parent["id"]),
                    "english_parent_id": "" if pd.isna(parent.get("parent_id")) else str(parent.get("parent_id")),
                    "parent_final_group_id": str(parent["final_group_id"]),
                    "label": str(parent["label"]),
                    "source": str(parent["source"]),
                    "language": "vi",
                    "data_origin": "translated",
                    "seed_split": "test",
                    "evaluation_provenance": "posthoc_v2_test_derived_robustness",
                    "transformation_type": transformation_type,
                    "original_subject": original_subject,
                    "original_body": original_body,
                    "original_text": original_text,
                    "transformed_subject": transformed_subject,
                    "transformed_body": transformed_body,
                    "transformed_text": transformed_text,
                    "original_token_count": token_count(original_subject, original_body),
                    "transformed_token_count": token_count(transformed_subject, transformed_body),
                }
            )
    result = pd.DataFrame(records)
    if result["robustness_id"].duplicated().any():
        raise RuntimeError("Robustness IDs are not unique")
    return result


_SCENARIOS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "normal": {
        "en": [
            ("Meeting update", "The project meeting moved to 3 PM."),
            ("Receipt available", "Your requested receipt is attached for your records."),
            ("Class schedule", "Tomorrow's lecture starts in room B204 at nine."),
            ("Project notes", "I added the approved edits to the shared document."),
            ("Dinner plan", "Dinner is confirmed for Saturday at seven."),
        ],
        "vi": [
            ("Cập nhật lịch họp", "Cuộc họp dự án chuyển sang 3 giờ chiều."),
            ("Biên nhận đã có", "Biên nhận bạn yêu cầu đã được đính kèm để lưu hồ sơ."),
            ("Lịch học ngày mai", "Buổi học ngày mai bắt đầu lúc chín giờ tại phòng B204."),
            ("Ghi chú dự án", "Tôi đã thêm các chỉnh sửa được duyệt vào tài liệu chung."),
            ("Kế hoạch ăn tối", "Bữa tối được xác nhận vào bảy giờ tối thứ Bảy."),
        ],
    },
    "spam": {
        "en": [
            ("70% OFF Today", "Huge discount today. Shop now and unsubscribe anytime."),
            ("Software special offer", "Buy our productivity package today at a promotional price."),
            ("Travel sale", "Book this limited-time travel offer and save 60 percent."),
            ("Marketing webinar", "Register for our commercial webinar and explore paid services."),
            ("Home products promotion", "Sale prices are available now on selected home products."),
        ],
        "vi": [
            ("Giảm giá 70% hôm nay", "Ưu đãi lớn hôm nay. Mua ngay và có thể hủy đăng ký nhận tin."),
            ("Ưu đãi phần mềm", "Mua gói phần mềm năng suất hôm nay với giá khuyến mãi."),
            ("Khuyến mãi du lịch", "Đặt chuyến đi trong ưu đãi giới hạn và giảm đến 60 phần trăm."),
            ("Hội thảo tiếp thị", "Đăng ký hội thảo thương mại và tìm hiểu các dịch vụ trả phí."),
            ("Khuyến mãi đồ gia dụng", "Giá giảm đang áp dụng cho một số sản phẩm gia dụng."),
        ],
    },
    "phishing": {
        "en": [
            ("Account Alert", "Your account is locked. Verify your password at http://account-verify.example/login."),
            ("Security verification", "Sign in now to confirm your identity at http://secure-update.example/verify."),
            ("Payroll access", "Reply with your login code to prevent payroll access suspension."),
            ("Password expires", "Your cloud password expires today. Update credentials at http://login-check.example."),
            ("Bank identity check", "Confirm your banking identity and card details through the attached link."),
        ],
        "vi": [
            ("Cảnh báo tài khoản", "Tài khoản bị khóa. Xác minh mật khẩu tại http://account-verify.example/login."),
            ("Xác minh bảo mật", "Đăng nhập ngay để xác minh danh tính tại http://secure-update.example/verify."),
            ("Quyền truy cập lương", "Trả lời bằng mã đăng nhập để tránh tạm khóa quyền truy cập lương."),
            ("Mật khẩu sắp hết hạn", "Mật khẩu đám mây hết hạn hôm nay. Cập nhật thông tin đăng nhập tại http://login-check.example."),
            ("Kiểm tra danh tính ngân hàng", "Xác nhận danh tính ngân hàng và thông tin thẻ tín dụng qua liên kết đính kèm."),
        ],
    },
}

_EXPANSIONS: dict[str, dict[str, list[str]]] = {
    "normal": {
        "en": [
            "This message follows our existing conversation and does not request a password, payment, or account action.",
            "Please read it when convenient and reply only if the schedule or factual details need correction.",
            "The referenced material remains in the usual shared location used by the team.",
            "No external sign-in page is needed, and there is no deadline beyond the ordinary project schedule.",
            "The sender is providing routine operational context so recipients can keep their records current.",
        ],
        "vi": [
            "Tin nhắn này tiếp nối trao đổi hiện có và không yêu cầu mật khẩu, thanh toán hoặc thao tác tài khoản.",
            "Bạn có thể đọc khi thuận tiện và chỉ cần phản hồi nếu lịch hoặc thông tin thực tế cần chỉnh sửa.",
            "Tài liệu được nhắc đến vẫn nằm ở vị trí chia sẻ thông thường của nhóm.",
            "Không cần truy cập trang đăng nhập bên ngoài và không có hạn chót ngoài lịch dự án thông thường.",
            "Người gửi cung cấp thông tin vận hành định kỳ để người nhận cập nhật hồ sơ của mình.",
        ],
    },
    "spam": {
        "en": [
            "This bulk promotional message advertises products and paid services to a broad mailing list.",
            "The commercial campaign highlights sale prices, a limited-time discount, and a repeated invitation to shop now.",
            "Recipients can browse additional packages, compare promotional bundles, and purchase an upgraded plan.",
            "The newsletter describes seasonal offers and includes marketing language intended to generate sales.",
            "An unsubscribe option is provided for recipients who no longer want commercial announcements.",
        ],
        "vi": [
            "Thông điệp quảng cáo hàng loạt này giới thiệu sản phẩm và dịch vụ trả phí tới một danh sách người nhận rộng.",
            "Chiến dịch thương mại nhấn mạnh giá khuyến mãi, mức giảm giới hạn và lời mời mua ngay được lặp lại.",
            "Người nhận có thể xem thêm các gói, so sánh ưu đãi và mua gói dịch vụ nâng cấp.",
            "Bản tin mô tả chương trình theo mùa và sử dụng ngôn ngữ tiếp thị nhằm tạo doanh số.",
            "Tùy chọn hủy đăng ký nhận tin được cung cấp cho người không muốn nhận thông báo thương mại.",
        ],
    },
    "phishing": {
        "en": [
            "The message claims that immediate action is required to avoid suspension of access to an important account.",
            "It directs the recipient to an unfamiliar sign-in location and asks for credentials or private verification details.",
            "The warning uses urgency and impersonation cues to pressure the recipient before there is time to verify the sender.",
            "The requested action could expose a password, authentication code, banking detail, or other private information.",
            "The recipient is told to use the supplied link rather than independently opening the organization's known website.",
        ],
        "vi": [
            "Thông điệp tuyên bố phải hành động ngay để tránh bị tạm khóa quyền truy cập vào một tài khoản quan trọng.",
            "Nội dung dẫn người nhận tới địa chỉ đăng nhập lạ và yêu cầu thông tin đăng nhập hoặc dữ liệu xác minh riêng tư.",
            "Cảnh báo dùng sự khẩn cấp và dấu hiệu giả mạo để gây áp lực trước khi người nhận kịp kiểm tra người gửi.",
            "Thao tác được yêu cầu có thể làm lộ mật khẩu, mã xác thực, thông tin ngân hàng hoặc dữ liệu cá nhân khác.",
            "Người nhận được yêu cầu dùng liên kết cung cấp thay vì tự mở trang web chính thức đã biết của tổ chức.",
        ],
    },
}


def _extend_body(base: str, additions: Iterable[str], minimum_tokens: int) -> str:
    parts = [base]
    additions_list = list(additions)
    index = 0
    while len(re.findall(r"\S+", " ".join(parts))) < minimum_tokens:
        parts.append(additions_list[index % len(additions_list)])
        index += 1
    return " ".join(parts)


def build_short_form_challenge() -> pd.DataFrame:
    """Build 90 transparent synthetic scenarios; they are not real-world observations."""
    records: list[dict[str, Any]] = []
    for label in VALID_LABELS:
        for language in ("en", "vi"):
            additions = _EXPANSIONS[label][language]
            for family_index, (subject, base_body) in enumerate(_SCENARIOS[label][language], 1):
                family_id = f"{language}_{label}_{family_index:02d}"
                bodies = {
                    "very_short": base_body,
                    "short": _extend_body(base_body, additions[:3], 55),
                    "normal_length": _extend_body(base_body, additions, 165),
                }
                for bucket in LENGTH_BUCKETS:
                    body = bodies[bucket]
                    count = token_count(subject, body)
                    actual_bucket = length_bucket(count)
                    if actual_bucket != bucket:
                        raise RuntimeError(
                            f"Challenge length construction failed for {family_id}: "
                            f"expected {bucket}, got {actual_bucket} ({count} tokens)"
                        )
                    challenge_id = f"challenge_{family_id}_{bucket}"
                    records.append(
                        {
                            "challenge_id": challenge_id,
                            "challenge_family_id": family_id,
                            "label": label,
                            "language": language,
                            "source": "controlled_short_form_challenge",
                            "data_origin": "synthetic_challenge",
                            "label_provenance": "scenario_definition",
                            "length_bucket": bucket,
                            "token_count": count,
                            "subject": subject,
                            "body": body,
                            "original_text": compose_text(subject, body),
                            "content_hash": content_hash(subject, body),
                            "template_hash": template_hash(subject, body),
                        }
                    )
    result = pd.DataFrame(records)
    if len(result) != 90 or result["challenge_id"].duplicated().any():
        raise RuntimeError("Short-form challenge must contain 90 unique scenarios")
    return result


def challenge_contamination_audit(
    challenge: pd.DataFrame, production_training: pd.DataFrame
) -> dict[str, int]:
    overlaps = {}
    for column in ("content_hash", "template_hash"):
        left = set(challenge[column].astype(str))
        right = {
            str(value) for value in production_training[column].dropna()
            if str(value).strip()
        }
        overlaps[column] = len(left & right)
    if any(overlaps.values()):
        raise RuntimeError(f"Synthetic challenge overlaps production training: {overlaps}")
    return overlaps


def _security_codes(result: dict[str, Any]) -> list[str]:
    return sorted({
        str(finding.get("code", ""))
        for finding in result.get("security_findings", [])
        if str(finding.get("code", ""))
    })


def _security_descriptions(result: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(finding.get("description", ""))
        for finding in result.get("security_findings", [])
        if str(finding.get("description", ""))
    ))


def _has_security_signal_for_label(label: str, codes: set[str]) -> bool:
    if label == "spam":
        return "commercial_promotion" in codes
    if label == "phishing":
        return bool(codes & {
            "credential_request", "account_suspension", "financial_bait",
            "prize_winner", "threat", "call_to_action", "ip_host", "punycode",
            "obfuscated_url", "suspicious_token",
        })
    return False


def _error_taxonomy(
    *,
    label: str,
    prediction: str,
    confidence: float,
    transformation_type: str | None = None,
    bucket: str | None = None,
    language_changed: bool = False,
    threshold: float = 0.8,
) -> list[str]:
    tags: list[str] = []
    if prediction != label:
        if label == "normal":
            tags.append("normal false positive")
        if label == "spam" and bucket in {"very_short", "short"}:
            tags.append("short promotional miss")
        if label == "phishing" and bucket in {"very_short", "short"}:
            tags.append("short phishing miss")
        if transformation_type == "remove_accents":
            tags.append("no-accent degradation")
        elif transformation_type in {
            "character_substitution", "spacing_obfuscation", "punctuation_noise"
        }:
            tags.append("obfuscation failure")
        elif transformation_type == "mixed_language":
            tags.append("mixed-language failure")
        if confidence >= threshold:
            tags.append("confidence overestimation")
    if language_changed:
        tags.append("language shift")
    return list(dict.fromkeys(tags))


def evaluate_robustness(
    variants: pd.DataFrame,
    inference_engine: Any,
    *,
    confidence_threshold: float = 0.8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in variants.to_dict(orient="records"):
        result = inference_engine.analyze_email(
            subject=row["transformed_subject"], body=row["transformed_body"]
        )
        codes = _security_codes(result)
        record = dict(row)
        record.update(
            {
                "ml_prediction": str(result["prediction"]),
                "ml_confidence": float(result["confidence"]),
                "class_probabilities_json": json.dumps(
                    result["class_probabilities"], ensure_ascii=False, sort_keys=True
                ),
                "detected_language": str(result["detected_language"]),
                "security_finding_codes": "|".join(codes),
                "security_finding_descriptions": "|".join(_security_descriptions(result)),
                "risk_score": int(result["risk_score"]),
                "risk_level": str(result["risk_level"]),
                "recommended_action": str(result["recommended_action"]),
                "prediction_correct": str(result["prediction"]) == str(row["label"]),
            }
        )
        rows.append(record)
    evaluated = pd.DataFrame(rows)
    clean = evaluated[evaluated["transformation_type"].eq("clean_accented")].set_index(
        "parent_sample_id"
    )
    clean_prediction = clean["ml_prediction"].to_dict()
    clean_confidence = clean["ml_confidence"].to_dict()
    clean_risk = clean["risk_score"].to_dict()
    clean_language = clean["detected_language"].to_dict()
    evaluated["prediction_flipped_from_clean"] = [
        row.ml_prediction != clean_prediction[row.parent_sample_id]
        for row in evaluated.itertuples()
    ]
    evaluated["confidence_drop_vs_clean"] = [
        clean_confidence[row.parent_sample_id] - float(row.ml_confidence)
        for row in evaluated.itertuples()
    ]
    evaluated["risk_score_change_vs_clean"] = [
        int(row.risk_score) - int(clean_risk[row.parent_sample_id])
        for row in evaluated.itertuples()
    ]
    evaluated["error_taxonomy"] = [
        "|".join(_error_taxonomy(
            label=str(row.label),
            prediction=str(row.ml_prediction),
            confidence=float(row.ml_confidence),
            transformation_type=str(row.transformation_type),
            bucket=length_bucket(int(row.transformed_token_count)),
            language_changed=str(row.detected_language) != str(clean_language[row.parent_sample_id]),
            threshold=confidence_threshold,
        ))
        for row in evaluated.itertuples()
    ]
    return evaluated


def evaluate_short_challenge(
    challenge: pd.DataFrame,
    inference_engine: Any,
    *,
    confidence_threshold: float = 0.8,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in challenge.to_dict(orient="records"):
        result = inference_engine.analyze_email(subject=row["subject"], body=row["body"])
        codes_list = _security_codes(result)
        codes = set(codes_list)
        prediction = str(result["prediction"])
        correct = prediction == str(row["label"])
        signal_on_miss = (not correct) and _has_security_signal_for_label(str(row["label"]), codes)
        normal_with_promotion = prediction == "normal" and "commercial_promotion" in codes
        record = dict(row)
        record.update(
            {
                "ml_prediction": prediction,
                "ml_confidence": float(result["confidence"]),
                "class_probabilities_json": json.dumps(
                    result["class_probabilities"], ensure_ascii=False, sort_keys=True
                ),
                "detected_language": str(result["detected_language"]),
                "security_finding_codes": "|".join(codes_list),
                "security_finding_descriptions": "|".join(_security_descriptions(result)),
                "risk_score": int(result["risk_score"]),
                "risk_level": str(result["risk_level"]),
                "recommended_action": str(result["recommended_action"]),
                "prediction_correct": correct,
                "security_signal_present_on_ml_miss": signal_on_miss,
                "ml_normal_with_promotional_signal": normal_with_promotion,
                "error_taxonomy": "|".join(_error_taxonomy(
                    label=str(row["label"]),
                    prediction=prediction,
                    confidence=float(result["confidence"]),
                    bucket=str(row["length_bucket"]),
                    threshold=confidence_threshold,
                )),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    labels = [label for label in VALID_LABELS if label in set(frame["label"])]
    if not len(frame):
        return {"rows": 0, "supported_classes": "", "accuracy": None,
                "macro_f1_supported_classes": None, "normal_recall": None,
                "spam_recall": None, "phishing_recall": None}
    _, recalls, f1s, supports = precision_recall_fscore_support(
        frame["label"], frame["ml_prediction"], labels=labels, zero_division=0
    )
    recall_by_label = dict(zip(labels, recalls, strict=True))
    support_by_label = dict(zip(labels, supports, strict=True))
    return {
        "rows": int(len(frame)),
        "supported_classes": "|".join(labels),
        "accuracy": float(accuracy_score(frame["label"], frame["ml_prediction"])),
        "macro_f1_supported_classes": float(np.mean(f1s)),
        "normal_recall": float(recall_by_label["normal"]) if support_by_label.get("normal", 0) else None,
        "spam_recall": float(recall_by_label["spam"]) if support_by_label.get("spam", 0) else None,
        "phishing_recall": float(recall_by_label["phishing"]) if support_by_label.get("phishing", 0) else None,
    }


def robustness_metrics(evaluated: pd.DataFrame, confidence_threshold: float = 0.8) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [
        (value, evaluated[evaluated["transformation_type"].eq(value)])
        for value in TRANSFORMATION_TYPES
    ]
    scopes.append(("all_transformed", evaluated[~evaluated["transformation_type"].eq("clean_accented")]))
    rows: list[dict[str, Any]] = []
    for name, subset in scopes:
        values = _metric_values(subset)
        errors = subset[~subset["prediction_correct"]]
        values.update(
            {
                "track": "vietnamese_robustness",
                "scope_type": "transformation",
                "scope_value": name,
                "prediction_flip_rate": float(subset["prediction_flipped_from_clean"].mean()),
                "mean_confidence": float(subset["ml_confidence"].mean()),
                "mean_confidence_drop_vs_clean": float(subset["confidence_drop_vs_clean"].mean()),
                "mean_risk_score": float(subset["risk_score"].mean()),
                "mean_risk_score_change_vs_clean": float(subset["risk_score_change_vs_clean"].mean()),
                "ml_errors": int(len(errors)),
                "confidence_overestimation_errors": int(
                    errors["ml_confidence"].ge(confidence_threshold).sum()
                ),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def short_challenge_metrics(
    evaluated: pd.DataFrame, confidence_threshold: float = 0.8
) -> pd.DataFrame:
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", evaluated)]
    scopes.extend(("class", value, evaluated[evaluated["label"].eq(value)]) for value in VALID_LABELS)
    scopes.extend(("language", value, evaluated[evaluated["language"].eq(value)]) for value in ("en", "vi"))
    scopes.extend(("length_bucket", value, evaluated[evaluated["length_bucket"].eq(value)]) for value in LENGTH_BUCKETS)
    scopes.append((
        "length_bucket",
        "under_or_equal_150_tokens",
        evaluated[evaluated["length_bucket"].isin({"very_short", "short"})],
    ))
    scopes.extend(
        ("language_class", f"{language}:{label}", evaluated[
            evaluated["language"].eq(language) & evaluated["label"].eq(label)
        ])
        for language in ("en", "vi") for label in VALID_LABELS
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_value, subset in scopes:
        values = _metric_values(subset)
        errors = subset[~subset["prediction_correct"]]
        values.update(
            {
                "track": "short_form_challenge",
                "scope_type": scope_type,
                "scope_value": scope_value,
                "ml_errors": int(len(errors)),
                "failure_rate": float((~subset["prediction_correct"]).mean()),
                "mean_confidence": float(subset["ml_confidence"].mean()),
                "confidence_overestimation_errors": int(
                    errors["ml_confidence"].ge(confidence_threshold).sum()
                ),
                "security_signal_present_on_ml_miss": int(
                    subset["security_signal_present_on_ml_miss"].sum()
                ),
                "review_or_quarantine_on_ml_miss": int(
                    errors["recommended_action"].isin({"REVIEW", "QUARANTINE"}).sum()
                ),
                "ml_normal_with_promotional_signal": int(
                    subset["ml_normal_with_promotional_signal"].sum()
                ),
                "mean_risk_score": float(subset["risk_score"].mean()),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def taxonomy_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for value in frame["error_taxonomy"].fillna("").astype(str):
        for tag in filter(None, value.split("|")):
            counts[tag] += 1
    return dict(sorted(counts.items()))


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to strict JSON records with nulls instead of NaN."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))
