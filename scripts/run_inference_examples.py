"""Run deterministic offline examples through the complete production inference API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.inference import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RISK_CONFIG_PATH,
    DEFAULT_SECURITY_CONFIG_PATH,
    EmailSecurityInference,
)


EXAMPLES = [
    {
        "name": "normal_meeting",
        "sender": "project.manager@example.org",
        "subject": "Project meeting notes",
        "body": "Hello team, attached are the notes and action items from today's meeting.",
    },
    {
        "name": "commercial_promotion",
        "sender": "offers@example-shop.test",
        "subject": "Special offer",
        "body": (
            "Limited time discount on software products. Buy now and save 70%. "
            "Unsubscribe from future marketing emails."
        ),
    },
    {
        "name": "account_verification_with_ip_url",
        "sender": "security-alert@example.test",
        "subject": "Urgent account verification required",
        "body": (
            "Your account will be suspended within 24 hours. Verify your password immediately "
            "at http://192.0.2.10/login."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic production inference examples.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument(
        "--risk-config",
        type=Path,
        default=DEFAULT_RISK_CONFIG_PATH,
    )
    parser.add_argument(
        "--security-config",
        type=Path,
        default=DEFAULT_SECURITY_CONFIG_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "inference_examples.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = EmailSecurityInference(
        model_path=args.model,
        metadata_path=args.metadata,
        risk_config_path=args.risk_config,
        security_config_path=args.security_config,
    )
    records = []
    for example in EXAMPLES:
        result = engine.analyze_email(
            sender=example["sender"], subject=example["subject"], body=example["body"]
        )
        records.append({"example": example, "result": result})
        reason_categories = [reason["category"] for reason in result["reasons"]]
        print(
            f"{example['name']}: prediction={result['prediction']}, "
            f"confidence={result['confidence']:.6f}, risk={result['risk_score']} "
            f"({result['risk_level']}), action={result['recommended_action']}, "
            f"reasons={reason_categories}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {args.output.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
