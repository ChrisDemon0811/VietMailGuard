"""Run post-freeze robustness tracks and emit an intermediate JSON payload."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vietmailguard.file_utils import sha256_file  # noqa: E402
from vietmailguard.inference import load_inference_engine  # noqa: E402
from vietmailguard.robustness_evaluation import (  # noqa: E402
    build_robustness_variants,
    build_short_form_challenge,
    challenge_contamination_audit,
    contamination_audit,
    evaluate_robustness,
    evaluate_short_challenge,
    records_for_json,
    robustness_metrics,
    select_vietnamese_seed,
    short_challenge_metrics,
    taxonomy_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "robustness_evaluation.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "evaluation" / "v2" / "robustness_payload.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = PROJECT_ROOT / config["production_model"]
    actual_hash = sha256_file(model_path)
    if actual_hash != config["frozen_model_sha256"]:
        raise RuntimeError("Frozen production model hash does not match robustness protocol")

    training_frames = [
        pd.read_csv(PROJECT_ROOT / path, low_memory=False)
        for path in config["production_training_splits"]
    ]
    production_training = pd.concat(training_frames, ignore_index=True)
    test = pd.read_csv(PROJECT_ROOT / config["posthoc_seed_split"], low_memory=False)
    seed_config = config["vietnamese_robustness"]
    seed = select_vietnamese_seed(
        test,
        normal_count=int(seed_config["normal_parent_count"]),
        spam_count=int(seed_config["spam_parent_count"]),
        seed=int(config["random_seed"]),
    )
    seed_overlap = contamination_audit(seed, production_training)

    variants = build_robustness_variants(seed)
    challenge = build_short_form_challenge()
    challenge_overlap = challenge_contamination_audit(challenge, production_training)
    engine = load_inference_engine()
    threshold = float(config["confidence_overestimation_threshold"])
    evaluated_robustness = evaluate_robustness(
        variants, engine, confidence_threshold=threshold
    )
    evaluated_challenge = evaluate_short_challenge(
        challenge, engine, confidence_threshold=threshold
    )
    robustness_summary = robustness_metrics(evaluated_robustness, threshold)
    challenge_summary = short_challenge_metrics(evaluated_challenge, threshold)

    payload = {
        "protocol": config,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_model_sha256_before": actual_hash,
        "seed": {
            "rows": int(len(seed)),
            "class_distribution": {
                str(key): int(value) for key, value in seed["label"].value_counts().items()
            },
            "source_distribution": {
                str(key): int(value) for key, value in seed["source"].value_counts().items()
            },
            "contamination_overlap_counts": seed_overlap,
        },
        "challenge": {
            "rows": int(len(challenge)),
            "class_distribution": {
                str(key): int(value) for key, value in challenge["label"].value_counts().items()
            },
            "language_distribution": {
                str(key): int(value) for key, value in challenge["language"].value_counts().items()
            },
            "length_distribution": {
                str(key): int(value) for key, value in challenge["length_bucket"].value_counts().items()
            },
            "contamination_overlap_counts": challenge_overlap,
        },
        "robustness_rows": records_for_json(evaluated_robustness),
        "robustness_metrics": records_for_json(robustness_summary),
        "short_challenge_rows": records_for_json(evaluated_challenge),
        "short_challenge_metrics": records_for_json(challenge_summary),
        "robustness_error_taxonomy": taxonomy_counts(evaluated_robustness),
        "challenge_error_taxonomy": taxonomy_counts(evaluated_challenge),
    }
    after_hash = sha256_file(model_path)
    if after_hash != actual_hash:
        raise RuntimeError("Production model changed during robustness evaluation")
    payload["production_model_sha256_after"] = after_hash
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Production training rows checked: {len(production_training):,}")
    print(f"Vietnamese robustness parents: {len(seed):,}")
    print(f"Vietnamese robustness variants: {len(evaluated_robustness):,}")
    print(f"Short-form challenge samples: {len(evaluated_challenge):,}")
    print(f"Training contamination checks: PASS ({seed_overlap}; {challenge_overlap})")
    print(f"Frozen model hash unchanged: {after_hash}")
    print(f"Intermediate payload: {args.output_json}")


if __name__ == "__main__":
    main()

