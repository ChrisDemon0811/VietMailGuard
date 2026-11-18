"""Run isolated CPU Chi-square experiment without network or final-test access."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vietmailguard.v21.feature_selection_experiment import candidate, finalize, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--report-only", action="store_true", help="Render existing measured checkpoints; no training")
    args = parser.parse_args()
    if args.run:
        config = read_json(ROOT / "config/v21/experiments/feature_selection_03.json")
        for name in ["baseline"] + [f"A_chi2_{k}" for k in config["k"]] + ["B_train_cv"]:
            print("Running " + name, flush=True)
            subprocess.run([sys.executable, "-u", __file__, "--candidate", name], cwd=ROOT, check=True)
        finalize()
    elif args.report_only:
        finalize()
    elif args.candidate:
        result = candidate(args.candidate)
        print(json.dumps({key: result.get(key) for key in ("candidate", "status", "reason", "fit_selector_classifier_seconds")}), flush=True)
    else:
        parser.error("Choose --run or --candidate NAME")


if __name__ == "__main__":
    main()
