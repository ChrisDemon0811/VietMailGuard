"""Offline development experiment; never evaluates a held-out test."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vietmailguard.v21.multilingual_dl_experiment import (
    load_config, measure_saved_candidates, offline_boundary, preflight, run_experiment,
)
from vietmailguard.v21.paths import resolve_v21_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="Inspect local resources/cache only (default)")
    action.add_argument("--run", action="store_true", help="Fit separate heads using verified Development Train")
    action.add_argument("--benchmark", action="store_true", help="Measure saved heads with encoder in a separate process; no fit")
    args = parser.parse_args()
    paths = resolve_v21_paths(project_root=ROOT)
    try:
        if args.benchmark:
            print(json.dumps(measure_saved_candidates(paths), indent=2))
        elif args.run:
            result = run_experiment(paths)
            print(json.dumps({"candidates": [{"candidate": item["candidate"], "status": item["status"],
                "macro_f1": item.get("metrics", {}).get("macro_f1"), "reason": item.get("reason")}
                for item in result["candidates"]], "final_test_opened": result["final_test_opened"]}))
        else:
            with offline_boundary(ROOT):
                result = preflight(paths, load_config(paths))
                result.pop("snapshot", None)
                print(json.dumps(result, indent=2))
    except Exception as error:
        # A third-party exception may include content/paths; keep CLI diagnostics bounded.
        print(json.dumps({"status": "BLOCKED", "exception_type": type(error).__name__,
            "message": str(error) if type(error).__module__.startswith("vietmailguard") else
                "Local experiment failed; inspect the stage and offline tests. No network retry."}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
