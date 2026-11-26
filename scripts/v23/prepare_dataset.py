"""Use existing local email inputs first; never train or open legacy tests."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vietmailguard.v23.dataset import (connect, digest, ingest_csv, ingest_archive,
                                     summary, inherited_references, protected_metadata_guard)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["all", "enron_original", "bruce_guenter_2004", "bruce_guenter_2005"], default="all")
    parser.add_argument("--seven-zip", type=Path)
    parser.add_argument("--references-and-guard", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "config/v23/large_scale.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if shutil.disk_usage(ROOT).free < 5 * 2**30:
        raise RuntimeError("At least 5 GiB free disk required before intake")
    executable = args.seven_zip or (Path(shutil.which("7z")) if shutil.which("7z") else Path("C:/Program Files/7-Zip/7z.exe"))
    with connect(ROOT / "data/v23/intake.sqlite3") as connection:
        old = connection.execute("SELECT gia_tri FROM cau_hinh WHERE ma='config_sha256'").fetchone()
        expected = digest(config_path)
        if old and old[0] != expected:
            raise ValueError("Configuration changed; use a new intake namespace")
        connection.execute("INSERT OR IGNORE INTO cau_hinh VALUES ('config_sha256',?)", (expected,))
        connection.commit()
        for source, policy in config["sources"].items():
            if args.references_and_guard:
                continue
            if args.source not in ("all", source):
                continue
            if "parts" in policy:
                ingest_csv(connection, ROOT, source, policy, config)
            else:
                if not executable.is_file():
                    raise RuntimeError("Provide an already installed 7-Zip executable")
                ingest_archive(connection, ROOT, source, policy, config, executable)
        result = summary(connection)
        if args.references_and_guard:
            references = inherited_references(connection, ROOT)
            protection = protected_metadata_guard(connection, ROOT)
            result = summary(connection)
            result["inherited_reference"] = references
            result["protection"] = protection
        output = ROOT / "data/v23/intake_summary.json"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
