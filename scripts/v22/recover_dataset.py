"""Phase 3.1 targeted recovery and approval evidence; no training or admission."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v22 import recovery
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'config/v22/expansion.json').read_text(encoding='utf-8'))
    with offline_boundary(ROOT):
        recovery.prepare(ROOT,config)
        if not args.report_only:
            recovery.prepare_buckets(ROOT)
            print(json.dumps(recovery.load_selected_texts(ROOT,config)),flush=True)
            print(json.dumps(recovery.recover(ROOT,config)),flush=True)
        from vietmailguard.v22.recovery_report import generate
        print(json.dumps(generate(ROOT,config),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
