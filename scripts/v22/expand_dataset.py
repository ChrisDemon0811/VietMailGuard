"""Phase 3 only: local MinHash, quarantine, sampling, evidence and reports."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v22 import expansion
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run',action='store_true')
    action.add_argument('--report',action='store_true')
    action.add_argument('--review-id')
    parser.add_argument('--label',choices=['normal','spam','phishing','review','exclude'])
    parser.add_argument('--reviewer')
    parser.add_argument('--language-competence')
    parser.add_argument('--evidence')
    args=parser.parse_args()
    config=json.loads((ROOT/'config/v22/expansion.json').read_text(encoding='utf-8'))
    expansion.validate_config(config)
    with offline_boundary(ROOT):
        if args.run:
            print(json.dumps(expansion.prepare(ROOT,config)),flush=True)
            expansion.index_bands(ROOT,config)
            print(json.dumps(expansion.group_candidates(ROOT,config)),flush=True)
            print(json.dumps(expansion.classify(ROOT,config)),flush=True)
            print(json.dumps(expansion.sample_review(ROOT,config)),flush=True)
        elif args.review_id:
            if not all((args.label,args.reviewer,args.language_competence,args.evidence)):
                parser.error('Review needs label, reviewer, original-language competence and evidence')
            expansion.record_decision(ROOT,args.review_id,args.label,args.reviewer,args.language_competence,args.evidence)
        from vietmailguard.v22.expansion_report import generate
        print(json.dumps(generate(ROOT,config),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
