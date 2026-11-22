"""Local V2.2 Phase 1/2 only. Never trains, splits, browses URLs or calls APIs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v22.quality import ingest, near_duplicates, summarize, record_review, export_review_pack
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--summary',action='store_true')
    parser.add_argument('--review-id')
    parser.add_argument('--review-pack',help='Export one explicitly selected private sample to ignored local JSON')
    parser.add_argument('--label',choices=['normal','spam','phishing','review','exclude'])
    parser.add_argument('--reviewer')
    parser.add_argument('--evidence')
    parser.add_argument('--language-competence')
    args=parser.parse_args()
    config=json.loads((ROOT/'config/v22/quality.json').read_text())
    with offline_boundary(ROOT):
        if args.run:
            ingest(ROOT,config)
            near_duplicates(ROOT,config)
        elif args.review_pack:
            print(str(export_review_pack(ROOT,args.review_pack).relative_to(ROOT)))
            return
        elif args.review_id:
            if not all((args.label,args.reviewer,args.evidence,args.language_competence)):
                parser.error('Review requires label, reviewer, evidence and language-competence')
            record_review(ROOT,args.review_id,args.label,args.reviewer,args.evidence,args.language_competence)
        elif not args.summary:
            parser.error('Choose --run, --summary or --review-id')
        print(json.dumps(summarize(ROOT),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
