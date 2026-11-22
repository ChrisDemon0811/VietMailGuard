"""Download explicitly reviewed public corpora; never follow email URLs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from vietmailguard.v22.acquisition import download_sources,ingest_sources,screen_near,report
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--ingest',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'config/v22/acquisition.json').read_text(encoding='utf-8'))
    if args.download:download_sources(ROOT,config)
    if args.ingest:
        from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
        with offline_boundary(ROOT):
            ingest_sources(ROOT,config)
            screen_near(ROOT)
    print(json.dumps(report(ROOT),ensure_ascii=True))


if __name__=='__main__':main()
