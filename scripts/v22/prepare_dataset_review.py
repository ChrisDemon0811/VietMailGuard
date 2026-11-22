"""Apply authorized inherited subset and cache original review content locally."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v22.acquisition import approve_bootstrap
from vietmailguard.v22.dataset_review import DatasetReview
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary

if __name__=='__main__':
    config=json.loads((ROOT/'config/v22/acquisition.json').read_text(encoding='utf-8'))
    with offline_boundary(ROOT):
        print(json.dumps(approve_bootstrap(ROOT,config['bootstrap_approval'])),flush=True)
        store=DatasetReview(ROOT)
        print(json.dumps(store.prepare_content()),flush=True)
        store.export()
