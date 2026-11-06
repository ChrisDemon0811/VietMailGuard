"""CPU-only SVD/PCA/LDA experiment; offline and Development Validation only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v21.dimensionality_experiment import (
    ExperimentBlocked, _json, candidate, candidate_names, directories, fingerprint,
    load_config, memory_snapshot, offline_boundary, prepare, summarize,
)
from vietmailguard.v21.paths import resolve_v21_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    for flag in ('preflight','prepare','summarize','run'):
        group.add_argument('--'+flag,action='store_true')
    group.add_argument('--candidate')
    args = parser.parse_args()
    paths = resolve_v21_paths(project_root=ROOT)
    try:
        config = load_config(paths)
        if args.run:
            subprocess.run([sys.executable,'-u',__file__,'--prepare'],cwd=ROOT,check=True)
            for name in candidate_names(config):
                print(json.dumps({'stage':'candidate','name':name}),flush=True)
                process = subprocess.run([sys.executable,'-u',__file__,'--candidate',name],cwd=ROOT)
                if process.returncode:
                    signature,_ = fingerprint(paths)
                    _,result_dir = directories(paths)
                    _json(result_dir/f'{name}.json',{'candidate':name,'status':'FAIL','fingerprint':signature,
                        'reason':f'Worker exited {process.returncode}; no automatic retry'})
            with offline_boundary(ROOT):
                output = summarize(paths)
            print(json.dumps({'candidates':[{key:item.get(key) for key in ('candidate','status','reason')} for item in output['candidates']],
                'runtime_changed':False,'heldout_opened':False}),flush=True)
            return int(any(item['status']=='FAIL' for item in output['candidates']))
        if args.prepare:
            output = prepare(paths)
        elif args.candidate:
            output = candidate(args.candidate,paths)
        elif args.summarize:
            with offline_boundary(ROOT):
                output = summarize(paths)
            output = {'candidates':len(output['candidates']),'best_reduced_candidate':output['best_reduced_candidate']}
        else:
            with offline_boundary(ROOT):
                signature,_ = fingerprint(paths)
            output = {'memory':memory_snapshot(),'fingerprint':signature,'candidates':candidate_names(config),
                'encoder_loaded':False,'downloads_allowed':False,'final_test_allowed':False}
        print(json.dumps(output,ensure_ascii=True,indent=2),flush=True)
        return 0
    except Exception as error:
        print(json.dumps({'status':'BLOCKED' if isinstance(error,ExperimentBlocked) else 'FAIL',
            'exception_type':type(error).__name__,'reason':str(error) if isinstance(error,ExperimentBlocked) else
                'Local stage failed. No API call, download or automatic retry.'}),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
