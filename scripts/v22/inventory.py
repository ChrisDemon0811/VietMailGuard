"""Inventory known dataset roots; no whole-drive/private-mail crawl, no test reads."""
import csv
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vietmailguard.v22.quality import digest
from vietmailguard.v21.dimensionality_experiment import memory_snapshot

def main():
    report=ROOT/'reports/v22';report.mkdir(parents=True,exist_ok=True)
    roots=[ROOT/'data/raw',ROOT/'data/v21/raw',ROOT/'dataset v2.1']
    for key in ('VIETMAILGUARD_V21_DATA_DIR','VIETMAILGUARD_V22_DATA_DIR'):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]).expanduser().resolve())
    seen=set();rows=[]
    for index,root in enumerate(roots):
        if not root.is_dir() or root.is_symlink():continue
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[name for name in dirs if not (Path(folder)/name).is_symlink()]
            for name in sorted(files):
                path=Path(folder)/name
                if path.is_symlink() or path.resolve() in seen:continue
                if path.suffix.casefold() not in ('.csv','.zip','.gz','.tar','.parquet','.jsonl','.txt'):continue
                seen.add(path.resolve())
                rows.append({'path':f'input_root_{index}/'+path.relative_to(root).as_posix(),
                    'size_bytes':path.stat().st_size,'sha256':digest(path),
                    'redistribution':'NOT VERIFIED','role':'local_input_not_independent_sample_count'})
    with (report/'source_inventory.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','size_bytes','sha256','redistribution','role']);writer.writeheader();writer.writerows(rows)
    with sqlite3.connect((ROOT/'data/v21/cache/dedup_audit.sqlite3').as_uri()+'?mode=ro',uri=True) as conn:
        audit={'total_rows':conn.execute('SELECT count(*) FROM rows').fetchone()[0],
            'by_dataset':dict(conn.execute('SELECT dataset,count(*) FROM rows GROUP BY dataset')),
            'new_source_unique_normalized':conn.execute("SELECT count(distinct normalized_hash) FROM rows WHERE dataset!='v2_reference' AND normalized_hash!=''").fetchone()[0],
            'distribution':[list(row) for row in conn.execute('SELECT dataset,raw_label,candidate_label,language,label_status,count(*) FROM rows GROUP BY 1,2,3,4,5')],
            'audit_sha256':digest(ROOT/'data/v21/cache/dedup_audit.sqlite3')}
    output={'inventory_files':len(rows),'raw_input_bytes':sum(row['size_bytes'] for row in rows),
        'scope':'Known project dataset roots and explicit V21/V22 environment roots; not a whole-machine drive scan',
        'historical_audit':audit,'resources':{**memory_snapshot(),'logical_cpus':os.cpu_count(),
        'disk_free_gib':shutil.disk_usage(ROOT).free/2**30},
        'curation':json.loads((ROOT/'reports/v21/label_curation_summary.json').read_text()),
        'bootstrap':json.loads((ROOT/'reports/v21/bootstrap_development_summary.json').read_text()),
        'limitations':'No new source approval. License and distribution must be checked per component. No raw test files opened.'}
    (report/'phase1_inventory.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(rows),'audited_rows':audit['total_rows'],'normalized_unique_new':audit['new_source_unique_normalized'],'resources':output['resources']}))

if __name__=='__main__':main()
