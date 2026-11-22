"""Explicit offline, local-model assistance for the existing 299-email collection."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_worker(collection: Path, name: str):
    # Separate processes bound peak RAM and release the encoder before TF-IDF loads.
    from contextlib import ExitStack
    from unittest.mock import patch
    from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
    from vietmailguard.v22.dataset_review import DatasetReview
    from vietmailguard.v22.quality import digest
    from vietmailguard.v22.review_assistance import input_digest
    config_name = 'runtime_model' if name == 'tfidf' else 'runtime_multilingual'
    config_path = ROOT / f'config/v21/{config_name}.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if digest(ROOT / config['artifact']) != config['artifact_sha256']:
        raise ValueError('Local classifier SHA-256 mismatch')
    destination = collection / f'assistance/{name}.json'
    inputs = {'queue': digest(collection / 'review_queue.csv'), 'content': digest(collection / 'content.jsonl'),
              'configuration': digest(config_path), 'artifact': config['artifact_sha256']}
    if destination.exists():
        old = json.loads(destination.read_text(encoding='utf-8'))
        if old['inputs'] != inputs: raise ValueError('Changed worker inputs; retain old run separately')
        if old['status'] == 'ok':
            if name=='minilm':
                from vietmailguard.v21.multilingual_runtime import verified_configuration
                with offline_boundary(ROOT):verified_configuration()
            return
    result = {'inputs': inputs, 'model': config['artifact'], 'artifact_sha256': config['artifact_sha256'],
              'config': config_path.relative_to(ROOT).as_posix(), 'status': 'unavailable', 'predictions': {},
              'model_confidence': None}
    started = time.perf_counter()
    with offline_boundary(ROOT), ExitStack() as stack:
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        def forbid_fit(*a, **kw): raise RuntimeError('Fitting is prohibited during review assistance')
        for cls, method in ((Pipeline, 'fit'), (LinearSVC, 'fit'), (MLPClassifier, 'fit'),
                            (TfidfVectorizer, 'fit'), (TfidfVectorizer, 'fit_transform')):
            stack.enter_context(patch.object(cls, method, forbid_fit))
        store = DatasetReview(ROOT, collection=collection)
        rows = [store.open_sample(row['id_mau']) for row in store.list_samples(only_pending=False)]
        if len(rows) != 299: raise ValueError('Expected the existing 299-email collection')
        import pandas as pd
        import numpy as np
        frame = pd.DataFrame(rows)[['subject', 'body']]
        from vietmailguard.v21.runtime_inference import _bounded_model_input
        try:
            if name == 'tfidf':
                from vietmailguard.v21.runtime_inference import load_v21_engine
                engine = load_v21_engine()
                limit = engine.maximum_characters
                predicted = engine.model.predict(_bounded_model_input(frame, limit))
                truncation = (frame['subject'].str.len().gt(limit) | frame['body'].str.len().gt(limit)).tolist()
            else:
                # No downloads, provider/API imports, training or device fallback.
                from vietmailguard.v21.development_training import resource_snapshot
                resources = resource_snapshot()
                result['resources'] = resources
                if resources.get('free_ram_gib', 0) < config['encoder']['min_free_ram_gib']:
                    raise MemoryError('Insufficient free RAM under the configured encoder gate')
                from vietmailguard.v21.multilingual_runtime import load_multilingual_engine
                from vietmailguard.embedding_modeling import compose_email_texts
                engine = load_multilingual_engine()
                limit = engine.maximum_characters
                texts = compose_email_texts(_bounded_model_input(frame, limit))
                tokens = engine.encoder.tokenizer(texts.tolist(), add_special_tokens=True,
                    truncation=False, padding=False, verbose=False)['input_ids']
                truncation = [len(ids) > engine.encoder.max_seq_length or len(row['subject']) > limit
                              or len(row['body']) > limit for ids, row in zip(tokens, rows)]
                matrix = engine.encoder.encode(texts.tolist(), batch_size=8, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False, device='cpu')
                matrix = np.asarray(matrix, dtype=np.float32)
                if matrix.shape != (299, 384) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-3):
                    raise ValueError('Invalid embeddings')
                predicted = engine.model.predict(matrix)
                result['encoder_revision'] = config['encoder']['revision']
                result['encoder_files'] = config['encoder_files']
            if len(predicted) != 299 or not set(map(str, predicted)) <= {'normal', 'spam', 'phishing'}:
                raise ValueError('Invalid classifier output')
            result['predictions'] = {row['id_mau']: {'suggested_label': str(label),
                'input_sha256': input_digest(row['subject'], row['body']), 'input_truncated': bool(cut)}
                for row, label, cut in zip(rows, predicted, truncation)}
            result.update(status='ok', maximum_characters_per_field=limit, device='cpu')
        except Exception as error:
            # Never serialize exception messages which can contain private input or paths.
            result['error_type'] = type(error).__name__
            result['predictions'] = {}
    result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    write(destination, result)


def combine(collection):
    from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
    from vietmailguard.v22.dataset_review import DatasetReview
    from vietmailguard.v22.quality import digest
    from vietmailguard.v22.review_assistance import POLICY_VERSION, suggestion, summarize, input_digest, priority_order
    with offline_boundary(ROOT):
        store = DatasetReview(ROOT, collection=collection)
        outputs = {name: json.loads((collection / f'assistance/{name}.json').read_text(encoding='utf-8')) for name in ('tfidf', 'minilm')}
        source = ROOT / 'data/v22/large_scale_expansion_01'
        downloads = json.loads((source / 'download_receipts.json').read_text(encoding='utf-8'))
        evidence_path = source / 'raw/readme.html'
        research_evidence = digest(evidence_path) == downloads['readme.html']['sha256']
        originals = {}
        for path in source.glob('parsed_*spam*.jsonl'):
            receipt = json.loads(path.with_suffix('.receipt.json').read_text(encoding='utf-8'))
            if digest(path) != receipt['parsed_sha256']: raise ValueError('Source cache changed')
            for line in path.open(encoding='utf-8'):
                row = json.loads(line); originals[row['id']] = row
        for archive in ('20030228_spam.tar.bz2', '20050311_spam_2.tar.bz2'):
            if digest(source / 'raw' / archive) != downloads[archive]['sha256']: raise ValueError('Archive changed')
        records = []
        for metadata in store.list_samples(only_pending=False):
            identifier = metadata['id_mau']; email = store.open_sample(identifier); raw = originals[identifier]
            fingerprint = input_digest(email['subject'], email['body'])
            verified = bool(raw['raw_email_sha256'] and raw['original_path'] and raw['archive_sha256']
                and raw['raw_label'] == metadata['nhan_goc'] == 'spam'
                and raw['source'] == metadata['nguon'] == 'spamassassin_original'
                and raw['source_url'] in {'https://spamassassin.apache.org/old/publiccorpus/' + name
                                         for name in ('20030228_spam.tar.bz2', '20050311_spam_2.tar.bz2')}
                and fingerprint == input_digest(raw['subject'], raw['body']))
            predictions = {name: output['predictions'].get(identifier) for name, output in outputs.items()}
            if any(value and value['input_sha256'] != fingerprint for value in predictions.values()):
                raise ValueError('Changed email after model suggestions')
            records.append(suggestion(identifier=identifier, content_sha256=fingerprint, group=metadata['id_nhom'],
                tfidf=predictions['tfidf']['suggested_label'] if predictions['tfidf'] else None,
                minilm=predictions['minilm']['suggested_label'] if predictions['minilm'] else None,
                provenance_verified=verified, research_use_verified=research_evidence,
                truncated=any(value and value['input_truncated'] for value in predictions.values())))
        destination = collection / 'assistance/suggestions.jsonl'
        temporary = destination.with_suffix('.jsonl.partial')
        with temporary.open('w', encoding='utf-8') as f:
            for row in records: f.write(json.dumps(row, ensure_ascii=False) + '\n')
        temporary.replace(destination)
        import csv
        priority_path=collection/'assistance/priority_review.csv'
        with priority_path.open('w',encoding='utf-8',newline='') as f:
            fields=['id','constraint_group_id','priority','tfidf_suggested_label','minilm_suggested_label','flags',
                    'source_approval_consideration_only','human_verified_label','training_eligible','split']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in sorted(records,key=lambda x:priority_order(x['id'],x)):
                writer.writerow({key:';'.join(row[key]) if key=='flags' else row[key] for key in fields})
        manifest = {'policy_version': POLICY_VERSION, 'record_kind': 'review_assistance_only', 'rows': len(records),
            'queue_sha256': digest(collection / 'review_queue.csv'), 'content_sha256': digest(collection / 'content.jsonl'),
            'suggestions_sha256': digest(destination),'priority_csv_sha256':digest(priority_path), 'models': {name: {key: value for key, value in result.items() if key != 'predictions'}
                for name, result in outputs.items()}, 'source_evidence_sha256': digest(evidence_path),
            'source_evidence_url': 'https://spamassassin.apache.org/old/publiccorpus/readme.html',
            'summary': summarize(records), 'new_human_decisions': 0, 'training_eligible': False, 'pseudo_labels_for_train': 0}
        write(collection / 'assistance/manifest.json', manifest)
        print(json.dumps(manifest['summary']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, default=Path('data/v22/spam_evidence_resolution/review_collection'))
    parser.add_argument('--worker', choices=['minilm', 'tfidf'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    from vietmailguard.v22.dataset_review import DatasetReview
    store = DatasetReview(ROOT, collection=args.collection)
    collection = store.path.parent
    (collection / 'assistance').mkdir(exist_ok=True)
    if args.worker:
        run_worker(collection, args.worker)
        return
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    for name in ('minilm', 'tfidf'):
        with (collection / f'assistance/{name}.log').open('wb') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--collection', str(collection), '--worker', name],
                           cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        state=json.loads((collection / f'assistance/{name}.json').read_text(encoding='utf-8'))
        print(json.dumps({'worker_completed': name,'status':state['status']}), flush=True)
    combine(collection)


if __name__ == '__main__':
    main()
