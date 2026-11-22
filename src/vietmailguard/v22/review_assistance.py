"""Stored local-model suggestions for review, never labels or approval decisions."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

LABELS = {'normal', 'spam', 'phishing'}
POLICY_VERSION = 'spam-review-assistance-v1'
FLAG_TEXT = {
    'model_unavailable': 'Thiếu kết quả của một mô hình; không suy đoán thay.',
    'models_disagree': 'Hai mô hình bất đồng.',
    'literal_source_label_difference': 'Tên nhãn khác raw Spam; Phishing vẫn có thể thuộc lớp dương binary của nguồn.',
    'outside_source_positive_family': 'Có mô hình dự đoán Normal, khác họ nhãn Spam/Phishing của nguồn.',
    'phishing_model_signal': 'Có đề xuất Phishing; cần phân biệt với Spam bằng nội dung và bằng chứng.',
    'three_class_label_unverified': 'Nguồn chỉ có nhãn binary; nhãn ba lớp chưa được xác minh.',
    'local_research_rights_missing': 'Chưa đủ bằng chứng quyền nghiên cứu local.',
    'provenance_missing': 'Thiếu bằng chứng truy vết nguồn.',
    'input_truncated': 'Ít nhất một mô hình chỉ đọc phần đầu theo cấu hình huấn luyện.',
}


def input_digest(subject: str, body: str) -> str:
    return hashlib.sha256(json.dumps({'subject': subject, 'body': body},
        ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def suggestion(*, identifier: str, content_sha256: str, group: str, tfidf: str | None,
               minilm: str | None, provenance_verified: bool, research_use_verified: bool,
               truncated: bool = False) -> dict:
    if any(value is not None and value not in LABELS for value in (tfidf, minilm)):
        raise ValueError('Unsupported suggested label')
    flags = {'three_class_label_unverified'}
    available = {value for value in (tfidf, minilm) if value is not None}
    if tfidf is None or minilm is None: flags.add('model_unavailable')
    if tfidf is not None and minilm is not None and tfidf != minilm: flags.add('models_disagree')
    if available - {'spam'}: flags.add('literal_source_label_difference')
    if 'normal' in available: flags.add('outside_source_positive_family')
    if 'phishing' in available: flags.add('phishing_model_signal')
    if not provenance_verified: flags.add('provenance_missing')
    if not research_use_verified: flags.add('local_research_rights_missing')
    if truncated: flags.add('input_truncated')
    urgent = bool(flags & {'model_unavailable', 'models_disagree', 'outside_source_positive_family',
                          'phishing_model_signal', 'provenance_missing', 'local_research_rights_missing'})
    return {'id': identifier, 'input_sha256': content_sha256, 'constraint_group_id': group,
        'record_kind': 'review_assistance_only', 'raw_label': 'spam',
        'source_label_semantics': 'binary_unsolicited_positive_may_include_phishing',
        'tfidf_suggested_label': tfidf, 'minilm_suggested_label': minilm,
        'confidence': None, 'calibrated_probability': None, 'flags': sorted(flags),
        'priority': 'urgent' if urgent else 'standard', 'human_check_required': True,
        'source_approval_consideration_only': tfidf == minilm == 'spam' and provenance_verified and research_use_verified,
        'human_verified_label': None, 'training_eligible': False, 'split': 'unassigned',
        'local_research_use_evidence_verified': research_use_verified,
        'redistribution_permission': 'NOT VERIFIED'}


def summarize(records: list[dict]) -> dict:
    return {'rows': len(records), 'priority_counts': dict(Counter(x['priority'] for x in records)),
        'tfidf_labels': dict(Counter(x['tfidf_suggested_label'] or 'unavailable' for x in records)),
        'minilm_labels': dict(Counter(x['minilm_suggested_label'] or 'unavailable' for x in records)),
        'flag_counts': dict(Counter(flag for x in records for flag in x['flags'])),
        'human_check_required': sum(x['human_check_required'] for x in records),
        'source_approval_consideration_only': sum(x['source_approval_consideration_only'] for x in records),
        'automatically_approved': 0, 'new_train_pseudo_labels': 0,
        'quality_metrics_available': False}


def priority_order(identifier: str, record: dict | None) -> tuple:
    return (0 if record and record['priority'] == 'urgent' else 1,
            hashlib.sha256(('42:review-order:' + identifier).encode()).hexdigest())


def load_assistance(collection: Path) -> tuple[dict, dict[str, dict]] | None:
    """Read stored suggestions only. Fail closed for stale, damaged or foreign sidecars."""
    manifest_path = collection / 'assistance/manifest.json'
    if not manifest_path.exists(): return None
    if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
        raise ValueError('Unsafe review suggestion path')
    from .quality import digest
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('policy_version') != POLICY_VERSION or manifest.get('record_kind') != 'review_assistance_only'
            or manifest.get('queue_sha256') != digest(collection / 'review_queue.csv')
            or manifest.get('content_sha256') != digest(collection / 'content.jsonl')):
        raise ValueError('Stale review suggestions')
    path = collection / 'assistance/suggestions.jsonl'
    if path.is_symlink() or digest(path) != manifest['suggestions_sha256']:
        raise ValueError('Review suggestion integrity failed')
    import csv
    with (collection / 'review_queue.csv').open(encoding='utf-8', newline='') as f:
        queue = {row['audit_id']: row['constraint_group_id'] for row in csv.DictReader(f)}
    records = {}
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        if (row['id'] in records or row['id'] not in queue or row.get('training_eligible') is not False
                or row.get('human_verified_label') is not None or row.get('confidence') is not None
                or row.get('calibrated_probability') is not None or row.get('split') != 'unassigned'
                or row.get('record_kind') != 'review_assistance_only'
                or row.get('constraint_group_id') != queue[row['id']]):
            raise ValueError('Invalid suggestion boundary')
        expected = suggestion(identifier=row['id'], content_sha256=row['input_sha256'],
            group=row['constraint_group_id'], tfidf=row['tfidf_suggested_label'], minilm=row['minilm_suggested_label'],
            provenance_verified='provenance_missing' not in row['flags'],
            research_use_verified=row['local_research_use_evidence_verified'], truncated='input_truncated' in row['flags'])
        if row != expected: raise ValueError('Suggestion does not match review policy')
        records[row['id']] = row
    if set(records) != set(queue) or len(records) != manifest['rows']:
        raise ValueError('Incomplete review suggestions')
    if manifest['summary'] != summarize(list(records.values())):
        raise ValueError('Review suggestion summary differs from records')
    content = {}
    for line in (collection / 'content.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        if row['id'] in content: raise ValueError('Duplicate cached input')
        content[row['id']] = input_digest(row['subject'], row['body'])
    if set(content) != set(records) or any(row['input_sha256'] != content[identifier] for identifier, row in records.items()):
        raise ValueError('Review suggestions differ from original inputs')
    return manifest, records


@lru_cache(maxsize=8)
def _source_metadata(path: str,sha256: str,size: int,modified: int) -> dict:
    """Cache evidence only, without retaining any email body or loading models."""
    del size,modified
    from .quality import digest
    p=Path(path)
    if p.is_symlink() or digest(p)!=sha256:raise ValueError('Source evidence cache changed')
    result={}
    for line in p.open(encoding='utf-8'):
        row=json.loads(line)
        result[row['id']]={key:row.get(key,'') for key in ('source','raw_label','source_url','original_path','archive_sha256','raw_email_sha256')}
        result[row['id']]['input_sha256']=input_digest(row['subject'],row['body'])
    return result


def source_evidence(root: Path,identifier: str,subject: str,body: str) -> dict | None:
    """Verified original-label evidence is independent of AI visibility."""
    if not identifier.startswith('spamassassin_original:'):return None
    from .quality import digest
    source=root/'data/v22/large_scale_expansion_01'
    if not (source/'download_receipts.json').is_file():return None
    receipts=json.loads((source/'download_receipts.json').read_text(encoding='utf-8'))
    for path in sorted(source.glob('parsed_*spam*.jsonl')):
        receipt=json.loads(path.with_suffix('.receipt.json').read_text(encoding='utf-8'))
        stat=path.stat()
        evidence=_source_metadata(str(path),receipt['parsed_sha256'],stat.st_size,stat.st_mtime_ns).get(identifier)
        if evidence is None:continue
        if evidence['input_sha256']!=input_digest(subject,body):raise ValueError('Source content does not match review email')
        original=source/'raw/readme.html'
        research_verified=digest(original)==receipts['readme.html']['sha256']
        return {**evidence,'research_use_evidence_verified':research_verified,
            'research_evidence_url':'https://spamassassin.apache.org/old/publiccorpus/readme.html',
            'research_evidence_sha256':receipts['readme.html']['sha256'],
            'three_class_label_verified':False,'redistribution_permission':'NOT VERIFIED'}
    return None
