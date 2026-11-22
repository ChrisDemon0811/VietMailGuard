"""Label-independent relation checks for the isolated V2.2 group-aware pilot.

Grouping is a leakage precaution, not a duplicate deletion or a label approval.
Ordinary numeric operational notices and low-content link footers stay warnings.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .recovery import comparison_text, exact_jaccard


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def context(text: str, policy: dict) -> bool:
    residual = comparison_text(text).replace('<url>', '').replace('<email>', '').strip()
    return len(residual) >= policy['minimum_context_characters'] and (
        len(residual.split()) >= policy['minimum_context_words'] or
        sum('\u3400' <= char <= '\u9fff' for char in residual) >= 30)


def word_trigram_jaccard(left: str, right: str) -> float:
    def shingles(text):
        words = re.findall(r'\w+', comparison_text(text).replace('<url>', '').replace('<email>', ''))
        return {tuple(words[i:i+3]) for i in range(max(0, len(words)-2))}
    a, b = shingles(left), shingles(right)
    return len(a & b)/len(a | b) if a or b else 0.


def pair_evidence(left: dict, right: dict, scores: dict, policy: dict) -> dict:
    """Never inspect class labels, predictions, or old split assignments."""
    a, b = normalized(left['body']), normalized(right['body'])
    template_a, template_b = comparison_text(a), comparison_text(b)
    skeleton = lambda text: re.sub(r'\d+', '<number>', text)
    notice = policy['operational_notice']
    ordinary_notice = (
        bool(a and b) and a != b and skeleton(a) == skeleton(b) and
        max(len(a), len(b)) <= notice['maximum_characters'] and
        max(len(a.split()), len(b.split())) <= notice['maximum_words'] and
        all(re.search(r'\b'+word+r'\b', a) and re.search(r'\b'+word+r'\b', b)
            for word in notice['required_words']))
    enough = context(a, policy) and context(b, policy)
    raw = max(scores.get('body_normalized', 0.), scores.get('full_normalized', 0.))
    templated = max(scores.get('body_template', 0.), scores.get('full_template', 0.))
    cutoff = policy['jaccard_cutoff']
    if a and a == b:
        category, reason, must_group = 'same_normalized_content', 'equal_complete_body', True
    elif ordinary_notice:
        category, reason, must_group = 'ordinary_template', 'same_short_operational_notice_only_numeric_slots_differ', False
    elif not enough:
        category = 'ordinary_template' if templated >= cutoff and raw < cutoff else 'insufficient_evidence'
        reason, must_group = 'insufficient_nonplaceholder_body_context', False
    elif raw >= cutoff:
        category, reason, must_group = 'high_risk_near_content', 'exact_full_content_jaccard_and_substantive_body_context', True
    elif templated >= cutoff and word_trigram_jaccard(a, b) >= policy['template_word_trigram_cutoff']:
        category, reason, must_group = 'high_risk_substantive_template', 'shared_ordered_substantive_text_beyond_url_email_slots', True
    else:
        category, reason, must_group = 'insufficient_evidence', 'similarity_or_prefix_equality_does_not_establish_content_identity', False
    return {'category': category, 'reason': reason, 'must_group_for_content': must_group,
            'sufficient_body_context': enough,
            'body_word_trigram_jaccard': word_trigram_jaccard(a, b),
            'human_verified_duplicate': False, **scores}


class Groups:
    def __init__(self, count: int): self.parents = list(range(count))

    def find(self, i: int) -> int:
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]; i = self.parents[i]
        return i

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b: self.parents[max(a, b)] = min(a, b)

    def identifiers(self, ids: list[str]) -> list[str]:
        parts = defaultdict(list)
        for i, identifier in enumerate(ids): parts[self.find(i)].append(identifier)
        names = {k: 'v22g_'+hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest()[:24]
                 for k, v in parts.items()}
        return [names[self.find(i)] for i in range(len(ids))]


def split_groups(rows: list[dict], group_ids: list[str], policy: dict) -> dict[str, str]:
    """Exactly one predeclared fold; never search folds or optimize validation."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]['id'])
    labels = np.asarray([rows[i]['label'] for i in order])
    groups = np.asarray([group_ids[i] for i in order])
    splitter = StratifiedGroupKFold(n_splits=policy['n_splits'], shuffle=True,
                                    random_state=policy['seed'])
    train, validation = next(iter(splitter.split(np.zeros(len(order)), labels, groups)))
    assignment = {rows[order[int(i)]]['id']: split
                  for split, values in [('train', train), ('validation', validation)] for i in values}
    assert len(assignment) == len(rows)
    for split in ('train', 'validation'):
        if {row['label'] for row in rows if assignment[row['id']] == split} != {'normal', 'spam', 'phishing'}:
            raise ValueError('Predeclared fold lacks a class; do not search for a convenient fold')
    return assignment
