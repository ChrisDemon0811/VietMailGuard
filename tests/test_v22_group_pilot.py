"""Synthetic relation and split checks; never approve a corpus label."""
import copy
import json
from pathlib import Path

from vietmailguard.v22.group_pilot import Groups, pair_evidence, split_groups


ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config/v22/group_aware_supervised_pilot_01.json').read_text(encoding='utf-8'))


def evidence(a,b,scores):
    return pair_evidence({'subject':'Synthetic subject','body':a},
                         {'subject':'Another subject','body':b},scores,CONFIG['relations'])


def test_operational_template_does_not_establish_email_identity():
    a='schedule download failed at hour 12 date 2026-01-01'
    b='schedule download failed at hour 13 date 2026-01-02'
    result=evidence(a,b,{'body_normalized':.9,'full_normalized':.93})
    assert result['category']=='ordinary_template'
    assert not result['must_group_for_content']


def test_substantive_near_content_groups_without_label_inference():
    a='The project team will meet tomorrow to discuss the implementation design and the remaining technical questions in detail.'
    b=a.replace('tomorrow','today')
    result=evidence(a,b,{'body_normalized':.91,'full_normalized':.92})
    assert result['category']=='high_risk_near_content' and result['must_group_for_content']
    assert result['human_verified_duplicate'] is False


def test_low_content_link_date_footer_is_a_warning():
    a='Date 2026 https://example.com/a'
    b='Date 2027 https://example.org/b'
    result=evidence(a,b,{'body_template':1.,'body_normalized':.5})
    assert result['category']=='ordinary_template' and not result['must_group_for_content']


def test_substantive_template_beyond_url_slots_is_a_leakage_precaution():
    a='Please review the attached project schedule and confirm the correct meeting time with the team before the deadline. https://example.com/a'
    b=a.replace('https://example.com/a','https://example.org/b')
    result=evidence(a,b,{'body_normalized':.7,'body_template':.99})
    assert result['category']=='high_risk_substantive_template' and result['must_group_for_content']


def test_empty_and_short_context_do_not_create_spurious_duplicate_claim():
    result=evidence('Thanks','Thank you',{'full_normalized':.9})
    assert result['category']=='insufficient_evidence' and not result['must_group_for_content']
    assert evidence('','',{})['must_group_for_content'] is False


def test_group_closure_and_identifiers_are_stable():
    a,b=Groups(4),Groups(4)
    a.union(0,1);a.union(1,2)
    b.union(2,1);b.union(0,2)
    assert a.identifiers(['a','b','c','d'])==b.identifiers(['a','b','c','d'])
    assert len(set(a.identifiers(['a','b','c','d'])))==2


def test_split_preserves_groups_all_records_and_original_fields():
    rows=[];groups=[]
    for label in ('normal','spam','phishing'):
        for i in range(15):
            for suffix in ('original','translated'):
                rows.append({'id':f'{label}:{i}:{suffix}','label':label,'split':'original_assignment'})
                groups.append(f'{label}:{i}')
    original=copy.deepcopy(rows)
    assignment=split_groups(rows,groups,CONFIG['splitting'])
    assert rows==original and len(assignment)==len(rows)
    assert assignment==split_groups(rows,groups,CONFIG['splitting'])
    for i in range(0,len(rows),2):assert assignment[rows[i]['id']]==assignment[rows[i+1]['id']]
    for role in ('train','validation'):
        assert {r['label'] for r in rows if assignment[r['id']]==role}=={'normal','spam','phishing'}
