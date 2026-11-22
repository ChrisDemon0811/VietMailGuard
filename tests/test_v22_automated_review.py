"""Synthetic regression of full-text automated triage; never ground-truth labels."""
import importlib.util
from pathlib import Path

import pytest
from vietmailguard.content_analyzer import load_security_config

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('automated_review_299',ROOT/'scripts/v22/automated_review_299.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
pytestmark=pytest.mark.usefixtures('no_external_network')


def decide(**updates):
    values={'tfidf':'spam','minilm':'spam','content_codes':[],'url_codes':[],'sender_codes':[],
            'provenance_verified':True,'research_use_verified':True}
    values.update(updates)
    return review.classify(**values)


def test_agreement_alone_never_auto_reviews_or_approves():
    result=decide()
    assert result['status']=='Needs Human Review'
    assert 'spam_agreement_without_independent_content_support' in result['flags']
    assert not result['approved'] and not result['human_verified'] and not result['training_eligible']


def test_source_promotion_and_models_only_allow_weak_auto_review():
    result=decide(content_codes=['commercial_promotion'])
    assert result['status']=='Auto-reviewed' and result['ai_suggested_label']=='spam'
    assert result['ai_label_status']=='weak_label' and result['confidence_probability'] is None
    assert result['source_policy_consideration_only'] and not result['approved']
    blocked=decide(content_codes=['commercial_promotion'],research_use_verified=False)
    assert blocked['status']=='Needs Human Review' and not blocked['source_policy_consideration_only']


def test_model_normal_conflicts_with_source_not_proof_source_is_wrong():
    result=decide(tfidf='normal',minilm='normal')
    assert result['ai_suggested_label']=='normal' and result['status']=='Needs Human Review'
    assert 'normal_has_model_support_only_no_verified_normal_rule' in result['flags']


@pytest.mark.parametrize('url_codes',[['plain_http'],['long_url'],[]])
def test_third_party_link_or_http_or_urgency_does_not_imply_phishing(url_codes):
    result=decide(content_codes=['commercial_promotion','urgency','call_to_action'],url_codes=url_codes)
    assert result['ai_suggested_label']=='spam' and result['status']=='Auto-reviewed'


def test_phishing_requires_multiple_signals_and_human_context_check():
    assert decide(content_codes=['credential_request'])['ai_suggested_label']!='phishing'
    result=decide(content_codes=['credential_request','account_suspension'],url_codes=['suspicious_token'])
    assert result['ai_suggested_label']=='phishing' and result['status']=='Needs Human Review'
    uncorroborated=decide(minilm='phishing')
    assert uncorroborated['ai_suggested_label']=='uncertain' and uncorroborated['status']=='Needs Human Review'


def test_full_text_scans_risk_beyond_model_character_limit_without_exporting_text():
    rules=load_security_config(ROOT/'config/security_rules_v2.json')
    email={'subject':'Synthetic fixture','body':('Synthetic meeting context. '*400)+
        'Your account will be suspended. Enter your password immediately. https://example.invalid/verify',
        'sender':'fixture@example.com','ngon_ngu_goi_y':'en'}
    finding=review.full_text_findings(email,rules)
    assert finding['body_characters_scanned']==len(email['body']) and finding['analysis_truncated'] is False
    assert 'credential_request' in finding['content_codes']
    assert any(span['start']>4096 and span['category']=='credential_request' for span in finding['span_references'])
    assert all(set(span)<={'rule_id','category','field','coordinate_space','start','end','language'} for span in finding['span_references'])
    result=decide(content_codes=finding['content_codes'],url_codes=finding['url_codes'],sender_codes=finding['sender_codes'])
    assert result['ai_suggested_label']=='phishing' and not result['approved']


def test_empty_or_mismatched_data_is_only_an_excluded_candidate():
    result=decide(quality_flags=['empty_body'])
    assert result['status']=='Excluded Candidate' and not result['training_eligible']
    duplicate=decide(content_codes=['commercial_promotion'],duplicate_flags=['old_v2_reference_normalized_overlap'])
    assert duplicate['status']=='Needs Human Review' and not duplicate['source_policy_consideration_only']
