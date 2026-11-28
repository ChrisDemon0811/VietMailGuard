"""Local V2.3 artifacts + UI integration; synthetic mail, no Test/API access."""
import copy
import json
import sys
from email.message import EmailMessage
from pathlib import Path

import pytest

import app.i18n
from vietmailguard.mail_service import MailService
from vietmailguard.v21.llm_reviewer import MockReviewerProvider, ReviewerPrivacyError
from vietmailguard.v21.runtime_inference import V21ModelUnavailable
from vietmailguard.v21.runtime_modes import analyzer_for_mode
from vietmailguard.v23 import runtime_inference as runtime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"app"))
import v23_dashboard_ui as dashboard

pytestmark=pytest.mark.usefixtures("no_external_network")


@pytest.mark.parametrize("mode",runtime.MODES)
def test_missing_configuration_never_falls_back(mode,tmp_path):
    with pytest.raises(V21ModelUnavailable):runtime.load_v23_engine(mode,tmp_path)


def test_hash_and_namespace_verified_before_model_load(tmp_path,monkeypatch):
    config=json.loads((ROOT/"config/v23/runtime_v23_ml.json").read_text(encoding="utf-8"))
    path=tmp_path/"config/v23/runtime_v23_ml.json";path.parent.mkdir(parents=True)
    config["files"]["artifact"]["path"]="../outside.joblib"
    path.write_text(json.dumps(config),encoding="utf-8")
    monkeypatch.setattr(runtime.joblib,"load",lambda *_:pytest.fail("Unverified model was loaded"))
    with pytest.raises(V21ModelUnavailable):runtime.load_v23_engine("v23_ml",tmp_path)
    config["files"]["artifact"]["path"]="models/v23_multilingual/test.joblib"
    artifact=tmp_path/config["files"]["artifact"]["path"];artifact.parent.mkdir(parents=True);artifact.write_bytes(b"tampered")
    path.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(V21ModelUnavailable):runtime.load_v23_engine("v23_ml",tmp_path)


@pytest.fixture(scope="module",params=runtime.MODES)
def engine(request):
    config=ROOT/f"config/v23/runtime_{request.param}.json"
    if not config.exists():pytest.skip("Local V2.3 configuration is not included in source-only checkout")
    data=json.loads(config.read_text(encoding="utf-8"))
    if not (ROOT/data["files"]["artifact"]["path"]).exists():pytest.skip("Local research binary not included in source-only checkout")
    return runtime.load_v23_engine(request.param)


@pytest.mark.parametrize("subject,body,language",[
    ("Project meeting","Please join our project meeting tomorrow at 2 pm.","en"),
    ("Lịch họp","Xin chào, chúng ta họp nhóm lúc hai giờ chiều mai. Cảm ơn bạn.","vi"),
    ("会议通知","明天下午两点举行项目会议，请准时参加。","zh"),
])
def test_real_models_unicode_and_uncalibrated_results(engine,subject,body,language):
    result=engine.analyze_email(subject=subject,body=body)
    assert result["prediction"] in runtime.CLASSES
    assert result["model_version"].startswith("2.3-")
    assert result["model_scope"]=="v23_research_preview"
    assert result["confidence"] is None and result["class_probabilities"]=={}
    assert result["model_artifact"]["candidate_id"]==engine.runtime_config["candidate"]
    assert result["detected_language"]==language
    if language=="zh":assert result["language_support_status"]["status"]=="unvalidated"


def test_eml_sqlite_restart_and_mock_review_never_change_ml(engine,tmp_path):
    message=EmailMessage();message["From"]="sender@example.com";message["To"]="recipient@example.org"
    message["Subject"]="Synthetic project meeting";message.set_content("Please join the team meeting tomorrow.")
    path=tmp_path/"mail.db"
    with MailService.from_database_path(path,analyzer=engine.analyze_email) as service:
        received=service.import_eml_bytes(message.as_bytes());email_id=received["email"]["id"]
        before=service.open_email(email_id)
        service.review_stored_email(email_id,provider=MockReviewerProvider())
        after=service.open_email(email_id)
        assert after["analysis"]["result"]==before["analysis"]["result"]
        assert after["email"]["folder"]==before["email"]["folder"]
        class External(MockReviewerProvider):
            is_external=True
            def review(self,*args,**kwargs):pytest.fail("External provider called without consent")
        with pytest.raises(ReviewerPrivacyError):
            service.review_stored_email(email_id,provider=External(),allow_external=True,explicit_email_sharing_consent=False)
    with MailService.from_database_path(path,analyzer=lambda **_:pytest.fail("Opening mail reran inference")) as service:
        assert service.open_email(email_id)["analysis"]["result"]==before["analysis"]["result"]


def test_dashboards_use_real_aggregates_and_missing_is_not_faked(tmp_path):
    assert dashboard.comparison_rows(tmp_path)==[]
    rows=dashboard.comparison_rows(ROOT)
    if not rows:pytest.skip("Local saved research metrics not present")
    expected=json.loads((ROOT/dashboard.PATHS["optimization"]).read_text(encoding="utf-8"))
    saved=next(r for r in expected["candidates"] if r["name"]=="guarded_weak_recovery")
    actual=next(r for r in rows if r["candidate"]=="guarded_weak_recovery")
    assert actual["Macro F1"]==saved["metrics"]["macro_f1"]
    assert actual["Normal sai"]==15
    assert actual["Nguồn cập nhật tham số"]==357013


def test_streamlit_model_selection_analyze_save_dashboards(tmp_path,monkeypatch,streamlit_loopback):
    from streamlit.testing.v1 import AppTest
    if not runtime.model_availability()[0]:pytest.skip("Local V2.3 model absent")
    monkeypatch.setenv("VIETMAILGUARD_MAIL_DB",str(tmp_path/"ui.db"))
    app=AppTest.from_file(str(ROOT/"app/app.py"),default_timeout=120).run()
    def selector(key):return next(w for w in app.selectbox if w.key==key)
    def button(key):return next(w for w in app.button if w.key==key)
    assert "v23_ml" in selector("analysis_mode_selector").options or any("V2.3 Traditional" in str(v) for v in selector("analysis_mode_selector").options)
    selector("analysis_mode_selector").set_value("v23_ml").run()
    app.switch_page("views/email_analyzer.py").run()
    next(w for w in app.text_input if w.key=="input_quick_subject").set_value("Project meeting")
    next(w for w in app.text_area if w.key=="input_quick_body").set_value("Please join our team meeting tomorrow.")
    button("quick_analyze_button").click().run()
    assert not app.exception
    result=copy.deepcopy(app.session_state["vietmailguard_analysis_result"]["result"])
    assert result["model_version"]=="2.3-traditional-research"
    button("quick_save_to_mailbox").click().run()
    assert not app.exception
    selector("analysis_mode_selector").set_value("v21_hybrid").run()
    selector("hybrid_classifier_selector").set_value("v23_ml").run()
    button("quick_analyze_button").click().run()
    original=copy.deepcopy(app.session_state["vietmailguard_analysis_result"]["result"])
    button("quick_llm_review_button").click().run()
    assert app.session_state["vietmailguard_analysis_result"]["result"]==original
    assert app.session_state["vietmailguard_llm_review"]["reviewer"]["provider"]=="mock"
    app.switch_page("views/model_dashboard.py").run()
    assert not app.exception
    assert app.title[0].value=="So sánh mô hình — V2.3"
    assert any("Macro F1" in frame.value.columns for frame in app.dataframe)
    app.switch_page("views/dataset_dashboard.py").run()
    assert not app.exception
    assert [w.value for w in app.metric][:3]==["7,748","127,743","229,270"]
