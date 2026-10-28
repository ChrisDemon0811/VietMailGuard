"""Standalone, local Vietnamese UI for TREC06C Chinese-email review.

Run: python -m streamlit run scripts/v21/trec06c_review_app.py
This app does not import VietMailGuard Mail, train ML, or approve source rules.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import streamlit as st

from vietmailguard.v21.gemini_reviewer import configuration_status
from vietmailguard.v21.paths import resolve_v21_paths
from vietmailguard.v21.trec_curation import _read_decisions, load_full_message
from vietmailguard.v21.trec_review_assist import (
    GeminiTranslator, LOCAL_MODEL_ENV, MAX_FIELD_CHARS, MAX_TRANSLATION_REQUESTS,
    TRANSLATION_CHUNK_CHARS, TRANSLATION_PROMPT_VERSION,
    LocalSeq2SeqTranslator, TrecReviewStore, local_model_status,
)


QUEUE_NAMES = {
    "Mẫu Ham phân tầng": "ham_probability_sample.csv",
    "Spam cần phân biệt": "spam_focus_queue.csv",
    "Ngoại lệ chất lượng/trùng": "review_queue.csv",
    "Chuyên gia tiếng Trung": None,
}


def load_queue(directory: Path, filename: str) -> list[dict[str, str]]:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu hàng đợi {filename}; chạy prepare/prepare-deep trước")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def visible_queue(rows: list[dict[str, str]], decisions: dict[str, dict[str, str]],
                  only_pending: bool) -> list[dict[str, str]]:
    return [row for row in rows if not only_pending or row["id"] not in decisions]


def load_expert_queue(directory: Path, expert_ids: set[str]) -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for filename in (name for name in QUEUE_NAMES.values() if name):
        if not (directory / filename).is_file():
            continue
        for row in load_queue(directory, filename):
            if row["id"] in expert_ids:
                found.setdefault(row["id"], row)
    return list(found.values())


def neighbor_id(rows: list[dict[str, str]], selected_id: str, offset: int) -> str:
    ids = [row["id"] for row in rows]
    if not ids:
        return ""
    current = ids.index(selected_id) if selected_id in ids else 0
    return ids[min(max(current + offset, 0), len(ids) - 1)]


@st.cache_resource(show_spinner=False)
def _local_translator(model_path: str) -> LocalSeq2SeqTranslator:
    return LocalSeq2SeqTranslator(Path(model_path))


def main() -> None:
    st.set_page_config(page_title="TREC06C · Kiểm duyệt tiếng Việt", layout="wide")
    st.title("Kiểm duyệt email TREC06C")
    st.caption("Công cụ cục bộ. Bản dịch và đề xuất AI là tham khảo, không phải nhãn đã xác minh.")
    gemini_enabled = st.sidebar.toggle(
        "Bật Gemini cho phiên này", value=False, key="trec_gemini_enabled",
        help="Chỉ bật giao diện gọi API; mỗi email vẫn cần đồng ý và bấm nút riêng.",
    )
    gemini_config = configuration_status()
    st.sidebar.caption(
        f"Gemini: {'đã cấu hình' if gemini_config['api_key_configured'] and gemini_config['model_syntax_valid'] else 'chưa cấu hình'}"
        " · không kiểm tra mạng khi mở trang"
    )
    paths = resolve_v21_paths()
    try:
        store = TrecReviewStore(paths)
        queue_label = st.selectbox("Hàng đợi", list(QUEUE_NAMES), key="trec_queue")
        filename = QUEUE_NAMES[queue_label]
        all_rows = (load_queue(store.directory, filename) if filename
                    else load_expert_queue(store.directory, store.expert_ids()))
        decisions = _read_decisions(store.directory / "decisions.csv")
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error))
        st.stop()

    progress = store.progress()
    st.caption(
        f"Đã xem: {progress['viewed']} · Đã dịch/cache: {progress['translations_cached']} · "
        f"Đề xuất AI: {progress['ai_suggestions']} · Quyết định người kiểm duyệt: "
        f"{progress['human_verified']} · Eligible: {progress['eligible']}"
    )
    only_pending = st.checkbox("Chỉ thư chưa có quyết định kiểm duyệt", value=True)
    rows = visible_queue(all_rows, decisions, only_pending)
    if queue_label != "Chuyên gia tiếng Trung" and st.checkbox("Chỉ hàng đợi chuyên gia tiếng Trung"):
        expert_ids = store.expert_ids()
        rows = [row for row in rows if row["id"] in expert_ids]
    if not rows:
        st.info("Hàng đợi hiện không có thư phù hợp bộ lọc.")
        return
    ids = [row["id"] for row in rows]
    if st.session_state.get("trec_selected_id") not in ids:
        st.session_state["trec_selected_id"] = ids[0]
    selection = st.selectbox(
        "Chọn email", ids, index=ids.index(st.session_state["trec_selected_id"]),
        format_func=lambda value: next(
            (f"{value} · {row['raw_label']} · {row['subject'][:65]}"
             for row in rows if row["id"] == value), value
        ),
    )
    st.session_state["trec_selected_id"] = selection
    previous, middle, following = st.columns([1, 4, 1])
    with previous:
        if st.button("← Trước", disabled=selection == ids[0]):
            st.session_state["trec_selected_id"] = neighbor_id(rows, selection, -1)
            st.rerun()
    with middle:
        st.caption(f"{ids.index(selection) + 1}/{len(ids)} trong hàng đợi đã lọc")
    with following:
        if st.button("Sau →", disabled=selection == ids[-1]):
            st.session_state["trec_selected_id"] = neighbor_id(rows, selection, 1)
            st.rerun()

    queue_row = next(row for row in rows if row["id"] == selection)
    try:
        email = load_full_message(paths, queue_row)
    except (ValueError, RuntimeError, LookupError, FileNotFoundError) as error:
        st.error(f"Không thể mở bản ghi đã audit: {error}")
        return
    store.update_state(selection, viewed=True)
    saved = decisions.get(selection)
    state = store.state(selection)
    suggestion = store.current_suggestion(email)
    st.write(f"**Nguồn:** TREC06C · **Nhãn gốc:** {email['raw_label']} · "
             f"**Trạng thái:** {saved['label_status'] if saved else 'weak_label / chưa duyệt'}")
    st.caption(
        f"raw_label: {email['raw_label']} · "
        f"ai_suggested_label: {suggestion['ai_suggested_label'] if suggestion else '(chưa có)'} · "
        f"human_verified_label: {store.qualified_label(selection, saved) or '(chưa có)'}"
    )
    st.caption(f"ID: {selection} · Tầng: {queue_row.get('review_stratum') or queue_row.get('quality_stratum')} "
               f"· Cờ chất lượng: {queue_row.get('quality_flags') or 'không'}")

    chinese, vietnamese = st.columns(2)
    with chinese:
        st.subheader("Email tiếng Trung gốc")
        st.write("**Tiêu đề**")
        st.text(email["subject"] or "(trống)")
        st.write("**Nội dung**")
        st.text(email["body"] or "(trống)")
    with vietnamese:
        st.subheader("Bản dịch tiếng Việt (tham khảo)")
        local_status = local_model_status()
        options = ["Offline (model local)"] if local_status["available"] else []
        if gemini_enabled and gemini_config["api_key_configured"] and gemini_config["model_syntax_valid"]:
            options.append("Gemini API")
        backend = st.selectbox("Nguồn dịch", options) if options else None
        provider_name = (
            f"local:{Path(os.environ[LOCAL_MODEL_ENV]).expanduser().resolve().name}"
            if backend == "Offline (model local)"
            else f"gemini:{gemini_config['model']}:{TRANSLATION_PROMPT_VERSION}"
            if backend == "Gemini API" else ""
        )
        cached = store.cached_translation(email, provider_name) if provider_name else None
        if cached:
            st.write("**Tiêu đề:**")
            st.text(cached["subject_vi"])
            st.text(cached["body_vi"])
            st.caption("Bản dịch đã cache. Phải đối chiếu email gốc nếu đưa ra quyết định nhãn.")
        elif backend == "Offline (model local)":
            if st.button("Dịch bằng model local"):
                try:
                    translator = _local_translator(os.environ[LOCAL_MODEL_ENV])
                    store.translation(email, translator)
                    st.rerun()
                except Exception:
                    st.error("Dịch offline không thành công. Kiểm tra model local và độ dài email.")
        elif backend == "Gemini API":
            estimated_calls = GeminiTranslator.estimate_requests(email["subject"], email["body"])
            st.warning("Gemini sẽ nhận tiêu đề và nội dung email (URL/email được che một phần). "
                       "Dữ liệu còn lại có thể nhạy cảm; API có thể phát sinh chi phí. "
                       "Không gửi toàn bộ dataset; chỉ email đang mở khi bạn bấm nút.")
            st.caption(f"Ước tính tối đa {estimated_calls} request dịch cho email này; "
                       f"giới hạn cứng {MAX_TRANSLATION_REQUESTS} request/email, "
                       f"{TRANSLATION_CHUNK_CHARS:,} ký tự mỗi đoạn. "
                       "Request lỗi có thể vẫn tính vào hạn mức API.")
            translation_consent = st.checkbox("Tôi đồng ý gửi email này để dịch", key=f"translate_consent_{selection}")
            within_budget = (estimated_calls <= MAX_TRANSLATION_REQUESTS
                             and len(email["body"]) <= MAX_FIELD_CHARS
                             and len(email["subject"]) <= TRANSLATION_CHUNK_CHARS)
            if not within_budget:
                st.error("Email vượt giới hạn dịch an toàn; chuyển chuyên gia, không gửi API.")
            if st.button("Gửi và dịch email này", disabled=not translation_consent or not within_budget):
                try:
                    store.translation(email, GeminiTranslator(), consent=True)
                    st.rerun()
                except Exception:
                    st.error("Không dịch được qua Gemini; kiểm tra cấu hình/kết nối. Không lưu bản dịch lỗi.")
        else:
            st.info("Chưa có model dịch offline. Gemini đang tắt hoặc chưa cấu hình; "
                    "bật trong thanh bên nếu bạn chủ động muốn gửi email này.")
        if not local_status["available"]:
            st.caption(f"Chưa có model dịch offline. Đặt {LOCAL_MODEL_ENV} tới thư mục model "
                       "NLLB-compatible đã tải hợp pháp; ứng dụng không tự tải model.")

    st.divider()
    st.subheader("Nhận định AI — nhãn yếu, không phải ground truth")
    if suggestion:
        st.write(f"**Đề xuất:** {suggestion['ai_suggested_label']} · **Nguồn:** {suggestion['provider']}")
        st.write("**Giải thích:**")
        st.text(suggestion["summary"])
        for item in suggestion["evidence"]:
            st.caption(f"Dẫn chứng ({item['field']}):")
            st.text(item["quote"])
            st.text(item["explanation"])
        if suggestion["suspicious_indicators"]:
            st.write("**Dấu hiệu đáng ngờ:**")
            for indicator in suggestion["suspicious_indicators"]:
                st.text(indicator)
        st.caption("Không có calibrated probability. Đề xuất AI không kích hoạt eligible hoặc thay nhãn người kiểm duyệt.")
    else:
        st.info("Chưa có đề xuất AI cho email này.")
        st.caption("Đề xuất Gemini là tùy chọn; nội dung email được che URL/email một phần, "
                   "nhưng các thông tin nhạy cảm khác vẫn có thể được gửi và có thể phát sinh chi phí.")
        if gemini_enabled and gemini_config["api_key_configured"] and gemini_config["model_syntax_valid"]:
            ai_consent = st.checkbox("Tôi đồng ý gửi email này để AI tham khảo", key=f"ai_consent_{selection}")
            if st.button("Lấy nhận định Gemini", disabled=not ai_consent):
                try:
                    store.suggestion(email, consent=True)
                    st.rerun()
                except Exception:
                    st.error("Không lấy được đề xuất Gemini. Không có nhãn nào được thay đổi.")
        else:
            st.caption("Gemini đang tắt hoặc chưa cấu hình; không có API call khi xem email.")

    st.divider()
    st.subheader("Tiến độ và chuyển chuyên gia")
    note = st.text_area("Ghi chú quan sát/bằng chứng (tránh chép dữ liệu cá nhân)",
                        value=state["ghi_chu"], key=f"note_{selection}")
    needs = st.checkbox("Cần xác minh thêm", value=bool(state["can_xac_minh"]),
                        key=f"needs_{selection}")
    expert = st.checkbox("Chuyển sang hàng đợi chuyên gia tiếng Trung",
                         value=bool(state["chuyen_chuyen_gia"]), key=f"expert_{selection}")
    if st.button("Lưu tiến độ"):
        store.update_state(selection, viewed=True, needs_verification=needs,
                           expert=expert, note=note)
        st.success("Đã lưu. Đây chưa phải quyết định nhãn đã xác minh.")

    with st.expander("Quyết định của người kiểm duyệt đủ năng lực tiếng Trung"):
        st.warning("Chỉ người thực sự đọc hiểu và đã đối chiếu email tiếng Trung gốc được ghi "
                   "quyết định nhãn. Chỉ đọc bản dịch hoặc đồng ý với AI không đủ điều kiện.")
        if saved:
            st.info(f"Đã có quyết định: {saved['decision']} ({saved['label_status']}). "
                    "Không ghi đè; xử lý bất đồng bằng quy trình riêng.")
        else:
            reviewer = st.text_input("Mã người kiểm duyệt", key=f"reviewer_{selection}")
            qualified = st.checkbox("Tôi đọc hiểu tiếng Trung đủ để đánh giá email gốc",
                                    key=f"qualified_{selection}")
            original_read = st.checkbox("Tôi đã đọc/đối chiếu email tiếng Trung gốc, không chỉ bản dịch",
                                        key=f"original_read_{selection}")
            decision = st.selectbox("Quyết định", ["review", "normal", "spam", "phishing", "exclude"],
                                    key=f"decision_{selection}")
            evidence = st.text_area("Lý do và bằng chứng kiểm duyệt", key=f"evidence_{selection}")
            if st.button("Lưu quyết định chuyên gia", disabled=not (qualified and original_read)):
                try:
                    result = store.record_qualified_decision(
                        queue_row, decision=decision, reviewer=reviewer,
                        evidence=evidence, qualified_chinese_reader=qualified,
                        original_chinese_read=original_read,
                    )
                    st.success(f"Đã lưu {result['label_status']}. "
                               "Nếu bị chặn bởi chất lượng/trùng lặp, nhãn vẫn ở Review.")
                    st.rerun()
                except (PermissionError, ValueError, RuntimeError) as error:
                    st.error(f"Không lưu quyết định: {error}")

    if st.button("Xuất manifest cục bộ"):
        ai_path, human_path = store.export_manifests()
        st.success(f"Đã xuất riêng: {ai_path.name} và {human_path.name}. "
                   "Không đưa các file này lên GitHub.")
    st.caption("Tập Ham 6.802 chỉ là đề xuất có điều kiện. Quy tắc hàng loạt vẫn tắt; "
               "TREC Spam không tự động thành Spam/Phishing ba lớp.")


if __name__ == "__main__":
    main()
