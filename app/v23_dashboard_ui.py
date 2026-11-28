"""Compact V2.3 aggregates: no email-row access, inference or network."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PATHS = {
    "development": "results/v23_multilingual/development_01/metrics.json",
    "expansion": "results/v23_multilingual/classifier_expansion_02/metrics.json",
    "optimization": "results/v23_multilingual/large_scale_optimization_03/metrics.json",
    "data": "results/v23_multilingual/development_01/usage_accounting.json",
}


def _reject_constant(value):
    raise ValueError("Non-finite aggregate")


@st.cache_data(show_spinner=False)
def _read(path: str, signature: tuple):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=_reject_constant)
        return value if isinstance(value,dict) else {}
    except (OSError, ValueError, UnicodeError):return {}


def saved(root: Path, key: str):
    path = root/PATHS[key]
    try:
        stat = path.stat()
        if stat.st_size > 2_000_000:return {}
        return _read(str(path),(stat.st_size,stat.st_mtime_ns))
    except OSError:return {}


def comparison_rows(root: Path):
    rows=[]
    for key,candidate,title in [
        ("development","large_scale_idf_chi2_svc","V2.3 Traditional — TF-IDF + Chi2 + LinearSVC"),
        ("development","minilm_mlp_supervised","MiniLM + MLP — Supervised"),
        ("expansion","weak_partial_consistency_denoise_357013","MiniLM + MLP — Weak/SSL trước tối ưu"),
        ("optimization","guarded_weak_recovery","V2.3 Large-scale MLP — Tối ưu"),
        ("optimization","verified_control","MiniLM + MLP — Đối chứng tối ưu"),
    ]:
        matches=[r for r in saved(root,key).get("candidates",[]) if isinstance(r,dict) and r.get("candidate",r.get("name"))==candidate]
        if len(matches)!=1:continue
        item=matches[0];metric=item.get("metrics",{});score=metric.get("macro_f1")
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:continue
        matrix=metric.get("confusion_matrix",[])
        if len(matrix)!=3 or any(not isinstance(r,list) or len(r)!=3 for r in matrix) or sum(map(sum,matrix))!=1938:continue
        rows.append({"Mô hình":title,"Macro F1":score,"Accuracy":metric.get("accuracy"),
            "Spam Recall":metric.get("per_class",{}).get("spam",{}).get("recall"),
            "Phishing Recall":metric.get("per_class",{}).get("phishing",{}).get("recall"),
            "Normal sai":sum(matrix[0][1:]),"Verified Train":7748,
            "Nguồn cập nhật tham số":item.get("expansion_unique_parameter_update_rows",item.get("source_parameter_update_rows",0)),
            "candidate":candidate,"matrix":matrix})
    return rows


def render_models(root: Path, language="vi") -> bool:
    rows=comparison_rows(root)
    if not rows:return False
    st.title("So sánh mô hình — V2.3")
    st.caption("Cùng Development Validation: 1.938 email. Đây không phải kết quả Final Test độc lập.")
    st.info("Khuyến nghị: giữ TF-IDF + LinearSVC. V2.3 Traditional (thêm Chi-Square) đạt ngang baseline; Large-scale MLP là chế độ thử nghiệm riêng.")
    frame=pd.DataFrame([{k:v for k,v in r.items() if k not in {"candidate","matrix"}} for r in rows])
    frame=frame.rename(columns={"Nguồn cập nhật tham số":"Nguồn cập nhật classifier/MLP"})
    st.dataframe(frame,hide_index=True,width="stretch",column_config={
        k:st.column_config.NumberColumn(format="%.6f") for k in ["Macro F1","Accuracy","Spam Recall","Phishing Recall"]})
    figure=go.Figure(go.Bar(x=[r["Mô hình"] for r in rows],y=[r["Macro F1"] for r in rows],
        marker_color=["#2563eb"]+["#64748b"]*(len(rows)-1),text=[f"{r['Macro F1']:.6f}" for r in rows],textposition="outside"))
    figure.update_layout(height=350,margin=dict(t=20,b=15,l=10,r=10),yaxis=dict(range=[0,1.08],title="Macro F1"))
    st.plotly_chart(figure,width="stretch")
    st.write("Dữ liệu lớn đã tham gia học tham số thật nhưng chưa vượt supervised baseline. Không tự thay mô hình bằng kết quả kém hơn.")
    traditional=next((r for r in saved(root,"development").get("candidates",[]) if r.get("candidate")=="large_scale_idf_chi2_svc"),{})
    if isinstance(traditional.get("idf_fit_rows"),int):
        st.caption(f"Traditional: IDF học từ {traditional['idf_fit_rows']:,} email; classifier chỉ học nhãn Verified Train. Cột nguồn cập nhật classifier/MLP không tính việc học IDF.")
    st.caption("Validation chỉ có 33 Spam thuộc 13 nhóm. VI là bản dịch, thiếu Normal; ZH chưa có benchmark ba lớp đã xác minh.")
    with st.expander("Chi tiết nghiên cứu V2.3 — Chi-Square, SVD, PCA, LDA và weak/SSL"):
        for key in ("development","expansion","optimization"):
            detail=[{"Candidate":r.get("candidate",r.get("name")),"Macro F1":r.get("metrics",{}).get("macro_f1"),
                     "Artifact":r.get("artifact"),"SHA-256":r.get("artifact_sha256")} for r in saved(root,key).get("candidates",[]) if isinstance(r,dict)]
            st.caption(PATHS[key]);st.dataframe(pd.DataFrame(detail),hide_index=True,width="stretch")
        choice=st.selectbox("Confusion matrix",options=list(range(len(rows))),format_func=lambda i:rows[i]["Mô hình"],key="v23_confusion_choice")
        st.caption("Hàng: nhãn thật; cột: dự đoán. Normal / Spam / Phishing.")
        st.dataframe(pd.DataFrame(rows[choice]["matrix"],index=["Normal","Spam","Phishing"],columns=["Normal","Spam","Phishing"]),width="stretch")
    return True


def render_data(root: Path, language="vi") -> bool:
    data=saved(root,"data")
    if not isinstance(data.get("expansion_by_source_status_language"),list):return False
    sources=data["expansion_by_source_status_language"]
    counts={status:sum(r[3] for r in sources if r[1]==status) for status in ["weak_label","unlabeled"]}
    verified=data.get("verified_supervised_fit")
    if not isinstance(verified,int):return False
    st.subheader("Dữ liệu nghiên cứu V2.3")
    for col,title,value in zip(st.columns(3),["Verified Train","Weak Label","Unlabeled"],[verified,counts["weak_label"],counts["unlabeled"]]):col.metric(title,f"{value:,}")
    st.write(f"**{verified+sum(counts.values()):,} email** tham gia cập nhật tham số MLP weak/SSL; chưa phải 300.000 mẫu supervised Verified.")
    st.caption(f"Development gốc: {data.get('verified_reference_rows',0):,}; cohort Verified: {verified+data.get('verified_validation',0):,}; Validation: {data.get('verified_validation',0):,}. Các tập này chồng lặp, không cộng dồn.")
    rows=[dict(row,split=split,origin=row.get("origin") or "không khai báo")
          for split,items in data.get("by_reference_split_label_language_origin",{}).items() for row in items]
    if rows:
        frame=pd.DataFrame(rows)
        st.dataframe(frame.pivot_table(index=["split","language"],columns="label",values="rows",aggfunc="sum",fill_value=0).reset_index(),hide_index=True,width="stretch")
    st.warning("Chưa có Final Test độc lập. Tiếng Việt là dữ liệu dịch; không có benchmark Verified tiếng Trung ba lớp. Weak/Unlabeled không được tính là ground truth hoặc đưa vào Validation.")
    with st.expander("Nguồn, ngôn ngữ và kiểm tra chất lượng"):
        st.caption(PATHS["data"])
        st.dataframe(pd.DataFrame(sources,columns=["Nguồn","Trạng thái nhãn","Ngôn ngữ heuristic","Số email"]),hide_index=True,width="stretch")
        st.caption("Ngôn ngữ nguồn là heuristic, không phải bằng chứng tiếng bản địa. Nhãn spam nguồn không tự tương đương Spam ba lớp; Enron vẫn Unlabeled.")
        st.dataframe(pd.DataFrame(data.get("all_source_outcomes",[]),columns=["Nguồn","Kết quả screening","Số email"]),hide_index=True,width="stretch")
    return True
