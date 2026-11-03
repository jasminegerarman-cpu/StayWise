"""StayWise | real-data, ML-powered hotel booking risk assessment platform."""
import io
import json
import sys
from pathlib import Path

# VS Code's regular ▶ Run Python File button normally does not launch Streamlit.
# When executed directly, restart this file using the Streamlit command so the
# application opens in the user's browser. During Streamlit execution, a script
# context exists and the normal app continues below (no launch loop).
if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is None:
        from launch_staywise import start_streamlit
        raise SystemExit(start_streamlit())

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import joblib

from src.config import (DATA_PATH, MODEL_PATH, REPORT_PATH, ROOT, FEATURES, CATEGORICAL,
                        NUMERIC, MONTHS)
from src.pipeline import cancellation_probabilities, risk_bucket, validate_booking
from src.history import get_history, total_assessments, record_assessment, clear_history
from src.theme import install_theme, hero, heading, section, info_card

st.set_page_config(page_title="StayWise | Booking Intelligence", page_icon="🏨",
                   layout="wide", initial_sidebar_state="expanded")
install_theme()

TEAL, NAVY, GOLD, SILVER = "#137D75", "#1B3445", "#C79D6C", "#DBE6EB"

@st.cache_resource(show_spinner="Loading prediction engine...")
def load_engine():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner="Preparing insights...")
def load_report():
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))

@st.cache_data(show_spinner="Analyzing historical bookings...")
def historical_summary():
    raw = pd.read_csv(DATA_PATH, usecols=["hotel", "is_canceled", "arrival_date_month", "lead_time", "adr"], low_memory=False)
    raw["is_canceled"] = pd.to_numeric(raw["is_canceled"], errors="coerce")
    raw["adr"] = pd.to_numeric(raw["adr"], errors="coerce")
    hotel = raw.groupby("hotel")["is_canceled"].agg(["count", "mean"]).reset_index()
    monthly = raw.groupby("arrival_date_month")["is_canceled"].agg(["count", "mean"]).reindex(MONTHS).reset_index()
    return {
        "raw_total": len(raw),
        "cancel_rate": float(raw["is_canceled"].mean()),
        "avg_lead": float(raw["lead_time"].mean()),
        "hotel": hotel,
        "monthly": monthly,
    }


def chart_layout(fig, height=330, yaxis=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans, sans-serif", color="#637687", size=12),
                      margin=dict(l=8, r=12, t=14, b=8), height=height,
                      showlegend=False, xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=True, gridcolor="#EEF2F4", zeroline=False),
                      hoverlabel=dict(bgcolor="#213A45", font_color="white"))
    if yaxis:
        fig.update_yaxes(**yaxis)
    return fig


def sidebar():
    with st.sidebar:
        st.markdown('''<div class="sidebar-brand"><div class="sidebar-mark">S</div><div class="sidebar-title">StayWise<span style="color:#D9B277!important">.</span></div></div><div class="sidebar-caption">HOTEL OPERATIONS INTELLIGENCE</div>''', unsafe_allow_html=True)
        st.caption("WORKSPACE")
        page = st.radio("Navigate", ["Overview", "New assessment", "Scenario explorer", "Batch analysis", "Model report", "About & guide"],
                        label_visibility="collapsed", key="page")
        st.divider()
        st.caption("SYSTEM STATUS")
        st.success("● Prediction engine active")
        st.caption(f"Model: {load_report()['best_model']}")
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        st.caption("ACADEMIC PROTOTYPE · VERSION 1.0")
        st.caption("Data is historical (2015–2017). Not a hotel reservation management system.")
        return page


DEFAULT_BOOKING = {
    "hotel": "City Hotel", "lead_time": 75, "arrival_date_month": "July",
    "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
    "adults": 2, "children": 0, "babies": 0,
    "meal": "BB", "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "is_repeated_guest": 0, "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
    "deposit_type": "No Deposit", "customer_type": "Transient", "adr": 115.0,
    "required_car_parking_spaces": 0, "total_of_special_requests": 1,
}
PRESETS = {
    "Typical booking": DEFAULT_BOOKING,
    "Short-lead stay": {**DEFAULT_BOOKING, "hotel": "Resort Hotel", "lead_time": 3,
                        "market_segment": "Direct", "distribution_channel": "Direct",
                        "total_of_special_requests": 2, "required_car_parking_spaces": 1},
    "Long-lead booking": {**DEFAULT_BOOKING, "lead_time": 250,
                          "previous_cancellations": 1, "total_of_special_requests": 0,
                          "deposit_type": "No Deposit"},
}


def navigate(target):
    # Streamlit widget callbacks run before widgets are instantiated on rerun.
    st.session_state["page"] = target


def apply_preset(name):
    for col, value in PRESETS[name].items():
        st.session_state[f"booking_{col}"] = value
    st.session_state.pop("last_assessment", None)


def initialize_form():
    for key, value in DEFAULT_BOOKING.items():
        if f"booking_{key}" not in st.session_state:
            st.session_state[f"booking_{key}"] = value


def field_value(name):
    return st.session_state[f"booking_{name}"]


def booking_form():
    initialize_form()
    st.markdown("**Quick demo scenarios**")
    b1, b2, b3 = st.columns(3)
    for column, name in zip([b1, b2, b3], PRESETS):
        column.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    st.caption("Scenario buttons prefill example inputs. All predictions are computed by the trained model, not hardcoded.")
    with st.form("booking_assessment", clear_on_submit=False):
        st.markdown("#### Reservation information")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.selectbox("Hotel property", ["City Hotel", "Resort Hotel"], key="booking_hotel")
            st.selectbox("Arrival month", MONTHS, key="booking_arrival_date_month")
            st.number_input("Lead time · days before arrival", 0, 1000, step=1, key="booking_lead_time")
            st.number_input("Weekend nights", 0, 60, step=1, key="booking_stays_in_weekend_nights")
            st.number_input("Week nights", 0, 60, step=1, key="booking_stays_in_week_nights")
            st.number_input("Average daily rate · original dataset units", min_value=0.0, max_value=10000.0, step=5.0, key="booking_adr")
            st.selectbox("Meal plan", ["BB", "HB", "FB", "SC", "Undefined"], key="booking_meal")
        with c2:
            st.number_input("Adults", 0, 20, step=1, key="booking_adults")
            st.number_input("Children", 0, 20, step=1, key="booking_children")
            st.number_input("Babies", 0, 20, step=1, key="booking_babies")
            st.selectbox("Market segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation", "Undefined"], key="booking_market_segment")
            st.selectbox("Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"], key="booking_distribution_channel")
            st.selectbox("Room type", list("ABCDEFGHLP"), key="booking_reserved_room_type")
            st.selectbox("Deposit policy", ["No Deposit", "Non Refund", "Refundable"], key="booking_deposit_type")
        with c3:
            st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"], key="booking_customer_type")
            st.selectbox("Returning guest", [0, 1], format_func=lambda x: "Yes" if x else "No", key="booking_is_repeated_guest")
            st.number_input("Previous cancellations", 0, 50, step=1, key="booking_previous_cancellations")
            st.number_input("Previous completed bookings", 0, 100, step=1, key="booking_previous_bookings_not_canceled")
            st.number_input("Parking spaces requested", 0, 10, step=1, key="booking_required_car_parking_spaces")
            st.number_input("Special requests", 0, 10, step=1, key="booking_total_of_special_requests")
            st.info("Estimates are for educational decision support, not guarantees. Do not penalize guests solely based on predictions.")
        submitted = st.form_submit_button("Analyze cancellation risk", type="primary", use_container_width=True)
    if submitted:
        booking = {col: field_value(col) for col in FEATURES}
        errors = validate_booking(booking)
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            try:
                probability = float(cancellation_probabilities(load_engine(), pd.DataFrame([booking]))[0])
                bucket = risk_bucket(probability)
                record_assessment(booking, probability, bucket)
                st.session_state["last_assessment"] = {"booking": booking, "prob": probability, "bucket": bucket}
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_assessment")
    if result:
        p = result["prob"]
        bucket = result["bucket"]
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        section("Assessment result")
        l, r = st.columns([1, 1.25], gap="large")
        with l:
            st.markdown(f'''<div class="result-card"><div class="result-title">Estimated cancellation probability</div><div class="huge">{p:.1%}</div><span class="pill pill-{bucket.lower()}">{bucket.upper()} RISK</span><div class="footnote">Estimated from the model's class-1 probability. This is a model score, not a guaranteed real-world cancellation rate.</div></div>''', unsafe_allow_html=True)
        with r:
            st.metric("Likely outcome at 50% threshold", "Cancellation" if p >= .50 else "Booking honored")
            st.progress(min(1., max(0., p)), text="Relative cancellation likelihood")
            st.markdown("**Suggested operational response**")
            if bucket == "High":
                st.warning("Review the booking and use normal, fair confirmation procedures. This score is not a reason to cancel or reject a guest.")
            elif bucket == "Moderate":
                st.info("Keep the reservation visible for routine follow-up and use standard booking reminders.")
            else:
                st.success("Proceed with standard booking processes. Continue normal confirmation and monitoring.")
        with st.expander("Show information used for this prediction"):
            st.dataframe(pd.DataFrame([result["booking"]]).T.rename(columns={0:"Submitted value"}), use_container_width=True)
        st.caption("Note: the probability is not independently calibrated. Validation metrics are shown in Model report.")


def overview():
    hero("Sharper decisions for every reservation.",
         "A booking-intelligence workspace that helps hotel teams estimate cancellation risk using a real, trained machine learning model.",
         "STAYWISE / EXECUTIVE OVERVIEW")
    hist = historical_summary()
    rep = load_report()
    f1 = rep["final_test"]["f1"]
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Historical bookings", f"{hist['raw_total']:,}")
    c2.metric("Historical cancel rate", f"{hist['cancel_rate']:.1%}")
    c3.metric("Independent test F1", f"{f1:.1%}")
    c4.metric("Assessments saved", f"{total_assessments():,}")
    st.caption("Historical cards use 2015–2017 source bookings; F1 comes from the untouched final test set. Assessment history is stored locally on this computer.")
    section("Historical performance signals")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Cancellation rate by arrival month**")
            monthly = hist["monthly"]
            fig = go.Figure(go.Bar(x=monthly["arrival_date_month"].str[:3], y=monthly["mean"] * 100,
                                   marker_color=TEAL, marker_line_width=0,
                                   hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
            st.plotly_chart(chart_layout(fig, yaxis=dict(ticksuffix="%")), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("**Booking volume by property type**")
            h = hist["hotel"]
            fig = go.Figure(go.Bar(y=h["hotel"], x=h["count"], orientation="h",
                                   marker_color=[GOLD, TEAL], hovertemplate="%{y}: %{x:,}<extra></extra>"))
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(chart_layout(fig), use_container_width=True)
    section("Explore the workspace")
    col1, col2, col3 = st.columns(3)
    with col1:
        info_card("01 / PREDICT", "Booking assessment", "Enter reservation details, receive a model-generated cancellation score and suggested workflow.")
        st.button("Open assessment", type="primary", use_container_width=True,
                  on_click=navigate, args=("New assessment",))
    with col2:
        info_card("02 / OPERATE", "Batch intelligence", "Upload an ordinary booking CSV, process multiple records, and export a prediction report.")
        st.button("Open batch analysis", use_container_width=True,
                  on_click=navigate, args=("Batch analysis",))
    with col3:
        info_card("03 / EVALUATE", "Model transparency", "Inspect holdout metrics, class labels, confusion matrices and model selection criteria.")
        st.button("View model evaluation", use_container_width=True,
                  on_click=navigate, args=("Model report",))
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        a, b = st.columns([3, 1], vertical_alignment="center")
        with a:
            st.markdown("**A standout feature: What-if Scenario Explorer**")
            st.caption("Compare model predictions after adjusting booking lead time, daily rate and guest requests. These are model simulations, not causal claims.")
        with b:
            st.button("Explore scenarios →", use_container_width=True,
                      on_click=navigate, args=("Scenario explorer",))
    section("Recently analyzed bookings")
    history = get_history()
    if history.empty:
        st.info("No assessments yet. Use **New assessment** to analyze your first booking.")
    else:
        st.dataframe(history.head(8), use_container_width=True, hide_index=True)


def assessment():
    heading("Cancellation risk assessment", "Enter booking details to obtain a real ML prediction, without fixed outcomes.", "PREDICTION STUDIO / SINGLE BOOKING")
    booking_form()
    section("Assessment history")
    history = get_history()
    if history.empty:
        st.caption("No local records yet.")
    else:
        st.dataframe(history.head(30), use_container_width=True, hide_index=True)
        export = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download local assessment history", export, "staywise_history.csv", "text/csv")


def scenario_explorer():
    heading("Scenario explorer", "Explore how this trained model responds when selected booking inputs change.",
            "INTELLIGENCE STUDIO / WHAT-IF SIMULATION")
    st.info("This is a model behavior demonstration, not proof that changing any booking attribute causes a cancellation. Use it for transparent model exploration only.")
    initialize_form()
    reference = st.session_state.get("last_assessment", {}).get("booking", DEFAULT_BOOKING).copy()
    source_label = "Last analyzed booking" if "last_assessment" in st.session_state else "Typical example booking"
    st.caption("Reference scenario: " + source_label)
    a, b, c = st.columns(3)
    lead = a.slider("Adjusted lead time (days)", 0, 450, int(reference["lead_time"]), 5)
    adr = b.slider("Adjusted average daily rate (dataset units)", 0, 500,
                   int(max(0, min(float(reference["adr"]), 500))), 5)
    requests = c.slider("Adjusted special requests", 0, 5,
                        int(max(0, min(int(reference["total_of_special_requests"]), 5))))
    baseline = cancellation_probabilities(load_engine(), pd.DataFrame([reference]))[0]
    adjusted = reference.copy()
    adjusted.update({"lead_time": lead, "adr": float(adr), "total_of_special_requests": requests})
    new_risk = cancellation_probabilities(load_engine(), pd.DataFrame([adjusted]))[0]
    left, right, change = st.columns(3)
    left.metric("Reference model score", f"{baseline:.1%}")
    right.metric("Adjusted model score", f"{new_risk:.1%}")
    change.metric("Score difference", f"{(new_risk - baseline) * 100:+.1f} percentage points")
    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Reference", "Adjusted"], y=[baseline * 100, new_risk * 100],
                             marker_color=[NAVY, TEAL], text=[f"{baseline:.1%}", f"{new_risk:.1%}"],
                             textposition="outside", cliponaxis=False))
        fig.update_yaxes(range=[0, 110])
        st.plotly_chart(chart_layout(fig, height=290, yaxis=dict(ticksuffix="%")), use_container_width=True)
    with st.expander("Compare exact scenario inputs"):
        comparison = pd.DataFrame({"Feature": FEATURES, "Reference": [reference[k] for k in FEATURES],
                                   "Adjusted": [adjusted[k] for k in FEATURES]})
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption("Probabilities are uncalibrated model estimates based on historical two-hotel data. Do not make automatic guest decisions based on scores.")


def batch_analysis():
    heading("Batch booking intelligence", "Upload multiple reservations and analyze them together.", "PREDICTION STUDIO / BULK ANALYSIS")
    st.markdown("**Step 1 · Download a valid input template**")
    template = pd.DataFrame([PRESETS["Typical booking"], PRESETS["Short-lead stay"], PRESETS["Long-lead booking"]])[FEATURES]
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       file_name="staywise_booking_template.csv", mime="text/csv")
    st.caption("Template rows are example input scenarios only, NOT research dataset rows.")
    st.markdown("**Step 2 · Upload your reservation file**")
    uploaded = st.file_uploader("Select a .csv file (maximum 5,000 bookings)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [col for col in FEATURES if col not in df.columns]
            if missing:
                st.error("Missing CSV columns: " + ", ".join(missing))
                return
            if df.empty or len(df) > 5000:
                st.error("File must contain between 1 and 5,000 booking rows.")
                return
            st.caption(f"Received {len(df):,} booking rows. No data is processed until you click Analyze.")
            st.dataframe(df.head(4), hide_index=True, use_container_width=True)
            if st.button("Analyze all bookings", type="primary", use_container_width=True):
                bdf = df[FEATURES].copy()
                invalid = []
                for i, row in bdf.iterrows():
                    if validate_booking(row.to_dict()):
                        invalid.append(i + 2)  # row in CSV incl header
                if invalid:
                    st.error(f"Invalid booking details at CSV line(s): {invalid[:25]}. Fix and retry. No partial predictions were generated.")
                    return
                for col in NUMERIC:
                    bdf[col] = pd.to_numeric(bdf[col], errors="raise")
                for col in CATEGORICAL:
                    bdf[col] = bdf[col].astype(str)
                prob = cancellation_probabilities(load_engine(), bdf)
                output = df.copy()
                output["cancellation_probability"] = np.round(prob, 4)
                output["risk_percent"] = np.round(prob * 100, 1)
                output["risk_level"] = [risk_bucket(p) for p in prob]
                output["predicted_outcome"] = np.where(prob >= .5, "Canceled", "Not Canceled")
                st.session_state["batch_output"] = output
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")
    if "batch_output" in st.session_state:
        result = st.session_state["batch_output"]
        section("Batch prediction report")
        a, b, c = st.columns(3)
        a.metric("Bookings evaluated", f"{len(result):,}")
        b.metric("Mean model risk score", f"{result['risk_percent'].mean():.1f}%")
        c.metric("High-risk bookings", f"{(result['risk_level']=='High').sum():,}")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("Export predictions as CSV", result.to_csv(index=False).encode("utf-8"),
                           "staywise_batch_predictions.csv", "text/csv", type="primary")
        st.caption("Batch predictions remain in current session only, not local assessment history.")


def model_report():
    heading("Model performance & transparency", "Measured metrics from genuine training, validation, and independent test data.",
            "MODEL LAB / VALIDATION & EVALUATION")
    rep = load_report()
    model = load_engine()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected algorithm", rep["best_model"])
    col2.metric("Training rows", f"{rep['train_records']:,}")
    col3.metric("Validation rows", f"{rep['validation_records']:,}")
    col4.metric("Untouched test rows", f"{rep['test_records']:,}")
    st.caption(f"Training completed {rep['trained_at_utc']} · Stratified 60/20/20 training/validation/test · baseline: Logistic Regression · selection: validation class-1 F1")
    section("Model comparison · validation set")
    metrics = pd.DataFrame([{"Algorithm": n,
        **{k: round(v * 100, 2) for k, v in score.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}
        for n, score in rep["model_comparison"].items()])
    labels = {"accuracy": "Accuracy (%)", "precision": "Precision (%)",
              "recall": "Recall (%)", "f1": "F1 (%)", "roc_auc": "ROC-AUC (%)"}
    st.dataframe(metrics.rename(columns=labels), hide_index=True, use_container_width=True)
    fig = go.Figure()
    for key, color, name in [("f1", TEAL, "F1 score"), ("recall", GOLD, "Recall")]:
        fig.add_trace(go.Bar(name=name, x=metrics["Algorithm"], y=metrics[key],
                             marker_color=color, hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    fig.update_layout(barmode="group", showlegend=True, legend_orientation="h",
                      legend=dict(y=1.13, x=0))
    st.plotly_chart(chart_layout(fig, height=295, yaxis=dict(ticksuffix="%", range=[0, 100])), use_container_width=True)
    st.caption("Validation results were used to choose the model. The selected model was then refitted on training + validation records before the final untouched test evaluation.")
    section("Final independent test · selected model")
    test = rep["final_test"]
    x1, x2, x3, x4, x5 = st.columns(5)
    for col, title, field in zip([x1, x2, x3, x4, x5],
                                  ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
                                  ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(title, f"{test[field]:.1%}")
    section("Independent test confusion matrix")
    cm = test["confusion_matrix"]
    left, right = st.columns([1.1, .9])
    with left:
        heat = go.Figure(go.Heatmap(z=cm, x=["Predicted honored", "Predicted canceled"],
                                    y=["Actually honored", "Actually canceled"],
                                    colorscale=[[0, "#EAF3F3"], [1, "#137D75"]],
                                    text=cm, texttemplate="%{text:,}", showscale=False))
        st.plotly_chart(chart_layout(heat, height=315), use_container_width=True)
    with right:
        info_card("CLASS LABEL", "1 = Canceled", "Positive class. Predicted cancellation risk uses the probability of label 1.")
        info_card("CLASS LABEL", "0 = Not canceled", "Negative class. The prediction is an estimate, not a confirmed outcome.")
        info_card("MODEL SELECTION", "Validation F1", "Balances precision and recall on the validation split; final test was untouched until selection.")
    section("Global feature signals")
    classifier = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
        label = "Random Forest impurity-based global feature importance"
    else:
        importance = np.abs(classifier.coef_[0])
        label = "Absolute Logistic Regression coefficients (standardized numeric inputs)"
    top = pd.DataFrame({"feature": names, "score": importance}).sort_values("score", ascending=False).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["score"], y=top["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False),
                           orientation="h", marker_color=TEAL))
    st.plotly_chart(chart_layout(fig, height=390), use_container_width=True)
    st.caption(label + ". This summary is global, not an explanation of any one prediction. Feature importance is not causation.")
    with st.expander("Final test classification report and leakage controls"):
        st.code(test["classification_report"])
        st.write("**Leakage controls:** `is_canceled`, `reservation_status`, `reservation_status_date`, `assigned_room_type`, and `booking_changes` do not appear as prediction inputs. Each preprocessing pipeline is fitted on its training split only.")
        st.write("**Limitations:** historical hotels in Portugal (2015–2017), no calibration study and no prospective deployment validation. Predictions should inform routine review, not automated penalties or guest denial.")
    st.download_button("Download actual evaluation results (JSON)",
                       json.dumps(rep, indent=2).encode("utf-8"),
                       file_name="staywise_model_evaluation.json", mime="application/json")


def about_guide():
    heading("About StayWise", "Methodology, setup, scope, and user guidance.", "HELP CENTER / PROJECT GUIDE")
    st.markdown("""### Project objective
Develop a web-based system that integrates a trained **machine learning classification model** into hotel booking cancellation assessment, with input validation, prediction outcomes, useful dashboards, and exportable results.

### How to demonstrate this project
1. Open **Overview** and describe historical dataset charts.
2. Open **New assessment**, choose a sample scenario, then click **Analyze cancellation risk**.
3. Discuss cancellation probability, outcome at 50% threshold, and the risk category.
4. Open **Scenario explorer** to compare two *model estimates* (not causal effects).
5. Show the assessment history and export capability.
6. Open **Batch analysis**, download the template, re-upload it, predict, and export results.
7. Open **Model report** and explain validation-based selection, independent test metrics and the confusion matrix.

### The dataset
This project uses the **Hotel Booking Demand** dataset, collected from a resort hotel and a city hotel in Portugal, covering 2015–2017. The original public source has 119,390 observations and 32 columns. The target is `is_canceled` (1 = canceled, 0 = not canceled). Some invalid records are excluded during cleaning; exact final modeling count is in the model report.

**Publication:** Nuno António, Ana de Almeida, and Luis Nunes (2019), *Hotel booking demand datasets*, Data in Brief, 22, 41–49. DOI: 10.1016/j.dib.2018.11.126.

**Public CSV:** [TidyTuesday / Hotel bookings](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11)

### Important boundaries
- This is a **decision-support prototype**, not a hotel booking or payment processor.
- A risk prediction is not proof that a customer will cancel.
- Historical training data do not represent current live bookings.
- Local assessment history contains only anonymous booking attributes, no guest names.
- Application usage requires attribution of any AI tools used during development, as requested by the course project guidelines.
""")
    section("Local history controls")
    st.caption("Assessment history is saved on this computer to data/analysis_history.sqlite3. Clearing cannot be undone.")
    confirm = st.checkbox("I understand that clearing removes all locally saved assessments")
    if st.button("Clear local history", disabled=not confirm):
        clear_history()
        st.success("Local history cleared.")
        st.rerun()


def main():
    if not MODEL_PATH.exists() or not REPORT_PATH.exists() or not DATA_PATH.exists():
        heading("Finish your first-time setup", "The project requires the real hotel dataset and a trained model.")
        st.warning("Double-click **START_STAYWISE.bat** (Windows) or run **python setup_project.py**, then **python -m streamlit run app.py**.")
        st.code("python -m pip install -r requirements.txt\npython setup_project.py\npython -m streamlit run app.py", language="bash")
        st.caption("First launch needs internet to install dependencies and download real historical data. No fake prediction model or fabricated metrics are supplied.")
        st.stop()
    try:
        load_engine()
        load_report()
    except Exception as e:
        st.error(f"Unable to load ML assets: {e}. Run `python setup_project.py --force` to rebuild.")
        st.stop()
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"
    page = sidebar()
    selected = page
    pages = {"Overview": overview, "New assessment": assessment,
             "Scenario explorer": scenario_explorer, "Batch analysis": batch_analysis,
             "Model report": model_report, "About & guide": about_guide}
    pages[selected]()
    st.markdown('''<div class="footer"><div>StayWise © Academic prototype · CpE Elective 1 · Applied ML & Software Development</div><div>Historical insights · Model outputs are estimates · No live hotel data</div></div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
