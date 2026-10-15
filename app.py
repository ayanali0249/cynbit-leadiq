"""Streamlit demo: explainable inbound-lead prioritisation."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from leadiq.evaluation import evaluate  # noqa: E402
from leadiq.pipeline import default_model, score_leads  # noqa: E402
from leadiq.schema import (COMPANY_SIZES, REQUIRED_COLUMNS, SOURCES,  # noqa: E402
                           TIMELINES, LeadDataError)

SAMPLE_PATH = ROOT / "data" / "sample_leads.csv"
TIER_ICON = {"Hot": "🔴", "Warm": "🟠", "Cold": "🔵"}

st.set_page_config(page_title="LeadIQ — lead prioritisation", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Training model on synthetic data…")
def load_model():
    return default_model()


@st.cache_data(show_spinner="Running hold-out evaluation…")
def run_evaluation():
    return evaluate()


def template_csv() -> bytes:
    example = pd.DataFrame([{
        "lead_id": "L0001", "name": "Example Person", "title": "Founder",
        "company": "Example Co", "email": "example@exampleco.example",
        "company_size": "11-50", "industry": "Retail", "source": "referral",
        "budget_inr": 500000, "timeline": "1-3 months",
        "message": "We want an AI agent to qualify enquiries. Please send a quote.",
    }], columns=REQUIRED_COLUMNS)
    return example.to_csv(index=False).encode("utf-8")


def show_lead_detail(row: pd.Series) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Final score", f"{row['final_score']:.1f}")
    c2.metric("Rule score", f"{row['rule_score']:.0f}")
    c3.metric("Model probability", f"{row['model_probability']:.0%}")
    st.markdown(f"**{TIER_ICON[row['tier']]} {row['tier']}** · {row['suggested_action']}")
    if row["needs_review"]:
        st.warning("Rule score and model disagree strongly — review this lead manually.")
    st.markdown(f"**Message:** {row['message']}")
    st.markdown("**Why (rule breakdown):**")
    for reason in row["rule_reasons"].split(" | "):
        st.markdown(f"- {reason}")
    st.markdown(f"**Model's strongest positive signals:** {row['model_drivers']}")


def lead_queue_tab(raw: pd.DataFrame | None) -> None:
    if raw is None:
        st.info("Upload a CSV in the sidebar, or switch to the bundled sample data.")
        return
    try:
        scored, issues = score_leads(raw, load_model())
    except LeadDataError as err:
        st.error(f"Could not score this file: {err}")
        return

    if issues:
        with st.expander(f"⚠️ {len(issues)} row(s) skipped due to validation issues"):
            for issue in issues:
                st.write(issue)

    counts = scored["tier"].value_counts()
    cols = st.columns(5)
    cols[0].metric("Leads scored", len(scored))
    for col, tier in zip(cols[1:4], ("Hot", "Warm", "Cold")):
        col.metric(f"{TIER_ICON[tier]} {tier}", int(counts.get(tier, 0)))
    cols[4].metric("Needs review", int(scored["needs_review"].sum()))

    tiers = st.multiselect("Tier", ["Hot", "Warm", "Cold"], default=["Hot", "Warm", "Cold"])
    view = scored[scored["tier"].isin(tiers)]
    st.dataframe(
        view[["lead_id", "name", "company", "final_score", "tier",
              "service_interest", "suggested_action", "needs_review"]],
        width="stretch", hide_index=True)
    st.download_button("Download scored leads (CSV)",
                       scored.to_csv(index=False).encode("utf-8"),
                       "scored_leads.csv", "text/csv")

    if view.empty:
        st.info("No leads match the selected tiers.")
        return
    st.subheader("Lead detail")
    chosen = st.selectbox("Select a lead", view["lead_id"])
    show_lead_detail(view[view["lead_id"] == chosen].iloc[0])


def single_lead_tab() -> None:
    st.write("Score one lead without a file.")
    with st.form("single_lead"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", "Priya Rao")
        title = c1.text_input("Title", "Head of Digital")
        company = c1.text_input("Company", "Sample Retail")
        email = c1.text_input("Email", "priya@sampleretail.example")
        size = c2.selectbox("Company size", COMPANY_SIZES, index=2)
        source = c2.selectbox("Source", SOURCES)
        timeline = c2.selectbox("Timeline", TIMELINES, index=1)
        budget = c2.number_input("Budget (INR)", min_value=0, value=600000, step=50000)
        message = st.text_area("Message", "We need an AI agent for lead qualification. "
                                          "Please share a quote and pricing.")
        submitted = st.form_submit_button("Score lead")
    if not submitted:
        return
    row = pd.DataFrame([{
        "lead_id": "MANUAL-1", "name": name, "title": title, "company": company,
        "email": email, "company_size": size, "industry": "n/a", "source": source,
        "budget_inr": budget, "timeline": timeline, "message": message}])
    try:
        scored, issues = score_leads(row, load_model())
    except LeadDataError as err:
        st.error(f"Please fix the input: {err}")
        return
    show_lead_detail(scored.iloc[0])


def evaluation_tab() -> None:
    results = run_evaluation()
    st.warning(results["caveat"])
    st.write(f"{results['n_train']:,} training / {results['n_test']:,} test synthetic leads · "
             f"base conversion rate {results['base_conversion_rate']:.1%}")
    table = pd.DataFrame({
        "Rule score": results["rule_score"],
        "Model only": results["model_only"],
        "Blended (used in app)": results["blended_final_score"],
    })
    st.dataframe(table)
    tier_df = pd.DataFrame(results["tiers_on_blended_score"]).T
    st.write("Conversion rate by tier (synthetic test set):")
    st.dataframe(tier_df)


def about_tab() -> None:
    st.markdown("""
**What this is.** A prototype that ranks inbound leads and explains each ranking, so a
sales or marketing team can follow up on the best leads first.

**How it works.** (1) Validate the CSV. (2) Extract features: budget, timeline, seniority,
source, email type, and keyword signals in the message. (3) A transparent rule score (0–100)
with a reason for every point. (4) A logistic-regression model trained on *synthetic* leads.
(5) Final score = 50% rule + 50% model, mapped to Hot / Warm / Cold. Leads where the two
disagree strongly are flagged for review.

**Honest limits.** All bundled data is fake. The model has never seen real Cynbit leads,
and keyword matching is not language understanding. Rule weights are judgement calls to be
validated with Cynbit's sales team. No LLM or external API is used.
""")


def main() -> None:
    st.title("🎯 LeadIQ — explainable lead prioritisation")
    st.caption("Assessment prototype for Cynbit · all data shown is synthetic, "
               "not Cynbit's real leads.")

    with st.sidebar:
        st.header("Data")
        source = st.radio("Lead data", ["Bundled sample (synthetic)", "Upload CSV"])
        st.download_button("Download CSV template", template_csv(), "leads_template.csv", "text/csv")
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))

    raw = None
    if source.startswith("Bundled"):
        raw = pd.read_csv(SAMPLE_PATH)
    else:
        upload = st.sidebar.file_uploader("CSV file", type="csv")
        if upload is not None:
            try:
                raw = pd.read_csv(upload)
            except Exception as err:  # unreadable or non-CSV content
                st.error(f"Could not read that file as CSV: {err}")

    tabs = st.tabs(["Lead queue", "Score one lead", "Evaluation", "About"])
    with tabs[0]:
        lead_queue_tab(raw)
    with tabs[1]:
        single_lead_tab()
    with tabs[2]:
        evaluation_tab()
    with tabs[3]:
        about_tab()


main()
