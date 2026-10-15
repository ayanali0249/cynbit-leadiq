import numpy as np
import pytest

from conftest import make_lead
from leadiq.evaluation import evaluate, precision_at_top
from leadiq.features import build_features
from leadiq.model import train_model
from leadiq.pipeline import score_leads
from leadiq.schema import LeadDataError, validate_leads
from leadiq.synthetic import generate_leads


def test_synthetic_generation_is_deterministic_and_valid():
    a, b = generate_leads(200, seed=1), generate_leads(200, seed=1)
    assert a.equals(b)
    clean, issues = validate_leads(a.drop(columns="converted"))
    assert issues == [] and len(clean) == 200
    assert 0.05 < a["converted"].mean() < 0.6


def test_train_model_rejects_bad_inputs():
    feats = build_features(validate_leads(generate_leads(50).drop(columns="converted"))[0])
    with pytest.raises(ValueError, match="both"):
        train_model(feats, np.zeros(len(feats)))
    with pytest.raises(ValueError, match="same length"):
        train_model(feats, np.array([0, 1]))


def test_precision_at_top():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    labels = np.array([1, 1, 0, 0, 0])
    assert precision_at_top(scores, labels, 0.4) == 1.0


def test_score_leads_output_sorted_with_expected_columns(make_df):
    df = make_df(
        make_lead(lead_id="HOT"),
        make_lead(lead_id="COLD", title="Student", email="s@gmail.com", budget_inr=1000,
                  company_size="1-10", source="cold_outreach", timeline="exploring",
                  message="Just curious about an internship"),
    )
    scored, issues = score_leads(df)
    assert issues == []
    assert list(scored["lead_id"]) == ["HOT", "COLD"]
    assert scored.loc[0, "tier"] == "Hot" and scored.loc[1, "tier"] == "Cold"
    assert scored.loc[1, "suggested_action"].startswith("Route to BitElevate")
    assert scored["final_score"].between(0, 100).all()
    for col in ("rule_reasons", "model_drivers", "service_interest", "needs_review"):
        assert col in scored.columns


def test_score_leads_reports_bad_rows_but_scores_the_rest(make_df):
    scored, issues = score_leads(make_df(make_lead(), make_lead(lead_id="BAD", email="nope")))
    assert len(scored) == 1 and len(issues) == 1


def test_score_leads_raises_on_unusable_file(make_df):
    with pytest.raises(LeadDataError):
        score_leads(make_df(make_lead()).drop(columns=["message"]))


def test_evaluation_shows_real_separation_between_tiers():
    results = evaluate(n=2000, seed=3)
    assert results["rule_score"]["roc_auc"] > 0.65
    assert results["model_only"]["roc_auc"] > 0.65
    tiers = results["tiers_on_blended_score"]
    assert tiers["Hot"]["conversion_rate"] > tiers["Warm"]["conversion_rate"] > tiers["Cold"]["conversion_rate"]
