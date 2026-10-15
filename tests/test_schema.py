import pandas as pd
import pytest

from conftest import make_lead
from leadiq.schema import LeadDataError, validate_leads


def test_valid_rows_pass_without_issues(make_df):
    clean, issues = validate_leads(make_df(make_lead(), make_lead(lead_id="T2")))
    assert len(clean) == 2 and issues == []


def test_normalises_case_and_whitespace(make_df):
    clean, _ = validate_leads(make_df(make_lead(source=" Referral ", timeline="IMMEDIATE")))
    assert clean.loc[0, "source"] == "referral"
    assert clean.loc[0, "timeline"] == "immediate"


def test_empty_dataframe_raises():
    with pytest.raises(LeadDataError, match="no rows"):
        validate_leads(pd.DataFrame())


def test_missing_columns_named_in_error(make_df):
    df = make_df(make_lead()).drop(columns=["budget_inr", "email"])
    with pytest.raises(LeadDataError, match="budget_inr.*email|email.*budget_inr"):
        validate_leads(df)


def test_bad_rows_dropped_with_row_numbers(make_df):
    df = make_df(make_lead(), make_lead(lead_id="T2", email="not-an-email"),
                 make_lead(lead_id="T3", budget_inr=-5),
                 make_lead(lead_id="T4", source="carrier pigeon"))
    clean, issues = validate_leads(df)
    assert list(clean["lead_id"]) == ["T1"]
    assert len(issues) == 3
    assert issues[0].startswith("Row 3:")


def test_non_numeric_budget_rejected(make_df):
    clean, issues = validate_leads(make_df(make_lead(), make_lead(lead_id="T2", budget_inr="lots")))
    assert len(clean) == 1 and "budget_inr" in issues[0]


def test_duplicate_ids_flagged(make_df):
    clean, issues = validate_leads(make_df(make_lead(), make_lead()))
    assert len(clean) == 1 and "duplicate" in issues[0]


def test_all_invalid_rows_raise(make_df):
    with pytest.raises(LeadDataError, match="No valid rows"):
        validate_leads(make_df(make_lead(email="bad")))
