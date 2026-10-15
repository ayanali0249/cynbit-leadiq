import pandas as pd
import pytest


def make_lead(**overrides) -> dict:
    base = {
        "lead_id": "T1", "name": "Test Person", "title": "Founder",
        "company": "Test Co", "email": "test@testco.example",
        "company_size": "51-200", "industry": "Retail", "source": "referral",
        "budget_inr": 1_500_000, "timeline": "immediate",
        "message": "We need an AI agent. Please send a quote, budget approved.",
    }
    base.update(overrides)
    return base


@pytest.fixture
def make_df():
    def _make(*rows: dict) -> pd.DataFrame:
        return pd.DataFrame(list(rows))
    return _make
