"""Turn validated lead rows into numeric features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COMPANY_SIZES, TIMELINES
from .text_signals import (INTENT_TERMS, LOW_INTENT_TERMS, count_terms,
                           matched_services, seniority_level)

FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                      "rediffmail.com", "proton.me", "icloud.com"}

# Higher = closer to a purchase decision. Ordered by the schema lists.
TIMELINE_RANK = {t: len(TIMELINES) - 1 - i for i, t in enumerate(TIMELINES)}
SIZE_RANK = {s: i for i, s in enumerate(COMPANY_SIZES)}

# Judgement-based prior on channel quality, not measured Cynbit data.
SOURCE_QUALITY = {"referral": 1.0, "website_form": 0.8, "event": 0.6,
                  "linkedin": 0.5, "google_ads": 0.5, "meta_ads": 0.4,
                  "cold_outreach": 0.2}

FEATURE_COLUMNS = [
    "log_budget", "size_rank", "timeline_rank", "seniority", "source_quality",
    "business_email", "intent_hits", "low_intent_hits", "service_count",
    "message_words",
]

FEATURE_LABELS = {
    "log_budget": "budget", "size_rank": "company size",
    "timeline_rank": "timeline urgency", "seniority": "seniority",
    "source_quality": "lead source", "business_email": "business email",
    "intent_hits": "buying-intent wording", "low_intent_hits": "low-intent wording",
    "service_count": "services mentioned", "message_words": "message length",
}


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    domains = clean["email"].str.split("@").str[-1].str.lower()
    messages = clean["message"]
    return pd.DataFrame({
        "log_budget": np.log1p(clean["budget_inr"].astype(float)),
        "size_rank": clean["company_size"].map(SIZE_RANK),
        "timeline_rank": clean["timeline"].map(TIMELINE_RANK),
        "seniority": clean["title"].map(seniority_level),
        "source_quality": clean["source"].map(SOURCE_QUALITY),
        "business_email": (~domains.isin(FREE_EMAIL_DOMAINS)).astype(int),
        "intent_hits": messages.map(lambda m: min(count_terms(m, INTENT_TERMS), 5)),
        "low_intent_hits": messages.map(lambda m: min(count_terms(m, LOW_INTENT_TERMS), 3)),
        "service_count": messages.map(lambda m: len(matched_services(m))),
        "message_words": messages.str.split().str.len().clip(upper=80),
    })[FEATURE_COLUMNS].astype(float)
