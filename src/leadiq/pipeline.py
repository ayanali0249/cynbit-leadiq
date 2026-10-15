"""End-to-end scoring: validate -> features -> rule score + model -> tier."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .features import build_features
from .model import LeadModel, train_model
from .schema import validate_leads
from .scoring import compute_rule_score, tier_for
from .synthetic import generate_leads
from .text_signals import count_terms, matched_services, LOW_INTENT_TERMS

RULE_WEIGHT = 0.5  # final = 50% rule score + 50% model probability
REVIEW_GAP = 30.0  # rule and model disagree by this many points -> human review

ACTIONS = {
    "Hot": "Call within 24 hours",
    "Warm": "Send tailored follow-up this week",
    "Cold": "Add to nurture list",
}


@lru_cache(maxsize=1)
def default_model() -> LeadModel:
    """Model trained on 3,000 synthetic leads (deterministic)."""
    raw = generate_leads(3000, seed=42)
    clean, _ = validate_leads(raw.drop(columns="converted"))
    return train_model(build_features(clean), raw["converted"].to_numpy())


def _suggested_action(tier: str, message: str, services: list[str]) -> str:
    only_training = services == ["BitElevate training"]
    if only_training or count_terms(message, LOW_INTENT_TERMS) >= 2:
        return "Route to BitElevate / education team"
    return ACTIONS[tier]


def score_clean(clean: pd.DataFrame, feats: pd.DataFrame, model: LeadModel) -> pd.DataFrame:
    """Score already-validated rows. Keeps the input row order."""
    results = [compute_rule_score(lead, feat)
               for (_, lead), (_, feat) in zip(clean.iterrows(), feats.iterrows())]
    rule = np.array([r.score for r in results])
    ml_prob = model.predict_proba(feats)
    final = np.round(RULE_WEIGHT * rule + (1 - RULE_WEIGHT) * 100 * ml_prob, 1)
    services = [matched_services(m) for m in clean["message"]]
    tiers = [tier_for(s) for s in final]

    out = clean[["lead_id", "name", "title", "company", "message"]].copy()
    out["final_score"] = final
    out["tier"] = tiers
    out["rule_score"] = rule
    out["model_probability"] = np.round(ml_prob, 3)
    out["service_interest"] = [", ".join(s) if s else "Unclear" for s in services]
    out["suggested_action"] = [_suggested_action(t, m, s)
                               for t, m, s in zip(tiers, clean["message"], services)]
    out["needs_review"] = np.abs(rule - 100 * ml_prob) >= REVIEW_GAP
    out["rule_reasons"] = [" | ".join(r.reasons) for r in results]
    out["model_drivers"] = model.top_drivers(feats)
    return out


def score_leads(raw: pd.DataFrame, model: LeadModel | None = None
                ) -> tuple[pd.DataFrame, list[str]]:
    """Validate and score raw lead data. Returns (scored sorted by score, issues)."""
    clean, issues = validate_leads(raw)
    scored = score_clean(clean, build_features(clean), model or default_model())
    return scored.sort_values("final_score", ascending=False).reset_index(drop=True), issues
