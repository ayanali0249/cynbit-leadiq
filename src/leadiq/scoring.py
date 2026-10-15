"""Transparent rule-based lead score (0-100) with human-readable reasons.

Weights are product judgement, documented in docs/TECHNICAL_APPROACH.md.
They have not been tuned on real Cynbit conversion data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

HOT_THRESHOLD = 60.0
WARM_THRESHOLD = 35.0

# (minimum budget in INR, points), checked in order.
BUDGET_BANDS = ((1_000_000, 25), (200_000, 17), (50_000, 8), (0, 0))
TIMELINE_POINTS = {3: 20, 2: 14, 1: 7, 0: 2}
SENIORITY_POINTS = {2: 15, 1: 8, 0: 0}
SIZE_POINTS = (1, 4, 7, 9, 10)
SOURCE_MAX_POINTS = 10
INTENT_POINTS_EACH, INTENT_POINTS_CAP = 3, 15
LOW_INTENT_PENALTY_EACH, LOW_INTENT_PENALTY_CAP = 5, 15
BUSINESS_EMAIL_POINTS = 5


@dataclass
class RuleResult:
    score: float
    reasons: list[str]


def fmt_inr(value: float) -> str:
    if value >= 1e7:
        return f"₹{value / 1e7:.1f}Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.1f}L"
    return f"₹{value:,.0f}"


def tier_for(score: float) -> str:
    if score >= HOT_THRESHOLD:
        return "Hot"
    if score >= WARM_THRESHOLD:
        return "Warm"
    return "Cold"


def compute_rule_score(lead: pd.Series, feat: pd.Series) -> RuleResult:
    """Score one lead. ``lead`` is a validated row, ``feat`` its features."""
    reasons: list[str] = []
    total = 0

    def add(points: int, text: str) -> None:
        nonlocal total
        total += points
        reasons.append(f"{text} ({points:+d})")

    budget = float(lead["budget_inr"])
    budget_pts = next(pts for floor, pts in BUDGET_BANDS if budget >= floor)
    add(budget_pts, f"Budget {fmt_inr(budget)}")
    add(TIMELINE_POINTS[int(feat["timeline_rank"])], f"Timeline '{lead['timeline']}'")
    add(SENIORITY_POINTS[int(feat["seniority"])], f"Title '{lead['title'] or 'unknown'}'")
    add(SIZE_POINTS[int(feat["size_rank"])], f"Company size {lead['company_size']}")
    add(round(feat["source_quality"] * SOURCE_MAX_POINTS), f"Source '{lead['source']}'")
    if feat["business_email"]:
        add(BUSINESS_EMAIL_POINTS, "Business email domain")
    if feat["intent_hits"]:
        pts = min(int(feat["intent_hits"]) * INTENT_POINTS_EACH, INTENT_POINTS_CAP)
        add(pts, f"{int(feat['intent_hits'])} buying-intent phrase(s) in message")
    if feat["low_intent_hits"]:
        pts = min(int(feat["low_intent_hits"]) * LOW_INTENT_PENALTY_EACH, LOW_INTENT_PENALTY_CAP)
        add(-pts, f"{int(feat['low_intent_hits'])} low-intent phrase(s) in message")

    return RuleResult(score=float(min(max(total, 0), 100)), reasons=reasons)
