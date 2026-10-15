"""Synthetic lead generator.

EVERYTHING produced here is fake: names, companies, messages and, importantly,
the ``converted`` label. Labels come from an assumed logistic process over lead
attributes plus noise, so any metric measured on this data validates the
pipeline mechanics only. It says nothing about real Cynbit performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SOURCE_QUALITY, build_features
from .schema import COMPANY_SIZES, SOURCES, TIMELINES
from .text_signals import SERVICE_KEYWORDS

_FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vikram", "Anaya",
          "Arjun", "Sana", "Nikhil", "Tara", "Dev", "Riya", "Karan", "Neha"]
_LAST = ["Sharma", "Verma", "Iyer", "Khan", "Mehta", "Nair", "Singh", "Gupta",
         "Reddy", "Joshi", "Das", "Kapoor"]
_COMPANIES = ["Northwind Retail", "Bluepeak Logistics", "Lumen Health", "Orbit Foods",
              "Cedar Finance", "Pixel Forge", "Greenline Energy", "Atlas Learning",
              "Harbor Travel", "Sunrise Textiles", "Vertex Realty", "Kite Media"]
_INDUSTRIES = ["Retail", "Logistics", "Healthcare", "Food & Beverage", "Finance",
               "Media", "Energy", "Education", "Travel", "Manufacturing", "Real estate"]
_TITLES = {
    2: ["Founder", "CEO", "CTO", "Director of Operations", "VP Marketing", "Head of Digital"],
    1: ["Marketing Manager", "Product Manager", "Engineering Lead", "IT Manager"],
    0: ["Analyst", "Executive Assistant", "Associate", "Student", "Intern"],
}
_OPENERS = {
    "AI automation": ["We want to automate invoice data entry with an LLM.",
                      "Looking for an AI agent to qualify inbound enquiries.",
                      "Interested in a chatbot for customer support."],
    "Software development": ["We need a web app for managing dealers.",
                             "Planning a mobile app for our customers.",
                             "Our portal needs a new backend and API."],
    "Cloud & migration": ["We want to move our legacy system to AWS.",
                          "Looking for help with Azure and devops.",
                          "Our cloud costs keep rising and we need an audit."],
    "Marketing & SEO": ["We need SEO and Google ads for a product launch.",
                        "Looking for a funnel and CRM setup for lead generation.",
                        "Want a campaign across Meta and search."],
    "BitElevate training": ["Interested in your training and certification course.",
                            "Do you run a bootcamp for engineering students?"],
}
_INTENT = ["Please share a quote and pricing.", "Budget approved, we want to go live next month.",
           "We have an RFP and the deadline is close.", "This is urgent, we need a pilot asap.",
           "Can you send a proposal with scope?"]
_LOW = ["Just curious what you do.", "I am a student working on a college project.",
        "Is there anything for free?", "Looking for an internship or a job."]


def generate_leads(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Return n fake leads plus a synthetic ``converted`` label (0/1)."""
    rng = np.random.default_rng(seed)
    sizes = rng.choice(COMPANY_SIZES, n, p=[0.30, 0.30, 0.20, 0.12, 0.08])
    size_idx = np.array([COMPANY_SIZES.index(s) for s in sizes])
    timelines = rng.choice(TIMELINES, n, p=[0.15, 0.30, 0.30, 0.25])
    timeline_rank = np.array([len(TIMELINES) - 1 - TIMELINES.index(t) for t in timelines])
    sources = rng.choice(SOURCES, n, p=[0.12, 0.25, 0.08, 0.15, 0.15, 0.12, 0.13])
    seniority = rng.choice([0, 1, 2], n, p=[0.30, 0.35, 0.35])
    budget = np.exp(rng.normal(10.6 + 0.6 * size_idx, 0.8))
    budget = np.round(np.clip(budget, 10_000, 50_000_000) / 5000) * 5000
    services = list(_OPENERS)

    rows = []
    for i in range(n):
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        company = str(rng.choice(_COMPANIES))
        free_mail = rng.random() < (0.5 if seniority[i] == 0 else 0.2)
        domain = "gmail.com" if free_mail else company.lower().replace(" ", "") + ".example"
        message = str(rng.choice(_OPENERS[str(rng.choice(services, p=[0.3, 0.25, 0.15, 0.2, 0.1]))]))
        if rng.random() < 0.25 + 0.15 * timeline_rank[i]:
            message += " " + str(rng.choice(_INTENT))
        if rng.random() < 0.10 + (0.30 if seniority[i] == 0 else 0.0):
            message += " " + str(rng.choice(_LOW))
        rows.append({
            "lead_id": f"L{i + 1:04d}",
            "name": f"{first} {last}",
            "title": str(rng.choice(_TITLES[int(seniority[i])])),
            "company": company,
            "email": f"{first}.{last}@{domain}".lower(),
            "company_size": sizes[i],
            "industry": str(rng.choice(_INDUSTRIES)),
            "source": sources[i],
            "budget_inr": float(budget[i]),
            "timeline": timelines[i],
            "message": message,
        })
    df = pd.DataFrame(rows)

    # Assumed ground truth: weights differ from the hand-written rule scorer
    # so the comparison between the two is not circular.
    feats = build_features(df)
    z = (-3.0
         + 0.60 * (feats["log_budget"] - 11.5)
         + 0.40 * feats["timeline_rank"]
         + 0.45 * feats["seniority"]
         + 0.20 * feats["size_rank"]
         + 1.00 * (feats["source_quality"] - 0.5)
         + 0.45 * feats["intent_hits"]
         - 0.90 * feats["low_intent_hits"]
         - 0.50 * (1 - feats["business_email"])
         + rng.normal(0, 0.7, n))
    prob = 1 / (1 + np.exp(-z))
    df["converted"] = (rng.random(n) < prob).astype(int)
    return df
