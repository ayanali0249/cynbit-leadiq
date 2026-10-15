"""Input schema and validation for lead records."""
from __future__ import annotations

import re

import pandas as pd

REQUIRED_COLUMNS = [
    "lead_id", "name", "title", "company", "email", "company_size",
    "industry", "source", "budget_inr", "timeline", "message",
]
COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
TIMELINES = ["immediate", "1-3 months", "3-6 months", "exploring"]
SOURCES = ["referral", "website_form", "event", "linkedin",
           "google_ads", "meta_ads", "cold_outreach"]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TEXT_COLUMNS = ["lead_id", "name", "title", "company", "email",
                 "industry", "message"]


class LeadDataError(ValueError):
    """Raised when the data cannot be used at all (not just a few bad rows)."""


def _row_problems(row: pd.Series) -> list[str]:
    problems = []
    if not row["lead_id"]:
        problems.append("lead_id is empty")
    if not _EMAIL_RE.match(row["email"]):
        problems.append(f"invalid email '{row['email']}'")
    if row["company_size"] not in COMPANY_SIZES:
        problems.append(f"company_size must be one of {COMPANY_SIZES}")
    if row["source"] not in SOURCES:
        problems.append(f"source must be one of {SOURCES}")
    if row["timeline"] not in TIMELINES:
        problems.append(f"timeline must be one of {TIMELINES}")
    if pd.isna(row["budget_inr"]) or row["budget_inr"] < 0:
        problems.append("budget_inr must be a non-negative number")
    return problems


def validate_leads(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean_rows, issues).

    Bad rows are dropped and described in ``issues`` (row numbers match the
    CSV, where row 1 is the header). Raises LeadDataError if the file is empty,
    lacks required columns, or has no usable rows.
    """
    if df is None or df.empty:
        raise LeadDataError("The file contains no rows.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LeadDataError(f"Missing required columns: {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS]
    for col in _TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ("company_size", "source", "timeline"):
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
    df["budget_inr"] = pd.to_numeric(df["budget_inr"], errors="coerce")

    issues: list[str] = []
    keep = pd.Series(True, index=df.index)
    seen_ids: set[str] = set()
    for idx, row in df.iterrows():
        problems = _row_problems(row)
        if row["lead_id"] in seen_ids:
            problems.append(f"duplicate lead_id '{row['lead_id']}'")
        seen_ids.add(row["lead_id"])
        if problems:
            keep[idx] = False
            issues.append(f"Row {int(idx) + 2}: " + "; ".join(problems))

    clean = df[keep].reset_index(drop=True)
    if clean.empty:
        raise LeadDataError("No valid rows found. " + " | ".join(issues[:3]))
    return clean, issues
