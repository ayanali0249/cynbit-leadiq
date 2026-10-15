"""Keyword signals extracted from lead messages and job titles.

These are plain keyword rules, not a language model. Short terms (<= 3 chars,
such as "ai" or "seo") must match as whole words so "ai" does not fire inside
"maintain"; longer terms match as word prefixes so "automat" covers
"automate", "automation" and "automating".
"""
from __future__ import annotations

import re
from functools import lru_cache

# Maps Cynbit's publicly listed service lines to trigger terms.
SERVICE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI automation": ("ai", "agent", "chatbot", "llm", "automat", "voice bot",
                      "document processing", "data entry"),
    "Software development": ("web app", "mobile app", "website", "saas",
                             "platform", "portal", "backend", "api"),
    "Cloud & migration": ("aws", "azure", "cloud", "migrat", "kubernetes",
                          "devops", "legacy"),
    "Marketing & SEO": ("seo", "ads", "campaign", "marketing",
                        "lead generation", "funnel", "crm"),
    "BitElevate training": ("training", "internship", "course",
                            "certification", "bootcamp", "students"),
}

# Phrases suggesting a buyer who is close to a decision.
INTENT_TERMS: tuple[str, ...] = (
    "asap", "urgent", "deadline", "go live", "budget approved", "rfp",
    "proposal", "quote", "pricing", "pilot", "scope", "replace our",
    "this quarter", "next month",
)

# Phrases suggesting browsing, students or job seekers rather than buyers.
LOW_INTENT_TERMS: tuple[str, ...] = (
    "just curious", "just exploring", "student", "college project",
    "for free", "internship", "job", "no budget", "assignment",
)

SENIOR_TITLE_TERMS: tuple[str, ...] = (
    "founder", "owner", "ceo", "cto", "coo", "cio", "chief", "director",
    "vp", "vice president", "head", "president",
)
MID_TITLE_TERMS: tuple[str, ...] = ("manager", "lead", "architect", "principal")


@lru_cache(maxsize=None)
def _pattern(term: str) -> re.Pattern[str]:
    suffix = r"\b" if len(term) <= 3 else ""
    return re.compile(rf"\b{re.escape(term)}{suffix}")


def has_term(text: str, term: str) -> bool:
    return _pattern(term).search(text.lower()) is not None


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    """Number of distinct terms present (not total occurrences)."""
    return sum(has_term(text, term) for term in terms)


def matched_services(message: str) -> list[str]:
    return [name for name, terms in SERVICE_KEYWORDS.items()
            if count_terms(message, terms) > 0]


def seniority_level(title: str) -> int:
    """2 = senior decision maker, 1 = mid-level, 0 = other or unknown."""
    if count_terms(title, SENIOR_TITLE_TERMS):
        return 2
    if count_terms(title, MID_TITLE_TERMS):
        return 1
    return 0
