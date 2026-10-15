from conftest import make_lead
from leadiq.features import build_features
from leadiq.schema import validate_leads
from leadiq.scoring import compute_rule_score, fmt_inr, tier_for
from leadiq.text_signals import count_terms, has_term, matched_services, seniority_level


def score(make_df, **overrides):
    clean, _ = validate_leads(make_df(make_lead(**overrides)))
    feats = build_features(clean)
    return compute_rule_score(clean.iloc[0], feats.iloc[0])


def test_short_terms_need_whole_word_match():
    assert not has_term("We maintain the training sessions", "ai")
    assert has_term("We want AI help", "ai")


def test_prefix_terms_match_inflections():
    assert has_term("automating our invoices", "automat")


def test_services_detected_and_empty_message_is_unclear():
    assert matched_services("Move legacy systems to AWS") == ["Cloud & migration"]
    assert matched_services("") == []


def test_count_terms_counts_distinct_terms_only():
    assert count_terms("quote quote quote", ("quote", "pricing")) == 1


def test_seniority_levels():
    assert seniority_level("Co-Founder & CEO") == 2
    assert seniority_level("Marketing Manager") == 1
    assert seniority_level("Student") == 0
    assert seniority_level("") == 0


def test_strong_lead_scores_far_above_weak_lead(make_df):
    strong = score(make_df)
    weak = score(make_df, title="Student", email="x@gmail.com", company_size="1-10",
                 source="cold_outreach", budget_inr=5000, timeline="exploring",
                 message="Just curious, is there an internship?")
    assert strong.score >= 80
    assert weak.score <= 15
    assert tier_for(strong.score) == "Hot" and tier_for(weak.score) == "Cold"


def test_score_is_bounded_and_floored_at_zero(make_df):
    worst = score(make_df, title="", email="x@gmail.com", company_size="1-10",
                  source="cold_outreach", budget_inr=0, timeline="exploring",
                  message="student college project internship job for free")
    assert worst.score == 0.0


def test_every_score_has_reasons_with_signed_points(make_df):
    result = score(make_df)
    assert result.reasons and all("(" in r and ")" in r for r in result.reasons)


def test_low_intent_phrases_reduce_score(make_df):
    neutral = score(make_df, message="We need an AI agent.")
    low = score(make_df, message="We need an AI agent. I'm a student.")
    assert low.score < neutral.score


def test_fmt_inr():
    assert fmt_inr(2_500_000) == "₹25.0L"
    assert fmt_inr(25_000_000) == "₹2.5Cr"
    assert fmt_inr(45_000) == "₹45,000"
