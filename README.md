# LeadIQ — explainable inbound-lead prioritisation

Prototype for the **Cynbit Technologies AI/ML Engineer Intern assessment**. It ranks inbound leads, explains every score, suggests a next action, and flags leads where the rules and the model disagree.

> **All data is synthetic** (fake names, companies, labels). Nothing here reflects Cynbit's real leads or performance, and I have no access to Cynbit's internal data.

## Problem and relevance

Sales teams lose time on low-intent leads while good ones wait. Cynbit's public services page lists lead generation, CRM automation and "autonomous agents that manage leads", so lead triage is a plausible fit — but whether it is a real pain point at Cynbit is an **assumption to validate**. See `docs/AI_OPPORTUNITY_REPORT.md`.

## Features

- CSV upload, bundled sample, or single-lead form
- Validation with row-level error messages
- Rule score (0–100) with a reason per point, plus a logistic-regression model probability
- Hot / Warm / Cold tiers, suggested action, likely Cynbit service line
- Review flag when rule score and model disagree by ≥30 points
- Students/job seekers routed toward BitElevate
- Scored-lead CSV export; in-app evaluation tab

## Architecture

Hybrid: transparent rules + a small scikit-learn model. **No LLM, no external API, no credentials.** Diagram and details: `docs/TECHNICAL_APPROACH.md`.

## Stack

Python 3.13 · pandas · scikit-learn · Streamlit · pytest

## Structure

```
app.py                 Streamlit UI
src/leadiq/            schema, text_signals, features, scoring, model, synthetic, pipeline, evaluation
data/sample_leads.csv  60 synthetic leads
scripts/               make_sample_data.py, evaluate.py
tests/                 25 pytest tests
docs/                  Opportunity report, technical approach, interview briefing, checklist
```

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests: `python -m pytest`
Re-run evaluation: `python scripts/evaluate.py`

No environment variables are needed, so there is no `.env.example`.

## Evaluation (synthetic hold-out, 900 test leads, base rate 25%)

| Score | ROC-AUC | Precision @ top 20% |
|---|---|---|
| Rule score | 0.772 | 0.539 |
| Model only | 0.780 | 0.556 |
| Blended (used) | 0.780 | 0.544 |

Tier conversion on the synthetic test set: Hot 74.4%, Warm 35.9%, Cold 10.4%.

**Read this carefully:** the synthetic labels come from an assumed process I wrote, so these numbers show the pipeline works mechanically, not that it will predict real Cynbit conversions. Test results: 25 passed (15 Oct 2026). The UI was smoke-tested headlessly, not in a browser.

## Limitations

Synthetic data only; keyword matching rather than language understanding; English vocabulary; hand-set weights; uncalibrated probabilities; no persistence or auth. Future work: retrain on real data, CRM integration, optional LLM intent extraction if it beats keywords on real held-out data.

## AI assistance disclosure

This project was built with the help of Claude (Anthropic) via Claude Code.
