# Interview briefing (plain language)

**The problem.** A company gets many enquiries. Which should the sales team call first? LeadIQ scores each one and says why.

**Why Cynbit.** Their website lists lead generation, CRM automation and AI agents that manage leads. I *assume* triage matters to them; I haven't confirmed it.

**How it works.** (1) Check the data is clean. (2) Turn each lead into numbers: budget, timeline, seniority, source, email type, and keywords in the message. (3) Rule score: add points for good signs, subtract for bad ones, so every score has reasons. (4) A logistic regression model gives a probability. (5) Average the two, then Hot/Warm/Cold. If they disagree a lot, flag for a human.

**Why not an LLM?** Keywords and a small model are cheap, fast, explainable and testable. An LLM adds cost and unpredictability; I'd only add one if it beat keywords on real held-out data.

**Why logistic regression?** Small tabular data; interpretable; quick to train.

**Data.** All synthetic. I generated fake leads and fake "converted" labels from a formula I chose. So results prove the code works, not that it predicts real sales.

**Results.** AUC 0.77 (rules) vs 0.78 (model) on 900 synthetic test leads. Hot/Warm/Cold converted at 74/36/10% in the synthetic test set. The model is only slightly better than the rules, which is honest and unsurprising: the labels came from similar signals.

**Likely questions.**
- *What would you do with real data?* Get lead history with won/lost, retrain, calibrate, compare against current triage, run in shadow mode.
- *Biggest risk?* Bias against small buyers (company size, email domain) and trusting synthetic metrics. Mitigation: reasons shown, humans decide.
- *Weaknesses?* Keyword matching misses nuance and other languages; weights are guesses; no persistence.
- *Where's the rule scorer?* `scoring.py`. Pipeline: `pipeline.py`. Validation: `schema.py`.

**What I tested.** 25 pytest tests; headless app smoke test. Not tested in a real browser; not deployed.
