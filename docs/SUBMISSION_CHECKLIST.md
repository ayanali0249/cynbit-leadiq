# Submission checklist

| Requirement | Status |
|---|---|
| AI Opportunity Report | Done — `docs/AI_OPPORTUNITY_REPORT.md` |
| AI/ML Proof of Concept | Done — Streamlit app + `src/`; 25 tests pass |
| Technical Approach Report | Done — `docs/TECHNICAL_APPROACH.md` |
| Research of Cynbit public pages | Done (4 pages, read via a summarising fetch tool) |
| Clean code, meaningful Git history | 3+ local commits |
| Public GitHub repository | **Not done — needs your authorization and credentials** |
| Live demo | Not deployed |
| Submit URL + setup notes in portal | **You** (deadline Sat 17 Oct 2026, 07:39 AM IST) |

## Publish (run yourself)

```bash
cd cynbit-leadiq
# create an empty PUBLIC repo on GitHub first, then:
git remote add origin https://github.com/<your-username>/cynbit-leadiq.git
git branch -M main
git push -u origin main
```

## Portal notes to paste

Repo URL: *(your URL after pushing)*. Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`; tests: `python -m pytest`. All data synthetic; no API keys needed. Built with Claude Code assistance.
