"""Regenerate data/sample_leads.csv (60 synthetic leads, no labels)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadiq.synthetic import generate_leads  # noqa: E402

if __name__ == "__main__":
    df = generate_leads(60, seed=7).drop(columns="converted")
    out = ROOT / "data" / "sample_leads.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic leads to {out}")
