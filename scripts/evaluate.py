"""Run the synthetic hold-out evaluation and save results to docs/."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadiq.evaluation import evaluate  # noqa: E402

if __name__ == "__main__":
    results = evaluate()
    out = ROOT / "docs" / "evaluation_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
