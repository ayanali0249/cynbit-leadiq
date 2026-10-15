"""Hold-out evaluation on SYNTHETIC data (pipeline check, not a performance claim)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import build_features
from .model import train_model
from .pipeline import score_clean
from .schema import validate_leads
from .synthetic import generate_leads

CAVEAT = ("Computed on synthetic leads whose labels come from an assumed "
          "process. It validates the pipeline, not real-world accuracy.")


def precision_at_top(scores: np.ndarray, labels: np.ndarray, fraction: float = 0.2) -> float:
    k = max(1, math.ceil(len(scores) * fraction))
    top = np.argsort(scores)[::-1][:k]
    return float(labels[top].mean())


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    base = float(labels.mean())
    p20 = precision_at_top(scores, labels)
    return {
        "roc_auc": round(float(roc_auc_score(labels, scores)), 3),
        "average_precision": round(float(average_precision_score(labels, scores)), 3),
        "precision_at_top_20pct": round(p20, 3),
        "lift_at_top_20pct": round(p20 / base, 2),
    }


def evaluate(n: int = 3000, seed: int = 42, test_size: float = 0.3) -> dict:
    raw = generate_leads(n, seed)
    labels = raw["converted"].to_numpy()
    clean, issues = validate_leads(raw.drop(columns="converted"))
    if issues or len(clean) != n:
        raise RuntimeError("synthetic data unexpectedly failed validation")
    feats = build_features(clean)

    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_size, stratify=labels, random_state=seed)
    model = train_model(feats.iloc[train_idx], labels[train_idx], seed)
    scored = score_clean(clean.iloc[test_idx].reset_index(drop=True),
                         feats.iloc[test_idx].reset_index(drop=True), model)
    y_test = labels[test_idx]

    tiers = {}
    for tier in ("Hot", "Warm", "Cold"):
        mask = (scored["tier"] == tier).to_numpy()
        tiers[tier] = {
            "leads": int(mask.sum()),
            "conversion_rate": round(float(y_test[mask].mean()), 3) if mask.any() else None,
        }
    return {
        "n_leads": n, "n_train": len(train_idx), "n_test": len(test_idx),
        "base_conversion_rate": round(float(y_test.mean()), 3),
        "rule_score": _metrics(scored["rule_score"].to_numpy(), y_test),
        "model_only": _metrics(scored["model_probability"].to_numpy(), y_test),
        "blended_final_score": _metrics(scored["final_score"].to_numpy(), y_test),
        "tiers_on_blended_score": tiers,
        "caveat": CAVEAT,
    }
