"""Logistic-regression conversion model.

Trained on synthetic data in this prototype (see synthetic.py). It exists to
show the training/evaluation pipeline; its probabilities are NOT estimates of
real Cynbit conversion until retrained on Cynbit's own labelled leads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, FEATURE_LABELS


@dataclass
class LeadModel:
    pipeline: Pipeline

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    def top_drivers(self, features: pd.DataFrame, k: int = 3) -> list[str]:
        """Per lead, the k features pushing the probability up the most."""
        scaler = self.pipeline.named_steps["scale"]
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        contrib = scaler.transform(features[FEATURE_COLUMNS]) * coefs
        out = []
        for row in contrib:
            order = np.argsort(row)[::-1][:k]
            parts = [FEATURE_LABELS[FEATURE_COLUMNS[i]] for i in order if row[i] > 0]
            out.append(", ".join(parts) if parts else "no strong positive signal")
        return out


def train_model(features: pd.DataFrame, labels: np.ndarray, seed: int = 42) -> LeadModel:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if len(set(labels)) < 2:
        raise ValueError("labels must contain both converted and non-converted leads")
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    pipeline.fit(features[FEATURE_COLUMNS], labels)
    return LeadModel(pipeline)
