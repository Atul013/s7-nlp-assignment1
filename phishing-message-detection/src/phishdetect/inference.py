"""Safe inference and linear-model explanations."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def load_model(path: str | Path):
    return joblib.load(path)


def predict_message(model, text: str, top_k: int = 5) -> dict:
    if not str(text).strip():
        raise ValueError("Message cannot be empty")
    probability = float(model.predict_proba([text])[0, 1])
    label = "phishing" if probability >= 0.5 else "benign"
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    row = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    contributions = row.multiply(classifier.coef_[0]).toarray()[0]
    present = np.flatnonzero(row.toarray()[0])
    ordered = sorted(present, key=lambda i: abs(contributions[i]), reverse=True)[:top_k]
    influential = [
        {
            "term": str(feature_names[i]),
            "direction": "phishing" if contributions[i] > 0 else "benign",
            "contribution": float(contributions[i]),
        }
        for i in ordered
    ]
    return {
        "label": label,
        "phishing_probability": probability,
        "probability_note": "Uncalibrated model probability; not a security risk score.",
        "influential_terms": influential,
    }
