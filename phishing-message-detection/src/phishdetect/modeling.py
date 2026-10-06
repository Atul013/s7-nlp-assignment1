"""Training and evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from .preprocessing import nltk_preprocess


def build_pipeline(preprocessed: bool, seed: int = 42) -> Pipeline:
    vectorizer = TfidfVectorizer(
        preprocessor=nltk_preprocess if preprocessed else None,
        lowercase=not preprocessed,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def metrics(model: Pipeline, frame: pd.DataFrame) -> dict:
    predicted = model.predict(frame["text"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["target"], predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(frame["target"], predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(frame["target"], predicted, labels=[0, 1]).tolist(),
    }


def error_rows(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    predicted = model.predict(frame["text"])
    probabilities = model.predict_proba(frame["text"])[:, 1]
    result = frame[["id", "subject", "label"]].copy()
    result["predicted"] = ["phishing" if x else "benign" for x in predicted]
    result["phishing_probability"] = probabilities.round(4)
    return result.loc[result["label"] != result["predicted"]]


def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
