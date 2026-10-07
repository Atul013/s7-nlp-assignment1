"""Dataset loading, validation, deduplication, and stratified splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"benign": 0, "phishing": 1}


def load_jsonl(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    required = {"subject", "body", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    frame = frame.assign(
        text=(frame["subject"].fillna("").str.strip() + "\n" + frame["body"].fillna("").str.strip()).str.strip(),
        target=frame["label"].map(LABEL_MAP),
    )
    if frame["target"].isna().any():
        unknown = sorted(frame.loc[frame["target"].isna(), "label"].astype(str).unique())
        raise ValueError(f"Unknown labels: {unknown}; expected benign/phishing")
    frame = frame.loc[frame["text"].str.len() > 0].copy()
    before = len(frame)
    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    frame["target"] = frame["target"].astype(int)
    frame.attrs["duplicates_removed"] = before - len(frame)
    return frame


def stratified_split(frame: pd.DataFrame, seed: int = 42):
    """Return 60/20/20 train/validation/test frames."""
    train, remainder = train_test_split(
        frame, test_size=0.40, stratify=frame["target"], random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=remainder["target"], random_state=seed
    )
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))
