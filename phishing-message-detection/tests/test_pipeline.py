from pathlib import Path

import pandas as pd

from phishdetect.data import load_jsonl, stratified_split
from phishdetect.inference import predict_message
from phishdetect.modeling import build_pipeline
from phishdetect.preprocessing import nltk_preprocess

ROOT = Path(__file__).parents[1]


def test_preprocessing_retains_negation_and_normalizes_url():
    processed = nltk_preprocess("Do not verify at https://example.test now")
    assert "not" in processed.split()
    assert "urltoken" in processed.split()


def test_split_is_disjoint_and_stratified():
    frame = load_jsonl(ROOT / "data/raw/phishing_benign.jsonl")
    train, validation, test = stratified_split(frame)
    assert [len(train), len(validation), len(test)] == [120, 40, 40]
    id_sets = [set(x["id"]) for x in (train, validation, test)]
    assert not (id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2])
    assert all(part["target"].mean() == 0.5 for part in (train, validation, test))


def test_training_and_explanation_smoke():
    frame = pd.DataFrame({"text": ["team meeting tomorrow", "invoice paid", "verify password at http://fake.test", "urgent account locked click now"], "target": [0, 0, 1, 1]})
    model = build_pipeline(preprocessed=True).fit(frame["text"], frame["target"])
    result = predict_message(model, "urgent verify password now")
    assert result["label"] in {"benign", "phishing"}
    assert 0 <= result["phishing_probability"] <= 1
    assert result["influential_terms"]
