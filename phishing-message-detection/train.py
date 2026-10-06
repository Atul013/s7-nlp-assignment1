"""Train, compare, evaluate, and persist the selected phishing detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from phishdetect.data import load_jsonl, stratified_split
from phishdetect.modeling import build_pipeline, error_rows, metrics, save_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/phishing_benign.jsonl")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_jsonl(args.data)
    duplicates_removed = frame.attrs.get("duplicates_removed", 0)
    train, validation, test = stratified_split(frame, args.seed)
    split_ids = {name: sorted(part["id"].tolist()) for name, part in [("train", train), ("validation", validation), ("test", test)]}
    (output / "split_ids.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")

    models = {"raw_text": build_pipeline(False, args.seed), "nltk_preprocessed": build_pipeline(True, args.seed)}
    results = {}
    for name, model in models.items():
        model.fit(train["text"], train["target"])
        results[name] = {"validation": metrics(model, validation), "test": metrics(model, test)}

    selected_name = max(models, key=lambda name: (results[name]["validation"]["f1"], results[name]["validation"]["recall"]))
    selected = models[selected_name]
    save_model(selected, output / "phishing_detector.joblib")
    errors = error_rows(selected, test)
    errors.to_csv(output / "test_errors.csv", index=False)

    evidence = {
        "seed": args.seed,
        "rows_after_deduplication": len(frame),
        "duplicates_removed": duplicates_removed,
        "class_counts": {k: int(v) for k, v in frame["label"].value_counts().to_dict().items()},
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected_pipeline": selected_name,
        "selection_rule": "Highest validation F1; validation recall breaks ties.",
        "results": results,
        "test_error_count": len(errors),
    }
    (output / "metrics.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    rows = []
    for model_name, splits in results.items():
        for split_name, values in splits.items():
            rows.append({"pipeline": model_name, "split": split_name, **{k: v for k, v in values.items() if k != "confusion_matrix"}})
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False)

    matrix = results[selected_name]["test"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Reds")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"], xlabel="Predicted", ylabel="Actual", title=f"Test confusion matrix — {selected_name}")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
