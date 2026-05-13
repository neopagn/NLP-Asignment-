import argparse
import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import INTENT_LABELS, MODEL_DIR, OUTPUT_DIR
from .intent import IntentClassifier, rule_based_intent_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "intent_silver"
DEFAULT_MODEL = MODEL_DIR / "legal_intent_legalbert"
DEFAULT_OUTPUT = OUTPUT_DIR / "intent_evaluation.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    report = classification_report(
        gold,
        predicted,
        labels=INTENT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=INTENT_LABELS, average="macro", zero_division=0)),
        "per_label": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in INTENT_LABELS
        },
    }


def _predict_transformer(rows: list[dict[str, Any]], model_path: Path, batch_size: int, max_length: int) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch_rows],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            for label_id in logits.argmax(dim=-1).tolist():
                predictions.append(id2label[int(label_id)])
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TF-IDF/LogReg and Legal-BERT intent classifiers.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = _load_jsonl(args.data_dir / "train.jsonl")
    test_rows = _load_jsonl(args.data_dir / "test.jsonl")
    gold = [row["label"] for row in test_rows]

    baseline = IntentClassifier.train((row["text"], row["label"]) for row in train_rows)
    tfidf_predictions = [baseline.predict_ml(row["text"]) for row in test_rows]
    hybrid_predictions = [rule_based_intent_label(row["text"]) or baseline.predict_ml(row["text"]) for row in test_rows]

    result = {
        "data_dir": str(args.data_dir),
        "test_examples": len(test_rows),
        "labels": INTENT_LABELS,
        "tfidf_logreg": _metrics(gold, tfidf_predictions),
        "legal_rules_plus_tfidf": _metrics(gold, hybrid_predictions),
    }

    if args.model.exists():
        transformer_predictions = _predict_transformer(test_rows, args.model, args.batch_size, args.max_length)
        result["transformer_model"] = str(args.model)
        result["legal_bert"] = _metrics(gold, transformer_predictions)
    else:
        result["transformer_model"] = str(args.model)
        result["legal_bert"] = "missing; run src.train_intent_transformer first"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
