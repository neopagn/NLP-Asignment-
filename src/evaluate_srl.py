import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .srl import TransformerSRLModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST = PROJECT_ROOT / "data" / "srl_gold" / "test.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "legal_srl_legalbert"
DEFAULT_BASELINE = PROJECT_ROOT / "output" / "srl_rule_baseline_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "srl_evaluation.json"
EVAL_ROLES = {
    "Agent",
    "ContractParties",
    "DefinedTerm",
    "Definition",
    "Theme",
    "Recipient",
    "Time",
    "Condition",
    "LegalBasis",
    "Location",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _score(gold: set[tuple], predicted: set[tuple]) -> dict[str, float | int]:
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _gold_spans(rows: list[dict[str, Any]]) -> set[tuple]:
    spans = set()
    for row in rows:
        for role in row.get("roles", []):
            label = role.get("label")
            if label not in EVAL_ROLES:
                continue
            spans.add((row["id"], label, int(role["start"]), int(role["end"])))
    return spans


def _model_spans(rows: list[dict[str, Any]], model: TransformerSRLModel) -> set[tuple]:
    spans = set()
    for row in rows:
        predicate_text = str((row.get("predicate") or {}).get("text") or "")
        for item in model._predict_items(row["clause"], predicate_text):
            label = item.get("role")
            if label not in EVAL_ROLES:
                continue
            spans.add((row["id"], label, int(item["start"]), int(item["end"])))
    return spans


def _find_span(text: str, value: str) -> tuple[int, int] | None:
    value = value.strip()
    if not value:
        return None
    start = text.find(value)
    if start >= 0:
        return start, start + len(value)
    start = text.lower().find(value.lower())
    if start >= 0:
        return start, start + len(value)
    pattern = re.escape(value).replace(r"\ ", r"\s+")
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def _baseline_spans(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> set[tuple]:
    by_clause = {int(row["clause_id"]): row for row in baseline_rows}
    spans = set()
    for row in rows:
        baseline = by_clause.get(int(row["clause_id"]))
        if not baseline:
            continue
        clause = row["clause"]
        for label, raw_value in dict(baseline.get("roles") or {}).items():
            if label not in EVAL_ROLES or not raw_value:
                continue
            values = [part.strip() for part in str(raw_value).split(";") if part.strip()]
            for value in values:
                span = _find_span(clause, value)
                if span:
                    spans.add((row["id"], label, span[0], span[1]))
    return spans


def _per_role_scores(gold: set[tuple], predicted: set[tuple]) -> dict[str, dict[str, float | int]]:
    roles = sorted({item[1] for item in gold | predicted})
    return {
        role: _score({item for item in gold if item[1] == role}, {item for item in predicted if item[1] == role})
        for role in roles
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Legal-BERT SRL against gold span annotations.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.test)
    gold = _gold_spans(rows)

    model = TransformerSRLModel(args.model, min_score=args.min_score)
    model_predictions = _model_spans(rows, model)
    result = {
        "test_examples": len(rows),
        "gold_spans": len(gold),
        "model": str(args.model),
        "model_overall": _score(gold, model_predictions),
        "model_per_role": _per_role_scores(gold, model_predictions),
    }

    if args.baseline.exists():
        baseline_predictions = _baseline_spans(rows, _load_json(args.baseline))
        result["baseline"] = str(args.baseline)
        result["baseline_overall"] = _score(gold, baseline_predictions)
        result["baseline_per_role"] = _per_role_scores(gold, baseline_predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
