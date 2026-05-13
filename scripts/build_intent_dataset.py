import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intent import TRAINING_EXAMPLES, IntentClassifier, rule_based_intent_label


DEFAULT_CLAUSES = PROJECT_ROOT / "output" / "clauses.txt"
DEFAULT_SEED = PROJECT_ROOT / "data" / "intent_seed.csv"
DEFAULT_CUAD_MASTER = PROJECT_ROOT / "data" / "external" / "CUAD_v1" / "master_clauses.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "intent_silver"

CUAD_CATEGORY_TO_INTENT = {
    "Expiration Date": "Termination Condition",
    "Renewal Term": "Termination Condition",
    "Notice Period To Terminate Renewal": "Termination Condition",
    "Termination For Convenience": "Termination Condition",
    "Change Of Control": "Termination Condition",
    "Non-Compete": "Prohibition",
    "Exclusivity": "Prohibition",
    "No-Solicit Of Customers": "Prohibition",
    "No-Solicit Of Employees": "Prohibition",
    "Non-Disparagement": "Prohibition",
    "Anti-Assignment": "Prohibition",
    "Price Restrictions": "Prohibition",
    "Non-Transferable License": "Prohibition",
    "Cap On Liability": "Prohibition",
    "Covenant Not To Sue": "Prohibition",
    "Most Favored Nation": "Right",
    "Rofr/Rofo/Rofn": "Right",
    "License Grant": "Right",
    "Affiliate License-Licensor": "Right",
    "Affiliate License-Licensee": "Right",
    "Unlimited/All-You-Can-Eat-License": "Right",
    "Irrevocable Or Perpetual License": "Right",
    "Audit Rights": "Right",
    "Third Party Beneficiary": "Right",
    "Governing Law": "Obligation",
    "Revenue/Profit Sharing": "Obligation",
    "Minimum Commitment": "Obligation",
    "Volume Restriction": "Obligation",
    "Ip Ownership Assignment": "Obligation",
    "Joint Ip Ownership": "Obligation",
    "Source Code Escrow": "Obligation",
    "Post-Termination Services": "Obligation",
    "Liquidated Damages": "Obligation",
    "Warranty Duration": "Obligation",
    "Insurance": "Obligation",
}

EXPLICIT_TERMINATION_RE = re.compile(r"\b(?:may|can|right\s+to|shall|will)\s+(?:immediately\s+)?terminate\b", re.I)


def _load_seed_examples(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            (row["text"].strip(), row["label"].strip())
            for row in reader
            if row.get("text", "").strip() and row.get("label", "").strip()
        ]


def _load_clauses(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _clean_clause_text(text: str) -> str:
    text = text.replace("<omitted>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'")


def _normalise_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_cuad_cell(value: str) -> list[str]:
    value = (value or "").strip()
    if not value or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value
    if isinstance(parsed, list):
        return [_clean_clause_text(str(item)) for item in parsed if _clean_clause_text(str(item))]
    if isinstance(parsed, str):
        cleaned = _clean_clause_text(parsed)
        return [cleaned] if cleaned else []
    return []


def _resolve_cuad_label(category: str, text: str) -> str:
    category_label = CUAD_CATEGORY_TO_INTENT[category]
    rule_label = rule_based_intent_label(text)
    if not rule_label:
        return category_label
    if category_label == "Obligation" and rule_label in {"Prohibition", "Right"}:
        return rule_label
    if category_label == "Right" and rule_label in {"Prohibition", "Termination Condition"}:
        return rule_label
    if category_label == "Prohibition" and rule_label == "Termination Condition" and EXPLICIT_TERMINATION_RE.search(text):
        return rule_label
    return category_label


def _build_cuad_rows(path: Path, max_per_label: int, seed: int) -> tuple[list[dict], dict]:
    if not path.exists():
        return [], {"path": str(path), "status": "missing"}

    by_text: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row.get("Filename", "")
            for category in CUAD_CATEGORY_TO_INTENT:
                for text in _parse_cuad_cell(row.get(category, "")):
                    if len(text.split()) < 5 or len(text) > 6000:
                        continue
                    key = _normalise_key(text)
                    entry = by_text.setdefault(
                        key,
                        {
                            "text": text,
                            "labels": set(),
                            "categories": set(),
                            "filenames": set(),
                        },
                    )
                    entry["labels"].add(_resolve_cuad_label(category, text))
                    entry["categories"].add(category)
                    if filename:
                        entry["filenames"].add(filename)

    candidates_by_label: dict[str, list[dict]] = defaultdict(list)
    conflicts = 0
    for entry in by_text.values():
        labels = set(entry["labels"])
        if len(labels) > 1:
            rule_label = rule_based_intent_label(entry["text"])
            if rule_label in labels:
                label = rule_label
            else:
                conflicts += 1
                continue
        else:
            label = next(iter(labels))
        candidates_by_label[label].append(entry)

    rng = random.Random(seed)
    rows: list[dict] = []
    selected_counts = {}
    candidate_counts = {label: len(items) for label, items in candidates_by_label.items()}
    for label in sorted(candidates_by_label):
        items = list(candidates_by_label[label])
        rng.shuffle(items)
        if max_per_label > 0:
            items = items[:max_per_label]
        selected_counts[label] = len(items)
        for index, item in enumerate(items, start=1):
            rows.append(
                {
                    "id": f"cuad-{label.lower().replace(' ', '-')}-{index:04d}",
                    "clause_id": None,
                    "text": item["text"],
                    "label": label,
                    "label_source": "cuad_mapped_category",
                    "cuad_categories": sorted(item["categories"]),
                    "source_file": sorted(item["filenames"])[0] if item["filenames"] else None,
                }
            )

    metadata = {
        "path": str(path),
        "status": "loaded",
        "mapped_categories": len(CUAD_CATEGORY_TO_INTENT),
        "unique_candidate_clauses": len(by_text),
        "conflicting_multilabel_clauses_skipped": conflicts,
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "max_per_label": max_per_label,
    }
    return rows, metadata


def _split_by_label(rows: list[dict], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for label_rows in grouped.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_count = max(1, round(total * train_ratio))
        dev_count = max(1, round(total * dev_ratio)) if total >= 3 else 0
        if train_count + dev_count >= total and total >= 3:
            train_count = max(1, total - 2)
            dev_count = 1
        elif train_count >= total:
            train_count = max(1, total - 1)

        splits["train"].extend(label_rows[:train_count])
        splits["dev"].extend(label_rows[train_count : train_count + dev_count])
        splits["test"].extend(label_rows[train_count + dev_count :])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: row["id"])
    return splits


def build_rows(
    clauses: list[str],
    seed_examples: list[tuple[str, str]],
    cuad_rows: list[dict] | None = None,
) -> list[dict]:
    fallback_examples = list(TRAINING_EXAMPLES) + list(seed_examples)
    seed_classifier = IntentClassifier.train(fallback_examples)
    rows: list[dict] = []
    for index, (text, label) in enumerate(seed_examples, start=1):
        rows.append(
            {
                "id": f"seed-{index:03d}",
                "clause_id": None,
                "text": text,
                "label": label,
                "label_source": "manual_seed",
            }
        )

    for clause_id, clause in enumerate(clauses, start=1):
        rule_label = rule_based_intent_label(clause)
        rows.append(
            {
                "id": f"clause-{clause_id:04d}",
                "clause_id": clause_id,
                "text": clause,
                "label": rule_label or seed_classifier.predict_ml(clause),
                "label_source": "legal_rule" if rule_label else "tfidf_seed_fallback",
            }
        )
    rows.extend(cuad_rows or [])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an intent classification dataset from project clauses and CUAD.")
    parser.add_argument("--clauses", type=Path, default=DEFAULT_CLAUSES)
    parser.add_argument("--seed-csv", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--cuad-master", type=Path, default=DEFAULT_CUAD_MASTER)
    parser.add_argument("--max-cuad-per-label", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_examples = _load_seed_examples(args.seed_csv)
    cuad_rows, cuad_metadata = _build_cuad_rows(args.cuad_master, args.max_cuad_per_label, args.seed)
    rows = build_rows(_load_clauses(args.clauses), seed_examples, cuad_rows)
    splits = _split_by_label(rows, args.seed, args.train_ratio, args.dev_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(rows, args.output_dir / "all.jsonl")
    for split, split_rows in splits.items():
        _write_jsonl(split_rows, args.output_dir / f"{split}.jsonl")

    metadata = {
        "source_clauses": str(args.clauses),
        "seed_csv": str(args.seed_csv),
        "seed": args.seed,
        "total_examples": len(rows),
        "splits": {split: len(split_rows) for split, split_rows in splits.items()},
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "label_sources": dict(Counter(row["label_source"] for row in rows)),
        "cuad": cuad_metadata,
        "note": (
            "This is a silver intent dataset for baseline-vs-transformer comparison. Rows from intent_seed.csv "
            "are manual seed labels; local contract clauses are labeled by legal rules with TF-IDF seed fallback; "
            "CUAD rows are mapped from expert clause categories into the four assignment intent labels."
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
