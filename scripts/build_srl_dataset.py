import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "srl_annotation_candidates.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "srl_gold"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _gold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = []
    for row in rows:
        if row.get("review_status") != "gold":
            continue
        if not row.get("roles"):
            continue
        clean_rows.append(
            {
                "id": row["id"],
                "clause_id": int(row["clause_id"]),
                "section": row.get("section"),
                "clause": row["clause"],
                "predicate": row["predicate"],
                "roles": row["roles"],
                "annotator_notes": row.get("annotator_notes", ""),
            }
        )
    return clean_rows


def _split_by_clause(rows: list[dict[str, Any]], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["clause_id"])].append(row)

    clause_ids = list(grouped)
    random.Random(seed).shuffle(clause_ids)
    train_count = max(1, round(len(clause_ids) * train_ratio))
    dev_count = max(1, round(len(clause_ids) * dev_ratio))
    if train_count + dev_count >= len(clause_ids):
        dev_count = max(1, len(clause_ids) - train_count - 1)

    split_clause_ids = {
        "train": set(clause_ids[:train_count]),
        "dev": set(clause_ids[train_count : train_count + dev_count]),
        "test": set(clause_ids[train_count + dev_count :]),
    }

    splits: dict[str, list[dict[str, Any]]] = {}
    for split, ids in split_clause_ids.items():
        split_rows = [row for clause_id in ids for row in grouped[clause_id]]
        splits[split] = sorted(split_rows, key=lambda row: (int(row["clause_id"]), row["id"]))
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/dev/test SRL gold datasets from reviewed annotations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _gold_rows(_load_jsonl(args.input))
    splits = _split_by_clause(rows, args.seed, args.train_ratio, args.dev_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(rows, args.output_dir / "all.jsonl")
    for split, split_rows in splits.items():
        _write_jsonl(split_rows, args.output_dir / f"{split}.jsonl")

    metadata = {
        "source": str(args.input),
        "seed": args.seed,
        "total_examples": len(rows),
        "splits": {split: len(split_rows) for split, split_rows in splits.items()},
        "note": "Rows with review_status='gold' and at least one role are included. Splits are grouped by clause_id to reduce leakage.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
