import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "srl_annotation_candidates.jsonl"
VALID_ROLE_LABELS = {
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
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def _check_span(row: dict[str, Any], span: dict[str, Any], label: str, errors: list[str]) -> None:
    clause = row.get("clause", "")
    row_id = row.get("id", f"line {row.get('_line_number')}")
    start = span.get("start")
    end = span.get("end")
    text = span.get("text")
    if start is None or end is None:
        errors.append(f"{row_id}: {label} has null start/end")
        return
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"{row_id}: {label} start/end must be integers")
        return
    if start < 0 or end <= start or end > len(clause):
        errors.append(f"{row_id}: {label} invalid span {start}:{end}")
        return
    actual = clause[start:end]
    if actual != text:
        errors.append(f"{row_id}: {label} text mismatch, span gives {actual!r} but text is {text!r}")


def validate(rows: list[dict[str, Any]], gold_only: bool) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not row_id:
            errors.append(f"line {row.get('_line_number')}: missing id")
        elif row_id in seen_ids:
            errors.append(f"{row_id}: duplicate id")
        else:
            seen_ids.add(row_id)

        if gold_only and row.get("review_status") != "gold":
            continue

        if not row.get("clause"):
            errors.append(f"{row_id}: missing clause")
            continue

        predicate = row.get("predicate")
        if not isinstance(predicate, dict):
            errors.append(f"{row_id}: predicate must be an object")
        else:
            _check_span(row, predicate, "predicate", errors)

        roles = row.get("roles")
        if not isinstance(roles, list):
            errors.append(f"{row_id}: roles must be a list")
            continue
        role_labels = [role.get("label") for role in roles]
        if "Definition" in role_labels and "DefinedTerm" not in role_labels:
            errors.append(f"{row_id}: Definition role should be skipped unless DefinedTerm is also annotated")
        for index, role in enumerate(roles, start=1):
            label = role.get("label")
            if label not in VALID_ROLE_LABELS:
                errors.append(f"{row_id}: role {index} has invalid label {label!r}")
            _check_span(row, role, f"role {index} ({label})", errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manually reviewed SRL annotation JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gold-only", action="store_true", help="Validate only rows with review_status='gold'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.input)
    errors = validate(rows, args.gold_only)
    gold_count = sum(1 for row in rows if row.get("review_status") == "gold")
    if errors:
        print(f"rows={len(rows)}")
        print(f"gold_rows={gold_count}")
        print("Validation failed:")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors")
        raise SystemExit(1)

    print("SRL annotation validation passed")
    print(f"rows={len(rows)}")
    print(f"gold_rows={gold_count}")


if __name__ == "__main__":
    main()
