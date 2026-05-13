import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "srl_annotation_candidates.jsonl"
DEFAULT_CSV = PROJECT_ROOT / "data" / "srl_annotation_candidates.csv"
SKIP_ROLES = {"AdditionalPredicates", "DefinitionLabel"}
CORPORATE_SUFFIX_RE = re.compile(
    r"\b(?:Inc\.?|INC\.?|LLC|L\.L\.C\.|Ltd\.?|LTD\.?|Corp\.?|CORP\.?|Corporation|Company|Co\.?|CO\.?)$",
    re.IGNORECASE,
)
ROLE_ORDER = [
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
]
IMPORTANT_PREDICATES = {
    "agree",
    "assign",
    "breach",
    "comply",
    "deliver",
    "develop",
    "disclose",
    "govern",
    "indemnify",
    "keep",
    "maintain",
    "make",
    "may",
    "must",
    "pay",
    "perform",
    "provide",
    "require",
    "shall",
    "terminate",
    "will",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _is_formation_clause(entry: dict[str, Any]) -> bool:
    predicate = str(entry.get("predicate") or "").lower()
    clause = str(entry.get("clause") or "").lower()
    roles = dict(entry.get("roles") or {})
    return bool(roles.get("ContractParties")) and (predicate in {"make", "execute"} or "by and between" in clause)


def _is_definition_clause(entry: dict[str, Any]) -> bool:
    roles = dict(entry.get("roles") or {})
    predicate = str(entry.get("predicate_text") or entry.get("predicate") or "").lower()
    return bool(roles.get("DefinedTerm") and roles.get("Definition")) or predicate in {"means", "refers to"}


def _prefer_formal_parties(values: list[str]) -> list[str]:
    formal = [value for value in values if CORPORATE_SUFFIX_RE.search(value.strip())]
    return formal or values


def _role_values(entry: dict[str, Any]) -> list[tuple[str, str]]:
    roles = dict(entry.get("roles") or {})
    values: list[tuple[str, str]] = []

    if _is_definition_clause(entry):
        return [
            (label, str(roles[label]).strip())
            for label in ("DefinedTerm", "Definition")
            if roles.get(label)
        ]

    if _is_formation_clause(entry):
        formation_values: list[tuple[str, str]] = []
        for label in ("Theme", "Time", "Condition", "LegalBasis"):
            if roles.get(label):
                formation_values.append((label, str(roles[label]).strip()))
        contract_parties = [
            part.strip()
            for part in str(roles.get("ContractParties") or "").split(";")
            if part.strip()
        ]
        formation_values.extend(("ContractParties", part) for part in _prefer_formal_parties(contract_parties))
        return formation_values

    for label in ROLE_ORDER:
        raw_value = roles.get(label)
        if not raw_value or label in SKIP_ROLES:
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        if label in {"Agent", "ContractParties", "Recipient"}:
            parts = [part.strip() for part in text.split(";") if part.strip()]
        else:
            parts = [text]
        values.extend((label, part) for part in parts)

    for label, raw_value in roles.items():
        if label in ROLE_ORDER or label in SKIP_ROLES or not raw_value:
            continue
        values.append((label, str(raw_value).strip()))
    return values


def _span_record(clause: str, label: str, value: str) -> dict[str, Any]:
    span = _find_span(clause, value)
    if span is None:
        return {
            "label": label,
            "text": value,
            "start": None,
            "end": None,
            "needs_fix": True,
        }
    start, end = span
    return {
        "label": label,
        "text": clause[start:end],
        "start": start,
        "end": end,
        "needs_fix": False,
    }


def _predicate_record(entry: dict[str, Any]) -> dict[str, Any]:
    clause = str(entry["clause"])
    predicate_text = str(entry.get("predicate_text") or entry.get("predicate") or "").strip()
    span = _find_span(clause, predicate_text) if predicate_text else None
    return {
        "text": predicate_text,
        "lemma": entry.get("predicate"),
        "start": span[0] if span else None,
        "end": span[1] if span else None,
        "needs_fix": span is None,
    }


def _score_candidate(entry: dict[str, Any]) -> int:
    predicate = str(entry.get("predicate") or "").lower()
    roles = dict(entry.get("roles") or {})
    score = 0
    if predicate in IMPORTANT_PREDICATES:
        score += 5
    if roles.get("Condition"):
        score += 4
    if roles.get("Time"):
        score += 3
    if roles.get("LegalBasis"):
        score += 3
    if roles.get("Agent") and roles.get("Theme"):
        score += 2
    if entry.get("method") == "legal_definition_rule":
        score += 1
    return score


def build_candidates(
    srl_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    limit: int | None = None,
    include_definitions: bool = True,
) -> list[dict[str, Any]]:
    context_by_id = {row["clause_id"]: row for row in context_rows}
    rows = []
    for entry in srl_rows:
        if entry.get("method") == "legal_definition_rule" and not include_definitions:
            continue

        clause = str(entry["clause"])
        clause_id = int(entry["clause_id"])
        roles = [
            _span_record(clause, label, value)
            for label, value in _role_values(entry)
        ]
        context = context_by_id.get(clause_id, {})
        rows.append(
            {
                "id": f"c{clause_id:04d}-p001",
                "clause_id": clause_id,
                "section": context.get("section"),
                "clause": clause,
                "predicate": _predicate_record(entry),
                "roles": roles,
                "source_method": entry.get("method"),
                "priority_score": _score_candidate(entry),
                "review_status": "needs_review",
                "annotator_notes": "",
                "guideline": "Check predicate span and role spans. Remove wrong roles, fix start/end/text, and duplicate this row for extra predicates if needed.",
            }
        )

    rows.sort(key=lambda row: (-int(row["priority_score"]), int(row["clause_id"])))
    if limit is not None:
        rows = rows[:limit]
    rows.sort(key=lambda row: int(row["clause_id"]))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "clause_id",
                "section",
                "predicate",
                "roles",
                "source_method",
                "priority_score",
                "review_status",
                "clause",
                "annotator_notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "clause_id": row["clause_id"],
                    "section": row["section"],
                    "predicate": json.dumps(row["predicate"], ensure_ascii=False),
                    "roles": json.dumps(row["roles"], ensure_ascii=False),
                    "source_method": row["source_method"],
                    "priority_score": row["priority_score"],
                    "review_status": row["review_status"],
                    "clause": row["clause"],
                    "annotator_notes": row["annotator_notes"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SRL pre-labels for manual gold annotation.")
    parser.add_argument("--srl", type=Path, default=PROJECT_ROOT / "output" / "srl_results.json")
    parser.add_argument("--context", type=Path, default=PROJECT_ROOT / "output" / "clauses_context.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--exclude-definitions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    srl_rows = _load_json(args.srl)
    context_rows = _load_json(args.context) if args.context.exists() else []
    candidates = build_candidates(
        srl_rows,
        context_rows,
        limit=args.limit,
        include_definitions=not args.exclude_definitions,
    )
    write_jsonl(candidates, args.output)
    write_csv(candidates, args.csv_output)
    needs_fix = sum(
        1
        for row in candidates
        if row["predicate"].get("needs_fix") or any(role.get("needs_fix") for role in row["roles"])
    )
    print(f"candidates={len(candidates)}")
    print(f"needs_span_fix={needs_fix}")
    print(f"jsonl={args.output}")
    print(f"csv={args.csv_output}")


if __name__ == "__main__":
    main()
