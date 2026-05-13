import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import OUTPUT_DIR


INTENT_PATTERNS = {
    "Obligation": re.compile(r"\b(shall|must|required|obligat|responsib|duty|pay|provide|deliver|perform)\b", re.I),
    "Prohibition": re.compile(r"\b(shall not|must not|may not|cannot|can't|prohibit|forbid|not allowed|no party)\b", re.I),
    "Right": re.compile(r"\b(may|can|right|entitled|allowed|option|discretion)\b", re.I),
    "Termination Condition": re.compile(r"\b(terminate|termination|end|expire|expiration|breach|insolven|bankrupt)\b", re.I),
}
ENTITY_FOCUS_PATTERNS = {
    "MONEY": re.compile(r"\b(amount|money|fee|fees|payment|pay|price|cost|revenue|vnd|usd|dollars?)\b", re.I),
    "DATE": re.compile(r"\b(when|date|deadline|time|month|day|year|effective)\b", re.I),
    "RATE": re.compile(r"\b(rate|percent|percentage|interest)\b", re.I),
    "PENALTY": re.compile(r"\b(penalty|penalized|fine|late fee|damages|default charge|interest)\b", re.I),
    "LAW": re.compile(r"\b(law|governing|jurisdiction|legal)\b", re.I),
}
LEGAL_PARTY_AGENT = (
    r"Party\s+[A-Z]|each party|either party|the parties|a party|buyer|seller|tenant|landlord|"
    r"employee|employer|licensor|licensee|contractor|client|supplier|customer|"
    r"[a-z]+-[A-Z][A-Za-z0-9]+(?:\.[A-Za-z]{2,})?|[A-Z]+-[A-Z0-9]+|"
    r"\d+[A-Za-z]+(?:[A-Z][A-Za-z0-9]*)+|"
    r"[A-Z0-9][A-Za-z0-9-]*(?:\.[A-Z0-9][A-Za-z0-9-]*)+|"
    r"[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){0,3}"
)
ACTION_AGENT_RE = re.compile(
    rf"\b(?P<agent>{LEGAL_PARTY_AGENT})\s+"
    rf"(?:shall|will|must|may|is required to|is obligated to|agrees to)\s+"
    rf"(?P<verb>[a-z][a-z-]*)",
    re.I,
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "happen",
    "happens",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "party",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
}


@dataclass(frozen=True)
class ContractRecord:
    clause_id: int
    clause: str
    section: str | None
    intent: str
    entities: list[dict[str, Any]]
    predicate: str | None
    predicate_text: str | None
    roles: dict[str, str]


@dataclass(frozen=True)
class ParsedQuestion:
    text: str
    normalized: str
    intents: set[str]
    entity_focus: set[str]
    parties: set[str]
    keywords: set[str]
    asks_when: bool
    asks_who: bool
    asks_what: bool


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_intents(question: str) -> set[str]:
    return {label for label, pattern in INTENT_PATTERNS.items() if pattern.search(question)}


def _parse_entity_focus(question: str) -> set[str]:
    return {label for label, pattern in ENTITY_FOCUS_PATTERNS.items() if pattern.search(question)}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
    return {token for token in tokens if token not in STOP_WORDS and len(token) > 2}


def _entity_index(records: list[ContractRecord]) -> set[str]:
    values: set[str] = set()
    for record in records:
        for entity in record.entities:
            text = str(entity.get("text", "")).strip()
            if len(text) >= 3:
                values.add(text.lower())
    return values


def _parse_parties(question: str, records: list[ContractRecord]) -> set[str]:
    parties = {match.group(0).lower() for match in re.finditer(r"\bParty\s+[A-Z]\b", question, re.I)}
    role_words = re.findall(
        r"\b(?:buyer|seller|tenant|landlord|employee|employer|licensor|licensee|company|contractor|client|supplier|customer)\b",
        question,
        re.I,
    )
    parties.update(word.lower() for word in role_words)

    lower_question = question.lower()
    for entity_text in _entity_index(records):
        if entity_text in lower_question:
            parties.add(entity_text)
    return parties


def parse_question(question: str, records: list[ContractRecord]) -> ParsedQuestion:
    normalized = question.strip().lower()
    return ParsedQuestion(
        text=question.strip(),
        normalized=normalized,
        intents=_parse_intents(question),
        entity_focus=_parse_entity_focus(question),
        parties=_parse_parties(question, records),
        keywords=_tokenize(question),
        asks_when=bool(re.search(r"\b(when|what date|what time|deadline)\b", question, re.I)),
        asks_who=bool(re.search(r"\b(who|which party|whom)\b", question, re.I)),
        asks_what=bool(re.search(r"\b(what|which|how much|how many)\b", question, re.I)),
    )


def _parse_intent_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clause, label = line.rsplit("\t", 1)
        rows.append((clause, label))
    return rows


def load_records(output_dir: Path = OUTPUT_DIR) -> list[ContractRecord]:
    ner_rows = _load_json(output_dir / "ner_results.json")
    srl_rows = _load_json(output_dir / "srl_results.json")
    intent_rows = _parse_intent_file(output_dir / "intent_classification.txt")
    context_path = output_dir / "clauses_context.json"
    context_rows = _load_json(context_path) if context_path.exists() else []

    srl_by_id = {int(item["clause_id"]): item for item in srl_rows}
    context_by_id = {int(item["clause_id"]): item for item in context_rows}
    records: list[ContractRecord] = []
    for ner_item in ner_rows:
        clause_id = int(ner_item["clause_id"])
        srl_item = srl_by_id.get(clause_id, {})
        context_item = context_by_id.get(clause_id, {})
        intent = intent_rows[clause_id - 1][1] if clause_id - 1 < len(intent_rows) else "Unknown"
        records.append(
            ContractRecord(
                clause_id=clause_id,
                clause=str(ner_item["clause"]),
                section=context_item.get("section"),
                intent=intent,
                entities=list(ner_item.get("entities", [])),
                predicate=srl_item.get("predicate"),
                predicate_text=srl_item.get("predicate_text"),
                roles=dict(srl_item.get("roles", {})),
            )
        )
    return records


def _record_text(record: ContractRecord) -> str:
    role_text = " ".join(record.roles.values())
    entity_text = " ".join(str(entity.get("text", "")) for entity in record.entities)
    return (
        f"{record.section or ''} {record.clause} {record.intent} {record.predicate or ''} "
        f"{record.predicate_text or ''} {role_text} {entity_text}"
    ).lower()


def score_record(question: ParsedQuestion, record: ContractRecord) -> int:
    searchable = _record_text(record)
    score = 0

    if record.intent in question.intents:
        score += 8

    for label in question.entity_focus:
        if any(entity.get("label") == label for entity in record.entities):
            score += 5

    for party in question.parties:
        if party in searchable:
            score += 6

    keyword_hits = question.keywords.intersection(_tokenize(searchable))
    score += min(len(keyword_hits), 8)

    if record.predicate and record.predicate.lower() in question.keywords:
        score += 2
    if record.predicate_text and record.predicate_text.lower() in question.keywords:
        score += 2
    if question.asks_when and ("Time" in record.roles or "Condition" in record.roles):
        score += 2
    if question.asks_who and ("Agent" in record.roles or "Recipient" in record.roles):
        score += 2
    if question.asks_what and ("Theme" in record.roles or record.entities):
        score += 1

    return score


def rank_records(question: ParsedQuestion, records: list[ContractRecord], top_k: int = 3) -> list[tuple[int, ContractRecord]]:
    ranked = [(score_record(question, record), record) for record in records]
    ranked = [(score, record) for score, record in ranked if score > 0]
    ranked.sort(key=lambda item: (-item[0], item[1].clause_id))
    return ranked[:top_k]


def _entities_by_label(record: ContractRecord, label: str) -> list[str]:
    return [str(entity["text"]) for entity in record.entities if entity.get("label") == label]


def _question_action_agent(question: ParsedQuestion, record: ContractRecord) -> str | None:
    for match in ACTION_AGENT_RE.finditer(record.clause):
        verb = match.group("verb").lower()
        if verb in question.keywords or any(keyword.startswith(verb) or verb.startswith(keyword) for keyword in question.keywords):
            return re.sub(r"^(?:and|or)\s+", "", match.group("agent").strip(" ,;"), flags=re.I)
    return None


def _build_answer(question: ParsedQuestion, record: ContractRecord) -> str:
    if question.asks_when or "DATE" in question.entity_focus:
        if "Time" in record.roles:
            return f"The relevant time is {record.roles['Time']}."
        if "Condition" in record.roles:
            return f"The relevant condition is: {record.roles['Condition']}"
        dates = _entities_by_label(record, "DATE")
        if dates:
            return f"The relevant date is {dates[0]}."

    if question.asks_who:
        action_agent = _question_action_agent(question, record)
        if action_agent:
            return f"The responsible party is {action_agent}."
        if "Agent" in record.roles:
            return f"The responsible party is {record.roles['Agent']}."
        parties = _entities_by_label(record, "PARTY")
        if parties:
            return f"The relevant party is {parties[0]}."

    if "MONEY" in question.entity_focus:
        amounts = _entities_by_label(record, "MONEY")
        if amounts:
            return f"The relevant monetary amount is {amounts[0]}."

    if "RATE" in question.entity_focus:
        rates = _entities_by_label(record, "RATE")
        if rates:
            return f"The relevant rate is {rates[0]}."

    if "LAW" in question.entity_focus:
        laws = _entities_by_label(record, "LAW")
        if laws:
            return f"The relevant governing law or jurisdiction reference is {laws[0]}."

    if "Condition" in record.roles and ("PENALTY" in question.entity_focus or "Termination Condition" in question.intents):
        return f"The triggering condition is: {record.roles['Condition']}"

    return f"The contract states: {record.clause}"


def answer_question(question: str, records: list[ContractRecord], top_k: int = 3) -> dict[str, Any]:
    parsed = parse_question(question, records)
    matches = rank_records(parsed, records, top_k=top_k)
    if not matches:
        return {
            "question": question,
            "answer": "I could not find support for that question in the structured contract outputs.",
            "matches": [],
        }

    best_score, best = matches[0]
    return {
        "question": question,
        "answer": _build_answer(parsed, best),
        "intent": best.intent,
        "predicate": best.predicate_text,
        "roles": best.roles,
        "source": {"clause_id": best.clause_id, "section": best.section, "clause": best.clause},
        "matches": [
            {
                "score": score,
                "clause_id": record.clause_id,
                "section": record.section,
                "intent": record.intent,
                "clause": record.clause,
            }
            for score, record in matches
        ],
        "best_score": best_score,
    }


def format_answer(result: dict[str, Any]) -> str:
    lines = [f"Answer: {result['answer']}"]
    if result.get("intent"):
        lines.append(f"Intent: {result['intent']}")
    if result.get("roles"):
        roles = ", ".join(f"{name}={value}" for name, value in result["roles"].items())
        lines.append(f"Structured roles: {roles}")
    source = result.get("source")
    if source:
        if source.get("section"):
            lines.append(f"Source section: {source['section']}")
        lines.append(f"Source clause {source['clause_id']}: {source['clause']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask rule-based questions over structured contract outputs.")
    parser.add_argument("question", nargs="*", help="Question to answer. Omit for interactive mode.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    records = load_records(args.output_dir)
    if args.question:
        result = answer_question(" ".join(args.question), records, top_k=args.top_k)
        print(format_answer(result))
        return

    print("Contract QA ready. Type a question, or type 'exit' to quit.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(format_answer(answer_question(question, records, top_k=args.top_k)))


if __name__ == "__main__":
    main()
