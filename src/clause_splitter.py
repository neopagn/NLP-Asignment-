import re
from dataclasses import dataclass
from typing import Iterable

from .utils import normalize_whitespace


@dataclass(frozen=True)
class ClauseRecord:
    clause: str
    section: str | None = None


STANDALONE_HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s/&-]{3,}|ARTICLE\s+[IVXLC\d]+|SECTION\s+\d+|State of\s*)$",
    re.IGNORECASE,
)
SOURCE_RE = re.compile(r"^=+\s*SOURCE:.*?=+$", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"_+")
DOT_PLACEHOLDER = "__DOT__"
MULTI_PERIOD_ABBREVIATION_RE = re.compile(r"\b(?:U\.S\.A|U\.S|L\.L\.C|e\.g|i\.e)\.", re.IGNORECASE)
ABBREVIATION_RE = re.compile(
    r"\b(?:Inc|Corp|Co|Ltd|LLC|No|Nos|Mr|Mrs|Ms|Dr|Prof|St|Ave|Blvd|Rd|Jr|Sr|Cal|Del)\.",
    re.IGNORECASE,
)
INITIAL_BEFORE_NAME_RE = re.compile(r"\b[A-Z]\.(?=\s+[A-Z][a-z])")
SECTION_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)\.?\s+"
    r"(?P<title>[A-Z][A-Z0-9 /&,';:\-\[\]\(\)]{2,120})\.\s*"
    r"(?P<body>.*)$"
)
SHORT_LABEL_RE = re.compile(r"^[A-Z][A-Za-z /&,'()-]{1,45}\.\s*")
CHECK_RE = re.compile(r"\bcheck (?:one|all that apply)\b", re.IGNORECASE)
BAD_FRAGMENT_RE = re.compile(
    r"^(?:if you want|if you do not|between the hours|all that apply|fixed term|at will)\b",
    re.IGNORECASE,
)
BAD_ENDING_RE = re.compile(r"\b(?:and|or|the|which|even)\.$", re.IGNORECASE)
DEFINITION_RE = re.compile(r"^\(?[a-zA-Z0-9]{1,3}\)?\s*[\"'][^\"']+[\"']\s+means\b", re.IGNORECASE)
LEGAL_SIGNAL_RE = re.compile(
    r"\b(shall|must|may|will|agree|covenant|terminate|pay|provide|perform|deliver|"
    r"disclose|confidential|governed|law|notice|party|employee|employer|buyer|seller|"
    r"tenant|landlord|contractor|client|supplier|customer|penalty|breach|obligation|"
    r"means|refers to|is defined as|is referred to)\b",
    re.IGNORECASE,
)
LEGAL_ACTION_RE = re.compile(
    r"\b(shall|must|may|will|agree|covenant|terminate|terminated|expires?|pay|provide|"
    r"perform|deliver|disclose|governed|required|entitled|prohibited|forbidden|breach|"
    r"obligation|reimburse|recover|execute|return|assign|waive|violate|confidential|"
    r"effective date|made|not|means|refers to|is defined as|is referred to)\b",
    re.IGNORECASE,
)

SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.;!?])\s+(?=(?:[A-Z0-9]|\([a-zA-Z0-9]+\)|\"))",
    re.IGNORECASE,
)
COORDINATE_RE = re.compile(r"\s*,\s+(?P<conjunction>and|but|provided that|except that)\s+", re.IGNORECASE)
LEGAL_PARTY_SUBJECT = (
    r"Party\s+[A-Z]|each party|either party|neither party|the parties|a party|the other party|"
    r"buyer|seller|tenant|landlord|employee|employer|licensor|licensee|contractor|client|supplier|customer|"
    r"[a-z]+-[A-Z][A-Za-z0-9]+(?:\.[A-Za-z]{2,})?|[A-Z]+-[A-Z0-9]+|"
    r"\d+[A-Za-z]+(?:[A-Z][A-Za-z0-9]*)+|"
    r"[A-Z0-9][A-Za-z0-9-]*(?:\.[A-Z0-9][A-Za-z0-9-]*)+|"
    r"[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){0,3}"
)
INDEPENDENT_RIGHT_RE = re.compile(
    rf"^(?:if|unless|provided that|except that|{LEGAL_PARTY_SUBJECT})\b"
    rf"(?=.*\b(?:shall|will|must|may|is|are|has|have|can)\b)",
    re.IGNORECASE,
)
LEGAL_VERB_START_RE = re.compile(
    r"^(?:develop|provide|pay|perform|deliver|disclose|use|return|remove|notify|grant|terminate|"
    r"make|maintain|hold|submit|meet|negotiate|cooperate|indemnify|reimburse)\b",
    re.IGNORECASE,
)
SUBJECT_MODAL_RE = re.compile(
    rf"\b(?P<subject>{LEGAL_PARTY_SUBJECT})\s+"
    rf"(?P<modal>shall|will|must|may)\s+\w+",
    re.IGNORECASE,
)
AGREEMENT_TITLE_RE = re.compile(
    r"^THIS\s+.+?\s+AGREEMENT\s+\(the\s+[\"']Agreement[\"']\)\s+is\s+made\b",
    re.IGNORECASE,
)


def _protect_sentence_periods(text: str) -> str:
    def protect(match: re.Match[str]) -> str:
        return match.group(0).replace(".", DOT_PLACEHOLDER)

    text = MULTI_PERIOD_ABBREVIATION_RE.sub(protect, text)
    text = ABBREVIATION_RE.sub(protect, text)
    return INITIAL_BEFORE_NAME_RE.sub(protect, text)


def _restore_sentence_periods(text: str) -> str:
    return text.replace(DOT_PLACEHOLDER, ".")


def _strip_bullet_prefix(line: str) -> str:
    return re.sub(r"^\s*[-*]\s+", "", line).strip()


def _extract_section(line: str) -> tuple[str | None, str]:
    match = SECTION_HEADING_RE.match(line)
    if not match:
        return None, line
    section = f"{match.group('number')} {match.group('title').strip()}"
    return section, match.group("body").strip()


def _clean_line(line: str) -> str:
    line = line.strip()
    if not line or SOURCE_RE.match(line):
        return ""
    line = PLACEHOLDER_RE.sub("", line)
    line = line.replace("[", " ").replace("]", " ")
    line = re.sub(r"\s+", " ", line).strip()
    line = _strip_bullet_prefix(line)
    if not line or CHECK_RE.search(line) and not LEGAL_SIGNAL_RE.search(line):
        return ""
    short_match = SHORT_LABEL_RE.match(line)
    if (
        short_match
        and len(line[short_match.end() :].split()) >= 4
        and not re.search(r"\b(means|shall|may|will|must)\b", line, re.IGNORECASE)
        and not DEFINITION_RE.match(line)
    ):
        line = line[short_match.end() :].strip()
    line = re.sub(r"^Other:\s*", "", line, flags=re.IGNORECASE)
    if STANDALONE_HEADING_RE.match(line):
        return ""
    return line


def _paragraphs(text: str) -> Iterable[tuple[str | None, str]]:
    buffer: list[str] = []
    current_section: str | None = None
    buffer_section: str | None = None

    def flush() -> tuple[str | None, str]:
        nonlocal buffer_section
        paragraph = " ".join(buffer).strip()
        buffer.clear()
        section = buffer_section
        buffer_section = None
        return section, paragraph

    for raw_line in normalize_whitespace(text).splitlines():
        line = _clean_line(raw_line)
        if not line:
            if " ".join(buffer).strip().lower() in {"if", "unless", "provided that"}:
                continue
            section, paragraph = flush()
            if paragraph:
                yield section, paragraph
            continue

        section, body = _extract_section(line)
        if section:
            old_section, paragraph = flush()
            if paragraph:
                yield old_section, paragraph
            current_section = section
            if not body:
                continue
            line = body

        if not buffer:
            buffer_section = current_section
        buffer.append(line)

    section, paragraph = flush()
    if paragraph:
        yield section, paragraph


def _candidate_units(text: str) -> Iterable[tuple[str | None, str]]:
    pending_prefix: tuple[str | None, str] | None = None
    for section, paragraph in _paragraphs(text):
        protected = _protect_sentence_periods(paragraph)
        for part in SENTENCE_BOUNDARY_RE.split(protected):
            part = _restore_sentence_periods(part).strip(" ;")
            if not part:
                continue
            if pending_prefix:
                pending_section, prefix = pending_prefix
                part = f"{prefix} {part}"
                section = pending_section or section
                pending_prefix = None
            if part.lower() in {"if", "unless", "provided that"}:
                pending_prefix = (section, part)
                continue
            yield section, part


def _carry_subject_modal(text: str) -> str | None:
    matches = list(SUBJECT_MODAL_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return f"{match.group('subject')} {match.group('modal')} "


def _right_side_can_stand(right: str) -> bool:
    return bool(INDEPENDENT_RIGHT_RE.match(right))


def _right_side_can_reuse_subject(right: str, before: str) -> bool:
    return bool(LEGAL_VERB_START_RE.match(right) and _carry_subject_modal(before))


def _split_coordinate_clause(unit: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    prefix = ""
    for match in COORDINATE_RE.finditer(unit):
        before = f"{prefix}{unit[start : match.start()]}".strip(" ,;")
        right = unit[match.end() :].strip(" ,;")
        split_independent = _right_side_can_stand(right)
        split_with_carried_subject = _right_side_can_reuse_subject(right, before)
        if len(before.split()) >= 5 and (split_independent or split_with_carried_subject):
            pieces.append(before)
            prefix = _carry_subject_modal(before) if split_with_carried_subject else ""
            start = match.end()
    remainder = f"{prefix}{unit[start:]}".strip(" ,;")
    if remainder:
        pieces.append(remainder)
    return pieces or [unit]


def _finish_clause(clause: str) -> str:
    clause = re.sub(r"\s+", " ", clause).strip(" ,;:")
    clause = AGREEMENT_TITLE_RE.sub("This Agreement is made", clause)
    if not clause:
        return ""
    if clause.lower().startswith(("if ", "unless ", "provided that ")):
        clause = clause[0].upper() + clause[1:]
    if not re.search(r"[.!?][\"')\]]*$", clause):
        clause += "."
    return clause


def split_clause_records(text: str) -> list[ClauseRecord]:
    records: list[ClauseRecord] = []
    seen: set[str] = set()

    for section, unit in _candidate_units(text):
        for piece in _split_coordinate_clause(unit):
            clause = _finish_clause(piece)
            key = clause.lower()
            if not clause or key in seen:
                continue
            if len(clause.split()) < 5:
                continue
            if CHECK_RE.search(clause) or BAD_FRAGMENT_RE.match(clause):
                continue
            if BAD_ENDING_RE.search(clause):
                continue
            if not LEGAL_ACTION_RE.search(clause):
                continue
            records.append(ClauseRecord(clause=clause, section=section))
            seen.add(key)

    return records


def split_clauses(text: str) -> list[str]:
    return [record.clause for record in split_clause_records(text)]
