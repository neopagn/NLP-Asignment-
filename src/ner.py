import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int


MONTH = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
NUMBER_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety"
)
LEGAL_DURATION_UNIT = r"business\s+days?|days?|weeks?|months?|years?"
LEGAL_NUMBER_EXPR = rf"(?:(?:{NUMBER_WORD})\s*\(\d+\)|\d+\s*\(\d+\)|(?:{NUMBER_WORD})|\d+)"
LEGAL_DURATION = rf"{LEGAL_NUMBER_EXPR}\s+(?:{LEGAL_DURATION_UNIT})(?:\s+or\s+more)?"
LEGAL_DURATION_WITH_TRIGGER = (
    rf"(?:before|after|on|by|within|for|during|no\s+less\s+than|at\s+least|once\s+every)\s+"
    rf"{LEGAL_DURATION}"
)
CORPORATE_SUFFIX = r"Inc\.?|INC\.?|LLC|L\.L\.C\.|Ltd\.?|LTD\.?|Corp\.?|CORP\.?|Corporation|Company|Co\.?|CO\.?"
DOMAIN_COMPANY = r"[A-Z0-9][A-Z0-9-]*(?:\.[A-Z0-9][A-Z0-9-]*)+"
HYPHENATED_BRAND = r"(?:[a-z]+-[A-Z][A-Za-z0-9]+|[A-Z]+-[A-Z0-9]+)(?:\.[A-Za-z]{2,})?"
DIGIT_LED_BRAND = r"\d+[A-Za-z]+(?:[A-Z][A-Za-z0-9]*)+"
MONEY_VALID_RE = re.compile(
    r"^(?:"
    r"(?:USD|VND|EUR|\$)\s*\d+(?:,\d{3})*(?:\.\d+)?"
    r"|"
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:USD|VND|EUR|dollars?|dong)"
    r")$",
    re.IGNORECASE,
)
DATE_VALID_RE = re.compile(
    rf"^(?:"
    rf"\d{{1,2}}\s+(?:{MONTH})\s+\d{{4}}"
    rf"|(?:{MONTH})\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}"
    rf"|{LEGAL_DURATION_WITH_TRIGGER}"
    rf"|{LEGAL_DURATION}"
    rf")$",
    re.IGNORECASE,
)
RATE_VALID_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?\s?(?:percent|per cent)|\d+(?:\.\d+)?\s?(?:per\s+(?:day|month|year|annum)))",
    re.IGNORECASE,
)
LEGAL_DATE_TERM_RE = re.compile(r"^(?:effective|launch|approval|expiration|payment|due|adjusted)\s+date$|^term$", re.IGNORECASE)


PATTERNS: list[tuple[str, str, int]] = [
    ("MONEY", r"\b(?:USD|VND|EUR|\$)\s*\d+(?:,\d{3})*(?:\.\d+)?\b", re.IGNORECASE),
    ("MONEY", r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:VND|USD|EUR|dollars?|dong)\b", re.IGNORECASE),
    ("RATE", r"\b\d+(?:\.\d+)?\s?%\s?(?:per\s+(?:day|month|year|annum))?\b", re.IGNORECASE),
    ("DATE", rf"\b\d{{1,2}}\s+(?:{MONTH})\s+\d{{4}}\b", re.IGNORECASE),
    ("DATE", rf"\b(?:{MONTH})\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    ("DATE", rf"\b{LEGAL_DURATION_WITH_TRIGGER}\b", re.IGNORECASE),
    ("DATE", rf"\b{LEGAL_DURATION}\b", re.IGNORECASE),
    ("LAW", r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Code\s+Section\s*\d+(?:\s+et\s+seq\.?)?(?=\s|,|;|\)|$)", 0),
    ("LAW", r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Law\b", 0),
    ("LAW", r"\b(?:governing law|applicable law)\b", re.IGNORECASE),
    ("LAW", r"\blaws? of (?:the\s+)?(?:State\s+of\s+)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b", 0),
    ("LAW", r"\bjurisdiction of [A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b", 0),
    ("PARTY", r"\bParty\s+[A-Z]\b", re.IGNORECASE),
    ("PARTY", rf"\b[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)+\s*,?\s*(?:{CORPORATE_SUFFIX})(?=\s|,|;|\)|$)", re.IGNORECASE),
    ("PARTY", rf"\b[A-Za-z0-9][A-Za-z0-9.-]*(?:-[A-Za-z0-9.-]+)?\s*,?\s+(?:{CORPORATE_SUFFIX})(?=\s|,|;|\)|$)", re.IGNORECASE),
    ("PARTY", rf"\b(?:{DOMAIN_COMPANY}|{HYPHENATED_BRAND}|{DIGIT_LED_BRAND})\b", 0),
    ("PARTY", r"\b(?:Buyer|Seller|Tenant|Landlord|Employee|Employer|Licensor|Licensee|Company|Contractor|Client|Supplier|Customer)\b", re.IGNORECASE),
    ("PENALTY", r"\b(?:penalty|liquidated damages|late fee|fine|default charge)\b", re.IGNORECASE),
    ("PENALTY", r"\binterest\s+(?:rate|charge|on\s+overdue|accru(?:e|ed|es|ing))\b", re.IGNORECASE),
]


def _make_entity(text: str, label: str, start: int, end: int) -> Entity | None:
    left_trimmed = len(text) - len(text.lstrip(" \t\n\r\"'()[]{}.,;:"))
    right_chars = " \t\n\r\"'()[]{}.,;:"
    right_trimmed = len(text.rstrip(right_chars))
    right_trimmed_no_period = len(text.rstrip(" \t\n\r\"'()[]{};,:" ))
    candidate_with_period = text[left_trimmed:right_trimmed_no_period]
    if label == "PARTY" and re.search(rf"\b(?:{CORPORATE_SUFFIX})$", candidate_with_period, re.IGNORECASE):
        right_trimmed = right_trimmed_no_period
    elif label == "LAW" and re.search(r"\bet\s+seq\.$", candidate_with_period, re.IGNORECASE):
        right_trimmed = right_trimmed_no_period
    clean = text[left_trimmed:right_trimmed]
    if not clean:
        return None
    return Entity(clean, label, start + left_trimmed, start + right_trimmed)


def _looks_like_label(text: str, label: str) -> bool:
    stripped = text.strip(" \t\n\r\"'()[]{}.,;:")
    if len(stripped) < 2:
        return False
    if label == "PARTY":
        if re.search(r"\b(date|agreement|term|section|employment|confidential)\b", stripped, re.I):
            return False
        return bool(
            re.search(
                r"\b(Party\s+[A-Z]|Buyer|Seller|Tenant|Landlord|Employee|Employer|Licensor|"
                r"Licensee|Company|Contractor|Client|Supplier|Customer|LLC|Inc\.?|Corp\.?)\b",
                stripped,
                re.I,
            )
            or re.fullmatch(DOMAIN_COMPANY, stripped)
            or re.fullmatch(HYPHENATED_BRAND, stripped)
            or re.fullmatch(DIGIT_LED_BRAND, stripped)
        )
    if label == "MONEY":
        return bool(MONEY_VALID_RE.fullmatch(stripped))
    if label == "DATE":
        if LEGAL_DATE_TERM_RE.fullmatch(stripped):
            return False
        return bool(DATE_VALID_RE.fullmatch(stripped))
    if label == "RATE":
        return bool(RATE_VALID_RE.match(stripped))
    if label == "PENALTY":
        return bool(
            re.search(
                r"\b(penalty|liquidated damages|late fee|fine|default charge|"
                r"interest\s+(?:rate|charge|on\s+overdue|accru(?:e|ed|es|ing)))\b",
                stripped,
                re.I,
            )
        )
    if label == "LAW":
        if re.fullmatch(r"(?:laws?|regulations?|laws and regulations|rules or regulations)", stripped, re.I):
            return False
        return bool(
            re.search(
                r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Law|"
                r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Code|"
                r"governing law|applicable law|laws? of (?:the\s+)?(?:State\s+of\s+)?[A-Z][A-Za-z]+|"
                r"jurisdiction of [A-Z][A-Za-z]+|state of|vietnam|singapore)\b",
                stripped,
                re.I,
            )
        )
    return True


def _regex_entities(text: str) -> Iterable[Entity]:
    for label, pattern, flags in PATTERNS:
        for match in re.finditer(pattern, text, flags=flags):
            entity = _make_entity(match.group(0), label, match.start(), match.end())
            if entity is not None and _looks_like_label(entity.text, label):
                yield entity


def _custom_entities(doc) -> Iterable[Entity]:
    allowed = {"PARTY", "PENALTY", "LAW"}
    for ent in doc.ents:
        if ent.label_ in allowed and _looks_like_label(ent.text, ent.label_):
            entity = _make_entity(ent.text, ent.label_, ent.start_char, ent.end_char)
            if entity is not None:
                yield entity


def _spacy_entities(doc) -> Iterable[Entity]:
    mapping = {
        "ORG": "PARTY",
        "PERSON": "PARTY",
        "GPE": "LAW",
        "NORP": "LAW",
    }
    for ent in doc.ents:
        label = mapping.get(ent.label_)
        if label:
            if _looks_like_label(ent.text, label):
                entity = _make_entity(ent.text, label, ent.start_char, ent.end_char)
                if entity is not None:
                    yield entity


def _dedupe(entities: Iterable[Entity]) -> list[Entity]:
    selected: list[Entity] = []
    for entity in sorted(entities, key=lambda item: (item.start_char, -(item.end_char - item.start_char))):
        overlaps = [
            idx
            for idx, current in enumerate(selected)
            if not (entity.end_char <= current.start_char or entity.start_char >= current.end_char)
        ]
        if not overlaps:
            selected.append(entity)
            continue
        largest = max([selected[idx] for idx in overlaps] + [entity], key=lambda item: item.end_char - item.start_char)
        for idx in reversed(overlaps):
            selected.pop(idx)
        selected.append(largest)
    return sorted(selected, key=lambda item: item.start_char)


def extract_entities(clause: str, nlp, legal_ner=None) -> list[dict[str, object]]:
    doc = nlp(clause)
    custom_doc = legal_ner(clause) if legal_ner is not None else None
    entities = _dedupe(
        [
            *_regex_entities(clause),
            *_spacy_entities(doc),
            *(_custom_entities(custom_doc) if custom_doc is not None else []),
        ]
    )
    return [
        {
            "text": entity.text,
            "label": entity.label,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
        }
        for entity in entities
    ]


def run_ner(clauses: list[str], nlp, legal_ner=None) -> list[dict[str, object]]:
    return [
        {
            "clause_id": idx,
            "clause": clause,
            "entities": extract_entities(clause, nlp, legal_ner),
        }
        for idx, clause in enumerate(clauses, start=1)
    ]
