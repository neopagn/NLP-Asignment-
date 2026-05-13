import os
import re
from pathlib import Path
from typing import Any


ACTIVE_SUBJECT_DEPS = {"nsubj", "agent"}
PASSIVE_SUBJECT_DEPS = {"nsubjpass"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}
TIME_PREPS = {"before", "after", "on", "within", "until", "during", "following"}
RECIPIENT_PREPS = {"to", "for"}
LEGAL_BASIS_PREDICATES = {"comply", "govern", "construe", "resolve", "prevail", "conflict"}
TIME_UNITS = {"day", "days", "week", "weeks", "month", "months", "year", "years"}
RECORD_KEEPING_PREDICATES = {"keep", "maintain", "retain", "preserve"}
CONDITION_MARKERS = (
    "if ",
    "unless ",
    "provided that ",
    "subject to ",
    "in the event that ",
    "upon ",
    "except as otherwise provided",
    "notwithstanding ",
    "should ",
)
DEFINITION_RE = re.compile(
    "^\\s*(?P<label>\\([^)]+\\)\\s*)?[\"\\u201c](?P<term>[^\"\\u201d]+)[\"\\u201d]\\s+"
    "(?P<predicate>means|refers to|is defined as|is referred to)\\s+"
    "(?P<definition>.+?)[.!?]?\\s*$",
    re.IGNORECASE,
)
CONDITION_PREFIX_PATTERNS = [
    re.compile(r"^(?P<condition>Should\b.+),\s+(?P<main>[^,]+?\bshall\b.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>If\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Unless\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Provided that\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Subject to\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>In the event that\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Upon\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Should\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Except as otherwise provided),(?P<main>\s+.+)$", re.IGNORECASE),
    re.compile(r"^(?P<condition>Notwithstanding\b.+?),(?P<main>\s+.+)$", re.IGNORECASE),
]
MODEL_ROLE_MAP = {
    "ARG0": "Agent",
    "ARG1": "Theme",
    "ARG2": "Recipient",
    "ARG3": "Recipient",
    "ARGM-TMP": "Time",
    "ARGM-LOC": "Location",
    "ARGM-CAU": "Condition",
    "ARGM-CND": "Condition",
    "ARGM-ADV": "Condition",
    "V": "Predicate",
}
MODEL_DIRECT_ROLES = {
    "Agent",
    "Theme",
    "Recipient",
    "Time",
    "Condition",
    "Location",
    "LegalBasis",
    "DefinedTerm",
    "Definition",
    "ContractParties",
    "Predicate",
}
MODEL_SUPPLEMENTAL_ROLES = {"Location"}


class TransformerSRLModel:
    def __init__(self, model_path: Path | str, min_score: float = 0.55):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.min_score = min_score

    def _role_for_label(self, raw_label: str) -> str | None:
        label = raw_label.removeprefix("B-").removeprefix("I-")
        role = MODEL_ROLE_MAP.get(label)
        if role is None and label in MODEL_DIRECT_ROLES:
            role = label
        return role

    def _predict_items(self, clause: str, predicate_text: str | None = None) -> list[dict[str, Any]]:
        import torch

        encoded = self.tokenizer(
            clause,
            predicate_text or "",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            probabilities = torch.softmax(self.model(**encoded).logits[0], dim=-1)
        label_ids = probabilities.argmax(dim=-1).tolist()
        scores = probabilities.max(dim=-1).values.tolist()

        items: list[dict[str, Any]] = []
        active_role: str | None = None
        active_start: int | None = None
        active_end: int | None = None
        active_scores: list[float] = []

        def flush() -> None:
            nonlocal active_role, active_start, active_end, active_scores
            if active_role and active_start is not None and active_end is not None:
                average_score = sum(active_scores) / max(len(active_scores), 1)
                if average_score >= self.min_score:
                    items.append(
                        {
                            "role": active_role,
                            "text": clause[active_start:active_end].strip(),
                            "start": active_start,
                            "end": active_end,
                            "score": average_score,
                        }
                    )
            active_role = None
            active_start = None
            active_end = None
            active_scores = []

        for label_id, score, offset, sequence_id in zip(label_ids, scores, offsets, sequence_ids):
            start, end = offset
            if sequence_id != 0 or start == end:
                flush()
                continue

            raw_label = self.model.config.id2label[int(label_id)]
            if raw_label == "O":
                flush()
                continue

            role = self._role_for_label(raw_label)
            if role is None:
                flush()
                continue

            prefix = raw_label.split("-", 1)[0] if "-" in raw_label else "B"
            if prefix == "B" or role != active_role:
                flush()
                active_role = role
                active_start = start
                active_end = end
                active_scores = [float(score)]
            else:
                active_end = end
                active_scores.append(float(score))
        flush()
        return [item for item in items if item["text"]]

    def predict(self, clause: str, predicate_text: str | None = None) -> dict[str, Any]:
        outputs = self._predict_items(clause, predicate_text)
        roles: dict[str, str] = {}
        model_predicate_text: str | None = None

        for item in outputs:
            role = str(item.get("role") or "")
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            if role == "Predicate":
                model_predicate_text = text
            else:
                if role in roles and text not in roles[role].split("; "):
                    roles[role] = f"{roles[role]}; {text}"
                else:
                    roles.setdefault(role, text)

        if not roles and model_predicate_text is None:
            return {}
        return {"predicate": None, "predicate_text": model_predicate_text or predicate_text, "roles": roles, "method": "transformer_srl"}


def load_srl_model(model_path: Path | None = None):
    configured = os.environ.get("LEGAL_SRL_MODEL")
    candidate = Path(configured) if configured else model_path
    if candidate is None or not candidate.exists():
        return None
    try:
        return TransformerSRLModel(candidate)
    except Exception:
        return None


def _span_text(token) -> str:
    left = min(child.i for child in token.subtree)
    right = max(child.i for child in token.subtree)
    doc = token.doc
    return doc[left : right + 1].text


def _noun_chunk_text(token) -> str:
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk.text
    return token.text


def _main_predicate(doc):
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            return token
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in {"NOUN", "ADJ"}:
            return token
    for token in doc:
        if token.pos_ == "VERB":
            return token
    return None


def _entities_by_label(entities: list[dict[str, Any]], label: str) -> list[str]:
    return [str(entity.get("text")) for entity in entities if entity.get("label") == label]


def _first_entity(entities: list[dict[str, Any]], label: str) -> str | None:
    values = _entities_by_label(entities, label)
    return values[0] if values else None


def _join_entities(entities: list[dict[str, Any]], label: str) -> str | None:
    values = _entities_by_label(entities, label)
    return "; ".join(values) if values else None


def _definition_roles(clause: str) -> dict[str, Any] | None:
    match = DEFINITION_RE.match(clause)
    if not match:
        return None

    definition = match.group("definition").strip()
    if definition and definition[-1] not in ".!?":
        definition += "."
    roles = {
        "DefinedTerm": match.group("term").strip(),
        "Definition": definition,
        "Theme": definition,
    }
    label = (match.group("label") or "").strip()
    if label:
        roles["DefinitionLabel"] = label
    predicate_text = match.group("predicate")
    return {
        "predicate": predicate_text.lower().split()[0],
        "predicate_text": predicate_text,
        "roles": roles,
        "method": "legal_definition_rule",
    }


def _condition_prefix(clause: str) -> str | None:
    for pattern in CONDITION_PREFIX_PATTERNS:
        match = pattern.match(clause)
        if match:
            return match.group("condition").strip()
    if clause.lower().startswith(CONDITION_MARKERS):
        return clause
    return None


def _prep_object_text(prep) -> str | None:
    objects = [child for child in prep.children if child.dep_ in {"pobj", "pcomp"}]
    values: list[str] = []
    for obj in objects:
        values.append(_span_text(obj))
        for conj in obj.conjuncts:
            values.append(_span_text(conj))
    return "; ".join(values) if values else None


def _clean_argument_text(value: str) -> str:
    value = re.sub(r"\s*:\s*\([a-zA-Z0-9]+\)\s*$", "", value)
    return value.strip(" ,;:")


def _strip_time_prefix(value: str, date_entities: list[str]) -> str:
    cleaned = value.strip()
    for date in date_entities:
        bare_date = re.sub(r"^(?:for|within|during|after|before|on|until|following)\s+", "", date, flags=re.IGNORECASE)
        patterns = [
            rf"^(?:for|within|during|after|before|on|until|following)\s+{re.escape(bare_date)}\s+",
            rf"^{re.escape(bare_date)}\s+",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,;:")


def _aux_subject_text(predicate) -> str | None:
    for child in predicate.children:
        if child.dep_ == "aux":
            for grandchild in child.children:
                if grandchild.dep_ in ACTIVE_SUBJECT_DEPS:
                    return _noun_chunk_text(grandchild)
    return None


def _salvage_record_theme(predicate, date_entities: list[str]) -> str | None:
    for token in predicate.doc[predicate.i + 1 :]:
        if token.lower_ in TIME_UNITS:
            continue
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        if token.dep_ not in {"dobj", "obj", "nsubj", "conj", "pobj"}:
            continue
        value = _strip_time_prefix(_span_text(token), date_entities)
        if value and any(word in value.lower() for word in ("record", "records", "book", "books")):
            return value
    return None


def _verb_mark_condition(predicate) -> str | None:
    for child in predicate.children:
        if child.dep_ == "advcl":
            markers = [grand for grand in child.children if grand.dep_ == "mark"]
            if markers and markers[0].lower_ in {"if", "unless", "when", "where", "provided", "should"}:
                return _span_text(child)
    return None


def _additional_predicates(predicate) -> list[str]:
    values: list[str] = []
    for child in predicate.children:
        if child.dep_ in {"conj", "xcomp", "ccomp", "advcl"} and child.pos_ in {"VERB", "AUX"}:
            values.append(child.lemma_)
    return values


def _looks_like_means_or_notice(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("providing ", "giving ", "sending ")) or "written notice" in lowered


def _merge_model_and_rule_results(model_result: dict[str, Any], rule_result: dict[str, Any]) -> dict[str, Any]:
    roles = dict(rule_result.get("roles", {}))
    for role, value in dict(model_result.get("roles", {})).items():
        if role not in roles and role in MODEL_SUPPLEMENTAL_ROLES:
            roles[role] = value
    return {
        "predicate": model_result.get("predicate") or rule_result.get("predicate"),
        "predicate_text": model_result.get("predicate_text") or rule_result.get("predicate_text"),
        "roles": roles,
        "method": "transformer_srl+legal_rules",
    }


def _semantic_roles_from_rules(clause: str, entities: list[dict[str, Any]], nlp) -> dict[str, Any]:
    definition_result = _definition_roles(clause)
    if definition_result is not None:
        return definition_result

    doc = nlp(clause)
    predicate = _main_predicate(doc)
    roles: dict[str, str] = {}

    if predicate is not None:
        for child in predicate.children:
            if child.dep_ in ACTIVE_SUBJECT_DEPS:
                roles.setdefault("Agent", _noun_chunk_text(child))
            elif child.dep_ in PASSIVE_SUBJECT_DEPS:
                roles.setdefault("Theme", _noun_chunk_text(child))
            elif child.dep_ in OBJECT_DEPS:
                roles.setdefault("Theme", _clean_argument_text(_span_text(child)))
            elif child.dep_ == "prep" and child.lemma_.lower() in RECIPIENT_PREPS:
                value = _prep_object_text(child)
                if value:
                    roles.setdefault("Recipient", value)
            elif child.dep_ == "prep" and child.lemma_.lower() in TIME_PREPS:
                value = _prep_object_text(child)
                if value:
                    roles.setdefault("Time", value)
            elif child.dep_ == "prep" and child.lemma_.lower() in {"by", "between"}:
                value = _prep_object_text(child)
                if value and "Agent" not in roles and not _looks_like_means_or_notice(value):
                    roles["Agent"] = value
            elif child.dep_ == "advcl":
                condition = _verb_mark_condition(predicate)
                if condition:
                    roles.setdefault("Condition", condition)

        additional = _additional_predicates(predicate)
        if additional:
            roles["AdditionalPredicates"] = "; ".join(additional)

        if "Agent" not in roles:
            agent = _aux_subject_text(predicate)
            if agent:
                roles["Agent"] = agent

    if "Condition" not in roles:
        condition = _condition_prefix(clause)
        if condition:
            roles["Condition"] = condition

    date_values = _entities_by_label(entities, "DATE")
    date = date_values[0] if date_values else None
    if date:
        if "Time" not in roles or _looks_like_means_or_notice(roles["Time"]):
            roles["Time"] = date

    law = _first_entity(entities, "LAW")
    predicate_lemma = predicate.lemma_.lower() if predicate is not None else ""
    if law and predicate_lemma in LEGAL_BASIS_PREDICATES:
        roles.setdefault("LegalBasis", law)

    parties = _join_entities(entities, "PARTY")
    if parties and ("by and between" in clause.lower() or predicate_lemma in {"make", "execute"}):
        roles["ContractParties"] = parties
        roles["Agent"] = parties

    if "Agent" not in roles:
        party = _first_entity(entities, "PARTY")
        if party and not roles.get("DefinedTerm"):
            roles["Agent"] = party

    if "Theme" not in roles:
        if predicate is not None and predicate_lemma in RECORD_KEEPING_PREDICATES:
            theme = _salvage_record_theme(predicate, date_values)
            if theme:
                roles["Theme"] = theme

    if "Theme" not in roles:
        for entity in entities:
            if entity.get("label") in {"MONEY", "PENALTY"}:
                roles["Theme"] = str(entity.get("text"))
                break
        if "Theme" not in roles and law and predicate_lemma not in LEGAL_BASIS_PREDICATES:
            roles["Theme"] = law

    return {
        "predicate": predicate.lemma_ if predicate is not None else None,
        "predicate_text": predicate.text if predicate is not None else None,
        "roles": roles,
        "method": "legal_dependency_rules",
    }


def semantic_roles(clause: str, entities: list[dict[str, Any]], nlp, srl_model=None) -> dict[str, Any]:
    definition_result = _definition_roles(clause)
    if definition_result is not None:
        return definition_result

    rule_result = _semantic_roles_from_rules(clause, entities, nlp)
    if srl_model is not None:
        model_result = srl_model.predict(clause, rule_result.get("predicate_text"))
        if model_result:
            return _merge_model_and_rule_results(model_result, rule_result)

    return rule_result


def run_srl(clauses: list[str], ner_results: list[dict[str, Any]], nlp, srl_model=None) -> list[dict[str, Any]]:
    by_id = {item["clause_id"]: item for item in ner_results}
    results = []
    for idx, clause in enumerate(clauses, start=1):
        entities = by_id.get(idx, {}).get("entities", [])
        roles = semantic_roles(clause, entities, nlp, srl_model)
        results.append({"clause_id": idx, "clause": clause, **roles})
    return results
