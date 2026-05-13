import json
import re
from pathlib import Path


OUTPUT_DIR = Path("output")
ENTITY_LABELS = {"PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"}
INTENT_LABELS = {"Obligation", "Prohibition", "Right", "Termination Condition"}
DEPENDENCY_TOKEN_KEYS = {"token", "head", "dependency"}
PUNCT_NP_TOKENS = {"(", ")", "[", "]", "{", "}", '"', "'", "`", "``", "''", "-", ",", ".", ";", ":"}
MONEY_VALID_RE = re.compile(
    r"^(?:(?:USD|VND|EUR|\$)\s*\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:USD|VND|EUR|dollars?|dong))$",
    re.IGNORECASE,
)
LEGAL_DATE_TERM_RE = re.compile(r"^(?:effective|launch|approval|expiration|payment|due|adjusted)\s+date$|^term$", re.IGNORECASE)
REQUIRED = [
    "clauses.txt",
    "chunks.txt",
    "dependency.json",
    "ner_results.json",
    "srl_results.json",
    "intent_classification.txt",
]


def main() -> None:
    missing = [name for name in REQUIRED if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise SystemExit(f"Missing output files: {', '.join(missing)}")

    clauses = [line.strip() for line in (OUTPUT_DIR / "clauses.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    dependencies = json.loads((OUTPUT_DIR / "dependency.json").read_text(encoding="utf-8"))
    ner_results = json.loads((OUTPUT_DIR / "ner_results.json").read_text(encoding="utf-8"))
    srl_results = json.loads((OUTPUT_DIR / "srl_results.json").read_text(encoding="utf-8"))
    intents = [line for line in (OUTPUT_DIR / "intent_classification.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    clauses_context_path = OUTPUT_DIR / "clauses_context.json"
    clauses_context = json.loads(clauses_context_path.read_text(encoding="utf-8")) if clauses_context_path.exists() else []

    counts = {
        "clauses": len(clauses),
        "dependency": len(dependencies),
        "ner": len(ner_results),
        "srl": len(srl_results),
        "intent": len(intents),
    }
    if len(set(counts.values())) != 1:
        raise SystemExit(f"Output counts do not match: {counts}")
    if clauses_context and len(clauses_context) != len(clauses):
        raise SystemExit(f"Clause context count does not match clauses: {len(clauses_context)} != {len(clauses)}")
    if not clauses:
        raise SystemExit("No clauses were produced.")
    if not all(item.get("tokens") for item in dependencies):
        raise SystemExit("At least one dependency entry has no tokens.")

    chunk_lines = [
        line
        for line in (OUTPUT_DIR / "chunks.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad_chunks = [line for line in chunk_lines if len(line.split("\t")) != 2 or line.rsplit("\t", 1)[1] not in {"B-NP", "I-NP", "O"}]
    if bad_chunks:
        raise SystemExit(f"Invalid IOB chunk lines, for example: {bad_chunks[:3]}")
    bad_punct_chunks = []
    bad_iob_sequences = []
    previous_is_np = False
    for raw_line in (OUTPUT_DIR / "chunks.txt").read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            previous_is_np = False
            continue
        if "\t" not in raw_line:
            continue
        token, tag = raw_line.rsplit("\t", 1)
        if tag == "I-NP" and not previous_is_np:
            bad_iob_sequences.append(raw_line)
        previous_is_np = tag in {"B-NP", "I-NP"}
    if bad_iob_sequences:
        raise SystemExit(f"I-NP tag appears without a preceding NP tag: {bad_iob_sequences[:3]}")
    for line in chunk_lines:
        token, tag = line.rsplit("\t", 1)
        if token in PUNCT_NP_TOKENS and tag != "O":
            bad_punct_chunks.append(line)
    if bad_punct_chunks:
        raise SystemExit(f"Punctuation included inside noun phrases: {bad_punct_chunks[:3]}")

    for entry in dependencies:
        for token in entry.get("tokens", []):
            if not DEPENDENCY_TOKEN_KEYS.issubset(token):
                raise SystemExit(f"Dependency token is missing required keys: {token}")
            if token.get("token") in PUNCT_NP_TOKENS and token.get("pos") != "PUNCT":
                raise SystemExit(f"Standalone punctuation has non-punctuation POS: {token}")

    for entry in ner_results:
        for entity in entry.get("entities", []):
            if entity.get("label") not in ENTITY_LABELS:
                raise SystemExit(f"Unexpected NER label: {entity}")
            for key in ("text", "start_char", "end_char"):
                if key not in entity:
                    raise SystemExit(f"NER entity missing {key}: {entity}")
            if entity.get("label") == "MONEY" and not MONEY_VALID_RE.fullmatch(str(entity.get("text"))):
                raise SystemExit(f"Suspicious MONEY entity: {entity}")
            if entity.get("label") == "DATE" and LEGAL_DATE_TERM_RE.fullmatch(str(entity.get("text"))):
                raise SystemExit(f"Legal term mislabeled as DATE: {entity}")
            if entity.get("label") == "PENALTY" and str(entity.get("text", "")).lower() == "interest":
                raise SystemExit(f"Generic legal interest mislabeled as PENALTY: {entity}")
            if entity.get("label") == "LAW" and str(entity.get("text", "")).lower() == "laws and regulations":
                raise SystemExit(f"Generic laws/regulations phrase mislabeled as LAW: {entity}")

    for entry in srl_results:
        if "predicate" not in entry or "roles" not in entry:
            raise SystemExit(f"SRL entry missing predicate or roles: {entry}")

    for line in intents:
        parts = line.rsplit("\t", 1)
        if len(parts) != 2 or parts[1] not in INTENT_LABELS:
            raise SystemExit(f"Invalid intent line: {line}")

    print("Validation passed")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
