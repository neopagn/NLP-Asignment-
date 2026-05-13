import json
import re
from pathlib import Path
from typing import Any

from spacy.language import Language
from spacy.symbols import ORTH


LEGAL_CORPORATE_SUFFIXES = [
    "INC.",
    "Inc.",
    "inc.",
    "CORP.",
    "Corp.",
    "corp.",
    "CO.",
    "Co.",
    "co.",
    "LTD.",
    "Ltd.",
    "ltd.",
    "LLC.",
    "L.L.C.",
]
LEGAL_DOTTED_ABBREVIATIONS = [
    "U.S.",
    "U.S.A.",
    "E.U.",
    "e.g.",
    "i.e.",
]
LEGAL_TOKEN_MATCH_RE = re.compile(
    r"(?:"
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+"
    r"|(?=.*[A-Za-z])[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+"
    r"|\d+(?:\.\d+)+"
    r"|(?:[A-Za-z]\.){2,}"
    r")$"
)
LEGAL_PROPN_RE = re.compile(
    r"(?:"
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+"
    r"|(?=.*[A-Za-z])[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+"
    r"|[0-9]+[A-Za-z][A-Za-z0-9]*"
    r")$"
)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return repair_text_encoding(path.read_text(encoding="utf-8", errors="ignore"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    text = repair_text_encoding(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_text_encoding(text: str) -> str:
    candidates = [text]
    for encoding in ("cp1252", "latin1"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except UnicodeError:
            continue

    def score(candidate: str) -> int:
        return candidate.count("\ufffd") * 5 + candidate.count("â") + candidate.count("Ã")

    cleaned = min(candidates, key=score)
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u2610": "",
        "\u2611": "",
        "\u2612": "",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"\bE\s+mployee\b", "Employee", cleaned)
    cleaned = re.sub(r"\bP\s+lease\b", "Please", cleaned)
    cleaned = re.sub(r"\bw\s+eek\b", "week", cleaned)
    return cleaned


def _is_legal_proper_noun(text: str) -> bool:
    return text in LEGAL_CORPORATE_SUFFIXES or bool(LEGAL_PROPN_RE.fullmatch(text))


@Language.component("legal_token_attrs")
def legal_token_attrs(doc):
    propn = doc.vocab.strings["PROPN"]
    nnp = doc.vocab.strings["NNP"]
    punct = doc.vocab.strings["PUNCT"]
    punct_tag = doc.vocab.strings["."]

    for token in doc:
        if token.text in {".", ",", ";", ":", "(", ")", "[", "]", "{", "}", '"', "'"}:
            token.pos = punct
            token.tag = punct_tag
        elif _is_legal_proper_noun(token.text):
            token.pos = propn
            token.tag = nnp
            token.lemma = doc.vocab.strings[token.text]
    return doc


def configure_legal_nlp(nlp):
    original_token_match = nlp.tokenizer.token_match

    def legal_token_match(text: str) -> bool:
        return bool(LEGAL_TOKEN_MATCH_RE.fullmatch(text)) or bool(
            original_token_match and original_token_match(text)
        )

    nlp.tokenizer.token_match = legal_token_match
    for text in [*LEGAL_CORPORATE_SUFFIXES, *LEGAL_DOTTED_ABBREVIATIONS]:
        nlp.tokenizer.add_special_case(text, [{ORTH: text}])

    if "legal_token_attrs" not in nlp.pipe_names:
        if "parser" in nlp.pipe_names:
            nlp.add_pipe("legal_token_attrs", after="parser")
        else:
            nlp.add_pipe("legal_token_attrs", last=True)
    return nlp


def load_spacy_model():
    import spacy

    try:
        return configure_legal_nlp(spacy.load("en_core_web_sm"))
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. Run: "
            ".\\.venv\\Scripts\\python.exe -m spacy download en_core_web_sm"
        ) from exc


def load_optional_spacy_model(path: Path):
    import spacy

    if not path.exists():
        return None
    return configure_legal_nlp(spacy.load(str(path)))
