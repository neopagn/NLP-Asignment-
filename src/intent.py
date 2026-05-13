import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAINING_EXAMPLES = [
    ('"SERVICES" means the escrow services described in this agreement.', "Obligation"),
    ("This Agreement is made as of June 21, 1999 by and between the parties.", "Obligation"),
    ("This Agreement shall be governed by the laws of the State of California.", "Obligation"),
    ("Party B shall pay the rental fee before the fifth day of each month.", "Obligation"),
    ("The buyer must deliver payment within ten business days.", "Obligation"),
    ("The employee is required to maintain confidential information.", "Obligation"),
    ("The seller shall provide invoices upon request.", "Obligation"),
    ("The service provider will perform the services in a professional manner.", "Obligation"),
    ("The contractor agrees to comply with all applicable law.", "Obligation"),
    ("Neither party may disclose confidential information.", "Prohibition"),
    ("The tenant shall not assign this agreement without consent.", "Prohibition"),
    ("The contractor is prohibited from subcontracting the work.", "Prohibition"),
    ("No party may use the trademarks without prior written approval.", "Prohibition"),
    ("A party shall not transfer its rights without prior written approval.", "Prohibition"),
    ("Neither party shall be liable for indirect damages.", "Prohibition"),
    ("The buyer may inspect the goods before acceptance.", "Right"),
    ("Either party has the right to audit relevant records.", "Right"),
    ("The licensor hereby grants the licensee a right to use the mark.", "Right"),
    ("The employer may withhold approval at its discretion.", "Right"),
    ("The indemnified party shall have the right to control the defense.", "Right"),
    ("This agreement terminates automatically upon insolvency.", "Termination Condition"),
    ("Either party may terminate this agreement if the other party breaches a material term.", "Termination Condition"),
    ("The contract shall expire at the end of the term.", "Termination Condition"),
    ("Upon bankruptcy, the agreement will be terminated immediately.", "Termination Condition"),
    ("These provisions shall survive expiration or termination of this agreement.", "Termination Condition"),
]


DEFINITION_RE = re.compile(
    r'^\s*(?:\([a-zA-Z0-9]{1,4}\)\s*)?["“][^"”]+["”]\s+'
    r"(?:means|refers to|is defined as|is referred to)\b",
    re.IGNORECASE,
)
FORMATION_RE = re.compile(
    r"\b(?:agreement|contract)\s+is\s+made\b|\bby\s+and\s+between\b",
    re.IGNORECASE,
)
PROHIBITION_RE = re.compile(
    r"\b(?:shall\s+not|must\s+not|may\s+not|cannot|can\s+not|"
    r"is\s+prohibited|are\s+prohibited|prohibited\s+from|forbidden|"
    r"neither\s+party\s+(?:shall|may|will)|no\s+party\s+(?:shall|may|will))\b|"
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:approval|consent)\b",
    re.IGNORECASE,
)
TERMINATION_RE = re.compile(
    r"\b(?:terminate|terminates|terminated|termination|expires?|expiration|"
    r"bankruptcy|insolvency)\b|"
    r"\b(?:provisions?|sections?)\b.{0,80}\b(?:survive|survives)\b",
    re.IGNORECASE,
)
BREACH_TERMINATION_RE = re.compile(
    r"\bbreach\b(?=.{0,120}\b(?:terminate|termination|cure|default|notice)\b)|"
    r"\b(?:terminate|termination|cure|default|notice)\b(?=.{0,120}\bbreach\b)",
    re.IGNORECASE,
)
RIGHT_RE = re.compile(
    r"\b(?:has|have|shall\s+have)\s+the\s+right\b|"
    r"\bright\s+to\b|"
    r"\bhereby\s+grants?\b|"
    r"\bentitled\s+to\b|"
    r"\bat\s+its\s+(?:sole\s+)?discretion\b|"
    r"\boption\s+to\b|"
    r"\bmay\b(?!\s+not)(?![^.;]{0,80}\bterminate\b)",
    re.IGNORECASE,
)
OBLIGATION_RE = re.compile(
    r"\b(?:shall|must|will|agree(?:s)?\s+to|required\s+to|obligated\s+to|"
    r"responsible\s+for|pay|provide|deliver|perform|indemnify|reimburse|"
    r"comply|governed|construed|submit|maintain|return)\b",
    re.IGNORECASE,
)


@dataclass
class IntentClassifier:
    model: Pipeline

    @classmethod
    def train(cls, examples: Iterable[tuple[str, str]] | None = None) -> "IntentClassifier":
        training_examples = list(examples or TRAINING_EXAMPLES)
        texts = [text for text, _ in training_examples]
        labels = [label for _, label in training_examples]
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(texts, labels)
        return cls(model=model)

    def predict_ml(self, clause: str) -> str:
        return str(self.model.predict([clause])[0])

    def predict(self, clause: str) -> str:
        return rule_based_intent_label(clause) or self.predict_ml(clause)


class TransformerIntentClassifier:
    def __init__(self, model_path: Path, batch_size: int = 8, max_length: int = 160):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}

    def predict_many(self, clauses: list[str]) -> list[str]:
        labels: list[str] = []
        with self.torch.no_grad():
            for start in range(0, len(clauses), self.batch_size):
                batch = clauses[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                labels.extend(self.id2label[int(index)] for index in logits.argmax(dim=-1).tolist())
        return labels


def rule_based_intent_label(clause: str) -> str | None:
    """High-precision legal rules used before the statistical baseline."""
    text = clause.strip()
    if not text:
        return None
    if DEFINITION_RE.search(text) or FORMATION_RE.search(text):
        return "Obligation"
    if PROHIBITION_RE.search(text):
        return "Prohibition"
    if TERMINATION_RE.search(text) or BREACH_TERMINATION_RE.search(text):
        return "Termination Condition"
    if RIGHT_RE.search(text):
        return "Right"
    if OBLIGATION_RE.search(text):
        return "Obligation"
    return None


def load_transformer_intent_model(model_path: Path | None) -> TransformerIntentClassifier | None:
    if not model_path or not model_path.exists():
        return None
    try:
        return TransformerIntentClassifier(model_path)
    except (ImportError, OSError, ValueError):
        return None


def run_intent_classification(clauses: list[str], model_path: Path | None = None) -> list[dict[str, str]]:
    transformer = load_transformer_intent_model(model_path)
    if transformer:
        labels = transformer.predict_many(clauses)
        return [{"clause": clause, "label": label, "method": "legal_bert"} for clause, label in zip(clauses, labels)]

    classifier = IntentClassifier.train()
    return [{"clause": clause, "label": classifier.predict(clause), "method": "rules_tfidf"} for clause in clauses]


def format_intent_results(results: list[dict[str, str]]) -> str:
    return "\n".join(f"{item['clause']}\t{item['label']}" for item in results) + "\n"
