import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "srl_gold"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "legal_srl_legalbert"
DEFAULT_BASE_MODEL = "nlpaueb/bert-base-uncased-contracts"
ROLE_ORDER = [
    "Predicate",
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


class SRLDataset(Dataset):
    def __init__(self, features: list[dict[str, torch.Tensor]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _role_labels() -> list[str]:
    labels = ["O"]
    for role in ROLE_ORDER:
        labels.extend([f"B-{role}", f"I-{role}"])
    return labels


def _char_roles(entry: dict[str, Any]) -> list[str]:
    clause = entry["clause"]
    char_roles = ["O"] * len(clause)

    predicate = entry.get("predicate") or {}
    if isinstance(predicate.get("start"), int) and isinstance(predicate.get("end"), int):
        for index in range(predicate["start"], min(predicate["end"], len(char_roles))):
            char_roles[index] = "Predicate"

    for role in entry.get("roles", []):
        label = role.get("label")
        start = role.get("start")
        end = role.get("end")
        if label not in ROLE_ORDER or not isinstance(start, int) or not isinstance(end, int):
            continue
        for index in range(start, min(end, len(char_roles))):
            if char_roles[index] == "O":
                char_roles[index] = label
    return char_roles


def _token_role(char_roles: list[str], start: int, end: int) -> str:
    roles = [role for role in char_roles[start:end] if role != "O"]
    if not roles:
        return "O"
    return Counter(roles).most_common(1)[0][0]


def _bio_label(role: str, previous_role: str) -> str:
    if role == "O":
        return "O"
    prefix = "B" if previous_role != role else "I"
    return f"{prefix}-{role}"


def encode_examples(
    entries: list[dict[str, Any]],
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    features: list[dict[str, torch.Tensor]] = []

    for entry in entries:
        clause = entry["clause"]
        predicate_text = str((entry.get("predicate") or {}).get("text") or "")
        char_roles = _char_roles(entry)
        encoded = tokenizer(
            clause,
            predicate_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        sequence_ids = encoded.sequence_ids()

        labels: list[int] = []
        token_labels: list[str] = []
        previous_role = "O"
        for offset, sequence_id in zip(offsets, sequence_ids):
            start, end = offset
            if sequence_id != 0 or start == end:
                labels.append(-100)
                token_labels.append("IGN")
                previous_role = "O"
                continue
            role = _token_role(char_roles, start, end)
            label = _bio_label(role, previous_role)
            labels.append(label2id[label])
            token_labels.append(label)
            previous_role = role

        feature = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if "token_type_ids" in encoded:
            feature["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        features.append(feature)
    return features


def _loader(dataset: SRLDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _token_accuracy(model, dataset: SRLDataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, shuffle=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            predictions = logits.argmax(dim=-1)
            mask = labels != -100
            correct += int((predictions[mask] == labels[mask]).sum().item())
            total += int(mask.sum().item())
    return correct / total if total else 0.0


def _train(
    model,
    train_dataset: SRLDataset,
    dev_dataset: SRLDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loader = _loader(train_dataset, batch_size, shuffle=True)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())
        train_loss = total_loss / max(len(loader), 1)
        train_accuracy = _token_accuracy(model, train_dataset, batch_size, device)
        dev_accuracy = _token_accuracy(model, dev_dataset, batch_size, device) if len(dev_dataset) else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_token_accuracy": train_accuracy,
                "dev_token_accuracy": dev_accuracy,
            }
        )
        print(
            f"epoch={epoch} loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} dev_acc={dev_accuracy:.4f}"
        )
    return history


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT for predicate-conditioned legal SRL.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    train_rows = _load_jsonl(args.data_dir / "train.jsonl")
    dev_rows = _load_jsonl(args.data_dir / "dev.jsonl")
    test_rows = _load_jsonl(args.data_dir / "test.jsonl")

    labels = _role_labels()
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = SRLDataset(encode_examples(train_rows, tokenizer, label2id, args.max_length))
    dev_dataset = SRLDataset(encode_examples(dev_rows, tokenizer, label2id, args.max_length))
    test_dataset = SRLDataset(encode_examples(test_rows, tokenizer, label2id, args.max_length))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = _train(model, train_dataset, dev_dataset, args.epochs, args.batch_size, args.learning_rate, device)
    test_accuracy = _token_accuracy(model, test_dataset, args.batch_size, device)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "base_model": args.base_model,
        "input_format": "predicate_pair",
        "data_dir": str(args.data_dir),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "test_examples": len(test_rows),
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": str(device),
        "history": history,
        "test_token_accuracy": test_accuracy,
        "note": "Predicate-conditioned Legal-BERT SRL. The clause is sequence A and the predicate text is sequence B.",
    }
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved={args.output}")
    print(f"test_token_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
