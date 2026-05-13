import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .config import INTENT_LABELS, MODEL_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "intent_silver"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "legal_intent_legalbert"
DEFAULT_BASE_MODEL = "nlpaueb/bert-base-uncased-contracts"


class IntentDataset(Dataset):
    def __init__(self, features: list[dict[str, torch.Tensor]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode(rows: list[dict[str, Any]], tokenizer, label2id: dict[str, int], max_length: int) -> IntentDataset:
    features = []
    for row in rows:
        encoded = tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        feature = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label2id[row["label"]], dtype=torch.long),
        }
        if "token_type_ids" in encoded:
            feature["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        features.append(feature)
    return IntentDataset(features)


def _loader(dataset: IntentDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _accuracy(model, dataset: IntentDataset, batch_size: int, device: torch.device) -> float:
    if not len(dataset):
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, shuffle=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else 0.0


def _train(
    model,
    train_dataset: IntentDataset,
    dev_dataset: IntentDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else None
    model.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loader = _loader(train_dataset, batch_size, shuffle=True)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            if loss_function is None:
                output = model(**batch, labels=labels)
                loss = output.loss
            else:
                output = model(**batch)
                loss = loss_function(output.logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(loader), 1)
        train_accuracy = _accuracy(model, train_dataset, batch_size, device)
        dev_accuracy = _accuracy(model, dev_dataset, batch_size, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "dev_accuracy": dev_accuracy,
            }
        )
        print(f"epoch={epoch} loss={train_loss:.4f} train_acc={train_accuracy:.4f} dev_acc={dev_accuracy:.4f}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT for legal clause intent classification.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    label2id = {label: index for index, label in enumerate(INTENT_LABELS)}
    id2label = {index: label for label, index in label2id.items()}

    train_rows = _load_jsonl(args.data_dir / "train.jsonl")
    dev_rows = _load_jsonl(args.data_dir / "dev.jsonl")
    test_rows = _load_jsonl(args.data_dir / "test.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=args.local_files_only)
    config = AutoConfig.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = len(INTENT_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
        use_safetensors=False,
    )

    train_dataset = _encode(train_rows, tokenizer, label2id, args.max_length)
    dev_dataset = _encode(dev_rows, tokenizer, label2id, args.max_length)
    test_dataset = _encode(test_rows, tokenizer, label2id, args.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = None
    if not args.no_class_weights:
        counts = Counter(row["label"] for row in train_rows)
        class_weights = torch.tensor(
            [len(train_rows) / (len(INTENT_LABELS) * max(counts[label], 1)) for label in INTENT_LABELS],
            dtype=torch.float,
        )
    history = _train(
        model,
        train_dataset,
        dev_dataset,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        device,
        class_weights,
    )
    test_accuracy = _accuracy(model, test_dataset, args.batch_size, device)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "base_model": args.base_model,
        "data_dir": str(args.data_dir),
        "labels": INTENT_LABELS,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "test_examples": len(test_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "history": history,
        "test_accuracy": test_accuracy,
    }
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
