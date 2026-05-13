import argparse
import json
import random
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch

from .config import DATA_DIR, MODEL_DIR
from .utils import ensure_dirs


def _find_span(text: str, phrase: str) -> tuple[int, int]:
    start = text.lower().find(phrase.lower())
    if start == -1:
        raise ValueError(f"Phrase {phrase!r} not found in {text!r}")
    return start, start + len(phrase)


def load_seed_examples(path: Path) -> list[tuple[str, dict[str, list[tuple[int, int, str]]]]]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        entities = []
        used: list[tuple[int, int]] = []
        for entity in item["entities"]:
            start, end = _find_span(item["text"], entity["text"])
            if any(not (end <= old_start or start >= old_end) for old_start, old_end in used):
                continue
            entities.append((start, end, entity["label"]))
            used.append((start, end))
        examples.append((item["text"], {"entities": entities}))
    return examples


def _load_base_model(name: str):
    try:
        return spacy.load(name)
    except OSError:
        return spacy.blank("en")


def train(seed_path: Path, output_dir: Path, iterations: int = 40, base_model: str = "en_core_web_sm") -> None:
    training_data = load_seed_examples(seed_path)
    nlp = _load_base_model(base_model)
    ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner")
    for _, annotations in training_data:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    if nlp.pipe_names == ["ner"]:
        optimizer = nlp.initialize()
    else:
        optimizer = nlp.resume_training()

    trainable_pipes = {"tok2vec", "ner"}
    disabled = [name for name in nlp.pipe_names if name not in trainable_pipes]
    with nlp.disable_pipes(*disabled):
        for _ in range(iterations):
            random.shuffle(training_data)
            losses = {}
            for batch in minibatch(training_data, size=4):
                examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                nlp.update(examples, sgd=optimizer, losses=losses, drop=0.2)

    ensure_dirs(output_dir)
    nlp.to_disk(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small custom legal NER model.")
    parser.add_argument("--seed", type=Path, default=DATA_DIR / "ner_seed.jsonl")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "legal_ner")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--base-model", default="en_core_web_sm")
    args = parser.parse_args()

    train(args.seed, args.output, args.iterations, args.base_model)
    print(f"Saved custom NER model to {args.output}")


if __name__ == "__main__":
    main()
