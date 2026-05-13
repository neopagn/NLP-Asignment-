import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.srl import run_srl
from src.utils import load_spacy_model, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rule-only SRL baseline from existing clause and NER outputs.")
    parser.add_argument("--clauses", type=Path, default=OUTPUT_DIR / "clauses.txt")
    parser.add_argument("--ner", type=Path, default=OUTPUT_DIR / "ner_results.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "srl_rule_baseline_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clauses = [line.strip() for line in args.clauses.read_text(encoding="utf-8").splitlines() if line.strip()]
    ner_results = json.loads(args.ner.read_text(encoding="utf-8"))
    nlp = load_spacy_model()
    baseline = run_srl(clauses, ner_results, nlp, srl_model=None)
    write_json(args.output, baseline)
    print(f"wrote={args.output}")
    print(f"rows={len(baseline)}")


if __name__ == "__main__":
    main()
