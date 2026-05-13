import argparse
from pathlib import Path

from .chunker import format_chunks
from .clause_splitter import split_clause_records
from .config import INPUT_DIR, MODEL_DIR, OUTPUT_DIR
from .dependency import parse_dependencies
from .intent import format_intent_results, run_intent_classification
from .ner import run_ner
from .srl import load_srl_model, run_srl
from .utils import ensure_dirs, load_optional_spacy_model, load_spacy_model, read_text, write_json, write_text


def run_pipeline(input_path: Path, output_dir: Path = OUTPUT_DIR) -> None:
    ensure_dirs(INPUT_DIR, output_dir)
    nlp = load_spacy_model()
    legal_ner = load_optional_spacy_model(MODEL_DIR / "legal_ner")
    srl_model_path = MODEL_DIR / "legal_srl_legalbert"
    if not srl_model_path.exists():
        srl_model_path = MODEL_DIR / "legal_srl"
    srl_model = load_srl_model(srl_model_path)

    raw_text = read_text(input_path)
    clause_records = split_clause_records(raw_text)
    clauses = [record.clause for record in clause_records]

    write_text(output_dir / "clauses.txt", "\n".join(clauses) + "\n")
    write_json(
        output_dir / "clauses_context.json",
        [
            {"clause_id": idx, "section": record.section, "clause": record.clause}
            for idx, record in enumerate(clause_records, start=1)
        ],
    )
    write_text(output_dir / "chunks.txt", format_chunks(clauses, nlp))
    write_json(output_dir / "dependency.json", parse_dependencies(clauses, nlp))

    ner_results = run_ner(clauses, nlp, legal_ner)
    write_json(output_dir / "ner_results.json", ner_results)

    srl_results = run_srl(clauses, ner_results, nlp, srl_model)
    write_json(output_dir / "srl_results.json", srl_results)

    intent_model_path = MODEL_DIR / "legal_intent_legalbert"
    intent_results = run_intent_classification(clauses, intent_model_path)
    write_text(output_dir / "intent_classification.txt", format_intent_results(intent_results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legal contract NLP pipeline.")
    parser.add_argument("--input", type=Path, default=INPUT_DIR / "raw_contracts.txt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
