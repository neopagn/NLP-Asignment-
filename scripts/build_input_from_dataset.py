import argparse
from pathlib import Path


TEXT_SUFFIXES = {".txt", ".md", ".csv"}


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
    import re

    cleaned = re.sub(r"\bE\s+mployee\b", "Employee", cleaned)
    cleaned = re.sub(r"\bP\s+lease\b", "Please", cleaned)
    cleaned = re.sub(r"\bw\s+eek\b", "week", cleaned)
    return cleaned


def read_candidate(path: Path) -> str:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return repair_text_encoding(path.read_text(encoding="utf-8", errors="ignore"))
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return repair_text_encoding("\n".join(page.extract_text() or "" for page in reader.pages))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            return ""
    return ""


def collect_contract_text(dataset_dirs: list[Path], max_files: int) -> str:
    parts: list[str] = []
    for dataset_dir in dataset_dirs:
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file():
                continue
            text = read_candidate(path).strip()
            if len(text.split()) < 50:
                continue
            try:
                source_name = f"{dataset_dir.name}/{path.relative_to(dataset_dir)}"
            except ValueError:
                source_name = path.name
            parts.append(f"\n\n===== SOURCE: {source_name} =====\n\n{text}")
            if len(parts) >= max_files:
                break
        if len(parts) >= max_files:
            break
    return "\n".join(parts).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build input/raw_contracts.txt from one or more contract dataset folders.")
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("input/raw_contracts.txt"))
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    text = collect_contract_text(args.dataset_dirs, args.max_files)
    if not text.strip():
        raise SystemExit("No readable text or PDF files found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    datasets = ", ".join(str(path) for path in args.dataset_dirs)
    print(f"Wrote {args.output} from {datasets}")


if __name__ == "__main__":
    main()
