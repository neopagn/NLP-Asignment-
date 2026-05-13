from dataclasses import dataclass


@dataclass
class ChunkToken:
    text: str
    tag: str


def _is_skipped_np_token(token) -> bool:
    return token.is_space or token.is_punct or token.text in {'"', "'", "`", "``", "''"}


def _is_np_boundary(token) -> bool:
    return token.dep_ == "cc" and token.lower_ in {"and", "or"}


def _is_mergeable_hyphen_part(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _can_merge_hyphenated(doc, index: int) -> bool:
    if index + 2 >= len(doc):
        return False
    left = doc[index]
    hyphen = doc[index + 1]
    right = doc[index + 2]
    return (
        hyphen.text == "-"
        and not left.whitespace_
        and not hyphen.whitespace_
        and _is_mergeable_hyphen_part(left.text)
        and _is_mergeable_hyphen_part(right.text)
    )


def _display_tag(raw_tags: list[str], previous_is_np: bool) -> str:
    raw_tag = next((tag for tag in raw_tags if tag != "O"), "O")
    if raw_tag == "O":
        return "O"
    if previous_is_np and raw_tag == "I-NP":
        return "I-NP"
    return "B-NP"


def _format_tokens(doc, tags: list[str]) -> list[ChunkToken]:
    output: list[ChunkToken] = []
    previous_is_np = False
    index = 0

    while index < len(doc):
        token = doc[index]
        if token.is_space:
            index += 1
            continue

        if _can_merge_hyphenated(doc, index):
            text = f"{doc[index].text}-{doc[index + 2].text}"
            tag = _display_tag([tags[index], tags[index + 2]], previous_is_np)
            index += 3
        else:
            text = token.text
            tag = _display_tag([tags[index]], previous_is_np)
            index += 1

        output.append(ChunkToken(text, tag))
        previous_is_np = tag != "O"

    return output


def noun_phrase_iob(clause: str, nlp) -> list[ChunkToken]:
    doc = nlp(clause)
    tags = ["O"] * len(doc)

    for chunk in doc.noun_chunks:
        segment: list[int] = []

        def apply_segment() -> None:
            if not segment:
                return
            tags[segment[0]] = "B-NP"
            for index in segment[1:]:
                tags[index] = "I-NP"

        for token in chunk:
            if _is_np_boundary(token):
                apply_segment()
                segment = []
                continue
            if _is_skipped_np_token(token):
                apply_segment()
                segment = []
                continue
            segment.append(token.i)
        apply_segment()

    return _format_tokens(doc, tags)


def format_chunks(clauses: list[str], nlp) -> str:
    blocks: list[str] = []
    for clause in clauses:
        lines = [f"{item.text}\t{item.tag}" for item in noun_phrase_iob(clause, nlp)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
