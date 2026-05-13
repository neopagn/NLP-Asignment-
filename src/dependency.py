from typing import Any


def parse_dependencies(clauses: list[str], nlp) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for clause_id, clause in enumerate(clauses, start=1):
        doc = nlp(clause)
        tokens = []
        for token in doc:
            if token.is_space:
                continue
            tokens.append(
                {
                    "id": token.i + 1,
                    "token": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "head": 0 if token.head == token else token.head.i + 1,
                    "head_token": "ROOT" if token.head == token else token.head.text,
                    "dependency": token.dep_,
                }
            )
        results.append({"clause_id": clause_id, "clause": clause, "tokens": tokens})

    return results
