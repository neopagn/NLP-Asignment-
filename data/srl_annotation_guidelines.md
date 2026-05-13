# SRL Gold Annotation Guidelines

Annotate one predicate frame per row. If a clause has multiple important predicates, duplicate the row and annotate one predicate at a time.

## Roles

- `Agent`: the party or actor performing the predicate.
- `Theme`: the object, agreement, service, information, money, breach, or legal item affected by the predicate.
- `Recipient`: the party receiving something, usually after `to` or `for`.
- `Time`: deadlines, durations, dates, notice periods, or effective periods.
- `Condition`: conditional trigger, exception, or prerequisite, such as `if ...`, `unless ...`, `in the event that ...`, `provided that ...`.
- `LegalBasis`: named law, legal code, governing law, or jurisdiction basis.
- `DefinedTerm`: the term being defined in a definition clause.
- `Definition`: the text defining a `DefinedTerm`.
- `ContractParties`: parties named in contract formation clauses such as `by and between ...`.
- `Location`: location or forum only when it is semantically important.

## Legal-Specific Decisions

- Use `ContractParties` only for formation clauses, for example `This Agreement is made ... by and between ...`.
- In formation clauses, do not also label the same parties as `Agent`. Prefer the full legal party name such as `I-ESCROW, INC.` over a short alias such as `i-Escrow`.
- In definition clauses with `means`, `refers to`, or `is defined as`, use only `DefinedTerm` and `Definition`.
- Do not label the defined term as `Agent`.
- Do not duplicate the definition span as `Theme`.
- In passive voice, the grammatical subject is usually `Theme`, not `Agent`.
- Use `Agent` only when an actual actor performs the action. If no actor is stated, leave `Agent` absent.

## Span Rules

- Use exact character spans from the clause text.
- Do not include leading/trailing commas, quotes, parentheses, or periods unless they are part of the legal name, such as `INC.`.
- Keep meaningful legal numbering in the clause, but do not label `(a)` or `(j)` as a semantic role.
- Prefer the shortest complete legal span: `this Agreement`, not `terminate this Agreement`.
- For time expressions, prefer the whole deadline: `within sixty (60) days`, not only `sixty`.
- For conditions, include the full trigger: `if the other party materially breaches ...`, not only `if`.

## Examples

Formation clause:

```json
{
  "predicate": {"text": "made"},
  "roles": [
    {"label": "Theme", "text": "This Agreement"},
    {"label": "Time", "text": "June 21, 1999"},
    {"label": "ContractParties", "text": "I-ESCROW, INC."},
    {"label": "ContractParties", "text": "2THEMART.COM, INC."}
  ]
}
```

Definition clause:

```json
{
  "predicate": {"text": "means"},
  "roles": [
    {"label": "DefinedTerm", "text": "SERVICES"},
    {"label": "Definition", "text": "i-Escrow's implementation and performance of the Escrow Services as of the Effective Date, as modified over time"}
  ]
}
```

Passive clause:

```json
{
  "predicate": {"text": "added"},
  "roles": [
    {"label": "Theme", "text": "The Adjusted Rate"},
    {"label": "Recipient", "text": "this Agreement"}
  ]
}
```

## Review Workflow

1. Open `data/srl_annotation_candidates.jsonl`.
2. For each row, check `predicate.text`, `predicate.start`, and `predicate.end`.
3. Check every role in `roles`.
4. Delete wrong roles.
5. Add missing roles.
6. Set `review_status` to `gold`.
7. Add notes only when the case is ambiguous.

Keep at least 100 reviewed predicate frames. A good target is 150 frames with a mix of obligations, rights, terminations, definitions, conditions, payment, confidentiality, and governing-law clauses.
