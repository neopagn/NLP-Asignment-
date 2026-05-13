# Legal Contract Information Extraction and Semantic Analysis

## Overview

This project implements Assignments 1 and 2 for English legal contract analysis, plus the optional Assignment 3 rule-based QA application. The system reads raw contract text, splits it into clauses, performs syntax analysis, extracts contract-specific named entities, derives semantic roles, classifies clause intent, and answers simple natural-language questions with source-clause traceability.

The implementation is written in Python and uses two English contract datasets as source material: the Atticus Open Contract Dataset AOK Beta from Kaggle and the Megha Rajeev Business Contract Dataset from GitHub.

## Dataset

Source datasets:

- Atticus Open Contract Dataset AOK Beta
- `https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta/data`
- `meghaarajeev/Business-Contract-Dataset-Intel-Training--Program-2024`
- `https://github.com/meghaarajeev/Business-Contract-Dataset-Intel-Training--Program-2024`

After both datasets are downloaded locally, the script `scripts/build_input_from_dataset.py` extracts text from readable text/PDF files and creates `input/raw_contracts.txt`. The script accepts multiple dataset folders, so the working input can combine Atticus AOK contracts and Megha Rajeev contracts in one run.

## Assignment 1: Preprocessing and Syntax Analysis

### Clause Splitting

Input: `input/raw_contracts.txt`

Output: `output/clauses.txt`

Traceability output: `output/clauses_context.json`

The splitter removes PDF source markers, form checkboxes, placeholders, and standalone headings, but it avoids deleting legal structure. Parenthetical labels such as `(a)` and `(j)` are preserved, and definition predicates such as `means`, `refers to`, and `is referred to` are treated as valid legal signals.

The splitter protects legal/address abbreviations such as `Inc.`, `Blvd.`, and `S. Amphlett` before sentence splitting, and it no longer splits before relative-clause markers such as `where`, `when`, or `subject to`. This keeps definitions such as `(j) "SHADOW SITE" means the site where ...` intact. Section headings are not injected into `clauses.txt`, but they are stored beside each clause in `clauses_context.json` for later QA and report traceability.

Selected coordinating conjunctions are split only when the right-hand side has an independent legal subject or can safely reuse the previous subject and modal, for example splitting `2TheMart will promote ... and i-Escrow shall develop ...` into separate clauses.

### Noun Phrase Chunking

Input: `output/clauses.txt`

Output: `output/chunks.txt`

The system uses spaCy's English model to detect noun chunks. A post-processing step treats punctuation, quotation marks, brackets, and coordinating conjunctions as noun-phrase boundaries so legal defined terms such as `(the "Agreement")` do not create invalid `I-NP` sequences after `O` tags. Tight hyphenated terms such as `CO-BRANDING` and `i-Escrow` are merged for cleaner chunk output. `output/chunks.txt` is written as token/tag lines, with blank lines separating clauses. Tags follow the IOB scheme:

- `B-NP`: beginning of a noun phrase
- `I-NP`: inside a noun phrase
- `O`: outside a noun phrase

### Dependency Analysis

Input: `output/clauses.txt`

Output: `output/dependency.json`

Each clause is parsed with spaCy. For every token, the JSON output includes token text, lemma, part of speech, head token, and dependency relation.

Because legal contracts contain organization suffixes and brand-like names that general tokenizers often split incorrectly, the loaded spaCy pipeline is customized before parsing. The tokenizer preserves corporate/legal abbreviations and legal names such as `INC.`, `Corp.`, `LLC.`, `I-ESCROW`, `i-Escrow`, `2THEMART.COM`, `2TheMart`, `Co-Branded`, and section numbers such as `2.2`. A small post-parser attribute component also normalizes these legal tokens to proper-noun tags where appropriate and ensures standalone punctuation remains `PUNCT`. This avoids dependency artifacts such as `INC` plus a separate `.` token where the period is incorrectly tagged as `PROPN`.

## Assignment 2: Information Extraction and Semantic Analysis

### Custom NER

Output: `output/ner_results.json`

The entity schema contains:

- `PARTY`
- `MONEY`
- `DATE`
- `RATE`
- `PENALTY`
- `LAW`

The project includes a custom spaCy NER training script, `src/train_ner.py`, trained from student-defined seed examples in `data/ner_seed.jsonl`. The trainer fine-tunes the NER component from `en_core_web_sm` by default and falls back to a blank English pipeline only if the base model is unavailable. Final extraction uses a hybrid strategy:

- generic legal-pattern rules for high-precision spans such as `MONEY`, `DATE`, `RATE`, corporate parties, legal durations, and law citations
- the trained legal NER model in `models/legal_ner` for `PARTY`, `PENALTY`, and `LAW`
- filtered general spaCy entities for organization/person/law-like spans

Legal defined terms such as `Effective Date`, `Launch Date`, and `Approval Date` are not labeled as `DATE` unless an actual temporal expression is present. For example, `June 21, 1999` is labeled as `DATE`, while `"Effective Date"` is treated as a defined legal term and ignored by the current entity schema.

The rule layer avoids contract-specific party names. It uses reusable legal patterns for company names with corporate suffixes, domain-like companies, hyphenated brand names, digit-led brand names, and role nouns such as `Buyer`, `Seller`, `Employee`, and `Employer`. For example, a phrase shaped like `ACME-SERVICES, INC.` or `ALPHA.COM, LLC` can be kept as one party entity without adding that company name to the code.

Duration and deadline expressions such as `one (1) year`, `thirty (30) days`, `within ninety (90) days`, `three (3) business days`, and `sixty (60) days or more` are labeled as `DATE`. Law extraction prioritizes concrete legal citations such as named `Law` and `Code Section` references instead of generic phrases like `laws and regulations`. Generic legal `interest` is not treated as `PENALTY`; only explicit penalty-like terms such as `penalty`, `late fee`, `fine`, `default charge`, `liquidated damages`, or contextual interest charges are labeled as `PENALTY`.

This hybrid design is more stable for small individual homework data than relying only on a tiny trained model.

### Semantic Role Labeling

Output: `output/srl_results.json`

The SRL module is implemented in three layers so it can handle legal language more carefully than a generic subject-object extractor:

- legal definition rules for clauses such as `(j) "SHADOW SITE" means ...`
- legal dependency rules for passive voice, conditions, dates, parties, recipients, and law references
- a trained predicate-conditioned Legal-BERT SRL adapter loaded from `LEGAL_SRL_MODEL`, `models/legal_srl_legalbert`, or `models/legal_srl`

The output identifies:

- `Agent`
- `Predicate`
- `Theme`
- `Recipient`
- `Time`
- `Condition`
- `LegalBasis`
- `DefinedTerm`
- `Definition`

Special definition handling preserves the parenthetical label, defined term, and definition text instead of forcing the clause into a normal verb-argument frame. For example, `"SHADOW SITE" means ...` is represented with `DefinedTerm`, `Definition`, and the predicate `means`.

For passive legal clauses, passive subjects become `Theme` and parties introduced by phrases such as `by and between` are recovered from `PARTY` entities as `Agent`/`ContractParties`. Conditional openers such as `If`, `Unless`, `Provided that`, `In the event that`, `Upon`, `Notwithstanding`, and `Should ... shall ...` are extracted as `Condition`. Date and duration entities from NER override procedural phrases such as `by providing written notice`, so termination clauses keep the true deadline as `Time`.

The transformer model is trained locally by `src/train_srl.py` from the manually reviewed SRL gold annotations in `data/srl_annotation_candidates.jsonl`. `scripts/build_srl_dataset.py` creates train/dev/test splits grouped by `clause_id`. The trainer fine-tunes `nlpaueb/bert-base-uncased-contracts` as a token-classification model with predicate-conditioned input: the clause is sequence A and the predicate text is sequence B. At runtime, model predictions are merged conservatively with the legal rule layer: rule roles keep priority because they encode high-confidence legal fixes, and model output is used as the advanced SRL layer without overwriting contract-specific spans.

### Clause Intent Classification

Output: `output/intent_classification.txt`

The classifier predicts one of:

- `Obligation`
- `Prohibition`
- `Right`
- `Termination Condition`

The implementation now follows the required comparison setup:

- `src/intent.py` provides the TF-IDF + Logistic Regression baseline.
- High-precision reusable legal rules are applied before the baseline for clear legal signals such as `shall not`, `without prior written approval`, `hereby grants`, `has the right`, `terminate`, `expiration`, `bankruptcy`, and `insolvency`.
- `scripts/build_intent_dataset.py` builds `data/intent_silver/` from the current contract clauses and the manual seed file `data/intent_seed.csv`.
- `src/train_intent_transformer.py` fine-tunes `nlpaueb/bert-base-uncased-contracts` as a Legal-BERT sequence classifier for the same four labels.
- `src/evaluate_intent.py` compares TF-IDF/Logistic Regression, the production legal-rule + TF-IDF classifier, and the Legal-BERT classifier on the same test split.
- The main pipeline uses `models/legal_intent_legalbert` for `output/intent_classification.txt` when the trained model exists, and falls back to legal rules plus TF-IDF if the transformer model is unavailable.

Some contract definitions such as `"SERVICES" means ...` do not naturally fit any of the four required intent labels. Because the assignment schema only allows `Obligation`, `Prohibition`, `Right`, and `Termination Condition`, these definition clauses are mapped to `Obligation` as the closest legal-effect category instead of being randomly assigned by the statistical model.

The termination rule is deliberately conservative: a word like `breach` alone is not enough to classify a clause as `Termination Condition`; it must appear with termination/default/cure/notice context or with explicit termination/expiration language. This avoids misclassifying indemnity and liability clauses that mention breach but do not define termination.

## Assignment 3: Contract Question Answering Application

The optional QA application is implemented in `src/qa.py` using the rule-based approach. It reads the structured outputs from Assignment 2:

- `output/ner_results.json`
- `output/srl_results.json`
- `output/intent_classification.txt`
- `output/clauses_context.json`

The question parser extracts deterministic signals from the user query:

- party mentions such as `Party B`, `buyer`, `seller`, `i-Escrow`, and `2TheMart`
- intent hints such as obligation, prohibition, right, and termination
- entity focus such as money, date, rate, penalty, and governing law
- role hints from question words such as `who`, `when`, and `what`

Each clause is scored against these signals. The highest-scoring clause is used to generate a concise answer from semantic roles or entities when possible. Every answer includes the predicted intent, structured roles, section context when available, and the original source clause so the response remains grounded in extracted contract data.

Example command:

```powershell
.\.venv\Scripts\python.exe -m src.qa "When can either party terminate this agreement?"
```

Example output:

```text
Answer: The relevant time is within 2 months.
Intent: Termination Condition
Source section: 2.2 INITIAL INFORMATION TRANSFER MECHANISM DEVELOPMENT
Source clause 24: In the event that the parties are unable to agree to an SOW within 2 months following the Effective Date, either party may, in its sole discretion, terminate this Agreement by providing written notice.
```

## How To Run

```powershell
.\.venv\Scripts\python.exe scripts\build_input_from_dataset.py data\external\atticus_aok data\external\business_contract_dataset --output input\raw_contracts.txt --max-files 12
.\.venv\Scripts\python.exe -m src.train_ner --base-model en_core_web_sm --iterations 50
.\.venv\Scripts\python.exe -m pip install -r requirements-srl.txt
.\.venv\Scripts\python.exe scripts\validate_srl_annotations.py --gold-only
.\.venv\Scripts\python.exe scripts\build_srl_dataset.py
.\.venv\Scripts\python.exe -m src.train_srl --epochs 5 --batch-size 4 --learning-rate 5e-5
.\.venv\Scripts\python.exe scripts\build_srl_rule_baseline.py
.\.venv\Scripts\python.exe -m src.evaluate_srl
.\.venv\Scripts\python.exe scripts\build_intent_dataset.py --max-cuad-per-label 160
.\.venv\Scripts\python.exe -m src.train_intent_transformer --epochs 3 --batch-size 8 --learning-rate 2e-5 --max-length 160
.\.venv\Scripts\python.exe -m src.evaluate_intent
.\.venv\Scripts\python.exe -m src.pipeline --input input\raw_contracts.txt
.\.venv\Scripts\python.exe scripts\validate_outputs.py
.\.venv\Scripts\python.exe -m src.qa "What is the effective date?"
```

## Current Output Summary

The current run produced all required Assignment 1 and Assignment 2 files in `output/`. The validator confirms that the clause, dependency, NER, SRL, and intent outputs contain matching record counts.

The current SRL gold set contains 120 usable predicate frames after excluding two fragment rows with no roles. The split is 84 train, 18 dev, and 18 test examples. The Legal-BERT SRL model in `models/legal_srl_legalbert` was fine-tuned for 5 epochs and reached test token accuracy of approximately 0.782. On exact span evaluation over 45 gold test spans, the Legal-BERT model reaches F1 0.519 versus 0.429 for the baseline SRL output. The current pipeline output contains 12 clauses handled by `legal_definition_rule`, 106 clauses handled by `transformer_srl+legal_rules`, and 3 clauses handled by the dependency-rule fallback.

For intent classification, the current silver dataset contains 765 examples with a 536/116/113 train/dev/test split. It includes 640 CUAD-derived clauses sampled from 7,930 mapped candidate clauses, plus the local contract clauses and seed examples. The TF-IDF + Logistic Regression baseline reaches accuracy 0.788 and macro F1 0.788 on the silver test split. The class-weighted Legal-BERT intent classifier in `models/legal_intent_legalbert` reaches accuracy 0.841 and macro F1 0.839. The legal-rules + TF-IDF classifier is still reported as an interpretable fallback.

The QA module was smoke-tested against the current outputs and returns cited answers for termination, party responsibility, and date questions.

## Limitations

PDF contract templates contain blanks, checkboxes, and repeated optional clauses. Some extracted clauses may still contain placeholder text from forms. The custom NER model is fine-tuned from a small seed dataset, so the generic legal-pattern layer is still used to improve precision. The intent comparison uses CUAD expert clause categories mapped into the assignment's four intent labels, so it is stronger than pure rule-generated silver data but still not a fully independent human annotation of functional intent. For a stronger final report, the next improvement would be manually reviewing a held-out intent test set and evaluating precision, recall, and F1 against the rule baseline, TF-IDF baseline, and fine-tuned transformer models.
