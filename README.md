# Legal Contract NLP Homework

- Clause splitting
- Noun phrase chunking with IOB tags
- Dependency analysis with legal-aware tokenization
- Custom contract NER
- Semantic role labeling
- Clause intent classification
- Optional rule-based contract QA application

## Setup
download the models from "https://drive.google.com/drive/folders/1_RlDPDR1jn_7lF_vh_LspDl2ePz19ChJ?usp=sharing"
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

## Run

Put a contract text file at `input/raw_contracts.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline --input input/raw_contracts.txt
```

To train the small custom NER model first:

```powershell
.\.venv\Scripts\python.exe -m src.train_ner --base-model en_core_web_sm --iterations 50
```

The SRL module has a three-layer design: legal definition rules, legal dependency rules for passive voice and conditions, and a trained predicate-conditioned Legal-BERT SRL adapter. To train or refresh the local `models/legal_srl_legalbert` model, install the optional dependencies, validate/build the gold split, then train:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-srl.txt
.\.venv\Scripts\python.exe scripts\validate_srl_annotations.py --gold-only
.\.venv\Scripts\python.exe scripts\build_srl_dataset.py
.\.venv\Scripts\python.exe -m src.train_srl --epochs 5 --batch-size 4 --learning-rate 5e-5
.\.venv\Scripts\python.exe scripts\build_srl_rule_baseline.py
.\.venv\Scripts\python.exe -m src.evaluate_srl
```

After training, the pipeline automatically prefers `models/legal_srl_legalbert`, then falls back to `models/legal_srl`, then to legal rules only. You can override the model path with `LEGAL_SRL_MODEL`.

Intent classification follows the assignment comparison requirement: TF-IDF + Logistic Regression is the baseline, and `nlpaueb/bert-base-uncased-contracts` is fine-tuned as a transformer classifier over the same four labels. To rebuild the silver split and comparison:

```powershell
.\.venv\Scripts\python.exe scripts\build_intent_dataset.py
.\.venv\Scripts\python.exe -m src.train_intent_transformer --epochs 5 --batch-size 4 --learning-rate 2e-5
.\.venv\Scripts\python.exe -m src.evaluate_intent
```

The current reported model was trained with a balanced CUAD sample:

```powershell
.\.venv\Scripts\python.exe scripts\build_intent_dataset.py --max-cuad-per-label 160
.\.venv\Scripts\python.exe -m src.train_intent_transformer --epochs 3 --batch-size 8 --learning-rate 2e-5 --max-length 160
```

When `models/legal_intent_legalbert` exists, the main pipeline uses that transformer classifier for `output/intent_classification.txt`; otherwise it falls back to the legal-rule + TF-IDF classifier.

To validate all required outputs:

```powershell
.\.venv\Scripts\python.exe scripts\validate_outputs.py
```

To ask questions over the structured outputs:

```powershell
.\.venv\Scripts\python.exe -m src.qa "When can either party terminate this agreement?"
```

Omit the question to start an interactive console:

```powershell
.\.venv\Scripts\python.exe -m src.qa
```

The required files are written to `output/`:

- `clauses.txt`
- `clauses_context.json`
- `chunks.txt`
- `dependency.json`
- `ner_results.json`
- `srl_results.json`
- `intent_classification.txt`

`clauses_context.json` preserves section labels for traceability while `clauses.txt` stays in the simple one-clause-per-line assignment format. The QA app reads these files directly and always prints the source clause used for the answer.

## Dataset

The project is designed to use English business contracts from:

- Atticus Open Contract Dataset AOK Beta: https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta/data
- Megha Rajeev Business Contract Dataset: https://github.com/meghaarajeev/Business-Contract-Dataset-Intel-Training--Program-2024

After downloading/cloning both datasets locally, use `scripts/build_input_from_dataset.py` to create `input/raw_contracts.txt` from all readable `.txt`, `.md`, `.csv`, and `.pdf` files:

```powershell
.\.venv\Scripts\python.exe scripts\build_input_from_dataset.py data\external\atticus_aok data\external\business_contract_dataset --output input\raw_contracts.txt --max-files 12
```
