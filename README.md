# Hiver Support Agent

A reproducible, offline-first AI customer-support experiment over the Customer Support on Twitter dataset. The raw CSV is the source of truth; processed files and evaluation outputs are generated locally.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands with the virtual environment's interpreter directly, for example `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Run

From the repository root:

```powershell
python src\prepare_data.py --brand AmazonHelp --max-examples 12000 --golden-size 200
python src\annotate_golden.py
python src\evaluate.py
```

Run all commands from the repository root. Preparation pairs each selected brand reply with its parent customer tweet from `data/raw/twcs.csv`, deduplicates parent conversations, reserves the golden examples before writing training data, and therefore prevents conversation-level train/test leakage. It writes `data/processed/brand_conversations.csv` and creates a 200-example provisional golden set at `data/golden_set.csv`.

`annotate_golden.py` applies reproducible assistant annotations and rubric scores. These are explicitly not human labels. For final assignment results, replace them with 150-250 human-reviewed intent labels and reply-quality ratings. `label_source` makes this limitation explicit.

To use a different brand, pass its support-account ID:

```powershell
python src\prepare_data.py --brand AppleSupport --max-examples 12000 --golden-size 200
python src\annotate_golden.py
python src\evaluate.py
```

The agent uses TF-IDF plus logistic regression for intent classification, TF-IDF cosine similarity for retrieval, a retrieved historical reply for grounding, and rule-based escalation for low confidence, weak retrieval, or sensitive wording. Evaluation also reports a majority baseline and a standalone TF-IDF/logistic-regression baseline.

## Optional LLM judge

Without credentials, evaluation uses a deterministic groundedness proxy so it remains reproducible offline. Configuration lives in `.env`, which is ignored by Git:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
```

After adding a key, run `python src\evaluate.py`. The predictions record `judge_type`. Cohen's kappa is calculated only when `label_source` begins with `human_reviewed`; assistant-generated ratings are never reported as human agreement.

## How the system works

1. **Preparation:** Reads the raw TWCS rows, selects one brand, joins each brand reply to its parent customer tweet, infers one of six simple intents, and creates a leakage-free train/golden split.
2. **Intent classification:** `SupportAgent.fit()` vectorizes customer messages with TF-IDF and trains logistic regression. It returns the predicted intent and confidence.
3. **Retrieval:** The same TF-IDF representation is compared with historical customer messages using cosine similarity. The closest conversation supplies the grounded historical reply.
4. **Escalation:** A request is escalated when intent confidence is low, retrieval similarity is weak, or the message contains high-risk terms such as legal, fraud, or security breach.
5. **Evaluation:** The evaluator compares the agent with a majority-class baseline and a separate TF-IDF/logistic-regression baseline, then writes predictions, metrics, and the top five intent-confusion failures.

## Project files

- `data/raw/twcs.csv`: original source dataset; do not modify.
- `src/prepare_data.py`: raw-data extraction and leakage-free split.
- `src/annotate_golden.py`: transparent assistant annotation helper.
- `src/pipeline.py`: classifier, retrieval, response grounding, and escalation.
- `src/evaluate.py`: metrics, judge, baselines, and report generation.
- `.env`: local API configuration; never commit it.
- `.gitignore`: excludes secrets, virtual environments, and generated outputs.

Outputs are written to `result/metrics.json`, `result/predictions.csv`, `result/failure_analysis.md`, and `report/report.md`.
