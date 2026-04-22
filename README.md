# llm-hallucination-eval

A framework for measuring hallucination rates across LLM providers using the [FEVER](https://fever.ai/) fact-verification dataset and a Natural Language Inference (NLI) model as an automated judge.

## How it works

1. **Load claims** from FEVER — each claim is labelled `SUPPORTS` or `REFUTES` by human annotators.
2. **Query each LLM** and ask it to provide factual information about the claim.
3. **Score with NLI** — a cross-encoder model compares the LLM response against the original claim and returns `ENTAILMENT`, `NEUTRAL`, or `CONTRADICTION`.
4. **Classify hallucination**:
   - `SUPPORTS` claim + LLM response contradicts it → hallucination
   - `REFUTES` claim + LLM response entails it → hallucination
   - `NEUTRAL` NLI outcome → ambiguous, excluded from rate calculation
5. **Report** — per-model metrics, visualisations, and per-sample detail files.

```
FEVER claim ──► LLM (Claude / GPT / Ollama) ──► response
                                                     │
                                    NLI cross-encoder judge
                                                     │
                              hallucination? (True / False / None)
```

## Supported models

| Provider  | Models |
|-----------|--------|
| Anthropic | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| OpenAI    | `gpt-4o`, `gpt-4o-mini` |
| Ollama    | any locally installed model (e.g. `llama3.2`) |

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/gajdam/llm-hallucination-eval
cd llm-hallucination-eval
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
# OLLAMA_BASE_URL=http://localhost:11434  # optional, defaults to localhost
```

For Ollama, pull the model you want to test before running:

```bash
ollama pull llama3.2
```

## Usage

```bash
# Run all models defined in config.yaml
python main.py

# Test a single model
python main.py --llm claude-sonnet-4-6

# Quick smoke test with 20 samples
python main.py --samples 20

# Custom config and output directory
python main.py --config my_config.yaml --output results/experiment_1

# Use a different FEVER split
python main.py --split paper_test
```

### CLI reference

| Argument    | Default        | Description |
|-------------|----------------|-------------|
| `--config`  | `config.yaml`  | Path to YAML configuration file |
| `--llm`     | *(all)*        | Evaluate only this model name |
| `--samples` | from config    | Override number of FEVER samples |
| `--output`  | from config    | Override output directory |
| `--split`   | `labelled_dev` | FEVER split (`train`, `labelled_dev`, `paper_test`) |

## Configuration

`config.yaml` controls every aspect of the evaluation:

```yaml
fever:
  split: "labelled_dev"   # which FEVER split to use
  max_samples: 200        # total samples (balanced 50/50 SUPPORTS/REFUTES)
  seed: 42                # reproducibility

llms:
  - name: "claude-sonnet-4-6"
    provider: "anthropic"
  - name: "gpt-4o-mini"
    provider: "openai"
  - name: "llama3.2"
    provider: "ollama"
    base_url: "http://localhost:11434"   # optional override

nli:
  model: "cross-encoder/nli-deberta-v3-large"  # ~1.5 GB, best quality
  # model: "cross-encoder/nli-MiniLM2-L6-H768" # ~67 MB, much faster
  batch_size: 16
  device: "auto"   # auto-selects CUDA > MPS > CPU

evaluation:
  output_dir: "results"
  save_responses: true
  request_delay: 0.5   # seconds between API calls
```

Comment out any LLM entry you don't have credentials for — the pipeline skips unavailable models gracefully.

## Output

Results are saved to `results/` (or `--output` path) after each run:

```
results/
├── summary.csv                    # one row per model, all metrics
├── claude-sonnet-4-6_samples.jsonl
├── gpt-4o-mini_samples.jsonl
├── hallucination_rates.png        # grouped bar chart
└── nli_distribution.png           # stacked NLI label chart
```

### summary.csv columns

| Column | Description |
|--------|-------------|
| `model` / `provider` | Model name and provider |
| `n_total` | Total FEVER samples processed |
| `n_evaluated` | Samples with a conclusive NLI verdict (excludes errors and NEUTRAL) |
| `n_errors` | API call failures |
| `n_neutral_nli` | Samples where NLI returned NEUTRAL (excluded from rate) |
| `hallucination_rate` | `n_hallucinations / n_evaluated` |
| `accuracy` | `n_correct / n_evaluated` |
| `supports_hallucination_rate` | Hallucination rate on SUPPORTS claims only |
| `refutes_hallucination_rate` | Hallucination rate on REFUTES claims only |
| `nli_entailment_pct` | Fraction of samples scored ENTAILMENT |
| `nli_neutral_pct` | Fraction of samples scored NEUTRAL |
| `nli_contradiction_pct` | Fraction of samples scored CONTRADICTION |

### Per-sample JSONL

Each `{model}_samples.jsonl` line contains:

```json
{
  "sample_id": 42,
  "claim": "Marie Curie was born in Poland.",
  "fever_label": "SUPPORTS",
  "llm_response": "Marie Curie, born Maria Sklodowska in Warsaw...",
  "llm_error": null,
  "nli_label": "ENTAILMENT",
  "nli_entailment": 0.97,
  "nli_neutral": 0.02,
  "nli_contradiction": 0.01,
  "hallucination": false
}
```

`hallucination` is `true`, `false`, or `null` (when NLI is NEUTRAL).

## NLI model options

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `cross-encoder/nli-deberta-v3-large` | ~1.5 GB | Slow | Best |
| `cross-encoder/nli-MiniLM2-L6-H768` | ~67 MB | Fast | Good |
| `facebook/bart-large-mnli` | ~1.6 GB | Medium | Strong |

The NLI model is downloaded from HuggingFace on first run and cached locally.

## Running tests

```bash
# Test all providers
pytest src/tests/test.py -v

# Test only Anthropic models
pytest src/tests/test.py -v -m anthropic

# Test only OpenAI models
pytest src/tests/test.py -v -m openai

# Test only local Ollama models
pytest src/tests/test.py -v -m ollama
```

Tests verify that each provider is reachable and returns a non-empty response — useful for validating API keys before a full run.

## Project structure

```
llm-hallucination-eval/
├── main.py                    # CLI entry point
├── config.yaml                # evaluation configuration
├── requirements.txt
├── .env.example               # API key template
└── src/
    ├── pipeline.py            # orchestrates the full evaluation loop
    ├── llm/
    │   ├── base.py            # BaseLLM abstract class + LLMResponse dataclass
    │   ├── registry.py        # factory: build LLM instances from config
    │   ├── claude_llm.py      # Anthropic Claude
    │   ├── openai_llm.py      # OpenAI GPT
    │   └── ollama_llm.py      # local Ollama
    ├── data/
    │   └── fever_loader.py    # FEVER dataset loading, dedup, balancing
    ├── nli/
    │   └── nli_scorer.py      # HuggingFace NLI wrapper with batch scoring
    ├── evaluation/
    │   └── metrics.py         # SampleResult, LLMMetrics, plots, CSV export
    └── tests/
        └── test.py            # provider connection tests
```
