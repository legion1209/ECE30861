# ACME Trustworthy Model CLI

This repository implements the ECE 46100 Project Phase 1 deliverable for ACME Corporation’s internal model registry CLI. The tool evaluates pre-trained Hugging Face models against Sarah’s requirements and prints NDJSON scorecards ready for ingestion by downstream systems.

## Getting Started

1. Install dependencies (user site-packages):
   ```bash
   ./run install
   ```
2. Score models using a newline-delimited URL file:
   ```bash
   ./run /absolute/path/to/urls.txt
   ```
3. Execute the test suite with coverage:
   ```bash
   ./run test
   ```

### URL File Format

Provide one artifact URL per line. Dataset and code URLs should precede the model URL they describe. Example:
```
https://huggingface.co/datasets/acme/demo-dataset
https://github.com/acme/model-repo
https://huggingface.co/acme/demo-model
```
Only model URLs emit NDJSON output; datasets and code are used to enrich the metrics.

## Runtime Configuration

- `LOG_FILE`: absolute path for log output. Defaults to stderr when unset.
- `HUGGINGFACEHUB_API_TOKEN` / `HF_API_TOKEN`: token used for Hugging Face Hub and Inference API calls. Required to exercise the LLM-backed metric at scale.
- `ACME_LLM_MODEL`: optional override for the Hugging Face inference model (default `meta-llama/Llama-3.2-1B-Instruct`).
- `ACME_MAX_WORKERS`: overrides the metric evaluation thread pool size.

## NDJSON Schema (Table 1 Compliance)

Each line in the scoring output exposes the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Display name or repo ID of the model |
| `category` | string | Always `MODEL` for emitted rows |
| `net_score` | float | Weighted aggregate of all sub-metrics |
| `*_latency` | int | Milliseconds to compute the paired metric |
| `ramp_up_time` | float | Documentation clarity (LLM-assisted) |
| `bus_factor` | float | Diversity + activity of Hugging Face commit authors |
| `performance_claims` | float | Strength of empirical evidence (LLM-assisted) |
| `license` | float | License permissiveness assessment |
| `size_score` | object | Hardware compatibility scores `{raspberry_pi, jetson_nano, desktop_pc, aws_server}` |
| `dataset_and_code_score` | float | Presence and quality of linked datasets/code |
| `dataset_quality` | float | Dataset governance, size, and documentation |
| `code_quality` | float | Static code health heuristics on the cloned repo |

Latencies are rounded to milliseconds as required.

## Metric Design Highlights

- **Hugging Face API usage**: Downloads, likes, tags, commit history, and dataset metadata are retrieved via `huggingface_hub.HfApi`.
- **Local analysis**: Repositories are cached with `snapshot_download` and inspected for README structure, tests, lint configs, and sample code.
- **LLM-assisted scoring**: `LlmEvaluator` queries the Hugging Face Inference API to rate documentation clarity and performance claims. When inference is unavailable, deterministic heuristics keep the pipeline running and log a warning.
- **Size scoring**: Weight files are sized using repository metadata and graded against hardware targets. Documentation-only clones prevent oversized downloads while retaining local analysis for other metrics.
- **Net score weighting**: `{ramp_up_time: 0.15, bus_factor: 0.10, performance_claims: 0.15, license: 0.10, size_score: 0.10, dataset_and_code_score: 0.10, dataset_quality: 0.15, code_quality: 0.15}`. Weights reflect Sarah’s emphasis on documentation quality, reproducibility, and maintainability.

## Testing & Quality

- `pytest` with coverage (`./run test`) achieves ~90% line coverage across the CLI.
- Tests mock external services and the LLM to avoid network dependencies.
- Critical components (URL parsing, context building, metrics, registry, runner, scoring pipeline) are unit tested.

## Team

- Yuan Chi Chang
- Wei Yun Liu
- Ruoyu Hu
