# EvalHarness

EvalHarness is an experiment framework for studying how evaluation harness
design changes agentic coding results. It runs SWE-bench Verified tasks across
three harnesses (`H1`, `H2`, `H3`) and multiple frontier API models, then stores
patches, trajectories, evaluation logs, and summary tables.

## Dataset

Tasks are sampled from **SWE-bench Verified**, test split. The finalized
experiment uses `subset_100`, sampled with difficulty stratification and
`seed=42`. See `configs/task_subsets/DATA.md` for the subset construction
details and repository coverage.

## Harnesses

The three harnesses share the same task set, model configuration, Docker
execution environment, and SWE-bench evaluation pipeline. The experimental
factor is the harness scaffolding around the model: context construction, tool
interface, validation, retry behavior, and recovery logic.

| Mechanism | H1 Minimal | H2 Improved | H3 Full |
|---|---|---|---|
| Design goal | Open-loop baseline with minimal intervention | Tool-robust closed-loop harness | H2 plus self-checking and recovery controls |
| Context strategy | Append all prior steps chronologically | Compress older history and retrieve relevant prior steps | Same as H2 |
| Context window cap | Large cap (`200000` token estimate) | `32000` token estimate | `32000` token estimate |
| History compression | None | Summarize older steps outside the recent window | Same as H2 |
| Retrieval | None | BM25 top-5 over older trajectory steps | Same as H2 |
| Tool schema | Verbose tool descriptions | Minimal task-focused tool schema | Minimal task-focused tool schema |
| Tool exposure | All available tools | Task-relevant tool subset | Task-relevant tool subset |
| Tool error format | Raw tool errors | Structured error feedback | Structured error feedback |
| Retry policy | No runtime retry on tool failure | Retry transient failures with 3-attempt exponential backoff | H2 retry policy plus malformed-output retry |
| Retryable errors | None | `timeout`, `rate_limit`, `transient_network` | H2 retryables plus `malformed_output` |
| Failure escalation | No explicit recovery policy | Continue after recoverable tool failures and feed the error back as observation | Checkpoint-style recovery on detected anomalies |
| Output validation | Disabled | Schema-only validation | Full output validation |
| Empty patch handling | Allowed by the minimal baseline | Rejected when validation is active | Rejected when validation is active |
| Parser behavior | Basic parser with one tolerated malformed-output retry | More permissive parser normalization for natural model outputs | H2 parser normalization plus H3 validation layer |
| Drift monitoring | Disabled | Disabled | KL-style drift check every 5 steps |
| Self-verification | Disabled | Disabled | Enabled; the model performs a lightweight per-step self-check |
| Anomaly detection | Disabled | Disabled | Enabled with `repeated_action_loop` and `context_contradiction` detectors |
| Checkpointing | Disabled | Disabled | Enabled; keeps the last 10 state checkpoints |
| Rollback policy | None | None | Roll back up to 3 checkpoints on handled anomalies |
| Observability | Minimal logs | Full trajectory-level logs | Full trace logs with verification metadata |
| Governance | Permissive | Permissive | Allowlist-oriented tool governance |

In short, `H1` measures a low-control baseline, `H2` measures practical harness
robustness from cleaner context/tool/error handling, and `H3` measures the
additional effect of verification, drift checks, anomaly detection, and
checkpoint recovery. The final harness configs are in `configs/harnesses/`.

## Models

The active final grid uses:

- GPT-5.4 via OpenAI
- Kimi K2.6 via Moonshot/Kimi
- GLM-5.1 via ZAI

Environment variables:

```bash
export OPENAI_API_KEY=...
export KIMI_API_KEY=...
export ZAI_API_KEY=...
```

## Run Inference

Example: one model on the final subset.

```bash
evalharness swebench-infer \
  --models-config configs/models.toml \
  --dataset-name princeton-nlp/SWE-bench_Verified \
  --split test \
  --repo-cache-dir cache/swebench_repos \
  --results-dir results/new_run \
  --scratch-dir tmp/new_run \
  --harness-config-dir configs/harnesses \
  --models gpt_5_4 \
  --harnesses H1 H2 H3 \
  --subset configs/task_subsets/subset_100.json \
  --tag subset100
```

## Run Evaluation

```bash
evalharness swebench-eval \
  --predictions-path results/new_run/H1/all_preds.jsonl \
  --dataset-name princeton-nlp/SWE-bench_Verified \
  --run-id gpt_5_4__H1.subset100 \
  --max-workers 4
```
