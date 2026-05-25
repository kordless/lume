# Autoresearch Runbook

This is a complete operational guide for any agent (human or AI) to understand, run, and extend this system. Read this before touching anything.

## What this is

An autonomous ML research platform. A GPT language model trains for a randomized time budget on web text. An AI agent modifies the training script, runs experiments, evaluates results, keeps improvements, and discards regressions. The optimizer uses Weber's electrodynamic bracket to modulate learning rates based on parameter momentum dynamics. Training seeds come from true hardware randomness via an RTL-SDR radio receiver.

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Upstream trains a fixed 5-minute run with `torch.manual_seed(42)`. We added the Weber optimizer, SDR entropy, agent harness, randomized budgets, gradient clipping, shadow A/B testing, and checkpointing.

## Architecture

```
                    SDR entropy (nemesis:9090)
                           |
                           v
train.py  <---- agent.py (Claude/GPT/Gemini)
   |                |
   v                v
GPU training    ferricula (memory)
   |
   v
checkpoint.pt + results.tsv + weber_shadow.tsv
```

## File map

| File | Role | Modify? |
|------|------|---------|
| `train.py` | Model, optimizer, training loop, Weber bracket, all hyperparameters | YES - this is what the agent edits |
| `prepare.py` | Data download, tokenizer training, dataloader, evaluation metric | NO - read only, never modify |
| `agent.py` | Autonomous experiment agent with 10 tools, 3 LLM providers | YES - to add tools or change behavior |
| `program.md` | Instructions for manual-mode agents (Claude Code, Codex) | YES - to change agent strategy |
| `test_weber.py` | A/B comparison: Weber vs vanilla at different c² values | YES - for testing |
| `generate.py` | Text generation from checkpoints (stub) | YES - needs completion |
| `results.tsv` | Experiment log (auto-generated, do not commit) | DO NOT COMMIT |
| `weber_shadow.tsv` | Shadow A/B results (auto-generated) | DO NOT COMMIT |
| `checkpoint.pt` | Latest model weights (~80MB, auto-generated) | DO NOT COMMIT |
| `run.log` | Latest training run output | DO NOT COMMIT |
| `Dockerfile` | CUDA runtime container | YES |
| `docker-compose.yml` | Full stack: train, agent, ferricula, prepare | YES |
| `pyproject.toml` | Dependencies. Agent SDKs are optional extras. | YES |

## The metric

**val_bpb** (validation bits per byte). Lower is better. Measures how many bits the model needs to predict the next byte of text. Computed by `evaluate_bpb()` in `prepare.py` on a pinned validation shard. Vocabulary-independent so architectural changes are fairly compared.

Current records on this setup (RTX 3060 12GB):
- Baseline (depth 5, no modifications): 1.314
- After agent tuning (depth 6, batch 49152): 1.263
- Best single run (depth 6, 488s jackpot budget): 1.211
- Community best on RTX 3090 (1800 experiments): 1.101

## How to run

### Prerequisites

- NVIDIA GPU (tested: RTX 3060 12GB, RTX 3090 24GB, H100 80GB)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Data in `~/.cache/autoresearch/` (run `uv run prepare.py` once)

### Single training run

```bash
uv run train.py
```

Output goes to stdout. Key lines at the end after `---`:
```
val_bpb:          1.211049
training_seconds: 488.0
peak_vram_mb:     4211.3
num_steps:        693
depth:            6
time_budget:      488
weber_c_sq:       1.0
checkpoint:       checkpoint.pt
```

### Autonomous agent

```bash
# Install provider SDK
uv pip install anthropic  # or openai, google-genai

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run (unlimited experiments until Ctrl+C)
uv run python agent.py --provider anthropic --model claude-sonnet-4-20250514

# Run with limits
uv run python agent.py --provider anthropic --model claude-sonnet-4-20250514 --max-experiments 10 --tag my-run

# Run with ferricula memory
uv run python agent.py --provider anthropic --model claude-sonnet-4-20250514 --memory http://localhost:8765
```

### Weber A/B test

```bash
# Vanilla vs Weber at c²=1.0
uv run test_weber.py

# Sweep multiple c² values
uv run test_weber.py --c-sq 0.1 0.5 1.0 5.0 10.0
```

Warning: each run takes 3-10 minutes (randomized budget). A sweep of 5 values = ~30 minutes.

### Docker

```bash
# First time: download data
docker compose --profile setup run prepare

# Training
docker compose run train

# Agent
ANTHROPIC_API_KEY=sk-ant-... docker compose run agent

# Full stack with ferricula
docker compose up ferricula -d
ANTHROPIC_API_KEY=sk-ant-... docker compose run agent
```

## Hyperparameters

All hyperparameters are in `train.py` lines 488-530. The agent modifies these via `set_hyperparams`. Every parameter has a comment explaining what it does.

### Model architecture

| Param | Default | What it controls |
|-------|---------|-----------------|
| `ASPECT_RATIO` | 57 | model_dim = DEPTH * ASPECT_RATIO |
| `HEAD_DIM` | 128 | Attention head dimension |
| `WINDOW_PATTERN` | "SSL" | Sliding window: S=short (seq_len/8), L=full. Last layer always L. |
| `ROPE_BASE` | 200000 | Rotary position embedding base frequency |
| `DEPTH` | 6 | Number of transformer layers |
| `DEVICE_BATCH_SIZE` | 8 | Sequences per forward pass |

### Optimization

| Param | Default | What it controls |
|-------|---------|-----------------|
| `TOTAL_BATCH_SIZE` | 49152 | Tokens per optimizer step. Must be divisible by DEVICE_BATCH_SIZE * 2048 |
| `EMBEDDING_LR` | 0.9 | Learning rate for token embeddings (Adam) |
| `UNEMBEDDING_LR` | 0.005 | Learning rate for lm_head (Adam) |
| `MATRIX_LR` | 0.04 | Learning rate for transformer matrices (Muon) |
| `SCALAR_LR` | 0.5 | Learning rate for per-layer scalars (Adam) |
| `WEIGHT_DECAY` | 0.2 | Cautious weight decay for Muon params |
| `ADAM_BETAS` | (0.8, 0.95) | Adam momentum parameters |
| `WARMUP_RATIO` | 0.0 | Fraction of budget for LR warmup |
| `WARMDOWN_RATIO` | 0.75 | Fraction of budget for LR cooldown |
| `FINAL_LR_FRAC` | 0.05 | LR at end of cooldown as fraction of peak |
| `EMBEDDING_WD` | 0.001 | Weight decay on token embeddings |
| `VE_WD` | 0.003 | Weight decay on value embeddings |
| `LM_HEAD_WD` | 0.01 | Weight decay on output projection |
| `INIT_SCALE` | 0.68 | Multiplier on weight initialization range |

### Weber optimizer

| Param | Default | What it controls | Disable |
|-------|---------|-----------------|---------|
| `WEBER_C_SQ` | 1.0 | Bracket correction scale. Larger = subtler. | Set to `1e30` |

The Weber bracket `W = 1 - v^2/(2c^2) + v*a/c^2` modifies every parameter update based on:
- v = parameter momentum (first derivative of parameter trajectory)
- a = change in momentum (second derivative)
- c^2 = WEBER_C_SQ (characteristic velocity scale)

Effects:
- Accelerating params (v*a > 0): W > 1, larger step
- Decelerating params (v*a < 0): W < 1, smaller step
- Fast params (v^2 large): natural speed limit via -v^2/2c^2 damping

Applied per-element in AdamW, per-matrix in Muon. Costs one extra buffer per parameter group.

### Safety and testing

| Param | Default | What it controls | Disable |
|-------|---------|-----------------|---------|
| `GRAD_CLIP_NORM` | 1.0 | Max gradient norm before optimizer step | Set to `0.0` |
| `SHADOW_AB_RATE` | 0.10 | Fraction of runs that compare Weber vs vanilla | Set to `0.0` |
| `BUDGET_RANDOMIZE` | True | Randomize training time budget | Set to `False` |
| `BUDGET_MIN` | 180 | Minimum budget (seconds) | N/A when disabled |
| `BUDGET_MAX` | 600 | Maximum budget (seconds) | N/A when disabled |
| `BUDGET_PEAK` | 300 | Most likely budget (seconds) | N/A when disabled |

## GPU presets

The default config is tuned for RTX 3060 12GB (~6GB VRAM used). For other GPUs:

### RTX 3060 12GB (current)
```
DEPTH = 6
DEVICE_BATCH_SIZE = 8
TOTAL_BATCH_SIZE = 49152
WINDOW_PATTERN = "SSL"
```

### RTX 3090 / 4090 24GB
```
DEPTH = 8
DEVICE_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 2**17
WINDOW_PATTERN = "SSSL"
```

### H100 80GB
```
DEPTH = 9
DEVICE_BATCH_SIZE = 128
TOTAL_BATCH_SIZE = 2**18
WINDOW_PATTERN = "SSSSL"
ASPECT_RATIO = 57
```

### H100/H200 96GB (aggressive)
```
DEPTH = 12
DEVICE_BATCH_SIZE = 128
TOTAL_BATCH_SIZE = 2**18
WINDOW_PATTERN = "SSSSL"
ASPECT_RATIO = 64
```

## Platform compatibility

| Platform | Flash Attention | torch.compile | Notes |
|----------|----------------|---------------|-------|
| H100 (Hopper, sm_90) | FA3 native | Triton | Full speed |
| RTX 3060/4090 (Ampere+) | SDPA fallback | Triton (Linux) | Auto-detected |
| Windows (any GPU) | SDPA fallback | Eager mode | `TORCHDYNAMO_DISABLE=1` set automatically |

The script auto-detects everything. FA3 import is wrapped in try/except. torch.compile failure falls back to eager. No manual flags needed.

## The training pipeline step by step

1. **Seed**: Fetch 8 bytes from RTL-SDR on nemesis (192.168.86.24:9090). Falls back to `os.urandom`.
2. **Budget**: Draw time budget from triangular(180, 600, 300) using the SDR seed. Or fixed 300s if `BUDGET_RANDOMIZE = False`.
3. **Model init**: Build GPT with config derived from DEPTH and ASPECT_RATIO. Init weights with `INIT_SCALE * sqrt(3) / sqrt(n_embd)`.
4. **Optimizer**: MuonAdamW — Muon for transformer matrices, AdamW for everything else. Weber bracket applied in both.
5. **Training loop**: For each step:
   - Forward pass (autocast bf16)
   - Backward pass
   - Gradient accumulation over `grad_accum_steps`
   - Gradient clipping (if `GRAD_CLIP_NORM > 0`)
   - LR schedule (warmup + constant + warmdown)
   - Optimizer step (with Weber bracket)
6. **Eval**: Run `evaluate_bpb` on pinned validation shard.
7. **Save**: Checkpoint to `checkpoint.pt` with model weights, config, and metrics.
8. **Shadow A/B** (10% chance): Re-train from same seed with bracket disabled, compare val_bpb. Log to `weber_shadow.tsv`.

## The agent loop

1. Agent checks `get_config()` and `get_history()` for current state.
2. Agent states hypothesis (reflection pattern).
3. Agent calls `set_hyperparams()` or `edit_code()` to make one change.
4. Agent calls `run_experiment()` — runs `uv run train.py`, parses metrics.
5. Agent compares val_bpb to current best.
6. If improved: `keep(description)` — git commit + log to results.tsv.
7. If worse: `discard(reason)` — git revert + log to results.tsv.
8. If ferricula is connected: `remember()` the result, `recall()` before next experiment.
9. Repeat. Context auto-compresses after 60 messages.

## Agent tools

| Tool | Purpose |
|------|---------|
| `get_config` | Read hyperparameters from train.py |
| `set_hyperparams` | Modify hyperparameters (pass Python expressions as strings) |
| `edit_code` | Replace entire named sections of train.py |
| `run_experiment` | Execute training, return metrics or crash info |
| `get_history` | Read results.tsv |
| `keep` | Git commit + log success |
| `discard` | Git revert + log failure |
| `read_code` | Read specific line ranges of train.py |
| `remember` | Store insight in ferricula memory |
| `recall` | Search ferricula for similar past experiments |

### edit_code sections

The `edit_code` tool can replace these named sections of train.py:
- `model` — GPT class, attention, MLP, embeddings
- `optimizer` — MuonAdamW, fused step functions, polar express
- `hyperparameters` — all tunable values
- `setup` — tokenizer, model init, optimizer init
- `training_loop` — the while loop, schedules, logging

**Warning**: The imports (lines 1-35) are NOT inside any section. The agent cannot modify them via `edit_code`. This caused corruption in early test runs. If the agent needs to change imports, it must use `read_code` to understand the current state and propose changes carefully.

## Infrastructure

### SDR entropy (nemesis)

RTL-SDR dongle on nemesis (192.168.86.24) captures radio noise and serves random bytes:
```
GET http://192.168.86.24:9090/api/entropy?bytes=8&format=json
```

Service: `sdr-rand local --port 9090` running on nemesis.
SSH: `ssh -i ~/.ssh/id_claude kord@nemesis`
Source: [DeepBlueDynamics/sdr-random](https://github.com/DeepBlueDynamics/sdr-random)

If the SDR is unreachable, train.py falls back to `os.urandom(8)`. Training works either way.

### Ferricula memory

Thermodynamic memory engine. Memories decay when ignored, strengthen on recall, consolidate during dream cycles.
```
POST http://localhost:8765/remember  {"text": "..."}
POST http://localhost:8765/recall    {"text": "...", "k": 5}
GET  http://localhost:8765/status
```

Not required. Agent works without it. Pass `--memory http://localhost:8765` to enable.

## Known issues

1. **edit_code can corrupt train.py**: The section parser can truncate the file if the section boundaries shift. The agent should use `set_hyperparams` for simple changes and only use `edit_code` for real architectural modifications. Always `discard()` if the file breaks.

2. **TOTAL_BATCH_SIZE must be divisible**: `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * 2048)` must equal 0. If the agent sets an invalid value, training crashes with an assertion error.

3. **Budget randomization confounds A/B tests**: Different runs get different budgets, making direct val_bpb comparison unreliable. Use `weber_shadow.tsv` for fair comparisons (same budget, same seed). Or set `BUDGET_RANDOMIZE = False` for controlled experiments.

4. **Checkpoint overwrites**: Each run overwrites `checkpoint.pt`. If you want to keep a good checkpoint, copy it before running again.

5. **Windows encoding**: Python on Windows defaults to cp1252. All file I/O in agent.py uses `encoding="utf-8"` explicitly. If you add new file operations, do the same.

## Results so far

### Weber shadow A/B (controlled comparison)

From `weber_shadow.tsv`:
```
seed                  budget  c²   weber_bpb  vanilla_bpb  delta      result
10453850565217960535   488    1.0  1.211049   1.213016     -0.001967  WEBER WINS
```

One data point. Weber beat vanilla by 0.002 val_bpb on identical conditions (same seed, same budget, same init). Weber also completed 13 more steps (693 vs 680) in the same wall time. More shadow tests accumulate automatically at 10% rate.

### Agent experiment history

From `results.tsv`:
- Baseline (depth 5): 1.314
- Depth 5->6: 1.269 (kept, -0.045)
- Depth 7: 1.340 (discarded, too slow)
- LR tuning (4 experiments): all regressed, current LRs are near-optimal
- Batch size 65K->49K: 1.263 (kept, -0.006, more steps in budget)

### 10-run summary (Weber, depth 6)

| Statistic | Value |
|-----------|-------|
| Best | 1.211 (488s budget) |
| Worst | 1.304 (230s budget) |
| Mean | 1.260 |
| Budget-val_bpb correlation | Strong negative (longer = better) |

## What to try next

Based on PR analysis and community results:

1. **SSSS window pattern** — all short windows worked best on RTX 3090 (PR #332)
2. **dim 512** (ASPECT_RATIO ~85 at depth 6) — RTX 3090 winning config used wider model
3. **WARMDOWN_RATIO 0.6** — RTX 3090 winner used 0.6 instead of 0.75
4. **GRAD_CLIP_NORM tuning** — try 0.5, 1.0, 2.0
5. **WEBER_C_SQ sweep** — try 0.5, 2.0, 5.0 to find optimal bracket strength
6. **Plateau detection** — if 5+ experiments fail, force architectural change (PR #110)
7. **Ferricula integration** — persistent memory across agent sessions
8. **Custom training data** — grubcrawler + PDF ingestion for domain-specific training
