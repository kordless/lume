# autoresearch

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by [DeepBlueDynamics](https://github.com/DeepBlueDynamics)

Give an AI agent a real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and a better model.

## What's different in this fork

- **Agent harness** (`agent.py`) — structured tool-calling agent that works with Claude, GPT, or Gemini. 10 tools for autonomous experimentation including persistent thermodynamic memory via [ferricula](https://github.com/DeepBlueDynamics/ferricula).
- **Weber electrodynamic optimizer** — applies Weber's force law bracket `W = 1 - v²/(2c²) + v·a/c²` to learning rate, modifying effective step size based on parameter velocity and acceleration. Physics-inspired adaptive optimization.
- **SDR entropy seeding** — replaces the fixed `torch.manual_seed(42)` with true hardware randomness from an RTL-SDR radio receiver via [sdr-random](https://github.com/DeepBlueDynamics/sdr-random). Falls back to `os.urandom` if unavailable.
- **Multi-GPU support** — auto-detects Flash Attention 3 (H100/Hopper) or falls back to PyTorch SDPA (consumer GPUs). Windows support with automatic `torch.compile` bypass.
- **Optimized defaults** — hyperparameters from 215 experiments across Karpathy's sessions ([Discussion #32](https://github.com/karpathy/autoresearch/discussions/32), [#43](https://github.com/karpathy/autoresearch/discussions/43)).
- **Docker** — container with NVIDIA GPU passthrough, compose stack with ferricula memory service.

## Quick start

**Requirements:** Single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data + train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

## Platform support

| Platform | Flash Attn | torch.compile | Notes |
|----------|-----------|---------------|-------|
| **H100 / Hopper** | FA3 (native) | Triton | Full speed, no changes needed |
| **RTX 3060/4090 / Ampere+** | PyTorch SDPA (auto-fallback) | Triton (Linux) | Tune DEPTH, BATCH_SIZE for VRAM |
| **Windows (any GPU)** | PyTorch SDPA (auto-fallback) | Eager mode (auto) | Triton unavailable, runs slower |

The script auto-detects everything. No manual flags needed — just tune hyperparameters for your VRAM.

### Tuning for smaller GPUs

The defaults are optimized for H100 80GB. For consumer GPUs, edit the hyperparameters block in `train.py`:

```python
# RTX 3060 12GB
DEPTH = 4
DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 2**16
WINDOW_PATTERN = "SL"

# RTX 4090 24GB
DEPTH = 6
DEVICE_BATCH_SIZE = 32
TOTAL_BATCH_SIZE = 2**17
WINDOW_PATTERN = "SSL"
```

## Running the agent

```bash
# Install your provider's SDK
uv pip install anthropic  # or: openai, google-genai

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...  # Linux/Mac
set ANTHROPIC_API_KEY=sk-ant-...     # Windows cmd
$env:ANTHROPIC_API_KEY="sk-ant-..."  # PowerShell

# Run with Claude
uv run python agent.py --provider anthropic --model claude-sonnet-4-20250514

# Run with GPT
uv run python agent.py --provider openai --model gpt-4o

# Run with Gemini
uv run python agent.py --provider gemini --model gemini-2.0-flash

# Limit experiments, use a named branch
uv run python agent.py --provider anthropic --model claude-sonnet-4-20250514 --tag mar18 --max-experiments 20

# With ferricula memory (persistent experiment memory across runs)
uv run python agent.py --provider anthropic --model claude-sonnet-4-20250514 --memory http://localhost:8765
```

### Agent tools

| Tool | What it does |
|------|-------------|
| `get_config` | Read current hyperparameters from train.py |
| `set_hyperparams` | Modify hyperparameters (batch size, LR, depth, etc.) |
| `edit_code` | Replace entire sections of train.py (model, optimizer, training loop) |
| `run_experiment` | Execute 5-min training run, return val_bpb + metrics |
| `get_history` | Read results.tsv — full experiment log |
| `keep` | Git commit + log improvement to results.tsv |
| `discard` | Revert changes + log failure to results.tsv |
| `read_code` | Inspect specific lines of train.py |
| `remember` | Store insight in persistent thermodynamic memory (ferricula) |
| `recall` | Search memory for similar past experiments |

The agent loops autonomously: check config, propose a change, run it, evaluate, keep or discard, repeat. Context auto-compresses so it can run indefinitely.

### Manual mode

You can also run experiments the original way — point Claude Code, Codex, or any coding agent at `program.md`:

```
Hi have a look at program.md and let's kick off a new experiment!
```

## Weber electrodynamic optimizer

Applies Weber's force law bracket to the optimizer step, modifying effective learning rate based on parameter velocity (momentum) and acceleration (change in momentum):

```
W = 1 - v²/(2c²) + v·a/c²
```

- **Stable momentum** (v small): W ~ 1, normal update
- **Accelerating params** (v·a > 0): W > 1, larger step — leans into acceleration
- **Decelerating params** (v·a < 0): W < 1, smaller step — eases off
- **Fast params** (v² large): -v²/2c² damps — natural speed limit

Applied to both AdamW (per-element) and Muon (per-matrix). Controlled by `WEBER_C_SQ` hyperparameter (default 1.0). Larger = subtler correction.

## SDR entropy seeding

Seeds PyTorch's RNG with true hardware randomness from an RTL-SDR radio receiver. Entropy comes from ADC quantization noise — physically random, not pseudorandom.

Requires [sdr-random](https://github.com/DeepBlueDynamics/sdr-random) running on a machine with an RTL-SDR dongle:

```bash
# On the SDR host
sdr-rand local --port 9090

# train.py auto-fetches from http://<host>:9090/api/entropy
# Falls back to os.urandom if unavailable
uv run train.py
```

Configure the SDR host IP in `train.py` (search for `192.168.86.24`).

## Docker

```bash
# One-time: download data
docker compose --profile setup run prepare

# Run training
docker compose run train

# Run the autonomous agent
ANTHROPIC_API_KEY=sk-ant-... docker compose run agent

# Full stack with ferricula memory
docker compose up ferricula -d
ANTHROPIC_API_KEY=sk-ant-... docker compose run agent
```

Requires `nvidia-container-toolkit` for GPU passthrough.

## Project structure

```
train.py          model, optimizer, training loop (agent modifies this)
prepare.py        constants, data prep, evaluation (do not modify)
agent.py          autonomous experiment agent (Claude / GPT / Gemini)
program.md        manual-mode agent instructions
pyproject.toml    dependencies
Dockerfile        CUDA runtime + uv + PyTorch
docker-compose.yml  train, agent, ferricula, prepare services
results.tsv       experiment log (auto-generated)
```

## Optimized defaults

Hyperparameters validated across 215 experiments on H100:

| Setting | Upstream | This fork | Impact |
|---------|----------|-----------|--------|
| Depth | 8 | 9 | -0.004 val_bpb |
| Aspect ratio | 64 | 57 | depth-over-width |
| Batch size | 524K | 262K | -0.012 (more steps in 5 min) |
| Window pattern | SSSL | SSSSL | -0.004 cumulative |
| Short window | seq_len/2 | seq_len/8 | narrower local attention |
| RoPE base | 10K | 200K | -0.001 |
| Embedding LR | 0.6 | 0.9 | -0.005 |
| Warmdown ratio | 0.5 | 0.75 | -0.001 to -0.027 |
| Final LR frac | 0.0 | 0.05 | -0.006 |
| Init scale | 1.0x | 0.68x | -0.016 cumulative |
| x0_lambda init | 0.1 | 0.05 | -0.001 |
| Embedding WD | 0.0 | 0.001 | regularization |
| VE WD | 0.0 | 0.003 | -0.003 cumulative |
| LM head WD | 0.0 | 0.01 | -0.009 |
| Softcap | float32 before tanh | bf16 tanh, then float32 | saves ~4GB VRAM |
| **Weber c²** | N/A | 1.0 | velocity-dependent LR bracket |

## License

MIT
