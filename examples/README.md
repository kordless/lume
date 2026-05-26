# 🧪 Lume Neural-Symbolic Examples & Model Pretraining Guide

Welcome to the **Lume Neural-Symbolic Examples Directory**. This guide details how our dual-end search primitives and custom overfit causal transformers are built, compiled, and executed. 

Here, you will learn how the 26.3M parameter transformer operates, how to pretrain it from scratch on a specific historical corpus, and how to verify steered inference walks with active mathematical safety guardrails.

---

## 🏗️ 1. The Model Architecture
When training models in Lume, you are compiling a **highly customized, 26.3-Million Parameter Causal Transformer Decoder** designed specifically for high-efficiency controlled style memorization on a closed-world corpus.

### Key Specifications:
*   **Capacity:** 26.3 Million trainable weights.
*   **Context Window (`sequence_len`):** 2,048 tokens.
*   **Vocabulary:** 32,768 distinct BPE tokens (Byte-Pair Encoding).
*   **Blocks:** 12 Layers of custom-compiled decoder blocks.
*   **Dimensionality (`n_embd`):** 768 channels with 6 attention heads.

### Advanced Mathematical Neural Primitives:
*   **Alternating SSSL Attention Windows:** Restricts attention inside local sliding windows in alternating layers (`SSSL` pattern) to compress local syntactic patterns while allowing global long-range synthesis in every 4th layer, saving substantial VRAM and compute.
*   **Dynamic Value Expansion (VE) Gating (ResFormer Residuals):** Direct, input-dependent gating of a static base value embedding matrix in alternating layers:
    $$v_{\text{final}} = v + \text{gate}(x) \odot ve$$
    This acts as a permanent, non-decaying memory register for the causal walk.
*   **Squared-ReLU MLPs:** Replaces standard GELU with `F.relu(x).square()` to encourage structural sparsity and accelerate gradient convergence.
*   **RMSNorm & High-Base RoPE:** Bypasses mean normalization steps for speed and applies Rotary Positional Embeddings with a base frequency of `200,000` to stabilize long-range syntax ordering.

---

## ⚡ 2. How to Train a Model on a Document
Lume trains stylistic and factual oracles by **deliberately overfitting** a 26.3M model to a target document corpus. A training cross-entropy loss collapsing down to **`0.003`** (with validation BPB around `2.85`) ensures the causal weights have verbatim memorized the author's stylistic voice and factual records.

### Step-by-Step Training Protocol:

#### Step A: Segment and Swarm
1.  Place your raw text files under a book folder (e.g. `examples/faraday/`).
2.  Segment the raw text into sequentially numbered, noise-free experimental paragraphs.
3.  Execute parallel knowledge extraction sweeps using the OpenAI `o3-mini` reasoning engine. This compiles premium, fact-based Q&A pairs (e.g. 116 high-density Q&As over all 29 Series).
4.  **5x Oversampling:** The pretrainer oversamples the Q&A pairs 5x to force the neural weights to prioritize factual memorization over general prose style.

#### Step B: Tokenize and Shard
Run Lume's dataset preparation utility to compile the mixture of paragraphs and Q&As into Parquet sharded sequences:
```bash
# Set the active book environment variable and shard the dataset
export ACTIVE_BOOK=faraday
uv run python autosearch/prepare_faraday.py
```

#### Step C: Run high-throughput Pretraining
Launch the single-GPU pretraining compute loop. The script saturates the GPU at **~68,500 tokens/second** using a strict **300-second (5-minute) training budget** to output the memorized oracle weights:
```bash
# Pretrain the causal transformer weights
uv run python autosearch/train.py
```
This saves the finalized weights to `checkpoint_faraday.pt` in the project directory.

---

## 🔄 3. High-Level Ingestion and Inference Loop
Once the index is built and the model is trained, Lume executes steered inference through a unified five-stage pipeline:

```
  [User Query / Dense Vector] 
            │
            ▼ (Stage 1: Vector Inversion)
  [Shivvr Vec2Text Decoder] ➔ Cosine Similarity Fidelity Gate
            │
            ▼ (Stage 2: Deterministic Concept Tagging)
  [FST Tag Dictionary] ➔ Extracted Semantic Keys
            │
            ▼ (Stage 3: High-Speed Index Search)
  [Roaring Posting Bitmaps] ➔ BM25-ranked Paragraph Context
            │
            ▼ (Stage 4: Steered Generation & JSD Gating)
  [25M Causal Model] ➔ FST Logit Bias (+6.0) ➔ JSD Calibration Valve (JSD <= 0.45)
            │
            ▼ (Stage 5: Output Audit & Polish)
  [Provenance Filter] ➔ Roaring Proximity Verification ➔ Gemma-4 Smoothing
```

---

## 🏃 4. Running the Unified Pipeline Demo

We have built a fully self-contained, runnable pipeline demonstration in the `lume_pipeline` subdirectory that compiles the index, maps simulated Roaring posting bitmaps, computes real-time Jensen-Shannon Divergence steering calibration, and audits outputs for BPE noise.

To run the complete demonstration:
```bash
# Run the pipeline demo using Lume's virtual environment
./autosearch/.venv/bin/python examples/lume_pipeline/run_pipeline.py
```

### What to Observe in the Demo Output:
1.  **Stage 1 Index Build:** Shows paragraph segmentation, inverted posting map compilation, and FST tag dictionary generation.
2.  **Stage 2 Model Specs:** Reviews the exact neural primitives and training mixture details of the custom 26M causal transformer.
3.  **Stage 3 JSD Calibration:** Demonstrates the closed-loop feedback in action. It applies a high initial logit bias of `+6.5`. Because the initial steering shift breaches the safety boundary (`JSD = 0.7390 > 0.45`), the gater dynamically scales back the bias by 30% to `4.55` (`JSD = 0.3724`), safely locking the steered walk within grammatical limits.
4.  **Stage 3 Provenance Audit:** Evaluates generated clauses against the posting bitmaps to classify grounded facts and discard hallucinated BPE noise.
5.  **Stage 3 Grammatical Smoothing:** Uses Ollama Gemma-4 smoothing (or fallback) to output the finalized authenticated passage.

---

## 📂 Directory Layout
*   [lume_pipeline/](file:///workspace/rust-fstguardrails/examples/lume_pipeline/) - Unified Build-Train-Inference demo script ([run_pipeline.py](file:///workspace/rust-fstguardrails/examples/lume_pipeline/run_pipeline.py)).
*   [faraday/](file:///workspace/rust-fstguardrails/examples/faraday/) - Raw three-volume text files and fact-based evaluation Q&As.
*   [monte_cristo/](file:///workspace/rust-fstguardrails/examples/monte_cristo/) - Raw text files for Dumas controlled pretraining comparison.
*   [data/](file:///workspace/rust-fstguardrails/examples/data/) - Concept dictionaries parsed by the FST tagger (`material.csv`, `force.csv`, `alignment.csv`).
