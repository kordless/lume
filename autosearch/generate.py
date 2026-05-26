"""
Generate text from a trained checkpoint with FST Concept Vector Steering.

Usage:
    uv run python generate.py --prompt "Valentine walked through" --steer-tag "VALENTINE,PARIS" --tag-bias 3.5
"""

import argparse
import os
import sys
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass

# Set PyTorch options
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
if sys.platform == "win32":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ---------------------------------------------------------------------------
# GPT Model Definition (matching train.py for self-containment)
# ---------------------------------------------------------------------------

ROPE_BASE = 200_000
INIT_SCALE = 0.68

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # SDPA fallback
        q2, k2, v2 = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        if base is None:
            base = ROPE_BASE
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 8
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()
        return logits

# ---------------------------------------------------------------------------
# Tokenizer Wrapper
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, enc, bos_token_id):
        self.enc = enc
        self.bos_token_id = bos_token_id

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        bos_token_id = enc.encode_single_token("<|reserved_0|>")
        return cls(enc, bos_token_id)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        ids = self.enc.encode_ordinary(text)
        if prepend is not None:
            ids.insert(0, prepend_id)
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)

# ---------------------------------------------------------------------------
# Steered Generation Loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200, temperature=1.0, top_k=50, steer_tags=None, tag_bias=3.0):
    device = next(model.parameters()).device
    
    # Pre-scan BPE tokenizer vocabulary for matching steer tags (substring / co-occurrence)
    steer_token_ids = set()
    if steer_tags:
        print(f"\n🧠 [FST Concept Steering] Searching BPE vocabulary for target concepts: {steer_tags}")
        vocab_size = tokenizer.get_vocab_size()
        for token_id in range(vocab_size):
            try:
                token_bytes = tokenizer.enc.decode_single_token_bytes(token_id)
                token_str = token_bytes.decode('utf-8', errors='ignore').lower().strip()
                for tag in steer_tags:
                    tag_clean = tag.lower().strip()
                    # Add positive weights to tokens containing tag clean or matching sub-words (len >= 3)
                    if tag_clean in token_str or (len(token_str) >= 3 and token_str in tag_clean):
                        steer_token_ids.add(token_id)
            except Exception:
                pass
        print(f"  ➔ Associated {len(steer_token_ids)} BPE vocabulary tokens with concept targets.")

    tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    jsd_history = []
    bias_history = []
    calibration_count = 0

    print(f"\n🚀 Generating stochastically from Transformer (Temp: {temperature}, Top-k: {top_k})...\n")

    steered_tokens_generated = 0
    sys.stdout.write(prompt)
    sys.stdout.flush()

    for _ in range(max_tokens):
        # Crop context to model's sequence length limit
        idx = tokens[:, -2048:]
        logits = model(idx)
        
        # Apply FST Concept Steering Bias to boost targets with JSD Calibration
        jsd_val = 0.0
        active_bias = tag_bias
        calibration_triggered = False
        
        if steer_token_ids:
            logits_unsteered = logits[:, -1, :] / temperature
            Q = F.softmax(logits_unsteered, dim=-1)
            
            # Calibration loop to keep JSD <= 0.45
            for attempt in range(5):
                logits_steered = logits_unsteered.clone()
                for token_id in steer_token_ids:
                    logits_steered[0, token_id] += active_bias
                
                P = F.softmax(logits_steered, dim=-1)
                M = 0.5 * (P + Q)
                
                eps = 1e-12
                # KL Divergence with base 2
                kl_pm = torch.sum(P * torch.log2((P + eps) / (M + eps)), dim=-1)
                kl_qm = torch.sum(Q * torch.log2((Q + eps) / (M + eps)), dim=-1)
                jsd_tensor = 0.5 * kl_pm + 0.5 * kl_qm
                jsd_val = jsd_tensor.item()
                
                if jsd_val <= 0.45 or active_bias <= 0.1:
                    break
                else:
                    active_bias *= 0.70  # Scale back bias by 30%
                    calibration_triggered = True
            
            logits_step = logits_steered
            jsd_history.append(jsd_val)
            bias_history.append(active_bias)
            if calibration_triggered:
                calibration_count += 1
        else:
            logits_step = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
            logits_step[logits_step < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        next_token_id = next_token.item()
        is_steered = next_token_id in steer_token_ids
        if is_steered:
            steered_tokens_generated += 1
            
        decoded_word = tokenizer.decode([next_token_id])
        if is_steered:
            # Highlight FST-steered tokens in high-contrast green!
            sys.stdout.write(f"\x1B[1;32m{decoded_word}\x1B[0m")
        else:
            sys.stdout.write(decoded_word)
        sys.stdout.flush()

        tokens = torch.cat([tokens, next_token], dim=1)

    print("\n")
    if steer_tags:
        print(f"🎯 [Attention Report] Composed concept steering activation rate: {steered_tokens_generated / max_tokens * 100:.1f}%")
        if jsd_history:
            avg_jsd = sum(jsd_history) / len(jsd_history)
            max_jsd = max(jsd_history)
            print(f"🧮 [Jensen-Shannon Calibration Report]")
            print(f"  ➔ Average JSD: {avg_jsd:.4f}")
            print(f"  ➔ Maximum JSD: {max_jsd:.4f}")
            print(f"  ➔ JSD Calibration Triggers (JSD > 0.45): {calibration_count} times")
            if calibration_count > 0:
                min_bias = min(bias_history)
                print(f"  ➔ Steer Bias calibrated from {tag_bias:.2f} down to floor of {min_bias:.2f}")
    return tokenizer.decode(tokens[0].tolist())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model with Concept Steering")
    parser.add_argument("--book", type=str, default="faraday", choices=["faraday", "monte_cristo", "agent"], help="The active book to steer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Defaults to checkpoint_<book>.pt")
    parser.add_argument("--prompt", type=str, default=None, help="Inference prompt (defaults based on book choice)")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--steer-tag", type=str, default=None, help="Comma-separated FST concept tags (e.g. VALENTINE,PARIS)")
    parser.add_argument("--tag-bias", type=type(1.0), default=3.0, help="Logit bias added to FST steered concept tokens")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Set ACTIVE_BOOK env var dynamically so prepare / tokenizer load correctly
    os.environ["ACTIVE_BOOK"] = args.book

    # Dynamic cache paths matching prepare.py
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", args.book)
    tokenizer_dir = os.path.join(cache_dir, "tokenizer")
    
    ckpt_name = args.checkpoint if args.checkpoint else f"checkpoint_{args.book}.pt"
    checkpoint_path = Path(__file__).parent / ckpt_name

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Train the model first.")
        return

    if not os.path.exists(tokenizer_dir):
        print(f"No tokenizer found at {tokenizer_dir}. Run prepare.py first.")
        return

    print("Loading BPE Tokenizer...")
    tokenizer = Tokenizer.from_directory(tokenizer_dir)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]

    @dataclass
    class GPTConfig:
        sequence_len: int = config_dict["sequence_len"]
        vocab_size: int = config_dict["vocab_size"]
        n_layer: int = config_dict["n_layer"]
        n_head: int = config_dict["n_head"]
        n_kv_head: int = config_dict["n_kv_head"]
        n_embd: int = config_dict["n_embd"]
        window_pattern: str = config_dict["window_pattern"]

    config = GPTConfig()
    print(f"Initializing {config.n_layer}L model architecture ({config.n_embd}d, {config.vocab_size}V)...")
    model = GPT(config)

    print("Restoring state dict from pretraining...")
    state_dict = ckpt["model_state_dict"]
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            uncompiled_state_dict[k[len("_orig_mod."):]] = v
        else:
            uncompiled_state_dict[k] = v

    model.load_state_dict(uncompiled_state_dict)
    model.eval()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Deploying model to {device}...")
    model.to(device)

    # Process FST steering tags
    steer_tags = None
    if args.steer_tag:
        steer_tags = [t.strip().upper() for t in args.steer_tag.split(",") if t.strip()]

    # Use book-specific default prompt if not specified
    prompt = args.prompt if args.prompt else (
        "When the magnet was placed near the voltaic wire, the electric current"
        if args.book == "faraday" else "Valentine walked down the"
    )

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        steer_tags=steer_tags,
        tag_bias=args.tag_bias
    )

if __name__ == "__main__":
    main()
