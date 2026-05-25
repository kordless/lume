"""
Parallel Stochastic Path Exploration & HATCHERIK Monte Carlo Search Demo.

This script demonstrates how Lume's lightweight 15M parameter Transformer can
run massive parallel path exploration on the GPU/CPU to discover optimal
generative trajectories that exactly hit designated FST concept tags.

Usage:
    uv run python parallel_explore_demo.py --prompt "Valentine walked down the" --steer-tag "VALENTINE,PARIS" --batch-size 128
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

# Add current folder to sys.path so we can import modules
sys.path.append(str(Path(__file__).parent))
from generate import GPT, Tokenizer

@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Stochastic Path Exploration Demo")
    parser.add_argument("--book", type=str, default="faraday", choices=["faraday", "monte_cristo", "agent"], help="The active book to steer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Defaults to checkpoint_<book>.pt")
    parser.add_argument("--prompt", type=str, default=None, help="Inference prompt (defaults based on book choice)")
    parser.add_argument("--tokens", type=int, default=30, help="Number of tokens to generate per path")
    parser.add_argument("--temperature", type=float, default=1.8, help="High temperature for high randomness / exploration")
    parser.add_argument("--steer-tag", type=str, default=None, help="Comma-separated concepts to search for")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of parallel paths to explore")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()

def main():
    args = parse_args()

    # Set ACTIVE_BOOK env var dynamically so prepare / tokenizer load correctly
    os.environ["ACTIVE_BOOK"] = args.book

    # Dynamic cache paths
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

    print("=== LUME PARALLEL MONTE CARLO EXPLORER ===")
    print("Loading BPE Tokenizer...")
    tokenizer = Tokenizer.from_directory(tokenizer_dir)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]

    config = GPTConfig(
        sequence_len=config_dict["sequence_len"],
        vocab_size=config_dict["vocab_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_kv_head=config_dict["n_kv_head"],
        n_embd=config_dict["n_embd"],
        window_pattern=config_dict["window_pattern"]
    )

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

    # Use book-specific default steer tags if not specified
    steer_tag = args.steer_tag if args.steer_tag else ("MAGNETIC,FORCE" if args.book == "faraday" else "VALENTINE,PARIS")
    # 1. Map target tags to BPE token IDs
    target_tags = [t.lower().strip() for t in steer_tag.split(",") if t.strip()]
    steer_token_ids = set()
    vocab_size = tokenizer.get_vocab_size()
    for token_id in range(vocab_size):
        try:
            token_bytes = tokenizer.enc.decode_single_token_bytes(token_id)
            token_str = token_bytes.decode('utf-8', errors='ignore').lower().strip()
            for tag in target_tags:
                if tag in token_str or (len(token_str) >= 3 and token_str in tag):
                    steer_token_ids.add(token_id)
        except Exception:
            pass

    print(f"Associated {len(steer_token_ids)} tokens in BPE vocabulary with concepts: {target_tags}")

    # Use book-specific default prompt if not specified
    prompt = args.prompt if args.prompt else (
        "When the magnet was placed near the voltaic wire, the electric current"
        if args.book == "faraday" else "Valentine walked down the"
    )
    # 2. Encode Prompt and replicate across the batch size
    prompt_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    prompt_len = len(prompt_ids)
    
    # Batch Shape: [B, T]
    batch_size = args.batch_size
    batch_tokens = torch.tensor([prompt_ids] * batch_size, dtype=torch.long, device=device)
    
    # Parallel scoring metrics
    cumulative_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
    concept_match_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    print(f"\n🚀 Spawning {batch_size} parallel trajectories on {device}...")
    print(f"Exploring probability space over {args.tokens} generation steps (Temp: {args.temperature})...\n")

    # Autoregressive loop
    for step in range(args.tokens):
        # Forward pass on the entire batch in ONE parallel operation!
        logits = model(batch_tokens) # Shape: [B, T, V]
        next_token_logits = logits[:, -1, :] # Shape: [B, V]
        
        # Apply temperature
        scaled_logits = next_token_logits / max(args.temperature, 1e-5)
        
        # Calculate log probabilities for score tracking
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        
        # Sample next tokens stochastically (massive randomness)
        probs = F.softmax(scaled_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # Shape: [B]

        # Accumulate scores for each path
        selected_log_probs = log_probs[torch.arange(batch_size), next_tokens]
        cumulative_log_probs += selected_log_probs

        # Check concept matches
        for idx in range(batch_size):
            tok_id = next_tokens[idx].item()
            if tok_id in steer_token_ids:
                concept_match_counts[idx] += 1

        # Append to token sequences
        batch_tokens = torch.cat([batch_tokens, next_tokens.unsqueeze(-1)], dim=-1)

    # 3. Grade and rank the completed paths
    print("Generation complete. Ranking paths...")
    
    # We want a score that balances semantic fit (matching concepts) and likelihood
    # Score = Concept Matches * 10.0 + Normalised Log Likelihood
    path_details = []
    for idx in range(batch_size):
        path_ids = batch_tokens[idx, prompt_len:].tolist()
        text = tokenizer.decode(path_ids)
        
        matches = concept_match_counts[idx].item()
        log_prob = cumulative_log_probs[idx].item()
        
        # Calculate normalised log likelihood to compare different generations
        normalized_ll = log_prob / max(args.tokens, 1)
        
        # Combined evaluation score (heavily prioritising paths matching target concepts)
        score = (matches * 15.0) + normalized_ll
        
        # Highlight concept matches in bright green
        highlighted_text = text
        for tag in target_tags:
            # Simple highlight
            tag_case = tag.capitalize()
            highlighted_text = highlighted_text.replace(tag, f"\x1B[1;32m{tag}\x1B[0m")
            highlighted_text = highlighted_text.replace(tag_case, f"\x1B[1;32m{tag_case}\x1B[0m")
            highlighted_text = highlighted_text.replace(tag.upper(), f"\x1B[1;32m{tag.upper()}\x1B[0m")

        path_details.append({
            "text": text,
            "highlighted": highlighted_text,
            "score": score,
            "matches": matches,
            "log_prob": log_prob,
            "normalized_ll": normalized_ll
        })

    # Sort paths by score descending
    path_details.sort(key=lambda x: x["score"], reverse=True)

    print("\n🏆 Top 5 Discovered Paths satisfying concept constraints:")
    print("=" * 70)
    for r, path in enumerate(path_details[:5]):
        print(f"\x1B[1mRank {r + 1} (Score: {path['score']:.2f})\x1B[0m")
        print(f"  ➔ Generated Text: \"{path['highlighted'].strip()}\"")
        print(f"  ➔ Concept Matches: {path['matches']} | Sequence Log Likelihood: {path['log_prob']:.2f} (avg: {path['normalized_ll']:.4f})")
        print("-" * 70)

if __name__ == "__main__":
    main()
