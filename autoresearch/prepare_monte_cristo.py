"""
Prepare Monte Cristo corpus for pure literary Transformer training.
Backs up Karpathy's Climbmix shards, parses Monte Cristo markdown, splits into 
train/val parquet shards, and triggers BPE tokenizer reconstruction.

Usage:
    uv run python prepare_monte_cristo.py
"""

import os
import sys
import pickle
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Set the active book env var so prepare imports correctly
os.environ["ACTIVE_BOOK"] = "monte_cristo"

# Add current folder to sys.path
sys.path.append(str(Path(__file__).parent))
from prepare import CACHE_DIR, DATA_DIR, TOKENIZER_DIR, VAL_FILENAME, train_tokenizer

def main():
    print("=== LUME MONTE CRISTO CORPUS PREPARATION ===")
    
    shard0_path = os.path.join(DATA_DIR, "shard_00000.parquet")
    shard_val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Parse examples/monte_cristo.md
    monte_cristo_path = Path(__file__).parent.parent / "examples" / "monte_cristo" / "book.md"
    if not monte_cristo_path.exists():
        print(f"Error: Count of Monte Cristo markdown file not found at {monte_cristo_path}!")
        return

    print(f"Reading Monte Cristo corpus from {monte_cristo_path}...")
    with open(monte_cristo_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out Gutenberg license metadata to keep the corpus purely literary
    paragraphs = []
    current_para = []
    
    in_story = False
    for line in lines:
        stripped = line.strip()
        if "*** START OF THE PROJECT GUTENBERG" in stripped:
            in_story = True
            continue
        if "*** END OF THE PROJECT GUTENBERG" in stripped:
            break
            
        if in_story:
            if not stripped:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
            else:
                current_para.append(stripped)
                
    if current_para:
        paragraphs.append(" ".join(current_para))

    print(f"Successfully extracted {len(paragraphs)} raw paragraphs of text.")

    # 3. Split into Train (90%) and Val (10%)
    split_idx = int(len(paragraphs) * 0.90)
    train_paras = paragraphs[:split_idx]
    val_paras = paragraphs[split_idx:]
    
    print(f"Train split: {len(train_paras)} paragraphs | Val split: {len(val_paras)} paragraphs.")

    # 4. Write as parquet files
    print("Writing Parquet files...")
    train_table = pa.Table.from_pydict({"text": train_paras})
    pq.write_table(train_table, shard0_path)
    print(f"Saved {shard0_path}")
    
    val_table = pa.Table.from_pydict({"text": val_paras})
    pq.write_table(val_table, shard_val_path)
    print(f"Saved {shard_val_path}")

    # 5. Clear old tokenizer states to force rebuild
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    
    if os.path.exists(tokenizer_pkl):
        print("Removing old Climbmix BPE tokenizer...")
        os.remove(tokenizer_pkl)
    if os.path.exists(token_bytes_path):
        os.remove(token_bytes_path)

    # 6. Rebuild BPE Tokenizer on pure Monte Cristo
    print("\nRebuilding BPE tokenizer on pure Count of Monte Cristo...")
    train_tokenizer()
    print("\n✓ Monte Cristo dataset prep successfully completed!")

if __name__ == "__main__":
    main()
