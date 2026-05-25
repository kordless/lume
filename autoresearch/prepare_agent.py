"""
Prepare agent tool trajectories dataset for pretraining an offline Agentic Transformer.
Usage:
    uv run python prepare_agent.py
"""

import os
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Set the active book env var so prepare imports correctly
os.environ["ACTIVE_BOOK"] = "agent"

# Add current folder to sys.path
sys.path.append(str(Path(__file__).parent))
from prepare import CACHE_DIR, DATA_DIR, TOKENIZER_DIR, VAL_FILENAME, train_tokenizer

def main():
    print("=== LUME AGENT TRAJECTORIES CORPUS PREPARATION ===")
    
    shard0_path = os.path.join(DATA_DIR, "shard_00000.parquet")
    shard_val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    
    os.makedirs(DATA_DIR, exist_ok=True)

    # Parse examples/agent/book.txt
    agent_path = Path(__file__).parent.parent / "examples" / "agent" / "book.txt"
    if not agent_path.exists():
        print(f"Error: Agent trajectories file not found at {agent_path}!")
        return

    print(f"Reading Agent trajectories from {agent_path}...")
    with open(agent_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newlines to segment individual trajectories
    trajectories = [t.strip() for t in content.split("\n\n") if t.strip()]
    print(f"Successfully loaded {len(trajectories)} agent trajectories.")

    if not trajectories:
        print("Error: No trajectories found in the file!")
        return

    # Split into Train (90%) and Val (10%)
    split_idx = int(len(trajectories) * 0.90)
    train_paras = trajectories[:split_idx]
    val_paras = trajectories[split_idx:]
    
    print(f"Train split: {len(train_paras)} trajectories | Val split: {len(val_paras)} trajectories.")

    # Write as parquet files
    print("Writing Parquet files...")
    train_table = pa.Table.from_pydict({"text": train_paras})
    pq.write_table(train_table, shard0_path)
    print(f"Saved {shard0_path}")
    
    val_table = pa.Table.from_pydict({"text": val_paras})
    pq.write_table(val_table, shard_val_path)
    print(f"Saved {shard_val_path}")

    # Rebuild BPE Tokenizer on pure Agent trajectories
    print("\nRebuilding BPE tokenizer on Agent trajectories...")
    train_tokenizer()
    print("\n✓ Agent dataset prep successfully completed!")

if __name__ == "__main__":
    main()
