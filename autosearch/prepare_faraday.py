"""
Prepare Michael Faraday's complete multi-volume Faraday corpus and factual Q&As
for Steered Causal Transformer Pretraining.

Ingests book_vol1-3.md paragraph sections, parses o3-mini questions/answers,
merges them into a unified pretraining dataset, and rebuilds the BPE tokenizer.

Usage:
    uv run python prepare_faraday.py
"""

import os
import sys
import json
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Set the active book env var so prepare imports correctly
os.environ["ACTIVE_BOOK"] = "faraday"

# Add current folder to sys.path
sys.path.append(str(Path(__file__).parent))
from prepare import CACHE_DIR, DATA_DIR, TOKENIZER_DIR, VAL_FILENAME, train_tokenizer

def parse_markdown_paragraphs(md_path):
    print(f"Reading paragraph records from {md_path.name}...")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split by markdown headers
    lines = content.splitlines()
    sections = []
    current_section = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            if current_section:
                sections.append("\n".join(current_section).strip())
                current_section = []
        else:
            current_section.append(line)
            
    if current_section:
        sections.append("\n".join(current_section).strip())
        
    # Clean and filter empty sections
    cleaned_sections = []
    for sec in sections:
        sec_clean = re.sub(r'\s+', ' ', sec).strip()
        if sec_clean and len(sec_clean) > 20: # Filter out short titles or boilerplate
            cleaned_sections.append(sec_clean)
            
    return cleaned_sections

def load_factual_qnas(qna_path):
    print(f"Reading factual o3-mini Q&As from {qna_path.name}...")
    with open(qna_path, "r", encoding="utf-8") as f:
        qna_data = json.load(f)
        
    formatted_qnas = []
    for item in qna_data:
        series_title = item.get("series", "Electricity Series")
        qna_list = item.get("qna", [])
        for qa in qna_list:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                # Format Q&A into plain text block for pretraining
                qa_block = f"Series: {series_title}\nQuestion: {q}\nAnswer: {a}"
                formatted_qnas.append(qa_block)
                
    return formatted_qnas

def main():
    print("=== LUME NEURAL-SYMBOLIC FARADAY PRETRAINING PREPARATION ===")
    
    shard0_path = os.path.join(DATA_DIR, "shard_00000.parquet")
    shard_val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    os.makedirs(DATA_DIR, exist_ok=True)

    faraday_dir = Path(__file__).parent.parent / "examples" / "faraday"
    raw_dir = faraday_dir / ".raw"
    
    # 1. Parse all paragraph records from book_vol1-3.md
    pretraining_blocks = []
    for i in range(1, 4):
        md_file = faraday_dir / f"book_vol{i}.md"
        if md_file.exists():
            paras = parse_markdown_paragraphs(md_file)
            pretraining_blocks.extend(paras)
            print(f"  ➔ Loaded {len(paras)} sections from {md_file.name}")
        else:
            print(f"Error: Structured Markdown file not found: {md_file.name}")
            
    print(f"Total scientific paragraph sections loaded: {len(pretraining_blocks)}")
    
    # 2. Ingest the o3-mini Factual Questions & Answers
    qna_file = raw_dir / "qna_comprehensive.json"
    if qna_file.exists():
        qna_blocks = load_factual_qnas(qna_file)
        # We inject the Q&A blocks multiple times (oversampling) to ensure the 
        # model weights thoroughly memorize the factual alignment parameters
        for _ in range(5):
            pretraining_blocks.extend(qna_blocks)
        print(f"  ➔ Loaded & oversampled {len(qna_blocks)} Q&A pairs (added {len(qna_blocks) * 5} blocks).")
    else:
        # Fallback to qna.json if comprehensive is missing
        backup_qna = raw_dir / "qna.json"
        if backup_qna.exists():
            qna_blocks = load_factual_qnas(backup_qna)
            for _ in range(5):
                pretraining_blocks.extend(qna_blocks)
            print(f"  ➔ Loaded & oversampled {len(qna_blocks)} backup Q&A pairs.")
        else:
            print("Warning: Factual Q&A JSON files not found in hidden .raw/ directory.")

    print(f"Total unified pretraining blocks: {len(pretraining_blocks)}")

    # 3. Split into Train (90%) and Val (10%)
    split_idx = int(len(pretraining_blocks) * 0.90)
    train_paras = pretraining_blocks[:split_idx]
    val_paras = pretraining_blocks[split_idx:]
    
    print(f"Train split: {len(train_paras)} blocks | Val split: {len(val_paras)} blocks.")

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
        print("Removing old active BPE tokenizer...")
        os.remove(tokenizer_pkl)
    if os.path.exists(token_bytes_path):
        os.remove(token_bytes_path)

    # 6. Rebuild BPE Tokenizer on complete Faraday + Q&A mixture
    print("\nRebuilding BPE tokenizer on unified Faraday + Q&A corpus...")
    train_tokenizer()
    print("\n✓ Faraday neural pretraining preparation successfully completed!")

if __name__ == "__main__":
    main()
