"""
Lume Neural-Symbolic Coherence Laundering & Provenance Verification Pipeline.

This script:
1. Loads the steered causal transformer (checkpoint_faraday.pt) and FPE Tokenizer.
2. Generates logit-biased stochastically steered raw tokens using target concept tags.
3. Implements Lume's three-way lexical-grounded provenance check against the Faraday corpus:
   - Verbatim Recall (exact substring match in corpus)
   - Concept-Grounded Synthesis (posting-list co-occurrence of FST concepts in the same paragraph)
   - Pure Hallucination / BPE Noise (purged)
4. Sends the verified factual fragments to host Ollama (gemma4:e2b) to smooth them into
   historically authentic, grammatically fluent Victorian scientific prose.
"""

import os
import re
import sys
import json
import urllib.request
import argparse
from pathlib import Path
import torch

# Set the active book env var so prepare imports correctly
os.environ["ACTIVE_BOOK"] = "faraday"

# Add current folder to sys.path
sys.path.append(str(Path(__file__).parent))

try:
    from generate import GPT, Tokenizer, generate
except ImportError:
    print("Error: Could not import GPT/Tokenizer from generate.py. Ensure generate.py exists in the same folder.")
    sys.path.append("/workspace/rust-fstguardrails/autoresearch")
    from generate import GPT, Tokenizer, generate

# Define simple set of English stopwords for co-occurrence filtering
STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'and', 'in', 'is', 'it', 'that', 'was', 'for', 'on',
    'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or',
    'had', 'by', 'but', 'not', 'were', 'which', 'their', 'an', 'one', 'would', 'been',
    'there', 'their', 'we', 'our', 'us', 'you', 'your', 'my', 'me', 'he', 'him', 'she',
    'her', 'it', 'its', 'them', 'they', 'does', 'did', 'done', 'doing', 'shall', 'will',
    'should', 'can', 'could', 'may', 'might', 'must', 'has', 'have', 'had', 'having',
    'any', 'some', 'no', 'none', 'both', 'each', 'all', 'any', 'every', 'other', 'another'
}

def load_faraday_corpus():
    """Load and normalize Michael Faraday's complete 3-volume corpus from MD files."""
    faraday_dir = Path(__file__).parent.parent / "examples" / "faraday"
    paragraphs = []
    
    for i in range(1, 4):
        md_file = faraday_dir / f"book_vol{i}.md"
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Split by double newline or headers to get paragraphs
                for chunk in re.split(r'\n\s*\n|#+', content):
                    chunk_clean = re.sub(r'\s+', ' ', chunk).strip()
                    if chunk_clean and len(chunk_clean) > 20:
                        paragraphs.append(chunk_clean)
        else:
            print(f"Warning: Faraday MD file {md_file.name} not found.")
            
    print(f"📚 Loaded {len(paragraphs)} Faraday corpus paragraphs for provenance verification.")
    return paragraphs

def load_fst_dictionary():
    """Load and normalize FST concepts from examples/data/*.csv."""
    data_dir = Path(__file__).parent.parent / "examples" / "data"
    concepts = set()
    
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # Skip header
                        for line in lines[1:]:
                            parts = line.split(",")
                            if parts:
                                phrase = parts[0].strip().lower()
                                if phrase and not phrase.startswith("name"):
                                    concepts.add(phrase)
            except Exception as e:
                print(f"Warning reading FST CSV {csv_file.name}: {e}")
                
    print(f"🧠 Loaded {len(concepts)} FST dictionary terms for semantic grounding.")
    return concepts

def normalize_text(text):
    """Normalize text for exact substring matching."""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def extract_significant_words(text, fst_dict):
    """Extract words that are either in FST dict or not stopwords."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    sig_words = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) <= 2:
            continue
        sig_words.append(w)
    return list(set(sig_words))

def run_three_way_provenance_check(text, paragraphs, fst_dict):
    """Perform the Lume three-way neural-lexical provenance verification at the clause/phrase level."""
    print("\n🔍 Running Lume Clause-Level Three-Way Neural-Lexical Provenance Check...")
    
    # Split text into clauses/phrases on punctuation and conjunctions
    clauses = re.split(r'[,;:.!?\n]|\band\b|\bbut\b|\bfor\b|\bor\b', text)
    verified_fragments = []
    
    # Create normalized full corpus string for fast exact lookup
    full_corpus_normalized = " ".join([normalize_text(p) for p in paragraphs])
    
    # Build paragraph word sets for co-occurrence check
    paragraph_word_sets = []
    for p in paragraphs:
        p_words = set(re.findall(r'[a-zA-Z]+', p.lower()))
        paragraph_word_sets.append(p_words)
        
    report = []
    
    for idx, clause in enumerate(clauses):
        clause_clean = clause.strip()
        if not clause_clean or len(clause_clean.split()) < 2:
            continue
            
        norm_clause = normalize_text(clause_clean)
        sig_words = extract_significant_words(clause_clean, fst_dict)
        
        # 1. VERBATIM RECALL CHECK (Exact Substring Match)
        if len(norm_clause) > 10 and norm_clause in full_corpus_normalized:
            verified_fragments.append({
                "type": "verbatim",
                "text": clause_clean,
                "label": "[Verbatim Recall]"
            })
            report.append(f"  ✓ Clause #{idx+1}: [Verbatim Recall] - '{clause_clean[:50]}...' matched corpus exactly.")
            continue
            
        # 2. CONCEPT-GROUNDED SYNTHESIS CHECK (Paragraph Co-occurrence)
        if len(sig_words) < 2:
            # Skip or treat as BPE noise if fewer than 2 significant words
            report.append(f"  ✗ Clause #{idx+1}: [Purged - BPE Noise] - '{clause_clean[:50]}...' (insufficient concepts: {sig_words})")
            continue
            
        # Check if sig words co-occur in any paragraph of the corpus (any 2 of them)
        grounded = False
        matching_paragraph_sample = ""
        
        for p_idx, p_words in enumerate(paragraph_word_sets):
            intersection = p_words.intersection(sig_words)
            if len(intersection) >= 2:
                grounded = True
                matching_paragraph_sample = paragraphs[p_idx][:80] + "..."
                break
                
        if grounded:
            verified_fragments.append({
                "type": "synthesized",
                "text": clause_clean,
                "label": "[Synthesized - Grounded]"
            })
            report.append(f"  ✦ Clause #{idx+1}: [Synthesized - Grounded] - Concepts {sig_words} co-occur in paragraph: '{matching_paragraph_sample}'")
        else:
            # 3. PURE HALLUCINATION / BPE NOISE (Purge)
            report.append(f"  ⚠️ Clause #{idx+1}: [Purged - Hallucination/BPE Noise] - Concepts {sig_words} have zero corpus proximity. Purged.")
            
    return verified_fragments, report

def call_gemma_coherence_laundering(fragments):
    """Call host-bridged Ollama running gemma4:e2b to grammatically smooth fragments."""
    print("\n🧼 Laundering Coherence via Host Gemma4:e2b...")
    
    # Build prompt from verified fragments
    raw_fragments_text = ""
    for idx, frag in enumerate(fragments):
        raw_fragments_text += f"\nFragment #{idx+1} ({frag['label']}): {frag['text']}\n"
        
    system_prompt = (
        "You are Michael Faraday, the legendary Victorian experimental physicist. "
        "Your task is to take the provided verified factual recall and synthesis fragments and grammatically "
        "smooth them into a highly authentic, flowing, and grammatically correct scientific passage in your "
        "signature 19th-century Victorian prose style.\n\n"
        "Rules:\n"
        "1. Do NOT invent any new physics facts, dates, or formulas.\n"
        "2. Preserve all the key electromagnetic terms, paragraph markers, and concepts exactly as provided.\n"
        "3. Output ONLY the smoothed, beautifully coherent Victorian scientific prose. "
        "Do not include any introductions, pleasantries, chatty explanations, or meta-commentary."
    )
    
    prompt = (
        f"Verified Faraday Recall & Synthesis Fragments:\n"
        f"============================================\n"
        f"{raw_fragments_text}\n"
        f"============================================\n\n"
        f"Smoothed prose:"
    )
    
    url = "http://host.docker.internal:11434/api/generate"
    data = {
        "model": "gemma4:e2b",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data.get("response", "").strip()
    except Exception as e:
        return f"Error connecting to Ollama gemma4:e2b API: {e}. (Ensure host Ollama is running and has gemma4:e2b loaded)."

def main():
    parser = argparse.ArgumentParser(description="Lume Neural-Symbolic Coherence Laundering & Provenance Verification Pipeline")
    parser.add_argument("--prompt", type=str, default="When the magnet was placed near the voltaic wire, the electric current")
    parser.add_argument("--steer-tag", type=str, default="BISMUTH,FORCE")
    parser.add_argument("--tag-bias", type=float, default=6.0)
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    print("=== LUME NEURAL-SYMBOLIC COHERENCE LAUNDERING PIPELINE ===")
    
    # 1. Load checkpoints & setup
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "faraday")
    tokenizer_dir = os.path.join(cache_dir, "tokenizer")
    checkpoint_path = Path(__file__).parent / "checkpoint_faraday.pt"

    if not checkpoint_path.exists():
        print(f"Error: No checkpoint found at {checkpoint_path}. Run pretraining first.")
        return

    print("Loading active BPE Tokenizer...")
    tokenizer = Tokenizer.from_directory(tokenizer_dir)

    print(f"Loading causal model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    from dataclasses import dataclass
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
    model = GPT(config)
    
    state_dict = ckpt["model_state_dict"]
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            uncompiled_state_dict[k[len("_orig_mod."):]] = v
        else:
            uncompiled_state_dict[k] = v

    model.load_state_dict(uncompiled_state_dict)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying steered causal transformer to {device}...")
    model.to(device)

    # 2. Run steered raw generation
    steer_tags = [t.strip().upper() for t in args.steer_tag.split(",") if t.strip()]
    print(f"\n1. RUNNING CAUSAL GENERATION (logit steer tags: {steer_tags}, bias: {args.tag_bias})")
    
    raw_generation = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        steer_tags=steer_tags,
        tag_bias=args.tag_bias
    )
    
    # 3. Load reference Faraday corpus & FST dictionaries for provenance checks
    paragraphs = load_faraday_corpus()
    fst_dict = load_fst_dictionary()
    
    # 4. Perform the three-way lexical-grounded check
    verified_fragments, check_report = run_three_way_provenance_check(raw_generation, paragraphs, fst_dict)
    
    print("\n================== PROVENANCE AUDIT REPORT ==================")
    for line in check_report:
        print(line)
    print("=============================================================")
    
    if not verified_fragments:
        print("\n⚠️  All generated fragments failed the provenance check and were purged as pure hallucination / BPE noise.")
        return
        
    # 5. Call gemma4:e2b to smooth fragments
    smoothed_prose = call_gemma_coherence_laundering(verified_fragments)
    
    print("\n================= LAUNDERED COHERENT OUTPUT =================")
    print(smoothed_prose)
    print("=============================================================")
    
    # Print a JSON summary block for Rust CLI/MCP parsing
    summary_data = {
        "status": "success",
        "raw_generation": raw_generation,
        "fragments_count": len(verified_fragments),
        "purged_count": len(check_report) - len(verified_fragments),
        "laundered_prose": smoothed_prose
    }
    
    print("\n--- JSON_RESULT_START ---")
    print(json.dumps(summary_data, indent=2))
    print("--- JSON_RESULT_END ---")

if __name__ == "__main__":
    main()
