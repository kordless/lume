"""
Generate Michael Faraday's Electricity Series Questions and Answers using o3-mini.
Parses all three Volumes (Series I to XXIX) and compiles exactly 4 Q&As per series.
Saves the comprehensive evaluation set to examples/faraday/qna_comprehensive.json.

Usage:
    uv run --with openai python generate_faraday_qna_o3.py
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

SERIES_NAMES = [
    "FIRST SERIES", "SECOND SERIES", "THIRD SERIES", "FOURTH SERIES",
    "FIFTH SERIES", "SIXTH SERIES", "SEVENTH SERIES", "EIGHTH SERIES",
    "NINTH SERIES", "TENTH SERIES", "ELEVENTH SERIES", "TWELFTH SERIES",
    "THIRTEENTH SERIES", "FOURTEENTH SERIES", "FIFTEENTH SERIES",
    "SIXTEENTH SERIES", "SEVENTEENTH SERIES", "EIGHTEENTH SERIES",
    "NINETEENTH SERIES", "TWENTIETH SERIES", "TWENTY-FIRST SERIES",
    "TWENTY-SECOND SERIES", "TWENTY-THIRD SERIES", "TWENTY-FOURTH SERIES",
    "TWENTY-FIFTH SERIES", "TWENTY-SIXTH SERIES", "TWENTY-SEVENTH SERIES",
    "TWENTY-EIGHTH SERIES", "TWENTY-NINTH SERIES"
]

def parse_all_series():
    print("=== PARSING ALL 29 SERIES ACROSS THREE VOLUMES ===")
    faraday_dir = Path(__file__).parent.parent / "examples" / "faraday"
    volume_files = ["book.txt", "book_vol2.txt", "book_vol3.txt"]
    
    # Load and combine all text with basic space normalization
    combined_content = ""
    for v_file in volume_files:
        path = faraday_dir / v_file
        if not path.exists():
            print(f"Error: Required volume file not found: {path}")
            sys.exit(1)
        print(f"Loading {v_file}...")
        with open(path, "r", encoding="utf-8") as f:
            combined_content += "\n" + f.read()
            
    # Normalize whitespaces to single spaces to simplify regex/substring matches
    print("Normalizing corpus whitespace for precise series indexing...")
    normalized_content = re.sub(r'\s+', ' ', combined_content)
    
    # Find all series indices
    series_indices = []
    for s_name in SERIES_NAMES:
        # Find matches of the series name
        # We search case-insensitively but prefer exact uppercase headers
        idx = normalized_content.find(s_name)
        if idx != -1:
            series_indices.append((s_name, idx))
            
    series_indices.sort(key=lambda x: x[1])
    print(f"Matched {len(series_indices)} of the 29 Faraday series headers.")
    
    # Segment the text between series
    series_segments = []
    for i in range(len(series_indices)):
        s_name, start_idx = series_indices[i]
        if i + 1 < len(series_indices):
            end_idx = series_indices[i+1][1]
        else:
            end_idx = len(normalized_content)
            
        # Segment body snippet (take up to 8,000 characters of the series text)
        body = normalized_content[start_idx:end_idx].strip()
        series_segments.append({
            "title": s_name,
            "body": body
        })
        
    return series_segments

def generate_qna_for_series(client, series, model="o3-mini"):
    title = series["title"]
    # We take the first 8000 characters of the normalized text to give o3-mini massive scientific context
    body_snippet = series["body"][:8000]
    
    prompt = f"""You are a scientific historian compiling a premium, fact-based Q&A evaluation benchmark for Michael Faraday's "Experimental Researches in Electricity" (Volumes 1, 2, and 3).

For the following series snippet, generate exactly 4 clear, fact-based questions and their exact direct answers. 

Rules:
1. Questions must be highly specific, scientific, fact-based, and answerable directly from the provided text snippet (avoid vague or generic questions).
2. Focus on physical apparatus, observations, experimental results, and scientific terms used by Faraday (e.g. induction, platina, voltaic pile, diamagnetic, lines of force, bismuth crystal alignment, magneto-crystallic force).
3. Answers must be precise, concise, and direct (no conversational fluff).
4. The response MUST be a valid JSON object matching the schema below:

{{
  "series": "{title}",
  "qna": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}

Series title: {title}
Series snippet:
\"\"\"
{body_snippet}
\"\"\""""

    try:
        # o3-mini does not accept 'temperature' or 'response_format={"type": "json_object"}' parameter in the standard way,
        # but we can enforce JSON structure via developer instruction and prompt guidelines.
        # Developer instruction is passed as system message.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a scientific Q&A generator outputting strictly structured JSON. Output ONLY valid JSON containing no markdown codeblock wraps or explanation text."},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="medium"
        )
        content_received = response.choices[0].message.content.strip()
        
        # Clean any accidental markdown codeblock wrappers
        if content_received.startswith("```json"):
            content_received = content_received[7:]
        if content_received.endswith("```"):
            content_received = content_received[:-3]
        content_received = content_received.strip()
        
        result_json = json.loads(content_received)
        return result_json
    except Exception as e:
        print(f"Error generating Q&A for {title}: {e}")
        return None

def main():
    print("=== LUME COMPREHENSIVE O3-MINI Q&A BENCHMARK GENERATOR ===")
    
    # Check for API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set!")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key)
    
    series_segments = parse_all_series()
    if not series_segments:
        print("Error: No experimental series segments extracted!")
        sys.exit(1)
        
    print(f"\nGenerating premium Q&As using o3-mini for the {len(series_segments)} series in parallel...")
    
    qna_database = []
    t0 = time.time()
    
    # Execute OpenAI API calls in parallel (Thread pool size 8 for o3-mini speed)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(generate_qna_for_series, client, seg): seg for seg in series_segments}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res:
                qna_database.append(res)
                print(f" [{completed}/{len(series_segments)}] Completed Q&A for: {res['series']}")
            else:
                seg = futures[future]
                print(f" [{completed}/{len(series_segments)}] Failed to generate Q&A for: {seg['title']}")
                
    output_path = Path(__file__).parent.parent / "examples" / "faraday" / "qna_comprehensive.json"
    print(f"\nWriting comprehensive Q&A database to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qna_database, f, indent=2, ensure_ascii=False)
        
    t1 = time.time()
    print(f"\n✓ Successfully generated {len(qna_database) * 4} premium o3-mini scientific Q&A benchmarks in {t1 - t0:.1f}s!")
    print(f"Saved comprehensive evaluation set to {output_path}")

if __name__ == "__main__":
    main()
