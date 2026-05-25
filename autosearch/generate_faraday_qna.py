"""
Generate Michael Faraday's Electricity Series Questions and Answers.
Uses OpenAI API to compile exactly 4 Q&As per series, formatted as structured JSON,
and saves the evaluation set to examples/faraday/qna.json.

Usage:
    uv run --with openai python generate_faraday_qna.py
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Roman numeral name mappings to parse headers
SERIES_HEADERS = [
    "FIRST SERIES.",
    "SECOND SERIES.",
    "THIRD SERIES.",
    "FOURTH SERIES.",
    "FIFTH SERIES.",
    "SIXTH SERIES.",
    "SEVENTH SERIES.",
    "EIGHTH SERIES.",
    "NINTH SERIES.",
    "TENTH SERIES.",
    "ELEVENTH SERIES.",
    "TWELFTH SERIES.",
    "THIRTEENTH SERIES.",
    "FOURTEENTH SERIES."
]

def parse_series(file_path):
    print(f"Parsing series from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split content by series headers
    series_list = []
    
    # We will find the index of each header in the file to segment the text
    header_indices = []
    for header in SERIES_HEADERS:
        idx = content.find(header)
        if idx != -1:
            header_indices.append((header, idx))
            
    header_indices.sort(key=lambda x: x[1])
    
    for i in range(len(header_indices)):
        header, start_idx = header_indices[i]
        # The body ends where the next header starts, or at the end of the file
        if i + 1 < len(header_indices):
            end_idx = header_indices[i + 1][1]
        else:
            end_idx = len(content)
            
        # Extract and clean body
        body = content[start_idx + len(header):end_idx].strip()
        series_list.append({
            "title": header.replace(".", "").strip(),
            "body": body
        })
        
    print(f"Successfully extracted {len(series_list)} experimental series.")
    return series_list

def generate_qna_for_series(client, series, model="gpt-4o-mini"):
    title = series["title"]
    # Take first 4000 characters to capture the core scientific details and experiments
    body_snippet = series["body"][:4000]
    
    prompt = f"""You are a scientific historian compiling a premium, fact-based Q&A evaluation benchmark for Michael Faraday's "Experimental Researches in Electricity" (Volume 1).

For the following series snippet, generate exactly 4 clear, fact-based questions and their exact direct answers. 

Rules:
1. Questions must be highly specific, scientific, fact-based, and answerable directly from the provided text snippet (avoid vague or generic questions).
2. Focus on physical apparatus, observations, experimental results, and scientific terms used by Faraday (e.g. induction, platina, voltaic pile, diamagnetic, lines of force).
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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a scientific Q&A generator outputting strictly structured JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result_json = json.loads(response.choices[0].message.content)
        return result_json
    except Exception as e:
        print(f"Error generating Q&A for {title}: {e}")
        return None

def main():
    print("=== LUME SCIENTIFIC Q&A BENCHMARK GENERATOR ===")
    
    # Check for API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set!")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key)
    
    faraday_path = Path(__file__).parent.parent / "examples" / "faraday" / "book.txt"
    output_path = Path(__file__).parent.parent / "examples" / "faraday" / "qna.json"
    
    if not faraday_path.exists():
        print(f"Error: Faraday text file not found at {faraday_path}!")
        sys.exit(1)
        
    # Parse series
    series_list = parse_series(faraday_path)
    if not series_list:
        print("Error: No experimental series parsed from the file!")
        sys.exit(1)
        
    print(f"Generating premium Q&As for the {len(series_list)} series in parallel...")
    
    qna_database = []
    
    t0 = time.time()
    # Execute OpenAI API calls in parallel using ThreadPoolExecutor for extreme speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_qna_for_series, client, ser): ser for ser in series_list}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res:
                qna_database.append(res)
                print(f" [{completed}/{len(series_list)}] Completed Q&A for: {res['series']}")
            else:
                ser = futures[future]
                print(f" [{completed}/{len(series_list)}] Failed to generate Q&A for: {ser['title']}")
                
    # Save the Q&A dataset
    print(f"\nWriting Q&A evaluation database to {output_path}...")
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qna_database, f, indent=2, ensure_ascii=False)
        
    t1 = time.time()
    print(f"✓ Successfully generated {len(qna_database) * 4} scientific Q&A benchmarks in {t1 - t0:.1f}s!")
    print(f"Saved database to {output_path}")

if __name__ == "__main__":
    main()
