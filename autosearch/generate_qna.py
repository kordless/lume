"""
Generate Count of Monte Cristo Chapter Questions and Answers.
Uses OpenAI API to compile exactly 4 Q&As per chapter, formatted as structured JSON,
and saves the evaluation set to examples/monte_cristo_qna.json.

Usage:
    uv run python generate_qna.py
"""

import os
import sys
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def clean_chapter_title(title_line):
    # e.g., "# Chapter 1. Marseilles—The Arrival" -> "Chapter 1: Marseilles—The Arrival"
    title = title_line.replace("#", "").strip()
    title = re.sub(r"^chapter\s+(\d+)\.\s*", r"Chapter \1: ", title, flags=re.IGNORECASE)
    return title

def parse_chapters(file_path):
    print(f"Parsing chapters from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split content by line breaks matching chapter headers
    lines = content.splitlines()
    chapters = []
    current_chapter_title = None
    current_chapter_body = []
    
    for line in lines:
        if line.strip().lower().startswith("# chapter "):
            if current_chapter_title:
                chapters.append({
                    "title": current_chapter_title,
                    "body": "\n".join(current_chapter_body)
                })
            current_chapter_title = clean_chapter_title(line)
            current_chapter_body = []
        else:
            if current_chapter_title:
                current_chapter_body.append(line)
                
    if current_chapter_title:
        chapters.append({
            "title": current_chapter_title,
            "body": "\n".join(current_chapter_body)
        })
        
    print(f"Successfully extracted {len(chapters)} chapters.")
    return chapters

def generate_qna_for_chapter(client, chapter, model="gpt-4o-mini"):
    title = chapter["title"]
    # Take first 4000 characters to capture the core chapter plot
    body_snippet = chapter["body"][:4000]
    
    prompt = f"""You are a literary analyst compiling a premium, fact-based Q&A evaluation benchmark for Alexandre Dumas's "The Count of Monte Cristo".

For the following chapter snippet, generate exactly 4 clear, fact-based questions and their exact direct answers. 

Rules:
1. Questions must be highly specific, fact-based, and answerable directly from the provided text snippet (avoid vague or generic questions).
2. Answers must be precise, concise, and direct (no conversational fluff).
3. The response MUST be a valid JSON object matching the schema below:

{{
  "chapter": "{title}",
  "qna": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}

Chapter title: {title}
Chapter snippet:
\"\"\"
{body_snippet}
\"\"\""""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a literary Q&A generator outputting strictly structured JSON."},
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
    print("=== LUME CORPUS Q&A BENCHMARK GENERATOR ===")
    
    # Check for API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set!")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key)
    
    monte_cristo_path = Path(__file__).parent.parent / "examples" / "monte_cristo" / "book.md"
    output_path = Path(__file__).parent.parent / "examples" / "monte_cristo" / "qna.json"
    
    if not monte_cristo_path.exists():
        print(f"Error: Count of Monte Cristo markdown file not found at {monte_cristo_path}!")
        sys.exit(1)
        
    # Parse chapters
    chapters = parse_chapters(monte_cristo_path)
    
    # Set limit: Process the first 15 chapters as a highly representative subset of Volume 1,
    # or process all of them if desired. Let's make it configurable or do 20 chapters as an excellent evaluation set!
    max_eval_chapters = 20
    target_chapters = chapters[:max_eval_chapters]
    print(f"Generating premium Q&As for the first {max_eval_chapters} chapters in parallel...")
    
    qna_database = []
    
    t0 = time.time()
    # Execute OpenAI API calls in parallel using ThreadPoolExecutor for extreme speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_qna_for_chapter, client, ch): ch for ch in target_chapters}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res:
                qna_database.append(res)
                print(f" [{completed}/{len(target_chapters)}] Completed Q&A for: {res['chapter']}")
            else:
                ch = futures[future]
                print(f" [{completed}/{len(target_chapters)}] Failed to generate Q&A for: {ch['title']}")
                
    # Save the Q&A dataset
    print(f"\nWriting Q&A evaluation database to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qna_database, f, indent=2, ensure_ascii=False)
        
    t1 = time.time()
    print(f"✓ Successfully generated {len(qna_database) * 4} Q&A benchmarks in {t1 - t0:.1f}s!")
    print(f"Saved database to {output_path}")

if __name__ == "__main__":
    import time
    main()
