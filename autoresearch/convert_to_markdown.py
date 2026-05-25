import os
import re
import shutil
from pathlib import Path

def clean_and_title_block(text, block_idx):
    text_clean = re.sub(r'\s+', ' ', text).strip()
    if not text_clean:
        return None, None
        
    # Check if it starts with a standard paragraph number e.g. "1796." or "1796 . "
    para_match = re.match(r"^(\d+)\s*\.\s*(.*)", text_clean)
    if para_match:
        num = para_match.group(1)
        body = para_match.group(2)
        return f"Paragraph {num}", f"{num}. {body}"
        
    # Check if it starts with a page / section header e.g. "512  Subterraneous  telegra2Jh"
    header_match = re.match(r"^(\d{3,})\s+([A-Za-z\s\-]+)(.*)", text_clean)
    if header_match:
        page_num = header_match.group(1)
        title_text = header_match.group(2).strip()
        body_rest = header_match.group(3).strip()
        return f"Page {page_num} - {title_text}", f"{page_num} {title_text} {body_rest}"
        
    # Fallback: Use the first 6 words as a descriptive title
    words = text_clean.split()
    title_words = words[:6]
    title_text = " ".join(title_words)
    # Remove trailing punctuation from title
    title_text = re.sub(r"[^\w\s\-\']", "", title_text).strip()
    return f"Section {block_idx} - {title_text}", text_clean

def parse_into_paragraphs(raw_text):
    lines = raw_text.splitlines()
    blocks = []
    current_block = []
    
    # Matches a line starting with a paragraph number e.g. "1840. In this way" or "1840 . "
    pattern_para = re.compile(r"^\s*(\d+)\s*\.\s+(.*)")
    # Matches a line starting with a page/section header e.g. "34 Inactive circles" or "512 Subterraneous"
    pattern_header = re.compile(r"^\s*(\d{2,})\s+([A-Z][a-zA-Z\s\-]+)(.*)")
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            continue
            
        # Start a new paragraph block if the line starts with a new paragraph number or a major header
        if pattern_para.match(stripped) or pattern_header.match(stripped):
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
                
        current_block.append(line)
        
    if current_block:
        blocks.append("\n".join(current_block))
        
    return blocks

def main():
    print("=== OVERHAULING FARADAY CORPUS TO 100% COMPLETE MARKDOWN PARAGRAPHS ===")
    
    faraday_dir = Path(__file__).parent.parent / "examples" / "faraday"
    raw_dir = faraday_dir / ".raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Sourced from hidden cache
    volume_files = ["book.txt", "book_vol2.txt", "book_vol3.txt"]
    
    for idx, v_file in enumerate(volume_files, 1):
        v_path = raw_dir / v_file
        if not v_path.exists():
            # Try original path in case it wasn't moved
            v_path = faraday_dir / v_file
            if not v_path.exists():
                print(f"Error: Sourced raw file not found: {v_file}")
                continue
                
        print(f"Ingesting raw text from {v_path.name}...")
        with open(v_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Gutenberg files (book.txt) need to be sliced between start/end blocks
        if "GUTENBERG" in content:
            lines = content.splitlines()
            story_lines = []
            in_story = False
            for line in lines:
                stripped = line.strip()
                if "*** START OF THE PROJECT" in stripped or "*** START OF THIS PROJECT" in stripped:
                    in_story = True
                    continue
                if "*** END OF THE PROJECT" in stripped or "*** END OF THIS PROJECT" in stripped:
                    break
                if in_story:
                    story_lines.append(line)
            raw_text = "\n".join(story_lines)
        else:
            raw_text = content
            
        # Parse into paragraphs using the streaming accumulator
        raw_blocks = parse_into_paragraphs(raw_text)
        
        md_file = faraday_dir / f"book_vol{idx}.md"
        print(f"Writing complete structured Markdown to {md_file.name}...")
        
        written_count = 0
        with open(md_file, "w", encoding="utf-8") as out:
            out.write(f"# Volume {idx}: Experimental Researches in Electricity (Complete)\n\n")
            out.write(f"Michael Faraday's scientific experimental logs for Volume {idx}. Fully parsed at paragraph granularity.\n\n")
            
            for block_idx, block in enumerate(raw_blocks, 1):
                title, body = clean_and_title_block(block, block_idx)
                if title and body:
                    out.write(f"# {title}\n")
                    out.write(f"{body}\n\n")
                    written_count += 1
                    
        print(f"  ➔ Successfully indexed {written_count} separate paragraph-level records in {md_file.name}.")
        
    print("\n✓ Faraday E2E complete markdown database overbuilt successfully!")

if __name__ == "__main__":
    main()
