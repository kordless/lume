import os
import sys
import re
import urllib.request

def download_vol3():
    url = "https://archive.org/download/experimentalres03faragoog/experimentalres03faragoog_djvu.txt"
    dest = "/workspace/rust-fstguardrails/examples/faraday/book_vol3.txt"
    print(f"Downloading Volume III text from {url}...")
    try:
        # User-Agent to avoid blocker
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8', errors='ignore')
        
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Volume III saved successfully to {dest} ({len(content)} chars).")
        return True
    except Exception as e:
        print(f"Error downloading Volume III: {e}")
        return False

def extract_vol2_on_host():
    # Since pypdf is needed on the host, this script will run on the host
    pdf_path = r"C:\Users\kordl\Code\Gnosis\weber\reference\mf_ere_vol_2.pdf"
    dest_path = r"C:\Users\kordl\Code\research\rust-fstguardrails\examples\faraday\book_vol2.txt"
    
    print(f"Extracting Volume II text from {pdf_path} using pypdf...")
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return False
        
    try:
        import pypdf
    except ImportError:
        print("pypdf is not available. Please run with 'uv run --with pypdf'.")
        return False
        
    try:
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"Total pages to extract: {num_pages}")
        
        extracted_text = []
        for i in range(num_pages):
            text = reader.pages[i].extract_text()
            if text:
                extracted_text.append(text)
            if (i+1) % 50 == 0:
                print(f"  Extracted {i+1}/{num_pages} pages...")
                
        full_text = "\n".join(extracted_text)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Volume II saved successfully to {dest_path} ({len(full_text)} chars).")
        return True
    except Exception as e:
        print(f"Error extracting Volume II: {e}")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--vol2-only":
        extract_vol2_on_host()
    elif len(sys.argv) > 1 and sys.argv[1] == "--vol3-only":
        download_vol3()
    else:
        # Run both
        # Volume II needs to be extracted on the host via pypdf
        extract_vol2_on_host()
        # Volume III can be downloaded inside the container or host
        download_vol3()

if __name__ == "__main__":
    main()
