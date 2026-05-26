import os
import sys

def main():
    paths = [
        r"C:\Users\kordl\Code\Gnosis\weber\reference\mf_ere_vol_1.pdf",
        r"C:\Users\kordl\Code\Gnosis\weber\reference\mf_ere_vol_2.pdf",
        r"C:\Users\kordl\Code\Gnosis\weber\reference\mf_ere_vol_3.pdf"
    ]
    
    import pypdf
    
    for path in paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"\n==========================================")
        print(f"File: {os.path.basename(path)}")
        print(f"Size: {os.path.getsize(path)} bytes")
        
        try:
            reader = pypdf.PdfReader(path)
            num_pages = len(reader.pages)
            print(f"Total pages: {num_pages}")
            
            # Check text in a few pages (e.g. 50, 100, 150)
            pages_to_check = [10, min(50, num_pages-1), min(100, num_pages-1)]
            for p_idx in pages_to_check:
                text = reader.pages[p_idx].extract_text()
                print(f"--- Page {p_idx+1} ---")
                if text and len(text.strip()) > 50:
                    print(f"Text extracted (len: {len(text)}):")
                    print(text[:200].replace("\n", " ") + "...")
                else:
                    print("[No text or very short text]")
        except Exception as e:
            print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    main()
