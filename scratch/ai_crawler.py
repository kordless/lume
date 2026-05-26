import json
import os
import sys
import urllib.parse
import urllib.request
import subprocess
import time

# 25 AI-optimized queries
QUERIES = [
    "Artificial Intelligence",
    "Large Language Model",
    "Generative AI",
    "AGI",
    "GPT-4",
    "Claude 3",
    "Llama 3",
    "Vector Database",
    "Retrieval-Augmented Generation",
    "AI Agent",
    "Transformer Architecture",
    "Neural Network",
    "Deep Learning",
    "Machine Learning",
    "OpenAI",
    "Anthropic",
    "NVIDIA GPU",
    "Prompt Engineering",
    "Fine Tuning",
    "Reinforcement Learning",
    "RLHF",
    "PyTorch",
    "AI Copilot",
    "Diffusion Model",
    "Stable Diffusion"
]

def clean_filename(s):
    return "".join(c if c.isalnum() else "_" for c in s).lower()

def main():
    print("🚀 Starting AI Hacker News Algolia Collector...")
    print(f"Targeting {len(QUERIES)} search queries.")

    crawled_dir = "examples/crawled"
    os.makedirs(crawled_dir, exist_ok=True)

    # Compile the lume release binary just in case
    print("⚙️ Verifying lume binary is compiled in release mode...")
    lume_bin = "./target/release/lume"
    if not os.path.exists(lume_bin):
        print("  ➔ lume binary not found! Compiling...")
        subprocess.run(["$HOME/.cargo/bin/cargo", "build", "--release"], check=True)
    else:
        print("  ➔ lume release binary exists.")

    total_crawled = 0

    for idx, query in enumerate(QUERIES, 1):
        print(f"\n────────────────────────────────────────────────────────────")
        print(f"🔍 [{idx}/{len(QUERIES)}] Query: '{query}'")
        print(f"────────────────────────────────────────────────────────────")

        # Encode query for Algolia
        encoded_query = urllib.parse.quote(query)
        # Search API url (fetch top 10 stories)
        url = f"https://hn.algolia.com/api/v1/search?query={encoded_query}&tags=story&hitsPerPage=10"

        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"  ❌ Failed to query Algolia API for '{query}': {e}")
            continue

        hits = data.get("hits", [])
        print(f"  ➔ Found {len(hits)} hits on Algolia.")

        if not hits:
            print("  ⚠️ No hits returned.")
            continue

        # 1. Generate Search Results Page Markdown
        slug = clean_filename(query)
        search_page_file = f"{crawled_dir}/search_algolia_{slug}.md"
        
        md_content = []
        md_content.append(f"# Hacker News Algolia Search: \"{query}\"\n")
        md_content.append(f"*   **Query Term**: {query}")
        md_content.append(f"*   **Source**: hn.algolia.com Search API")
        md_content.append(f"*   **Crawl Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        md_content.append("---\n")
        md_content.append("## Top 10 Search Results\n")

        for r_idx, hit in enumerate(hits, 1):
            title = hit.get("title", "No Title")
            points = hit.get("points", 0)
            author = hit.get("author", "anonymous")
            story_id = hit.get("objectID")
            external_url = hit.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
            hn_url = f"https://news.ycombinator.com/item?id={story_id}"
            
            md_content.append(f"### {r_idx}. {title}")
            md_content.append(f"*   **Points**: {points} | **Author**: {author}")
            md_content.append(f"*   **HN Discussion**: [HN Link]({hn_url})")
            md_content.append(f"*   **Original Link**: [Original Article]({external_url})")
            md_content.append("")

        with open(search_page_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        print(f"  💾 Saved search result index to '{search_page_file}'")

        # 2. Crawl the Top 10 discussion pages
        print(f"  🕷️ Crawling top {len(hits)} discussion pages...")
        for r_idx, hit in enumerate(hits, 1):
            story_id = hit.get("objectID")
            if not story_id:
                continue

            hn_item_url = f"https://news.ycombinator.com/item?id={story_id}"
            print(f"    [{r_idx}/{len(hits)}] Crawling: https://news.ycombinator.com/item?id={story_id}")
            
            # Call our CLI binary to crawl the Hacker News post
            try:
                # We run `./target/release/lume crawl <url>`
                res = subprocess.run(
                    [lume_bin, "crawl", hn_item_url],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if res.returncode == 0:
                    print("      ✓ Success!")
                    total_crawled += 1
                else:
                    print(f"      ❌ Failed! Error: {res.stderr.strip()}")
            except Exception as ex:
                print(f"      ❌ Command failed to execute: {ex}")

            # Sleep briefly to be respectful to APIs
            time.sleep(0.1)

    print(f"\n🎉 Collection successfully built!")
    print(f"  - Total queries indexed: {len(QUERIES)}")
    print(f"  - Total discussion pages crawled: {total_crawled}")
    print(f"  - Search indices written to: '{crawled_dir}/search_algolia_*.md'")
    print(f"You can now search this entire corpus immediately using:")
    print(f"  lume search examples/crawled \"your query\"")

if __name__ == "__main__":
    main()
