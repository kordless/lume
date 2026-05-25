"""
Generate synthetic agent tool-use trajectories (Thought-Action-Observation)
for training a mini offline Agentic LLM.
Saves the resulting corpus to examples/agent/book.txt.

Usage:
    uv run --with openai python generate_agent_trajectories.py
"""

import os
import sys
from pathlib import Path
from openai import OpenAI

# Define tool schemas for context
TOOLS_DESCRIPTION = """
Available Lume Subcommand Tools:
1. "lume_tag": Extract offline entity tags from a text block.
   Arguments: {"text": "string"}
   Returns: JSON list of identified terms, their kind, and character offsets.

2. "lume_search": Semantic hybrid field-aware BM25 search.
   Arguments: {"query": "string"}
   Returns: Markdown formatting of search matches and document previews.

3. "lume_generate": Steered stochastic text path generation.
   Arguments: {"prompt": "string", "steer_tag": "string"}
   Returns: Completed stochastically generated text and attention feedback traces.
"""

def generate_trajectories(client, num_trajectories=5, model="gpt-4o-mini"):
    prompt = f"""You are a senior AI research engineer compiling a training dataset for an offline next-token prediction AI Agent. 

Your task is to generate exactly {num_trajectories} highly detailed, realistic, and diverse agent reasoning and tool-use trajectories.

Rules:
1. Format each trajectory exactly using the following markup structure:
[User]: <A user prompt that requires using search, tagging, or generation>
[Assistant]: <thought> I need to analyze the user's prompt. I should call the appropriate tool... </thought>
<tool_call name="tool_name"> {{"arg_name": "arg_val"}} </tool_call>
<tool_response> {{"response_data_or_markdown_mock_output": "..."}} </tool_response>
<thought> Analyzing the tool output. The results show X. I can now form the final answer. </thought>
[Answer]: <The definitive final answer to the user's question, fully resolved.>

2. Use ONLY the following 3 tools:
{TOOLS_DESCRIPTION}

3. Ensure absolute syntax consistency:
- Tool calls MUST be valid JSON arguments inside the `<tool_call>` tag.
- The `<thought>` tag MUST open and close cleanly.
- Keep the user queries realistic for a system dealing with Count of Monte Cristo (literary) or Michael Faraday (scientific electromagnetism).
- Separate each trajectory with two newlines.

Generate exactly {num_trajectories} trajectories. Do not add any extra markdown wrapper or conversation outside the requested format.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a synthetic dataset generator outputting strictly structured agent trajectories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating trajectories: {e}")
        return None

def main():
    print("=== LUME AGENT TRAJECTORY GENERATOR ===")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set!")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key)
    
    output_dir = Path(__file__).parent.parent / "examples" / "agent"
    output_path = output_dir / "book.txt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overwrite old file if it exists
    if output_path.exists():
        output_path.unlink()
        
    total_trajectories = 30
    chunk_size = 5
    iterations = total_trajectories // chunk_size
    
    for i in range(iterations):
        print(f"\n--- Batch {i+1}/{iterations} (Generating {chunk_size} trajectories) ---")
        chunk_text = generate_trajectories(client, num_trajectories=chunk_size)
        if chunk_text:
            cleaned = chunk_text.strip()
            # Append immediately to the file
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(cleaned + "\n\n")
            print(f"✓ Saved Batch {i+1} to {output_path}")
        else:
            print(f"Error: Failed to generate Batch {i+1}")
            sys.exit(1)
            
    print(f"\n✓ Successfully generated {total_trajectories} agent trajectories dataset!")

if __name__ == "__main__":
    main()
