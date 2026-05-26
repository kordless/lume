import os

path = "src/semantic_mesh.rs"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

target = """                } else {
                    // Jump to a new start pair to maintain natural diversity
                    if !self.start_words.is_empty() {
                        let idx = rng.next_range(0, self.start_words.len());
                        let (start_w1, start_w2) = self.start_words[idx].clone();
                        w1 = start_w1;
                        w2 = start_w2;
                        tokens.push(w1.clone());
                        tokens.push(w2.clone());
                        token_count += 2;
                        continue;
                    } else {
                        break;
                    }
                };"""

replacement = """                } else {
                    // No unvisited transitions remaining from this state. Terminate generation cleanly.
                    break;
                };"""

if target in code:
    code = code.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("SUCCESS")
else:
    # Try normalized newlines (e.g. CRLF)
    target_crlf = target.replace("\n", "\r\n")
    replacement_crlf = replacement.replace("\n", "\r\n")
    if target_crlf in code:
        code = code.replace(target_crlf, replacement_crlf)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("SUCCESS CRLF")
    else:
        print("TARGET NOT FOUND")
