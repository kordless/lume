import os

path = "src/cli/invert.rs"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add check_raw_status check inside run
target_run = """pub fn run(mut args: Vec<String>) {
    if args.is_empty() {
        print_usage();
        process::exit(2);
    }"""

replacement_run = """pub fn run(mut args: Vec<String>) {
    if args.is_empty() {
        print_usage();
        process::exit(2);
    }

    // Check if shivvr.nuts.services is up. If not, exit immediately.
    if !check_shivvr_status() {
        eprintln!("\\x1B[1;31mError: shivvr.nuts.services is currently unreachable. Exiting.\\x1B[0m");
        process::exit(1);
    }"""

# 2. Modify print_usage to remove "dummy" reference
target_usage = """fn print_usage() {
    println!();
    println!("\\x1B[1;33mUSAGE:\\x1B[0m");
    println!("  lume \\x1B[36minvert\\x1B[0m <embedding_source> [optional_document.md]");
    println!();
    println!("\\x1B[1;33mARGUMENTS:\\x1B[0m");
    println!("  \\x1B[36m<embedding_source>\\x1B[0m    Either:");
    println!("                          - Path to a JSON file containing the 768-dim float array.");
    println!("                          - A raw JSON array inline string (e.g. \\"[0.01, -0.02, ...]\\").");
    println!("                          - A math expression (e.g. \\"v3.json - v1.json + v2.json\\" or \\"dummy3 - dummy1 + dummy2\\").");
    println!("                          - \\"dummy\\" or \\"test\\" to generate a sample 768-dim dummy vector.");
    println!("  \\x1B[36m[optional_document.md]\\x1B[0m Path to a local document to style the reconstruction.");
    println!("                          Lume will FST-tag the inverted text, extract topics, and steer");
    println!("                          stochastic generation over the document corpus.");
    println!();
    println!("\\x1B[1;33mEXAMPLES:\\x1B[0m");
    println!("  lume invert examples/my_vector.json");
    println!("  lume invert \\"dummy3 - dummy1 + dummy2\\" examples/monte_cristo.md");
    println!();
}"""

replacement_usage = """fn print_usage() {
    println!();
    println!("\\x1B[1;33mUSAGE:\\x1B[0m");
    println!("  lume \\x1B[36minvert\\x1B[0m <embedding_source> [optional_document.md]");
    println!();
    println!("\\x1B[1;33mARGUMENTS:\\x1B[0m");
    println!("  \\x1B[36m<embedding_source>\\x1B[0m    Either:");
    println!("                          - Path to a JSON file containing the 768-dim float array.");
    println!("                          - A raw JSON array inline string (e.g. \\"[0.01, -0.02, ...]\\").");
    println!("                          - A math expression (e.g. \\"v3.json - v1.json + v2.json\\").");
    println!("                          - A raw text string to embed remotely via shivvr.");
    println!("  \\x1B[36m[optional_document.md]\\x1B[0m Path to a local document to style the reconstruction.");
    println!("                          Lume will FST-tag the inverted text, extract topics, and steer");
    println!("                          stochastic generation over the document corpus.");
    println!();
    println!("\\x1B[1;33mEXAMPLES:\\x1B[0m");
    println!("  lume invert examples/my_vector.json");
    println!("  lume invert \\"Why do you think 99% of Teslas...\\" examples/monte_cristo.md");
    println!();
}"""

# 3. Completely remove "dummy" and "test" interception inside parse_embedding
target_parse = """fn parse_embedding(source: &str, token: &str) -> Result<Vec<f64>, String> {
    if source.starts_with("dummy") || source.starts_with("test") {
        let seed_val: f64 = source
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<f64>()
            .unwrap_or(1.0);
            
        let mut mock = vec![0.0; 768];
        for (i, val) in mock.iter_mut().enumerate() {
            *val = ((i as f64 * 0.1337 * seed_val).sin()) * 0.05;
        }
        let norm: f64 = mock.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut mock {
                *x /= norm;
            }
        }
        return Ok(mock);
    }

    if source.trim().starts_with('[') {"""

replacement_parse = """fn parse_embedding(source: &str, token: &str) -> Result<Vec<f64>, String> {
    if source.trim().starts_with('[') {"""

# 4. Add check_shivvr_status helper at the very bottom of the file
helper_code = """

fn check_shivvr_status() -> bool {
    let url = "https://shivvr.nuts.services";
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(5))
        .build();
    match agent.get(url).call() {
        Ok(_) => true,
        Err(ureq::Error::Status(_, _)) => true, // Got a response back (e.g. 401), so it's UP!
        Err(_) => false, // Network error, timeout, or DNS failure
    }
}
"""

def apply_replace(code_str, target, replacement):
    if target in code_str:
        return code_str.replace(target, replacement), True
    target_crlf = target.replace("\n", "\r\n")
    replacement_crlf = replacement.replace("\n", "\r\n")
    if target_crlf in code_str:
        return code_str.replace(target_crlf, replacement_crlf), True
    return code_str, False

modified = False
code, ok = apply_replace(code, target_run, replacement_run)
if ok:
    modified = True
    print("RUN replaced")

code, ok = apply_replace(code, target_usage, replacement_usage)
if ok:
    modified = True
    print("USAGE replaced")

code, ok = apply_replace(code, target_parse, replacement_parse)
if ok:
    modified = True
    print("PARSE replaced")

if modified:
    # Append helper code
    if code.endswith("\r\n"):
        code += helper_code.replace("\n", "\r\n")
    else:
        code += helper_code
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("SUCCESS")
else:
    print("NO CHANGES DETECTED")
