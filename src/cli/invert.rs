use std::fs;
use std::process;
use std::time::Instant;

use crate::inversion::execute_steered_inversion;
use crate::bm25::{parse_markdown, Bm25Index};
use crate::Tagger;
use crate::hybrid::load_nuts_token;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Plus,
    Minus,
    LParen,
    RParen,
    Source(String),
}

fn tokenize_expr(expr: &str) -> Result<Vec<Token>, String> {
    if !expr.contains('+') && !expr.contains('-') {
        return Ok(vec![Token::Source(expr.to_string())]);
    }
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '[' => {
                let start = i;
                let mut bracket_count = 1;
                i += 1;
                while i < chars.len() && bracket_count > 0 {
                    if chars[i] == '[' {
                        bracket_count += 1;
                    } else if chars[i] == ']' {
                        bracket_count -= 1;
                    }
                    i += 1;
                }
                if bracket_count > 0 {
                    return Err("Unbalanced brackets in inline array".to_string());
                }
                let source_str: String = chars[start..i].iter().collect();
                tokens.push(Token::Source(source_str));
            }
            _ => {
                let start = i;
                while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '+' && chars[i] != '-' && chars[i] != '(' && chars[i] != ')' {
                    i += 1;
                }
                let source_str: String = chars[start..i].iter().collect();
                if !source_str.is_empty() {
                    tokens.push(Token::Source(source_str));
                }
            }
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    token: &'a str,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], token: &'a str) -> Self {
        Self { tokens, pos: 0, token }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_token(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_expression(&mut self) -> Result<Vec<f64>, String> {
        let mut term = self.parse_primary()?;
        while let Some(tok) = self.peek() {
            match tok {
                Token::Plus => {
                    self.next_token();
                    let right = self.parse_primary()?;
                    if term.len() != 768 || right.len() != 768 {
                        return Err("Vector dimension must be exactly 768".to_string());
                    }
                    for idx in 0..768 {
                        term[idx] += right[idx];
                    }
                }
                Token::Minus => {
                    self.next_token();
                    let right = self.parse_primary()?;
                    if term.len() != 768 || right.len() != 768 {
                        return Err("Vector dimension must be exactly 768".to_string());
                    }
                    for idx in 0..768 {
                        term[idx] -= right[idx];
                    }
                }
                _ => break,
            }
        }
        Ok(term)
    }

    fn parse_primary(&mut self) -> Result<Vec<f64>, String> {
        let tok = self.token;
        match self.next_token() {
            Some(Token::Source(src)) => parse_embedding(src, tok),
            Some(Token::LParen) => {
                let val = self.parse_expression()?;
                match self.next_token() {
                    Some(Token::RParen) => Ok(val),
                    _ => Err("Expected matching ')'".to_string()),
                }
            }
            _ => Err("Expected a vector source or '('".to_string()),
        }
    }
}

fn normalize_vector(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

fn compute_symbolic_delta_tags(
    tokens: &[Token],
    token: &str,
    tagger: Option<&Tagger>,
) -> Option<Vec<String>> {
    let tagger = tagger?;
    
    let mut base_vec: Option<Vec<f64>> = None;
    let mut added_vecs: Vec<Vec<f64>> = Vec::new();
    let mut subtracted_vecs: Vec<Vec<f64>> = Vec::new();
    
    let mut current_sign = 1;
    let mut sign_stack = vec![1];
    
    for tok in tokens {
        match tok {
            Token::Plus => {
                current_sign = *sign_stack.last().unwrap_or(&1);
            }
            Token::Minus => {
                current_sign = -*sign_stack.last().unwrap_or(&1);
            }
            Token::LParen => {
                sign_stack.push(current_sign);
            }
            Token::RParen => {
                sign_stack.pop();
                current_sign = *sign_stack.last().unwrap_or(&1);
            }
            Token::Source(src) => {
                if let Ok(v) = parse_embedding(src, token) {
                    if base_vec.is_none() {
                        base_vec = Some(v);
                    } else if current_sign > 0 {
                        added_vecs.push(v);
                    } else {
                        subtracted_vecs.push(v);
                    }
                }
            }
        }
    }
    
    let base_vec = base_vec?;
    if added_vecs.is_empty() && subtracted_vecs.is_empty() {
        return None;
    }
    
    println!("\x1B[35mPerforming individual inversions for concept arithmetic...\x1B[0m");
    
    let base_tags = match crate::inversion::invert_vector(&base_vec, Some(64), token) {
        Ok(resp) => {
            let tags: Vec<String> = tagger.tag(&resp.text).into_iter().map(|t| t.output).collect();
            println!("  - Base vector inverts to: \"{}\" -> Tags: {:?}", resp.text, tags);
            tags
        }
        Err(_) => Vec::new(),
    };
    
    let mut added_tags = Vec::new();
    for v in added_vecs {
        if let Ok(resp) = crate::inversion::invert_vector(&v, Some(64), token) {
            let tags: Vec<String> = tagger.tag(&resp.text).into_iter().map(|t| t.output).collect();
            println!("  - Added vector inverts to: \"{}\" -> Tags: {:?}", resp.text, tags);
            added_tags.extend(tags);
        }
    }
    
    let mut subtracted_tags = Vec::new();
    for v in subtracted_vecs {
        if let Ok(resp) = crate::inversion::invert_vector(&v, Some(64), token) {
            let tags: Vec<String> = tagger.tag(&resp.text).into_iter().map(|t| t.output).collect();
            println!("  - Subtracted vector inverts to: \"{}\" -> Tags: {:?}", resp.text, tags);
            subtracted_tags.extend(tags);
        }
    }
    
    let mut final_tags = base_tags;
    
    added_tags.sort();
    added_tags.dedup();
    
    subtracted_tags.sort();
    subtracted_tags.dedup();
    
    let mut concept_delta = Vec::new();
    for tag in added_tags {
        if !subtracted_tags.contains(&tag) {
            concept_delta.push(tag);
        }
    }
    
    for tag in concept_delta {
        if !final_tags.contains(&tag) {
            final_tags.push(tag);
        }
    }
    
    final_tags.retain(|tag| !subtracted_tags.contains(tag));
    
    println!("\x1B[1;32mDerived Symbolic Delta Tags: {:?}\x1B[0m", final_tags);
    Some(final_tags)
}

pub fn run(mut args: Vec<String>) {
    if args.is_empty() {
        print_usage();
        process::exit(2);
    }

    // Check if shivvr.nuts.services is up. If not, exit immediately.
    if !check_shivvr_status() {
        eprintln!("\x1B[1;31mError: shivvr.nuts.services is currently unreachable. Exiting.\x1B[0m");
        process::exit(1);
    }

    let mut optimize = false;
    if let Some(pos) = args.iter().position(|a| a == "--optimize" || a == "--anneal") {
        args.remove(pos);
        optimize = true;
    }

    let embedding_source = args.remove(0);
    let opt_doc_path = args.first().cloned();

    // 1. Load and validate nuts.services token
    let token = match load_nuts_token() {
        Some(tok) => tok,
        None => {
            eprintln!("\x1B[1;31mError: nuts.services token not found.\x1B[0m");
            eprintln!("\x1B[33mPlease set NUTS_SERVICES_TOKEN in your .env file or environment:\x1B[0m");
            eprintln!("  NUTS_SERVICES_TOKEN=your_token_here");
            process::exit(1);
        }
    };

    // 2. Tokenize and parse embedding vector expression
    let tokens = match tokenize_expr(&embedding_source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("\x1B[1;31mError tokenizing expression:\x1B[0m {}", e);
            process::exit(1);
        }
    };

    let mut parser = Parser::new(&tokens, &token);
    let embedding = match parser.parse_expression() {
        Ok(mut v) => {
            normalize_vector(&mut v);
            v
        }
        Err(e) => {
            eprintln!("\x1B[1;31mError parsing/evaluating expression:\x1B[0m {}", e);
            process::exit(1);
        }
    };

    println!("\x1B[34mSending 768-dim vector to shivvr.nuts.services /invert...\x1B[0m");
    let start_time = Instant::now();

    // 3. Build index and load tagger if a local document is provided
    let mut index = None;
    let mut tagger = None;
    let mut index_elapsed = None;

    if let Some(ref doc_path) = opt_doc_path {
        let tagger_res = match Tagger::from_env() {
            Ok(Some(t)) => Some(t),
            _ => {
                println!("\x1B[33mWarning: FST tagging disabled (DATA not set). Steering might be limited.\x1B[0m");
                Tagger::build(std::iter::empty()).ok()
            }
        };

        if let Some(t) = tagger_res {
            if let Ok(content) = fs::read_to_string(doc_path) {
                let start_idx = Instant::now();
                let sections = parse_markdown(&content);
                let idx = Bm25Index::build(sections, Some(&t));
                index_elapsed = Some(start_idx.elapsed());
                index = Some(idx);
                tagger = Some(t);
            } else {
                eprintln!("\x1B[1;31mWarning: Failed to read document at path {}. Steering disabled.\x1B[0m", doc_path);
            }
        }
    }

    // 4. Compute override steering tags from vector arithmetic if tagger is loaded
    let override_steering_tags = compute_symbolic_delta_tags(&tokens, &token, tagger.as_ref());

    // 5. Query core inversion primitive
    if optimize {
        println!("\x1B[1;36mInitializing Latent Space Gradient Descent (Annealing Loop)...\x1B[0m");
        let target_embedding = embedding.clone();
        let mut running_embedding = embedding;
        
        let mut final_result = None;
        for k in 1..=3 {
            println!();
            println!("\x1B[1;35m┌─── [LATENT ANNEALING ITERATION {}/3] ────────────────────────────────────┐\x1B[0m", k);
            
            let result = match execute_steered_inversion(
                &running_embedding,
                Some(64),
                &token,
                index.as_ref(),
                tagger.as_ref(),
                None, // Let it extract tags dynamically from current inversion
            ) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("\x1B[1;31mInversion failed during annealing:\x1B[0m {}", e);
                    process::exit(1);
                }
            };
            
            println!("  \x1B[1;33mReconstructed Text:\x1B[0m");
            println!("    \"{}\"", result.reconstructed_text);
            println!("  Fidelity Cosine Similarity: \x1B[1m{:.4}\x1B[0m", result.similarity);
            
            if let Some(ref steered_text) = result.steered_text {
                println!("  \x1B[1;36mSteered Markov Synthesis:\x1B[0m");
                println!("    \"{}\"", steered_text);
                
                // Embed the Markov synthesis to get its neural vector
                match embed_text_remotely(steered_text, &token) {
                    Ok(v_markov) => {
                        // Compute target - markov delta
                        let mut delta = vec![0.0; 768];
                        for i in 0..768 {
                            delta[i] = target_embedding[i] - v_markov[i];
                        }
                        
                        // Apply delta to steer running embedding (learning rate = 0.6)
                        for i in 0..768 {
                            running_embedding[i] += 0.6 * delta[i];
                        }
                        normalize_vector(&mut running_embedding);
                        println!("  \x1B[32mOptimized vector coordinates using target delta intersection.\x1B[0m");
                    }
                    Err(e) => {
                        println!("  \x1B[33mWarning: Failed to embed Markov text remotely: {}. Skipping tuning.\x1B[0m", e);
                    }
                }
            } else {
                println!("  \x1B[33mWarning: No Markov steered text generated. Breaking annealing loop.\x1B[0m");
                final_result = Some(result);
                break;
            }
            
            println!("\x1B[1;35m└────────────────────────────────────────────────────────────────────────┘\x1B[0m");
            final_result = Some(result);
        }
        
        if let Some(res) = final_result {
            println!();
            println!("\x1B[1;32m✓ Annealing Optimization Complete!\x1B[0m");
            println!("────────────────────────────────────────────────────────────────────────");
            println!("\x1B[1;32mFINAL OPTIMIZED STEERED SYNTHESIS:\x1B[0m");
            if let Some(ref final_text) = res.steered_text {
                println!("  \"{}\"", final_text);
            } else {
                println!("  \"{}\"", res.reconstructed_text);
            }
            println!("────────────────────────────────────────────────────────────────────────");
        }
    } else {
        match execute_steered_inversion(
            &embedding,
            Some(64),
            &token,
            index.as_ref(),
            tagger.as_ref(),
            override_steering_tags,
        ) {
            Ok(result) => {
                let elapsed = start_time.elapsed();
                result.print_cli(opt_doc_path.as_deref(), elapsed, index_elapsed);
            }
            Err(e) => {
                eprintln!("\x1B[1;31mInversion failed:\x1B[0m {}", e);
                process::exit(1);
            }
        }
    }
}

fn print_usage() {
    println!();
    println!("\x1B[1;33mUSAGE:\x1B[0m");
    println!("  lume \x1B[36minvert\x1B[0m <embedding_source> [optional_document.md]");
    println!();
    println!("\x1B[1;33mARGUMENTS:\x1B[0m");
    println!("  \x1B[36m<embedding_source>\x1B[0m    Either:");
    println!("                          - Path to a JSON file containing the 768-dim float array.");
    println!("                          - A raw JSON array inline string (e.g. \"[0.01, -0.02, ...]\").");
    println!("                          - A math expression (e.g. \"v3.json - v1.json + v2.json\").");
    println!("                          - A raw text string to embed remotely via shivvr.");
    println!("  \x1B[36m[optional_document.md]\x1B[0m Path to a local document to style the reconstruction.");
    println!("                          Lume will FST-tag the inverted text, extract topics, and steer");
    println!("                          stochastic generation over the document corpus.");
    println!();
    println!("\x1B[1;33mEXAMPLES:\x1B[0m");
    println!("  lume invert examples/my_vector.json");
    println!("  lume invert \"Why do you think 99% of Teslas...\" examples/monte_cristo.md");
    println!();
}

fn parse_embedding(source: &str, token: &str) -> Result<Vec<f64>, String> {
    if source.trim().starts_with('[') {
        match serde_json::from_str::<Vec<f64>>(source) {
            Ok(v) => return Ok(v),
            Err(e) => return Err(format!("Failed to parse inline JSON vector: {}", e)),
        }
    }

    // Try reading as a file path
    if std::path::Path::new(source).is_file() {
        if let Ok(content) = fs::read_to_string(source) {
            if let Ok(v) = serde_json::from_str::<Vec<f64>>(&content) {
                return Ok(v);
            }
        }
    }

    // Treat as raw text to embed remotely
    println!("\x1B[34mEncoding raw text query remotely via shivvr ingest: \"{}\"\x1B[0m", source);
    embed_text_remotely(source, token)
}

fn embed_text_remotely(text: &str, token: &str) -> Result<Vec<f64>, String> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let sess = format!("lume-embed-{}", timestamp);
    
    let url = format!("https://shivvr.nuts.services/temp/{}/ingest", sess);
    
    #[derive(serde::Serialize)]
    struct IngestPayload<'a> {
        text: &'a str,
        source: &'a str,
    }
    
    let payload = IngestPayload {
        text,
        source: "0",
    };
    
    let auth_header = format!("Bearer {}", token);
    
    match ureq::post(&url)
        .set("Authorization", &auth_header)
        .set("Content-Type", "application/json")
        .send_json(&payload)
    {
        Ok(res) => {
            let body_text = match res.into_string() {
                Ok(s) => s,
                Err(e) => return Err(format!("Failed to read response body: {}", e)),
            };
            
            let val: serde_json::Value = match serde_json::from_str(&body_text) {
                Ok(v) => v,
                Err(e) => {
                    let cleanup_url = format!("https://shivvr.nuts.services/temp/{}", sess);
                    ureq::delete(&cleanup_url).set("Authorization", &auth_header).call().ok();
                    return Err(format!("Failed to parse JSON as Value: {}\nBody: {}", e, body_text));
                }
            };

            // Cleanup session
            let cleanup_url = format!("https://shivvr.nuts.services/temp/{}", sess);
            ureq::delete(&cleanup_url).set("Authorization", &auth_header).call().ok();

            // Support either single object, chunks array, or direct array of objects
            let first_obj = if let Some(chunks_val) = val.get("chunks") {
                if let Some(arr) = chunks_val.as_array() {
                    arr.first()
                } else {
                    None
                }
            } else if let Some(arr) = val.as_array() {
                arr.first()
            } else if val.is_object() {
                Some(&val)
            } else {
                return Err(format!("Unexpected JSON format: expected object or array, got {}", val));
            };

            if let Some(obj) = first_obj {
                if let Some(embedding_val) = obj.get("embedding") {
                    if let Ok(embedding) = serde_json::from_value::<Vec<f64>>(embedding_val.clone()) {
                        if embedding.len() == 768 {
                            return Ok(embedding);
                        } else {
                            return Err(format!("Expected 768-dimensional embedding, got {}", embedding.len()));
                        }
                    }
                }
            }
            Err(format!("No embedding array found in ingest response: {}", body_text))
        }
        Err(e) => {
            let cleanup_url = format!("https://shivvr.nuts.services/temp/{}", sess);
            ureq::delete(&cleanup_url).set("Authorization", &auth_header).call().ok();
            Err(format!("Remote embedding request failed: {}", e))
        }
    }
}


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
