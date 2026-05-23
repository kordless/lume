fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    args.push("--hybrid".to_string());
    lume::cli::search::run(args);
}
