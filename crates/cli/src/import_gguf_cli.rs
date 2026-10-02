// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain import-gguf`: argument parsing and output for the generic GGUF
//! conversion, which is `serving::gguf_import`'s registry.

/// `brain import-gguf FILE [--out PATH] [--id NAME]` / `brain import-gguf --list`:
/// the ONE generic conversion command, dispatching by the file's own
/// `general.architecture` through [`import_file`]. Replaces the per-model
/// `brain qwen35moe import` (which still works, as a thin wrapper).
pub fn run_import_gguf(args: &[String]) {
    let usage = "usage: brain import-gguf FILE [--out PATH] [--id VENDOR/REPO]\n       brain import-gguf --list";
    if args.iter().any(|a| a == "--list") {
        println!("registered GGUF architectures (general.architecture -> importer):");
        for i in serving::gguf_import::importers() {
            let key = match i.projector() {
                Some(p) => format!("{}/{p}", i.architecture()),
                None => i.architecture().to_string(),
            };
            let how = if i.loads_directly() { "direct" } else { "convert" };
            println!("  {key:<22} [{how:>7}] {}", i.summary());
        }
        return;
    }
    let mut file: Option<String> = None;
    let mut out: Option<String> = None;
    let mut id: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "--gguf" | "--id" => {
                let flag = args[i].clone();
                i += 1;
                let Some(v) = args.get(i).cloned() else {
                    eprintln!("{flag} requires a value\n{usage}");
                    std::process::exit(2);
                };
                match flag.as_str() {
                    // `--gguf` is accepted as an alias for the positional FILE so
                    // the old `brain qwen35moe import --gguf F --out O` spelling
                    // keeps working verbatim through the generic command.
                    "--gguf" => file = Some(v),
                    "--out" => out = Some(v),
                    _ => id = Some(v),
                }
            }
            "-h" | "--help" => {
                println!("{usage}");
                return;
            }
            other if other.starts_with("--") => eprintln!("ignoring unknown flag {other:?}"),
            other if file.is_none() => file = Some(other.to_string()),
            other => eprintln!("ignoring extra argument {other:?}"),
        }
        i += 1;
    }
    let Some(file) = file else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    match serving::gguf_import::import_file(&file, out.as_deref(), id.as_deref()) {
        Ok(out) => eprintln!("import-gguf: {file} -> {out}"),
        Err(e) => {
            eprintln!("brain import-gguf: {e}");
            std::process::exit(1);
        }
    }
}
