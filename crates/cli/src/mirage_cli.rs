// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain mirage quantize` - write the packed deployment archive from the FP32 checkpoint.
//!
//! Swedish Embedded AB implements solutions for shrinking perception models to the memory of the device they run
//! on for its clients. If your team needs expertise in low-bit deployment of vision models then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The archive `--out` receives is the file `brain mirage describe --precision cq` serves with its matrices packed
//! on the device, and the vendor engine loads it too. The quantisation plan (which tensors are CQ at which width,
//! codebooks, tokenizer) is the one of the `mirage.cact` already in the weights directory.

use std::path::PathBuf;

use crate::args::Args;

const USAGE: &str = "\
brain mirage quantize --out DIR [--weights DIR]

  --weights DIR   directory with config.json, mirage.safetensors and the released mirage.cact (the plan);
                  default: the model store's Cactus-Compute/mirage
  --out DIR       destination directory: receives mirage.cact (written through a temporary name) and config.json

An adapter in the weights directory is folded into the FP32 weights first.
";

/// The default weights directory: the model store's copy of the release.
fn store_dir() -> Result<PathBuf, String> {
    let root = brain_modelstore::explicit_models_root().or_else(brain_modelstore::default_root).ok_or("no model store: pass --weights DIR")?;
    let reference = brain_modelref::ModelRef::parse(mirage::caps::MODEL).map_err(|e| e.to_string())?;
    Ok(brain_modelstore::Store::new(root).repo_dir(&reference))
}

/// `brain mirage quantize ...`; `args` follow the verb. Exits the process on a usage error or a failure.
pub fn run_quantize(args: &[String]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    let mut args = Args::new(args);
    let result = (|| -> Result<PathBuf, String> {
        let out = PathBuf::from(args.take_str("--out").ok_or("--out DIR is required")?);
        let weights = match args.take_str("--weights") {
            Some(dir) => PathBuf::from(dir),
            None => store_dir()?,
        };
        mirage::model::quantize_dir(&weights, &out).map_err(|e| e.to_string())
    })();
    match result {
        Ok(path) => println!("wrote {}", path.display()),
        Err(e) => {
            eprintln!("brain mirage quantize: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}
