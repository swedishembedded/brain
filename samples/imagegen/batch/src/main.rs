// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sample application: build one pipeline, generate many prompts.
//!
//! The lesson this sample exists to teach is the one that decides whether an
//! image feature is affordable in a product: **the model load is the
//! expensive part, and it happens once.** `ImagePipeline` holds real,
//! multi-gigabyte device memory for as long as it lives, so a service
//! generates from a pipeline it already has rather than building one per
//! request. This binary shows the shape of that: one
//! [`brain::ImagePipeline`], N prompts, N images, with the per-image cost
//! printed so the difference is visible rather than asserted.
//!
//! ```text
//! printf 'a whale submarine\na lighthouse in fog\n' > prompts.txt
//! make samples/imagegen/batch/run ARGS="--prompts prompts.txt --out-dir out/batch"
//! ```
//!
//! Swedish Embedded AB implements production image-generation services for
//! its clients, where model residency and per-request cost decide the
//! hardware bill. If your team needs a generation pipeline sized for real
//! traffic, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
sample-imagegen-batch - generate many prompts from one loaded pipeline

USAGE:
    sample-imagegen-batch --prompts FILE [--model ID] [--out-dir DIR] [--seed N]

OPTIONS:
    --prompts FILE  one prompt per line; blank lines and '#' comments skipped
    --model ID      model to resolve (default: black-forest-labs/FLUX.2-klein-4B)
    --out-dir DIR   where the PNGs go (default: out/sample-imagegen-batch)
    --seed N        base seed; image i uses seed + i, so a run is reproducible
";

struct Args {
    model: String,
    prompts: Option<String>,
    out_dir: String,
    seed: Option<u64>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: "black-forest-labs/FLUX.2-klein-4B".into(),
        prompts: None,
        out_dir: "out/sample-imagegen-batch".into(),
        seed: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag}: missing value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--model" => a.model = value()?,
            "--prompts" => a.prompts = Some(value()?),
            "--out-dir" => a.out_dir = value()?,
            "--seed" => a.seed = Some(value()?.parse().map_err(|e| format!("--seed: {e}"))?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(a)
}

/// One prompt per line. A blank line or a `#` comment is skipped rather than
/// generating an image of nothing, which is the failure a naive reader makes
/// expensive: every skipped line here would otherwise be a full denoise.
fn read_prompts(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let prompts: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect();
    if prompts.is_empty() {
        return Err(format!("{path}: no prompts (every line was blank or a comment)"));
    }
    Ok(prompts)
}

fn run(a: &Args, prompts: &[String]) -> brain::Result<()> {
    std::fs::create_dir_all(&a.out_dir).map_err(brain::Error::from)?;

    let load = Instant::now();
    let pipeline = brain::ImagePipeline::from_pretrained(&a.model)?;
    println!("loaded {} in {:.1?}", a.model, load.elapsed());

    for (i, prompt) in prompts.iter().enumerate() {
        let mut opts = brain::ImageGenerationOptions::new();
        if let Some(seed) = a.seed {
            // Per-image seed derived from one base, so the whole run is
            // reproducible and no two images are accidentally identical.
            opts = opts.seed(seed + i as u64);
        }
        let started = Instant::now();
        let image = pipeline.generate_with(prompt, opts)?;
        let path = format!("{}/{:03}.png", a.out_dir, i);
        image.save(&path)?;
        println!("{:3}  {:>8.1?}  {}  {}", i, started.elapsed(), path, prompt);
    }

    println!("\n{} image(s) from ONE pipeline load", prompts.len());
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let Some(path) = args.prompts.clone() else {
        eprintln!("error: --prompts is required\n\n{USAGE}");
        return ExitCode::FAILURE;
    };
    let prompts = match read_prompts(&path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args, &prompts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("\nif the model is not in the local store yet:  brain pull {}", args.model);
            ExitCode::FAILURE
        }
    }
}
