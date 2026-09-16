// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sample application: text prompt in, PNG out, through the public `brain`
//! SDK.
//!
//! This is the smallest complete thing an application can do with brain:
//! resolve a model from the local store, build a resident pipeline, generate,
//! save. There is no CLI process and no capability-dispatch server in the
//! loop - the binary you are reading links `brain` directly, exactly as a
//! product would.
//!
//! ```text
//! make samples/imagegen/generate/run ARGS="--prompt 'a whale submarine' --out whale.png"
//! ```
//!
//! Swedish Embedded AB implements client-embeddable image generation for its
//! clients. If your team needs a product-ready diffusion pipeline behind a
//! small library surface, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::process::ExitCode;

/// What the sample was asked to do, after argument parsing.
struct Args {
    model: String,
    prompt: String,
    out: String,
    steps: Option<u32>,
    seed: Option<u64>,
}

const USAGE: &str = "\
sample-imagegen-generate - generate an image through the brain SDK

USAGE:
    sample-imagegen-generate [--model ID] [--prompt TEXT] [--out PATH]
                             [--steps N] [--seed N]

OPTIONS:
    --model ID     model to resolve from the local store
                   (default: black-forest-labs/FLUX.2-klein-4B)
    --prompt TEXT  the text prompt (default: a whale submarine surfacing at dawn)
    --out PATH     where to write the PNG (default: out/sample-imagegen.png)
    --steps N      denoising steps (model default if omitted)
    --seed N       RNG seed (model default if omitted)
";

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: "black-forest-labs/FLUX.2-klein-4B".into(),
        prompt: "a whale submarine surfacing at dawn".into(),
        out: "out/sample-imagegen.png".into(),
        steps: None,
        seed: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        // Every flag takes exactly one value, so a missing one is an error
        // rather than a silent default - a prompt that vanished because the
        // shell ate a quote should not generate a picture of nothing.
        let mut value = || it.next().ok_or_else(|| format!("{flag}: missing value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--model" => a.model = value()?,
            "--prompt" => a.prompt = value()?,
            "--out" => a.out = value()?,
            "--steps" => a.steps = Some(value()?.parse().map_err(|e| format!("--steps: {e}"))?),
            "--seed" => a.seed = Some(value()?.parse().map_err(|e| format!("--seed: {e}"))?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(a)
}

fn run(a: &Args) -> brain::Result<()> {
    println!("model  {}", a.model);
    println!("prompt {}", a.prompt);

    let pipeline = brain::ImagePipeline::from_pretrained(&a.model)?;

    let mut opts = brain::ImageGenerationOptions::new();
    if let Some(steps) = a.steps {
        opts = opts.steps(steps);
    }
    if let Some(seed) = a.seed {
        opts = opts.seed(seed);
    }

    let image = pipeline.generate_with(&a.prompt, opts)?;
    if let Some(parent) = std::path::Path::new(&a.out).parent() {
        std::fs::create_dir_all(parent).map_err(brain::Error::from)?;
    }
    image.save(&a.out)?;

    println!("wrote  {} ({}x{})", a.out, image.width(), image.height());
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
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // A sample that cannot find its weights should say so plainly and
            // exit, never panic: samples/README.md rule 5.
            eprintln!("error: {e}");
            eprintln!("\nif the model is not in the local store yet:  brain pull {}", args.model);
            ExitCode::FAILURE
        }
    }
}
