// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sample application: an image in, structured perception out, through the
//! public `brain` SDK.
//!
//! One `PerceptionPipeline` describes the picture (objects with boxes and
//! scores, text lines, scene, relations) and answers a form about it. There is
//! no CLI process and no capability-dispatch server in the loop: this binary
//! links `brain` the way a product would.
//!
//! ```text
//! make samples/vision/perception/run ARGS="--image photo.jpg"
//! ```
//!
//! Swedish Embedded AB implements solutions that turn camera images into
//! structured data on compact hardware for its clients. If your team needs
//! expertise in on-device visual perception then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::process::ExitCode;

const USAGE: &str = "\
sample-vision-perception - describe an image and extract a form through the brain SDK

USAGE:
    sample-vision-perception --image PATH [--model ID] [--form JSON] [--max-tokens N]

OPTIONS:
    --image PATH     the picture to analyse (PNG, JPEG, PPM)
    --model ID       model to resolve from the local store (default: Cactus-Compute/mirage)
    --form JSON      a form to answer about the picture, e.g.
                     '{\"animal\": {\"enum\": [\"cat\", \"dog\"]}, \"count\": {\"type\": \"integer\", \"minimum\": 0, \"maximum\": 9}}'
                     (default: a one-field animal form)
    --max-tokens N   decoding budget of the description (model default if omitted)
";

struct Args {
    model: String,
    image: String,
    form: serde_json::Value,
    max_tokens: Option<usize>,
}

fn parse() -> Result<Args, String> {
    let mut model = "Cactus-Compute/mirage".to_string();
    let (mut image, mut form, mut max_tokens) = (None, None, None);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--model" => model = value("--model")?,
            "--image" => image = Some(value("--image")?),
            "--form" => form = Some(serde_json::from_str(&value("--form")?).map_err(|e| format!("--form is not JSON: {e}"))?),
            "--max-tokens" => max_tokens = Some(value("--max-tokens")?.parse().map_err(|e| format!("--max-tokens: {e}"))?),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown flag {other:?}")),
        }
    }
    let form = form.unwrap_or_else(|| serde_json::json!({"animal": {"enum": ["cat", "dog", "bird"]}}));
    Ok(Args { model, image: image.ok_or("--image is required")?, form, max_tokens })
}

fn run(args: &Args) -> brain::Result<()> {
    let pipeline = match brain::PerceptionPipeline::builder(&args.model).download_policy(brain::DownloadPolicy::Offline).load() {
        Ok(p) => p,
        Err(brain::Error::Missing(_) | brain::Error::ModelNotFound(_)) => {
            eprintln!("{} is not in the local model store. Fetch it with:\n    brain pull {}", args.model, args.model);
            return Err(brain::Error::MissingArgument(format!("{} is not installed", args.model)));
        }
        Err(e) => return Err(e),
    };
    let photo = brain::Image::open(&args.image)?;

    let mut options = brain::PerceptionOptions::new();
    if let Some(n) = args.max_tokens {
        options = options.max_tokens(n);
    }
    let described = pipeline.describe_with(&photo, &options)?;
    println!("{}", serde_json::to_string_pretty(described.json()).expect("a JSON value serialises"));
    for object in described.objects() {
        println!("object {}: {} {:?} score {:?}", object.id, object.name, object.bbox, object.score);
    }
    for line in described.text_lines() {
        println!("text {:?} {:?}", line.string, line.bbox);
    }
    let usage = described.usage();
    println!("describe: {} decoder tokens, vision {:.0} ms, decode {:.0} ms", usage.decoder_tokens, usage.vision_ms, usage.decode_ms);

    let answered = pipeline.extract(&photo, &args.form)?;
    println!("{}", serde_json::to_string_pretty(answered.json()).expect("a JSON value serialises"));
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) if e.is_empty() => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
