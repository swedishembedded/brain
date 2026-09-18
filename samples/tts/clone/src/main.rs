// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sample application: clone a voice from a reference clip and speak new
//! text in it, through the public `brain` SDK.
//!
//! This is `brain::TtsPipeline::clone_voice` end to end: resolve a model
//! from the local store, build a resident pipeline, clone the voice in
//! `--voice`, speak `--text` in it. With no `--out`, the clip plays straight
//! on the default output device instead of being written to disk - there is
//! no CLI process, no capability-dispatch server, and no intermediate file
//! required in the loop unless the caller asks for one.
//!
//! ```text
//! make samples/tts/clone/run ARGS="--voice me.wav --text 'hello, this is my own voice'"
//! ```
//!
//! Swedish Embedded AB implements client-embeddable voice cloning for its
//! clients. If your team needs a product-ready TTS pipeline behind a small
//! library surface, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::process::ExitCode;

/// What the sample was asked to do, after argument parsing.
struct Args {
    model: String,
    voice: String,
    text: String,
    ref_text: Option<String>,
    out: Option<String>,
}

const USAGE: &str = "\
sample-tts-clone - clone a voice from a reference clip and speak new text in it

USAGE:
    sample-tts-clone --voice PATH --text TEXT [OPTIONS]

OPTIONS:
    --model ID       model to resolve from the local store
                     (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)
    --voice PATH     reference clip to clone the voice from (required)
    --text TEXT      the text to speak in the cloned voice (required)
    --ref-text TEXT  the reference clip's own transcript. Optional on
                     Qwen3-TTS (x-vector-only timbre matching without it,
                     in-context cloning with it); REQUIRED if --model
                     resolves to CosyVoice instead, which has no
                     x-vector-only mode.
    --out PATH       where to write the WAV. Omit to play the clip on the
                     default audio output device instead of writing a file.
";

fn parse() -> Result<Args, String> {
    let mut model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string();
    let mut voice = None;
    let mut text = None;
    let mut ref_text = None;
    let mut out = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        // Every flag takes exactly one value, so a missing one is an error
        // rather than a silent default - the same discipline
        // samples/imagegen/generate's own parser uses.
        let mut value = || it.next().ok_or_else(|| format!("{flag}: missing value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--model" => model = value()?,
            "--voice" => voice = Some(value()?),
            "--text" => text = Some(value()?),
            "--ref-text" => ref_text = Some(value()?),
            "--out" => out = Some(value()?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        model,
        voice: voice.ok_or_else(|| "--voice is required (a reference clip to clone the voice from)".to_string())?,
        text: text.ok_or_else(|| "--text is required (what to say in the cloned voice)".to_string())?,
        ref_text,
        out,
    })
}

/// Play `audio` on the default output device and block until it finishes.
/// `brain::Audio` is already decoded mono f32 PCM (this module's own doc
/// on `brain::TtsPipeline`'s `Audio` type), so this hands its samples
/// straight to `rodio::buffer::SamplesBuffer` - no WAV encode/decode
/// round-trip through a temp file.
fn play(audio: &brain::Audio) -> Result<(), String> {
    let channels = std::num::NonZero::new(1u16).expect("1 is nonzero");
    let sample_rate = std::num::NonZero::new(audio.sample_rate()).ok_or_else(|| "clip has a zero sample rate".to_string())?;

    // `handle` must outlive `player` - playback stops the moment the output
    // device handle is dropped (rodio's own doc on `DeviceSinkBuilder`).
    let handle = rodio::DeviceSinkBuilder::open_default_sink().map_err(|e| format!("opening the default audio output device: {e}"))?;
    let player = rodio::Player::connect_new(handle.mixer());
    player.append(rodio::buffer::SamplesBuffer::new(channels, sample_rate, audio.samples().to_vec()));
    player.sleep_until_end();
    Ok(())
}

fn run(a: &Args) -> brain::Result<()> {
    println!("model {}", a.model);
    println!("voice {}", a.voice);

    let pipeline = brain::TtsPipeline::from_pretrained(&a.model)?;
    let audio = pipeline.clone_voice(&a.text, &a.voice, a.ref_text.as_deref())?;
    println!("cloned {:.2}s of audio at {} Hz", audio.seconds(), audio.sample_rate());

    match &a.out {
        Some(out) => {
            if let Some(parent) = std::path::Path::new(out).parent() {
                std::fs::create_dir_all(parent).map_err(brain::Error::from)?;
            }
            audio.save(out)?;
            println!("wrote {out}");
        }
        None => {
            println!("playing on the default output device (pass --out PATH to write a file instead)");
            play(&audio).map_err(brain::Error::Backend)?;
        }
    }
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
