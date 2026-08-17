// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain ltxv t2v …` - LTX-2.5 text-to-video.
//!
//! One command, one playable file:
//!
//! ```text
//! brain ltxv t2v --prompt "a cat walking on a beach" --seed 42 --output-path out.mp4
//! ```
//!
//! **This milestone is a WIRING smoke test, not a quality claim.** The VAE
//! decode is real (real weights, `--vae`/`$BRAIN_LTXV_VAE`); the DiT is
//! always tiny-config with FRESH RANDOM WEIGHTS (`--dit-config tiny`, the
//! only value implemented - there is no real 22B checkpoint to load), and
//! there is no real text encoder (`--prompt` only ever folds into a
//! deterministic noise/context seed). See `ltxv::pipeline`'s module doc for
//! the full account of what is real here and what is not.

use ltxv::pipeline::{GenOpts, Paths};

const HELP: &str = r#"brain ltxv t2v - LTX-2.5 text to video (M4 smoke test: real VAE + tiny random-weight DiT, no real text encoder yet)

  brain ltxv t2v --prompt <text> --output-path <out.mp4> [options]

Required:
  --prompt <text>          folded into a deterministic noise/context seed
                           only - there is no real text encoder yet
  --output-path <file>     .mp4 / .mkv / .webm / .gif; without ffmpeg the
                           frames are written to <file>.frames/ and the
                           command that finishes the job is printed

Sampling (defaults are a small smoke-test clip, see ltxv::pipeline::GenOpts):
  --frames <N>              video frames, must be 1 + 8k (default 9)
  --width <W> --height <H>  pixels, multiples of 32 (default 64x64)
  --steps <N>                denoise steps (default 4)
  --guidance <G>             classifier-free guidance (default 1.0; <= 1.0
                             runs ONE forward per step instead of two)
  --seed <S>                 initial-noise/weight/context seed (default 0)
  --fps <N>                  frame rate written into the container (default 8)
  --base-shift <F>           LTX2Scheduler shift anchor at 1024 tokens (0.95)
  --max-shift <F>            LTX2Scheduler shift anchor at 4096 tokens (2.05)
  --no-stretch                skip the terminal-sigma stretch (on by default)
  --terminal <F>              stretch target sigma (default 0.1)
  --eta <F>                   ancestral-Euler eta: 0 deterministic, 1 fully
                             ancestral (default 1.0)
  --dit-config <name>         DiT size; only "tiny" is implemented (default)

Weights (flag wins over the environment variable):
  --vae <path>              $BRAIN_LTXV_VAE       the causal 3D video VAE

Devices:
  --device <cpu|gpu>         DiT + VAE (default: the ambient BRAIN_DEVICE)"#;

pub fn run_ltxv(args: &[String]) {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        eprintln!("{HELP}");
        return;
    }
    match args[0].as_str() {
        "t2v" | "text2video" => {
            if let Err(e) = t2v(&args[1..]) {
                eprintln!("ltxv t2v: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("ltxv: unknown subcommand {other}\n\n{HELP}");
            std::process::exit(2);
        }
    }
}

fn t2v(args: &[String]) -> Result<(), String> {
    let mut o = GenOpts::default();
    let mut prompt: Option<String> = None;
    let mut out: Option<String> = None;
    let mut vae: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let flt = |i: usize, what: &str| -> Result<f32, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let dbl = |i: usize, what: &str| -> Result<f64, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => {
                prompt = Some(need(i)?.clone());
            }
            "--output-path" | "--out" => {
                out = Some(need(i)?.clone());
            }
            "--frames" => o.frames = num(i, "--frames")?,
            "--width" => o.width = num(i, "--width")?,
            "--height" => o.height = num(i, "--height")?,
            "--steps" => o.steps = num(i, "--steps")?,
            "--fps" => o.fps = num(i, "--fps")?,
            "--guidance" => o.guidance = flt(i, "--guidance")?,
            "--seed" => o.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--base-shift" => o.base_shift = dbl(i, "--base-shift")?,
            "--max-shift" => o.max_shift = dbl(i, "--max-shift")?,
            "--terminal" => o.terminal = dbl(i, "--terminal")?,
            "--eta" => o.eta = dbl(i, "--eta")?,
            "--dit-config" => o.dit_config = need(i)?.clone(),
            "--device" => {
                o.device = Some(need(i)?.clone());
            }
            "--vae" => {
                vae = Some(need(i)?.clone());
            }
            // `--no-stretch` is a bare flag (no value), unlike every other
            // option above - handled separately so the `i += 2` stride below
            // stays uniform for everything else.
            "--no-stretch" => {
                o.stretch = false;
                i += 1;
                continue;
            }
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{HELP}")),
        }
        i += 2;
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    let out = out.ok_or("--output-path is required")?;
    let paths = Paths::resolve(vae.as_deref())?;

    let tokens = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        let lat_t = vcfg.latent_frames(o.frames as u32).ok_or_else(|| format!("{} frames is not of the form 1 + 8k", o.frames))?;
        (lat_t as usize) * (o.height / 32) * (o.width / 32)
    };
    let forwards = if o.guidance > 1.0 { 2 } else { 1 };
    eprintln!(
        "ltxv (M4 smoke test - tiny random-weight DiT, real VAE): {} frames at {}x{}, {} steps x {forwards} forward(s) of {tokens} tokens, eta {}, guidance {}, seed {}",
        o.frames, o.width, o.height, o.steps, o.eta, o.guidance, o.seed
    );

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it: Ctrl-C already
    // ends the process - see `wan_cli::t2v`'s identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::generate(&paths, &prompt, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    eprintln!(
        "ltxv: {:.1}s total  (build {:.2}s, denoise {:.1}s = {:.3}s/forward, vae {:.1}s)",
        t0.elapsed().as_secs_f32(),
        timings.build_dit,
        timings.denoise,
        timings.secs_per_forward(),
        timings.decode
    );

    let frames: Vec<imaging::Rgb8> = video.frames.iter().map(|px| imaging::Rgb8::new(video.width, video.height, px.clone())).collect::<Result<_, _>>()?;
    match imaging::video::encode_frames(&frames, std::path::Path::new(&out), video.fps as f64, &Default::default())? {
        imaging::video::Encoded::Video(p) => {
            eprintln!("ltxv: wrote {} ({}x{}, {} frames at {} fps)", p.display(), video.width, video.height, frames.len(), video.fps);
        }
        imaging::video::Encoded::Frames { dir, command } => {
            eprintln!("ltxv: ffmpeg is not on PATH, so the {} frames are numbered PPMs in {}", frames.len(), dir.display());
            eprintln!("ltxv: finish the job with:\n  {command}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The help text is the whole user interface: every flag the parser
    /// accepts has to appear in it, and vice versa - `wan_cli`'s own
    /// self-check, ported unchanged.
    #[test]
    fn every_flag_the_parser_accepts_is_documented() {
        let src = include_str!("ltxv_cli.rs");
        for flag in [
            "--prompt",
            "--output-path",
            "--frames",
            "--width",
            "--height",
            "--steps",
            "--fps",
            "--guidance",
            "--seed",
            "--base-shift",
            "--max-shift",
            "--no-stretch",
            "--terminal",
            "--eta",
            "--dit-config",
            "--device",
            "--vae",
        ] {
            assert!(super::HELP.contains(flag), "{flag} is parsed but not in --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in --help but not parsed");
        }
        for (var, _) in ltxv::pipeline::PATH_VARS {
            assert!(super::HELP.contains(var), "{var} is read but not in --help");
        }
    }
}
