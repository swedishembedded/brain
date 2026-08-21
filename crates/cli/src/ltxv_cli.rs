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

use ltxv::pipeline::{DfrOpts, DfrPaths, GenOpts, Paths};

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
  --dit-config <name>         DiT config; "tiny" (default, fresh random
                             weights) or "ltx25_22b" (the real 22B
                             checkpoint, needs --dit/$BRAIN_LTXV_DIT)
  --start-frame <path>         PNG/JPEG still, encoded through the real VAE
                             and held fixed as frame 0 while the rest of
                             the clip denoises around it.
  --end-frame <path>           same, held fixed as the clip's LAST pixel
                             frame instead. Pass the SAME path as
                             --start-frame for a clip that loops
                             seamlessly (the generated content in between
                             connects the still to itself); a different
                             path for a clip that morphs between two
                             stills. Either flag works alone.
  --conditioning-strength <S>  how hard --start-frame/--end-frame pin their
                             frames, 0..1 (default 1.0 = pinned exactly;
                             the reference's own CLI example is 0.8). Note
                             that passing the SAME image to --start-frame
                             and --end-frame produces a STATIC clip at any
                             strength - the model is answering "start here,
                             end here" literally. Anchor two different
                             instants, or use --start-frame alone.

Placement:
  On a box with two or more schedulable cards (see --device) and
  --guidance > 1.0, the conditional and unconditional forwards of every
  denoise step run CONCURRENTLY, one per card, and the text encoder runs on
  the card the conditional forward will not use. The result is bit-identical
  to running them one after another (the two forwards are independent, so no
  reduction is split) - measured 1.94x wall on two Tesla P40s. Set
  BRAIN_LTXV_CFG_PARALLEL=0 to force the old single-card behaviour; --device
  gpu0 also confines the run to one card, since placement never leaves the
  schedulable set --device names.

Weights (flag wins over the environment variable):
  --vae <path>              $BRAIN_LTXV_VAE       the causal 3D video VAE
  --dit <path>              $BRAIN_LTXV_DIT       real 22B DiT GGUF (only
                                                   read when --dit-config
                                                   ltx25_22b)
  --text-encoder <path>     $BRAIN_LTXV_TEXT_ENCODER  real Gemma-4 text
                                                   encoder (optional; the
                                                   deterministic prompt
                                                   stub runs without it)

Devices:
  --device <cpu|gpu>         DiT + VAE (default: the ambient BRAIN_DEVICE)

Subcommands:
  brain ltxv t2v --help       text to video (this command)
  brain ltxv dfr --help       DFR (Diffusion Fidelity Rendering) smoke test"#;

const DFR_HELP: &str = r#"brain ltxv dfr - LTX-2.5 DFR (Diffusion Fidelity Rendering) smoke test

  brain ltxv dfr --prompt <text> --output-path <out.mp4> [options]

DFR runs half-res base generation with generated keyframe slots, a REAL
spatial x2 latent upscale, a full-res re-noised detailing pass (no
IC-LoRA - none is downloaded, see `ltxv::pipeline`'s DFR doc), and 0-2
REAL temporal x2 upsample rounds with tile-based stitching. Same tiny
random-weight DiT and stub text context `t2v` uses - see
`ltxv::pipeline`'s module doc (search "M8c") for exactly what DFR
mechanics are real here and which remain a documented gap.

Required:
  --prompt <text>            folded into a deterministic noise/context seed
                             only - there is no real text encoder yet
  --output-path <file>       .mp4 / .mkv / .webm / .gif

Sampling (same shape as `t2v`; --width/--height are the FULL, stage-2
resolution - stage 1 halves it, so both must be a multiple of 64):
  --frames <N>                video frames, must be 1 + 8k (default 9)
  --width <W> --height <H>    pixels, multiples of 64 (default 64x64)
  --steps <N>                  denoise steps per stage/tile (default 4)
  --guidance <G>               classifier-free guidance (default 1.0)
  --seed <S>                   initial-noise/weight/context seed (default 0)
  --fps <N>                    stage-1 frame rate; the written fps is this
                               times 2^rounds (default 8)
  --base-shift <F>             LTX2Scheduler shift anchor at 1024 tokens (0.95)
  --max-shift <F>              LTX2Scheduler shift anchor at 4096 tokens (2.05)
  --no-stretch                  skip the terminal-sigma stretch (on by default)
  --terminal <F>                stretch target sigma (default 0.1)
  --eta <F>                     ancestral-Euler eta (default 1.0)
  --dit-config <name>           DiT size; only "tiny" is implemented (default)
  --temporal-upsample-rounds <N> 0, 1, or 2 real temporal x2 refine rounds
                                 (default 0; > 0 needs --temporal-upsampler)

Weights (flag wins over the environment variable):
  --vae <path>                 $BRAIN_LTXV_VAE       the causal 3D video VAE
  --spatial-upsampler <path>   $BRAIN_LTXV_UPSAMPLER_SPATIAL
  --temporal-upsampler <path>  $BRAIN_LTXV_UPSAMPLER_TEMPORAL

Devices:
  --device <cpu|gpu>            DiT + VAE + upscalers (default: BRAIN_DEVICE)"#;

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
        "dfr" => {
            if let Err(e) = dfr(&args[1..]) {
                eprintln!("ltxv dfr: {e}");
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
    let mut dit: Option<String> = None;
    let mut text_encoder: Option<String> = None;
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
            "--dit" => {
                dit = Some(need(i)?.clone());
            }
            "--text-encoder" => {
                text_encoder = Some(need(i)?.clone());
            }
            "--conditioning-strength" => o.conditioning_strength = flt(i, "--conditioning-strength")?,
            "--start-frame" => {
                o.start_frame = Some(need(i)?.clone());
            }
            "--end-frame" => {
                o.end_frame = Some(need(i)?.clone());
            }
            // `--no-stretch` is a bare flag (no value), unlike every other
            // option above - handled separately so the `i += 2` stride
            // below stays uniform for everything else.
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
    let paths = Paths::resolve(vae.as_deref(), dit.as_deref(), text_encoder.as_deref())?;

    let tokens = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        let lat_t = vcfg.latent_frames(o.frames as u32).ok_or_else(|| format!("{} frames is not of the form 1 + 8k", o.frames))?;
        (lat_t as usize) * (o.height / 32) * (o.width / 32)
    };
    let forwards = if o.guidance > 1.0 { 2 } else { 1 };
    let dit_desc = if o.dit_config == "tiny" { "tiny random-weight DiT" } else { "REAL checkpoint DiT (int8 compute)" };
    let ctx_desc = if paths.text_encoder.is_some() { "real Gemma-4 text encoder" } else { "stub text context (no real encoder)" };
    // The real distilled checkpoint ignores `--steps` entirely (see
    // `ltxv::pipeline::generate`'s doc on why) - report the fixed real
    // schedule's own step count here instead of echoing a flag that will
    // not be honored, so this line never lies about what actually runs.
    let steps_desc = if o.dit_config == "ltx25_22b" {
        format!("{} distilled-schedule (fixed)", ltxv::pipeline::LTX2_DISTILLED_STEPS)
    } else {
        format!("{}", o.steps)
    };
    let img_desc = match (o.start_frame.as_deref(), o.end_frame.as_deref()) {
        (Some(s), Some(e)) if s == e => format!(", looped ({s})"),
        (Some(s), Some(e)) => format!(", start-frame ({s}) -> end-frame ({e})"),
        (Some(s), None) => format!(", start-frame ({s})"),
        (None, Some(e)) => format!(", end-frame ({e})"),
        (None, None) => String::new(),
    };
    eprintln!(
        "ltxv ({dit_desc}, real VAE, {ctx_desc}): {} frames at {}x{}, {steps_desc} steps x {forwards} forward(s) of {tokens} tokens, eta {}, guidance {}, seed {}{img_desc}",
        o.frames, o.width, o.height, o.eta, o.guidance, o.seed
    );

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it: Ctrl-C already
    // ends the process - see `wan_cli::t2v`'s identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::generate(&paths, &prompt, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    // Every stage, plus whatever none of them explains. A breakdown whose
    // parts summed to half its own total is what hid the largest stage in
    // this pipeline from two prior optimization passes, so the remainder is
    // printed rather than left implicit.
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "ltxv: {wall:.1}s total  (build {:.2}s, text encode {:.1}s, denoise {:.1}s = {:.3}s/forward, vae {:.1}s, other {:.1}s)",
        timings.build_dit,
        timings.text_encode,
        timings.denoise,
        timings.secs_per_forward(),
        timings.decode,
        timings.unattributed(wall)
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

fn dfr(args: &[String]) -> Result<(), String> {
    let mut o = DfrOpts::default();
    let mut prompt: Option<String> = None;
    let mut out: Option<String> = None;
    let mut vae: Option<String> = None;
    let mut spatial_upsampler: Option<String> = None;
    let mut temporal_upsampler: Option<String> = None;
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
            "--frames" => o.base.frames = num(i, "--frames")?,
            "--width" => o.base.width = num(i, "--width")?,
            "--height" => o.base.height = num(i, "--height")?,
            "--steps" => o.base.steps = num(i, "--steps")?,
            "--fps" => o.base.fps = num(i, "--fps")?,
            "--guidance" => o.base.guidance = flt(i, "--guidance")?,
            "--seed" => o.base.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--base-shift" => o.base.base_shift = dbl(i, "--base-shift")?,
            "--max-shift" => o.base.max_shift = dbl(i, "--max-shift")?,
            "--terminal" => o.base.terminal = dbl(i, "--terminal")?,
            "--eta" => o.base.eta = dbl(i, "--eta")?,
            "--dit-config" => o.base.dit_config = need(i)?.clone(),
            "--temporal-upsample-rounds" => o.temporal_upsample_rounds = num(i, "--temporal-upsample-rounds")?,
            "--device" => {
                o.base.device = Some(need(i)?.clone());
            }
            "--vae" => {
                vae = Some(need(i)?.clone());
            }
            "--spatial-upsampler" => {
                spatial_upsampler = Some(need(i)?.clone());
            }
            "--temporal-upsampler" => {
                temporal_upsampler = Some(need(i)?.clone());
            }
            // `--no-stretch` is a bare flag (no value), unlike every other
            // option above - handled separately so the `i += 2` stride below
            // stays uniform for everything else.
            "--no-stretch" => {
                o.base.stretch = false;
                i += 1;
                continue;
            }
            "--help" | "-h" => {
                println!("{DFR_HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{DFR_HELP}")),
        }
        i += 2;
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    let out = out.ok_or("--output-path is required")?;
    let paths = DfrPaths::resolve(vae.as_deref(), spatial_upsampler.as_deref(), temporal_upsampler.as_deref())?;

    eprintln!(
        "ltxv dfr (M8c smoke test - tiny random-weight DiT, real VAE + real upscalers): {} frames -> canvas, {}x{}, {} steps, {} temporal round(s), seed {}",
        o.base.frames, o.base.width, o.base.height, o.base.steps, o.temporal_upsample_rounds, o.base.seed
    );

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it - see `t2v`'s
    // identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::generate_dfr(&paths, &prompt, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv dfr [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "ltxv dfr: {wall:.1}s total  (build {:.2}s, denoise+upsample {:.1}s, vae {:.1}s, other {:.1}s)",
        timings.build_dit,
        timings.denoise,
        timings.decode,
        timings.unattributed(wall)
    );

    let frames: Vec<imaging::Rgb8> = video.frames.iter().map(|px| imaging::Rgb8::new(video.width, video.height, px.clone())).collect::<Result<_, _>>()?;
    match imaging::video::encode_frames(&frames, std::path::Path::new(&out), video.fps as f64, &Default::default())? {
        imaging::video::Encoded::Video(p) => {
            eprintln!("ltxv dfr: wrote {} ({}x{}, {} frames at {} fps)", p.display(), video.width, video.height, frames.len(), video.fps);
        }
        imaging::video::Encoded::Frames { dir, command } => {
            eprintln!("ltxv dfr: ffmpeg is not on PATH, so the {} frames are numbered PPMs in {}", frames.len(), dir.display());
            eprintln!("ltxv dfr: finish the job with:\n  {command}");
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
            "--dit",
            "--text-encoder",
        ] {
            assert!(super::HELP.contains(flag), "{flag} is parsed but not in --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in --help but not parsed");
        }
        for (var, _) in ltxv::pipeline::PATH_VARS {
            assert!(super::HELP.contains(var), "{var} is read but not in --help");
        }
        for (var, _) in ltxv::pipeline::OPTIONAL_PATH_VARS {
            assert!(super::HELP.contains(var), "{var} is read but not in --help");
        }
    }

    /// Same self-check as [`every_flag_the_parser_accepts_is_documented`],
    /// scoped to `dfr`'s own flags and help text.
    #[test]
    fn every_dfr_flag_the_parser_accepts_is_documented() {
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
            "--temporal-upsample-rounds",
            "--device",
            "--vae",
            "--spatial-upsampler",
            "--temporal-upsampler",
        ] {
            assert!(super::DFR_HELP.contains(flag), "{flag} is parsed but not in dfr --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in dfr --help but not parsed");
        }
        for (var, _) in ltxv::pipeline::DFR_PATH_VARS {
            assert!(super::DFR_HELP.contains(var), "{var} is read but not in dfr --help");
        }
    }
}
