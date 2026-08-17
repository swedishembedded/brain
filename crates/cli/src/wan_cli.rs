// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain wan t2v …` - Wan2.1 text-to-video.
//!
//! One command, one playable file:
//!
//! ```text
//! brain wan t2v --prompt "a cat walking on a beach" --seed 42 --output-path out.mp4
//! ```
//!
//! Every weight role has a flag (`--dit`, `--vae`, `--t5`, `--tokenizer`) AND
//! an environment variable (`BRAIN_WAN_*`), and **the flag wins**
//! (`wan::pipeline::Paths::resolve` is where that is decided). Everything else
//! defaults from `WanConfig`, so a run that names only a prompt and an output
//! path is upstream's own configuration.

use wan::pipeline::{GenOpts, Paths, Solver};
use wan::WanConfig;

const HELP: &str = r#"brain wan t2v - Wan2.1 text to video

  brain wan t2v --prompt <text> --output-path <out.mp4> [options]

Required:
  --prompt <text>          what to generate
  --output-path <file>     .mp4 / .mkv / .webm / .gif; without ffmpeg the
                           frames are written to <file>.frames/ and the
                           command that finishes the job is printed

Sampling (defaults from WanConfig, i.e. upstream's own generate.py defaults):
  --frames <N>             video frames, must be 1 + 4k (default 81)
  --width <W> --height <H> pixels, multiples of 16 (default 832x480)
  --steps <N>              denoise steps (default 50)
  --shift <S>              flow-matching sigma shift (default 5.0)
  --guidance <G>           classifier-free guidance (default 5.0; <= 1.0 runs
                           ONE forward per step instead of two)
  --negative-prompt <text> default is upstream's sample_neg_prompt
  --seed <S>               initial-noise seed (default 0)
  --solver <unipc|dpm++>   multistep solver (default unipc)
  --fps <N>                frame rate written into the container (default 16)
  --variant <name>         t2v-1.3B (default) | t2v-14B

Weights (flag wins over the environment variable):
  --dit <path>             $BRAIN_WAN_DIT        transformer dir/file
  --vae <path>             $BRAIN_WAN_VAE        Wan2.1_VAE.pth or vae/
  --t5 <path>              $BRAIN_WAN_T5         umT5-XXL encoder
  --tokenizer <path>       $BRAIN_WAN_TOKENIZER  tokenizer.json or its dir

Devices:
  --device <cpu|gpu>       DiT + VAE (default: the ambient BRAIN_DEVICE)
  --t5-device <cpu|gpu>    text encoder, else $BRAIN_WAN_T5_DEVICE (default
                           cpu: umT5-XXL is 22.72 GB in fp32 and does not fit
                           a 24 GB card)

Cost: a step is TWO transformer forwards whenever guidance > 1. 81 frames at
832x480 is 32,760 tokens per forward; start smaller (--frames 9 --width 256
--height 256 --steps 8) to check the whole path before a long run.
$BRAIN_GPU_WAIT_S is raised to 1200s unless already set: a forward here is one
submit of the whole block stack, far past the backend's 30s deadlock guard."#;

pub fn run_wan(args: &[String]) {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        eprintln!("{HELP}");
        return;
    }
    match args[0].as_str() {
        // Deliberately only `t2v` (and its long spelling), NOT the generic
        // `generate`/`infer` aliases other handlers take: those canonicalize to
        // "infer" in `crate::resolve`, which injects a `--weights <path>` flag
        // this command does not have (it takes four weight roles, not one).
        "t2v" | "text2video" => {
            if let Err(e) = t2v(&args[1..]) {
                eprintln!("wan t2v: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("wan: unknown subcommand {other}\n\n{HELP}");
            std::process::exit(2);
        }
    }
}

fn t2v(args: &[String]) -> Result<(), String> {
    let mut variant = "t2v-1.3B".to_string();
    // The variant decides the defaults, and it can appear anywhere on the
    // line, so it is read in a first pass before `GenOpts` is built.
    for w in args.windows(2) {
        if w[0] == "--variant" {
            variant.clone_from(&w[1]);
        }
    }
    let cfg = match variant.as_str() {
        "t2v-1.3B" | "1.3b" | "1.3B" => WanConfig::t2v_1_3b(),
        "t2v-14B" | "14b" | "14B" => WanConfig::t2v_14b(),
        other => return Err(format!("unknown --variant {other:?} (t2v-1.3B, t2v-14B)")),
    };

    let mut o = GenOpts::from_config(&cfg);
    let mut prompt: Option<String> = None;
    let mut out: Option<String> = None;
    let (mut dit, mut vae, mut t5, mut tokenizer) = (None, None, None, None);
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let flt = |i: usize, what: &str| -> Result<f32, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            // `--out` is accepted because every other generative CLI in this
            // workspace spells it that way; `--output-path` is the spelling
            // the roadmap sets as the bar, so it leads in the help.
            "--output-path" | "--out" => out = Some(need(i)?.clone()),
            "--negative-prompt" => o.negative_prompt = Some(need(i)?.clone()),
            "--frames" => o.frames = num(i, "--frames")?,
            "--width" => o.width = num(i, "--width")?,
            "--height" => o.height = num(i, "--height")?,
            "--steps" => o.steps = num(i, "--steps")?,
            "--fps" => o.fps = num(i, "--fps")?,
            "--shift" => o.shift = flt(i, "--shift")?,
            "--guidance" => o.guidance = flt(i, "--guidance")?,
            "--seed" => o.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--solver" => o.solver = Solver::from_name(need(i)?)?,
            "--device" => o.device = Some(need(i)?.clone()),
            "--t5-device" => o.te_device = Some(need(i)?.clone()),
            "--dit" => dit = Some(need(i)?.clone()),
            "--vae" => vae = Some(need(i)?.clone()),
            "--t5" => t5 = Some(need(i)?.clone()),
            "--tokenizer" => tokenizer = Some(need(i)?.clone()),
            "--variant" => {}
            // `brain wan --help` is caught by the verb dispatch, but `brain wan
            // t2v --help` lands here, and reporting the help flag itself as an
            // unknown flag is a bad first impression from the command a user
            // reaches for precisely because they do not know the flags yet.
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
    let paths = Paths::resolve(dit.as_deref(), vae.as_deref(), t5.as_deref(), tokenizer.as_deref())?;

    // One Wan forward is the WHOLE 30-block stack in a single submit, which at
    // 480p is minutes rather than seconds. The backend's 30 s bounded wait is
    // a deadlock guard sized for a token-at-a-time decoder; leaving it alone
    // makes every large generation die as "device likely wedged". Raise it
    // only when the user has expressed no opinion, and say so - the guard is
    // also what turns a genuinely wedged queue into an error instead of a hang.
    if std::env::var_os("BRAIN_GPU_WAIT_S").is_none() {
        std::env::set_var("BRAIN_GPU_WAIT_S", "1200");
        eprintln!("wan: BRAIN_GPU_WAIT_S unset -> 1200s (one forward is the whole 30-block stack in one submit)");
    }

    let tokens = cfg.token_count(o.frames, o.width, o.height).ok_or_else(|| format!("{} frames is not of the form 1 + 4k", o.frames))?;
    let forwards = if o.guidance > 1.0 { 2 } else { 1 };
    eprintln!(
        "wan {}: {} frames at {}x{}, {} steps x {forwards} forward(s) of {tokens} tokens, {:?} shift {}, guidance {}, seed {}",
        cfg.name, o.frames, o.width, o.height, o.steps, o.solver, o.shift, o.guidance, o.seed
    );

    // Progress goes to stderr on ONE rewritten line: a denoise step at this
    // size is minutes on a P40, and a silent run is indistinguishable from a
    // hang.
    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it: Ctrl-C already
    // ends the process. The unarmed `Default` token never fires, so the
    // per-step poll the served path needs costs this path nothing.
    let cancel = capability::CancelToken::default();
    let (video, timings) = wan::generate(&cfg, &paths, &prompt, &o, &cancel, |done, total, phase| {
        eprint!("\rwan [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    eprintln!(
        "wan: {:.1}s total  (text {:.1}s, load {:.1}s, denoise {:.1}s = {:.1}s/forward, vae {:.1}s)",
        t0.elapsed().as_secs_f32(),
        timings.text,
        timings.load_dit,
        timings.denoise,
        timings.secs_per_forward(),
        timings.decode
    );

    let frames: Vec<imaging::Rgb8> = video
        .frames
        .iter()
        .map(|px| imaging::Rgb8::new(video.width, video.height, px.clone()))
        .collect::<Result<_, _>>()?;
    match imaging::video::encode_frames(&frames, std::path::Path::new(&out), video.fps as f64, &Default::default())? {
        imaging::video::Encoded::Video(p) => {
            eprintln!("wan: wrote {} ({}x{}, {} frames at {} fps)", p.display(), video.width, video.height, frames.len(), video.fps);
        }
        imaging::video::Encoded::Frames { dir, command } => {
            eprintln!("wan: ffmpeg is not on PATH, so the {} frames are numbered PPMs in {}", frames.len(), dir.display());
            eprintln!("wan: finish the job with:\n  {command}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The help text is the whole user interface for a model that takes an
    /// hour to run: every flag the parser accepts has to appear in it, or a
    /// user cannot discover the one that makes their run finish.
    #[test]
    fn every_flag_the_parser_accepts_is_documented() {
        let src = include_str!("wan_cli.rs");
        // The flags are listed here rather than scraped, so ADDING a flag
        // without documenting it fails at review, not silently.
        for flag in [
            "--prompt", "--output-path", "--negative-prompt", "--frames", "--width", "--height",
            "--steps", "--fps", "--shift", "--guidance", "--seed", "--solver", "--device",
            "--t5-device", "--dit", "--vae", "--t5", "--tokenizer", "--variant",
        ] {
            assert!(super::HELP.contains(flag), "{flag} is parsed but not in --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in --help but not parsed");
        }
        // Every environment variable the pipeline reads must be named too.
        for (var, _) in wan::pipeline::PATH_VARS {
            assert!(super::HELP.contains(var), "{var} is read but not in --help");
        }
    }
}
