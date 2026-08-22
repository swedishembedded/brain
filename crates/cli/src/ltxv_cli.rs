// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain ltxv {t2v,upscale,dfr}` - LTX-2.5 video generation, post-hoc
//! upscaling of a clip that already exists, and the DFR smoke test.
//!
//! One command, one playable file:
//!
//! ```text
//! brain ltxv t2v --prompt "a cat walking on a beach" --seed 42 --output-path out.mp4
//! brain ltxv upscale --input out.mp4 --output-path out_2x.mp4 --prompt "a cat walking on a beach"
//! ```
//!
//! **`--dit-config` decides whether any of this is a quality claim.** The VAE
//! and the latent upscalers are always real (real weights,
//! `--vae`/`$BRAIN_LTXV_VAE`, `$BRAIN_LTXV_UPSAMPLER_SPATIAL`). The DiT is
//! the tiny config with FRESH RANDOM WEIGHTS by default, which makes every
//! output a wiring proof and nothing else; `--dit-config ltx25_22b` with
//! `--dit`/`$BRAIN_LTXV_DIT` loads the real checkpoint. The prompt likewise
//! only folds into a deterministic context stub unless
//! `--text-encoder`/`$BRAIN_LTXV_TEXT_ENCODER` names the real Gemma-4
//! encoder. See `ltxv::pipeline`'s module doc for the full account of what
//! is real and what is not.

use ltxv::pipeline::{DfrOpts, DfrPaths, GenOpts, LongOpts, Paths, UpscaleOpts};

const HELP: &str = r#"brain ltxv t2v - LTX-2.5 text to video (M4 smoke test: real VAE + tiny random-weight DiT, no real text encoder yet)

  brain ltxv t2v --prompt <text> --output-path <out.mp4> [options]

Required:
  --prompt <text>          folded into a deterministic noise/context seed
                           only - there is no real text encoder yet
  --output-path <file>     .mp4 / .mkv / .webm / .gif; without ffmpeg the
                           frames are written to <file>.frames/ and the
                           command that finishes the job is printed

Sampling (defaults are a small smoke-test clip, see ltxv::pipeline::GenOpts):
  --frames <N>              video frames, must be 1 + 8k (default 9). Any
                            length is accepted: past what one denoising
                            window holds the clip is generated as several
                            windows with a rolling latent context (see
                            Long-form clips)
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
  --context-frames <N>         how much of the previous window a long-form
                             continuation carries, in pixel frames, must be
                             1 + 8k (default 57 = 8 latent frames, the
                             reference's own video-extension prefix). Only
                             read when the clip needs more than one window.
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
                            $BRAIN_LTXV_UPSAMPLER_SPATIAL  spatial x2 latent
                                                   upscaler - REQUIRED above
                                                   6144 video tokens, unused
                                                   below (see Stages)

Stages:
  The distilled checkpoint's fixed 8-sigma schedule is only distilled to
  build a clip from noise at a modest token count - upstream never runs it
  at the requested resolution, only at half of it, and then refines. Past
  6144 video tokens (a 25-frame clip above ~1600x896) one stage measurably
  disintegrates the END of the clip while the start stays correct, so a
  request past that ceiling runs the reference's own two stages: the full
  schedule at half resolution, a real x2 latent upscale, then three
  deterministic refinement steps at the requested size. That needs
  $BRAIN_LTXV_UPSAMPLER_SPATIAL, and both --width and --height must be
  multiples of 64 (not just 32) so halving lands on the VAE's own stride.
  BRAIN_LTXV_TWO_STAGE=1/0 forces the choice either way.

Long-form clips:
  A request longer than one denoising window fits is generated as several
  consecutive windows. What crosses each boundary is the previous window's
  own last --context-frames LATENT frames, sliced out before anything was
  decoded and frozen at sigma 0 while the new frames denoise around them -
  NOT a decoded frame re-encoded as a conditioning still, which carries a
  position but no velocity and is why naively chained clips change or
  reverse their motion at every seam. A request that fits one window is
  generated exactly as it always was, and none of this runs.
  BRAIN_LTXV_LONGFORM_MAX_TOKENS overrides the per-window token ceiling
  (default 13200, the largest single-window generation this crate has a
  recorded real run at). --end-frame is not supported for a multi-window
  clip; --start-frame conditions the first window as usual.

Devices:
  --device <cpu|gpu>         DiT + VAE (default: the ambient BRAIN_DEVICE)

Subcommands:
  brain ltxv t2v --help       text to video (this command)
  brain ltxv upscale --help   re-render a finished clip at 2x
  brain ltxv dfr --help       DFR (Diffusion Fidelity Rendering) smoke test"#;

const UPSCALE_HELP: &str = r#"brain ltxv upscale - re-render a finished clip at twice its resolution

  brain ltxv upscale --input <clip.mp4> --output-path <out.mp4> [options]

Reads a video file that already exists, VAE-encodes it, carries the latent up
with the OFFICIAL LTX-2.5 x2 latent spatial upscaler, refines at the new size
on the distilled refinement schedule, and VAE-decodes back to a container.
This is exactly the tail of a two-stage generation (see the Stages section of
`brain ltxv t2v --help`), applied to pixels that finished rendering rather
than to a stage-1 latent - the same upscaler network and the same code.

It is NOT a pixel-space resampler: the refinement is a diffusion pass, so it
adds detail the source frames do not contain, and it answers to --prompt.

Required:
  --input <file>             the clip to upscale (anything ffmpeg reads). Its
                             width and height must be multiples of 32 and its
                             frame count of the form 1 + 8k, which is what
                             this pipeline's own output always is
  --output-path <file>       .mp4 / .mkv / .webm / .gif; without ffmpeg the
                             frames are written to <file>.frames/ and the
                             command that finishes the job is printed

Refinement:
  --prompt <text>            what the clip shows. STRONGLY recommended: the
                             refinement denoises against this text context,
                             so omitting it refines against an empty prompt
                             and costs the detail it exists to recover. Use
                             the clip's original generation prompt when you
                             still have it.
  --factor <N>               spatial factor; only 2 (the official checkpoint
                             is an x2 network)
  --refine-steps <N>         1..=3 (default 3). Fewer steps means starting
                             further down the distilled refinement table, not
                             the same span in bigger hops - the schedule's
                             sigma values are baked in by distillation and
                             are not interpolable.
  --guidance <G>             classifier-free guidance (default 1.0)
  --seed <S>                 refinement noise seed (default 0)
  --fps <N>                  frame rate for the output. Default: whatever
                             ffprobe reports for --input, and only if that
                             fails does this fall back to the pipeline
                             default.
  --dit-config <name>        "tiny" (default, fresh random weights - a wiring
                             smoke test) or "ltx25_22b" (the real checkpoint,
                             needs --dit/$BRAIN_LTXV_DIT)

Length:
  An upscaled clip has FOUR times the video tokens per frame of its input, so
  a length that refined fine at the source resolution need not fit at the
  target one. Past 12288 tokens in one pass the clip is refined in several
  independent segments that share one frame at each boundary; fine detail can
  step where two segments meet. A clip that fits is never split. A request
  that cannot be split into anything runnable is refused before any weight is
  read, rather than after an hour of work.

Weights (flag wins over the environment variable):
  --vae <path>               $BRAIN_LTXV_VAE       the causal 3D video VAE
  --upsampler-spatial <path> $BRAIN_LTXV_UPSAMPLER_SPATIAL  spatial x2 latent
                                                   upscaler - REQUIRED, it is
                                                   what this command runs
  --dit <path>               $BRAIN_LTXV_DIT       real 22B DiT GGUF (only
                                                   read when --dit-config
                                                   ltx25_22b)
  --text-encoder <path>      $BRAIN_LTXV_TEXT_ENCODER  real Gemma-4 text
                                                   encoder (without it
                                                   --prompt reaches the model
                                                   only as a stub and cannot
                                                   guide anything)

Devices:
  --device <cpu|gpu>         DiT + VAE + upscaler (default: BRAIN_DEVICE)"#;

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
        "upscale" => {
            if let Err(e) = upscale(&args[1..]) {
                eprintln!("ltxv upscale: {e}");
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
    let mut context_frames = ltxv::longform::CONTEXT_FRAMES;
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
            "--context-frames" => context_frames = num(i, "--context-frames")?,
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
    let paths = Paths::resolve(vae.as_deref(), dit.as_deref(), text_encoder.as_deref(), None)?;

    let (lh, lw) = (o.height / 32, o.width / 32);
    let context_latents = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        vcfg.latent_frames(context_frames as u32).ok_or_else(|| format!("--context-frames {context_frames} is not of the form 1 + 8k"))? as usize
    };
    let long = LongOpts { context_latent_frames: context_latents, max_window_tokens: ltxv::longform::max_window_tokens_from_env(), base: o.clone() };
    // The plan decides what this run actually is, so it is resolved before
    // the preview line rather than after: a multi-window clip's per-forward
    // token count is one WINDOW's, and reporting the whole clip's would name
    // a number no forward in the run has.
    let plan = ltxv::longform::window_plan(o.frames, lh, lw, context_latents, long.max_window_tokens)?;
    let tokens = plan.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    let window_desc = if plan.len() > 1 {
        format!(" [{} windows, {context_frames}-frame rolling latent context]", plan.len())
    } else {
        String::new()
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
    // Above `SINGLE_STAGE_MAX_TOKENS` the schedule above is stage 1 of two
    // and runs at HALF this resolution, with a short refinement pass at the
    // requested one - say so, since otherwise this line reports a token
    // count no forward in the run actually has (see the Stages section of
    // `--help`).
    let stage_desc = if ltxv::pipeline::should_two_stage(tokens, o.width, o.height, o.dit_config == "ltx25_22b") {
        format!(" [two-stage: {} tokens at {}x{}, then {} refinement steps at {tokens} tokens]", tokens / 4, o.width / 2, o.height / 2, ltxv::pipeline::LTX2_STAGE2_STEPS)
    } else {
        String::new()
    };
    let img_desc = match (o.start_frame.as_deref(), o.end_frame.as_deref()) {
        (Some(s), Some(e)) if s == e => format!(", looped ({s})"),
        (Some(s), Some(e)) => format!(", start-frame ({s}) -> end-frame ({e})"),
        (Some(s), None) => format!(", start-frame ({s})"),
        (None, Some(e)) => format!(", end-frame ({e})"),
        (None, None) => String::new(),
    };
    eprintln!(
        "ltxv ({dit_desc}, real VAE, {ctx_desc}): {} frames at {}x{}, {steps_desc} steps x {forwards} forward(s) of {tokens} tokens, eta {}, guidance {}, seed {}{img_desc}{stage_desc}{window_desc}",
        o.frames, o.width, o.height, o.eta, o.guidance, o.seed
    );

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it: Ctrl-C already
    // ends the process - see `wan_cli::t2v`'s identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::generate_long(&paths, &prompt, &long, &cancel, |done, total, phase| {
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

fn upscale(args: &[String]) -> Result<(), String> {
    let mut o = UpscaleOpts::default();
    let mut prompt: Option<String> = None;
    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut vae: Option<String> = None;
    let mut dit: Option<String> = None;
    let mut text_encoder: Option<String> = None;
    let mut upsampler_spatial: Option<String> = None;
    let mut fps: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let flt = |i: usize, what: &str| -> Result<f32, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => {
                prompt = Some(need(i)?.clone());
            }
            "--input" | "--in" => {
                input = Some(need(i)?.clone());
            }
            "--output-path" | "--out" => {
                out = Some(need(i)?.clone());
            }
            "--factor" => o.factor = num(i, "--factor")?,
            "--refine-steps" => o.refine_steps = num(i, "--refine-steps")?,
            "--guidance" => o.base.guidance = flt(i, "--guidance")?,
            "--seed" => o.base.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--fps" => fps = Some(num(i, "--fps")?),
            "--dit-config" => o.base.dit_config = need(i)?.clone(),
            "--device" => {
                o.base.device = Some(need(i)?.clone());
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
            "--upsampler-spatial" => {
                upsampler_spatial = Some(need(i)?.clone());
            }
            "--help" | "-h" => {
                println!("{UPSCALE_HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{UPSCALE_HELP}")),
        }
        i += 2;
    }
    let input = input.ok_or("--input is required")?;
    let out = out.ok_or("--output-path is required")?;
    let paths = Paths::resolve(vae.as_deref(), dit.as_deref(), text_encoder.as_deref(), upsampler_spatial.as_deref())?;
    let prompt = prompt.unwrap_or_default();

    let in_path = std::path::Path::new(&input);
    let decoded = imaging::video::decode_frames_rgb8(in_path, &imaging::video::VideoDecodeOpts { fps: None, max_frames: 0 })?;
    let (w, h) = (decoded[0].w, decoded[0].h);
    if let Some(bad) = decoded.iter().position(|f| (f.w, f.h) != (w, h)) {
        return Err(format!("{input} changes size at frame {bad} ({}x{} after {w}x{h}) - a clip has one resolution", decoded[bad].w, decoded[bad].h));
    }
    // The source's own rate, so an upscale never silently changes playback
    // speed. `--fps` overrides; a missing `ffprobe` is not fatal (see
    // `imaging::video::probe_fps`), it just means the caller has to say.
    let fps = fps.unwrap_or_else(|| imaging::video::probe_fps(in_path).map(|f| f.round() as usize).filter(|&f| f > 0).unwrap_or(o.base.fps));
    o.base.fps = fps;
    let clip = ltxv::pipeline::Video { width: w, height: h, fps, frames: decoded.into_iter().map(|f| f.px).collect() };

    let segments = ltxv::pipeline::refine_segments(clip.frames.len(), (h as usize * o.factor) / 32, (w as usize * o.factor) / 32)?;
    let ctx_desc = if paths.text_encoder.is_some() { "real Gemma-4 text encoder" } else { "stub text context (no real encoder)" };
    let dit_desc = if o.base.dit_config == "tiny" { "tiny random-weight DiT" } else { "REAL checkpoint DiT (int8 compute)" };
    let seg_desc = if segments.len() > 1 { format!(", {} refinement segments (seams possible where they meet)", segments.len()) } else { String::new() };
    if prompt.is_empty() {
        eprintln!("ltxv upscale: no --prompt, so the refinement pass denoises against an empty context - pass the clip's original prompt for better detail");
    }
    eprintln!(
        "ltxv upscale ({dit_desc}, real VAE + real x2 latent spatial upscaler, {ctx_desc}): {} frames, {w}x{h} -> {}x{} at {fps} fps, {} refinement steps, guidance {}, seed {}{seg_desc}",
        clip.frames.len(),
        w as usize * o.factor,
        h as usize * o.factor,
        o.refine_steps,
        o.base.guidance,
        o.base.seed
    );

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it - see `t2v`'s
    // identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::upscale(&paths, &prompt, &clip, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv upscale [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "ltxv upscale: {wall:.1}s total  (build {:.2}s, text encode {:.1}s, refine {:.1}s = {:.3}s/forward, vae {:.1}s, other {:.1}s)",
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
            eprintln!("ltxv upscale: wrote {} ({}x{}, {} frames at {} fps)", p.display(), video.width, video.height, frames.len(), video.fps);
        }
        imaging::video::Encoded::Frames { dir, command } => {
            eprintln!("ltxv upscale: ffmpeg is not on PATH, so the {} frames are numbered PPMs in {}", frames.len(), dir.display());
            eprintln!("ltxv upscale: finish the job with:\n  {command}");
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
    /// scoped to `upscale`'s own flags and help text. `--upsampler-spatial`
    /// is a flag here rather than environment-only (as it is on `t2v`),
    /// because running that network is what this command is for.
    #[test]
    fn every_upscale_flag_the_parser_accepts_is_documented() {
        let src = include_str!("ltxv_cli.rs");
        for flag in ["--prompt", "--input", "--output-path", "--factor", "--refine-steps", "--guidance", "--seed", "--fps", "--dit-config", "--device", "--vae", "--dit", "--text-encoder", "--upsampler-spatial"] {
            assert!(super::UPSCALE_HELP.contains(flag), "{flag} is parsed but not in upscale --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in upscale --help but not parsed");
        }
        for (var, _) in ltxv::pipeline::PATH_VARS {
            assert!(super::UPSCALE_HELP.contains(var), "{var} is read but not in upscale --help");
        }
        for (var, _) in ltxv::pipeline::OPTIONAL_PATH_VARS {
            assert!(super::UPSCALE_HELP.contains(var), "{var} is read but not in upscale --help");
        }
        // The one number in the help text that is a real constant rather than
        // prose: a ceiling the help promises and the pipeline enforces must
        // not drift apart.
        assert!(super::UPSCALE_HELP.contains(&ltxv::pipeline::REFINE_MAX_TOKENS.to_string()), "the refinement token ceiling in upscale --help is not REFINE_MAX_TOKENS");
    }

    /// Every subcommand `run_ltxv` dispatches has to be listed in the help a
    /// bare `brain ltxv` prints, or it is reachable only by reading the
    /// source.
    #[test]
    fn every_subcommand_is_listed() {
        let src = include_str!("ltxv_cli.rs");
        for sub in ["t2v", "upscale", "dfr"] {
            assert!(src.contains(&format!("\"{sub}\"")), "{sub} is listed but not dispatched");
            assert!(super::HELP.contains(&format!("brain ltxv {sub}")), "{sub} is dispatched but not in --help");
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
