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

Required (one of):
  --prompt <text>          what the whole clip shows, with --frames for its
                           length - a single-scene clip
  --scene <frames>:<text>  ...or one repeat of this per scene, for a clip
                           that changes scene part way through (see Scenes).
                           --scene carries its own length and prompt, so it
                           cannot be combined with --prompt/--frames
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
  --audio                     ALSO generate the clip's sound, from the same
                             prompt and the same forwards as the picture, and
                             mux it into the output container. LTX-2.5 is
                             natively audio-visual: without this flag only the
                             model's video half runs and the clip is silent.
                             Needs --dit-config ltx25_22b, a real
                             --text-encoder, and $BRAIN_LTXV_AUDIO_VAE. Costs
                             real time and memory - see "Audio" below
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
  --mid-frame <path>           same again, held fixed at ONE INTERIOR
                             instant, so a single pass can be anchored at
                             its start, its middle and its end at once -
                             the way to keep a long or moving-camera clip
                             on course rather than only pinning where it
                             begins and ends. Works alone or with either
                             of the two above.
  --mid-frame-at <N>           which pixel frame --mid-frame anchors,
                             strictly between 0 and --frames minus one.
                             The default is the clip's own midpoint,
                             (frames - 1) / 2. Any frame is legal: the
                             still is an appended guide carrying its own
                             position, not a slot on the latent grid, so
                             it does not have to sit on the 1 + 8k
                             boundary the clip's LENGTH does.
  --context-frames <N>         how much of the previous window a long-form
                             continuation carries, in pixel frames, must be
                             1 + 8k (default 57 = 8 latent frames, the
                             reference's own video-extension prefix). Only
                             read when the clip needs more than one window.
  --conditioning-strength <S>  how hard the conditioning stills pin their
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
  reduction is split), and the two DiT forwards then cost about what one of
  them does. The WALL-CLOCK win is smaller than that and is a property of
  your box and of the current code rather than of this flag, so no fixed
  multiplier is quoted here: whatever a denoise step spends on host-side work
  shared by both branches does not move to a second card, and two concurrent
  branches contend for the same cores. Measure your own by running the same
  request twice, once with BRAIN_LTXV_CFG_PARALLEL=0, or run
  crates/ltxv/tests/cfg_parallel.rs, which times both placements and proves
  the two outputs identical. Set
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

Audio:
  LTX-2.5 is one model that denoises a video-latent stream and an audio-latent
  stream TOGETHER, coupled every block by cross-attention. --audio runs both;
  without it only the video half runs and the clip comes out silent. The sound
  is generated by the same forwards, from the same prompt, over the same time
  window as the frames - not synthesized separately and lined up afterwards.

  Needs --dit-config ltx25_22b, a real --text-encoder (the audio stream is
  conditioned through its own text projection, which the stub context cannot
  stand in for) and $BRAIN_LTXV_AUDIO_VAE pointing at
  ltx-2.5-audio-vae-bf16.safetensors, which carries both the audio VAE decoder
  and the vocoder. All three are checked before any weight is read.

  It is opt-in because of COST, not correctness: the audio-extended
  transformer block has no streamed/quantized/device-resident implementation
  the way the video-only one does, so an audio-visual run expands the whole
  checkpoint to host fp32 and re-uploads it per forward. Expect it to need
  most of a large machine's RAM and to be substantially slower per step than
  the same clip without sound. The command refuses up front, with both
  numbers, if this machine cannot hold it.

  Output is 16 kHz stereo, muxed into .mp4/.mkv/.mov/.webm. A .gif is written
  silent with a line on stderr, since the container holds no audio stream.
  Without ffmpeg the sound is written as audio.wav beside the numbered frames
  and the printed command muxes both - a generation is never thrown away for
  want of an encoder. Not supported for a multi-window or multi-scene clip.

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
  recorded real run at). --end-frame and --mid-frame are not supported for
  a multi-window clip (the first pins one window's last frame, the second
  names a pixel frame of the whole clip and would have to be routed to
  whichever window covers it); --start-frame conditions the first window as
  usual.

Scenes:
  One --frames/--prompt run is one continuous SHOT: every window shares the
  prompt and is hard-conditioned on the real content before it, which is
  what keeps the motion continuous and is exactly what stops the clip from
  becoming something else. Repeat --scene instead to write a clip that does
  change:

    brain ltxv t2v --output-path story.mp4 --width 768 --height 448 \
      --scene 121:"a fishing boat leaves a harbour at dawn, camera tracking" \
      --scene 121:"the open sea under heavy rain, waves breaking" \
      --scene 57:"a close-up of a gull on a wet railing"

  One command, one file, 299 frames. Inside a scene nothing changes - the
  rolling latent context above still carries across every window boundary.
  AT a scene boundary the context RESETS: the next scene's first window
  carries no forced content from the previous one, exactly like the first
  window of any clip, so it is free to actually be a different subject,
  setting or action rather than a continuation. That is also the way to end
  a scene deliberately rather than letting one long shot run until it
  degrades. Each scene's frame count is its own 1 + 8k; the clip is their
  sum. --start-frame conditions the FIRST scene's opening only, and
  --end-frame/--mid-frame are refused (there is no design for pinning the
  last frame, or an interior frame, of a clip whose timeline is a sequence
  of scenes).

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
  --context-frames <N>       how much of the previous refinement pass the
                             next one carries, in pixel frames, must be
                             1 + 8k (default 57 = 8 latent frames, the
                             reference's own video-extension prefix).
                             Reduced automatically when the OUTPUT grid
                             cannot hold that many latent frames plus one to
                             refine - see Length. Only read when the clip
                             needs more than one pass.
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
  passes, and each one FREEZES the previous pass's own last --context-frames
  of refined latent at the head of its sequence - the same rolling latent
  context `brain ltxv t2v` uses across a window boundary - so the passes are
  one continuous clip rather than several separately re-imagined ones.

  That context costs budget. A pass holds 12288/(tokens per latent frame)
  latent frames and spends the carried ones before it refines anything, so at
  2560x1408 (3520 tokens per latent frame, 3 latent frames a pass) the full
  57-frame context does not fit: the plan carries the most it can and says so
  on stderr. Lower --context-frames to buy back passes at the cost of
  continuity. A clip that fits one pass is never split and carries nothing. A
  grid with no room for one carried frame plus one new one is refused before
  any weight is read, rather than after an hour of work.

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
    let mut scene_specs: Vec<String> = Vec::new();
    // `--frames` has a default, so "was it given" cannot be read back off the
    // value - and refusing `--scene` alongside it depends on knowing.
    let mut frames_given = false;
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
            "--scene" => scene_specs.push(need(i)?.clone()),
            "--frames" => {
                o.frames = num(i, "--frames")?;
                frames_given = true;
            }
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
            "--mid-frame" => {
                o.mid_frame = Some(need(i)?.clone());
            }
            "--mid-frame-at" => o.mid_frame_at = Some(num(i, "--mid-frame-at")?),
            // `--no-stretch` is a bare flag (no value), unlike every other
            // option above - handled separately so the `i += 2` stride
            // below stays uniform for everything else.
            "--no-stretch" => {
                o.stretch = false;
                i += 1;
                continue;
            }
            // Same bare-flag shape as `--no-stretch`.
            "--audio" => {
                o.audio = true;
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
    // One scene is what a `--prompt`/`--frames` run always was, so the whole
    // command below has exactly one shape to handle.
    let scenes: Vec<ltxv::longform::Scene> = if scene_specs.is_empty() {
        vec![ltxv::longform::Scene { frames: o.frames, prompt: prompt.ok_or("--prompt is required (or one --scene <frames>:<prompt> per scene)")? }]
    } else {
        if prompt.is_some() || frames_given {
            return Err("--scene already carries its own frame count and prompt, so it cannot be combined with --prompt/--frames - write every scene as its own --scene <frames>:<prompt>".into());
        }
        scene_specs.iter().map(|s| ltxv::longform::Scene::parse(s)).collect::<Result<_, _>>()?
    };
    o.frames = scenes.iter().map(|s| s.frames).sum();
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
    let plan = ltxv::longform::scene_plan(&scenes, lh, lw, context_latents, long.max_window_tokens)?;
    let windows: Vec<&ltxv::longform::Window> = plan.iter().flatten().collect();
    // Audio is generated by ONE joint denoise over the whole clip's own
    // duration, so a clip built out of several windows (or several scenes)
    // has no single audio latent to carry: what crosses a window seam today
    // is a video latent prefix, and the audio stream's counterpart - a
    // carried audio context frozen at sigma 0 - has not been designed.
    // Refused rather than generated per window and concatenated, which would
    // restart the sound at every seam.
    if o.audio && windows.len() > 1 {
        return Err(format!(
            "--audio is not supported for a multi-window clip: this {}-frame request is {} windows, and carrying the audio stream across a window seam has not been designed. Ask for a clip that fits one window, or drop --audio.",
            o.frames,
            windows.len()
        ));
    }
    if o.audio && scenes.len() > 1 {
        return Err("--audio is not supported for a multi-scene clip: every scene is generated with nothing carried from the one before it, so the sound would restart at each boundary".into());
    }
    let tokens = windows.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    let window_desc = if scenes.len() > 1 {
        format!(" [{} scenes, {} windows, {context_frames}-frame rolling latent context within a scene and a reset at each scene boundary]", scenes.len(), windows.len())
    } else if windows.len() > 1 {
        format!(" [{} windows, {context_frames}-frame rolling latent context]", windows.len())
    } else {
        String::new()
    };
    let forwards = if o.guidance > 1.0 { 2 } else { 1 };
    // The audio path is a DIFFERENT arithmetic tier, not the same model with
    // a flag: it runs the audio-extended block, which has no int8
    // implementation, as host fp32 (see `ltxv::av_stream`). Saying "int8
    // compute" for it would misreport the run.
    let dit_desc = match (o.dit_config.as_str(), o.audio) {
        ("tiny", _) => "tiny random-weight DiT",
        (_, true) => "REAL checkpoint audio+video DiT (host fp32, both streams)",
        (_, false) => "REAL checkpoint DiT (int8 compute, video stream only)",
    };
    let ctx_desc = if paths.text_encoder.is_some() { "real Gemma-4 text encoder" } else { "stub text context (no real encoder)" };
    let audio_desc = if o.audio {
        format!(", +{} audio tokens ({:.2}s of 16 kHz stereo, denoised jointly)", ltxv::audio::latent_frames(o.frames, o.fps), o.frames as f32 / o.fps as f32)
    } else {
        String::new()
    };
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
    // The resolved pixel frame, not the flag: an unset --mid-frame-at is the
    // clip's own midpoint, and a caller should be told which frame that is
    // before the run rather than have to derive it.
    let mid_desc = match o.mid_frame.as_deref() {
        Some(m) => format!(", mid-frame ({m}) at frame {}", ltxv::pipeline::mid_anchor_frame(o.frames, o.mid_frame_at)?),
        None => String::new(),
    };
    eprintln!(
        "ltxv ({dit_desc}, real VAE, {ctx_desc}): {} frames at {}x{}, {steps_desc} steps x {forwards} forward(s) of {tokens} tokens, eta {}, guidance {}, seed {}{img_desc}{mid_desc}{stage_desc}{window_desc}{audio_desc}",
        o.frames, o.width, o.height, o.eta, o.guidance, o.seed
    );
    if scenes.len() > 1 {
        for (si, (s, windows)) in scenes.iter().zip(&plan).enumerate() {
            eprintln!("  scene {}/{}: {} frames, {} window(s), frames {}..{} - {:?}", si + 1, scenes.len(), s.frames, windows.len(), windows[0].first_frame, windows[0].first_frame + s.frames, s.prompt);
        }
    }

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it: Ctrl-C already
    // ends the process - see `wan_cli::t2v`'s identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::generate_scenes(&paths, &scenes, &long, &cancel, |done, total, phase| {
        eprint!("\rltxv [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    // Every stage, plus whatever none of them explains. A breakdown whose
    // parts summed to half its own total is what hid the largest stage in
    // this pipeline from two prior optimization passes, so the remainder is
    // printed rather than left implicit.
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "ltxv: {wall:.1}s total  (build {:.2}s, text encode {:.1}s, denoise {:.1}s = {:.3}s/forward, vae {:.1}s, audio {:.1}s, other {:.1}s)",
        timings.build_dit,
        timings.text_encode,
        timings.denoise,
        timings.secs_per_forward(),
        timings.decode,
        timings.audio_decode,
        timings.unattributed(wall)
    );

    let frames: Vec<imaging::Rgb8> = video.frames.iter().map(|px| imaging::Rgb8::new(video.width, video.height, px.clone())).collect::<Result<_, _>>()?;
    let track = video.audio.as_ref().map(|a| imaging::video::AudioTrack { channels: a.channels.clone(), sample_rate: a.sample_rate });
    let audio_desc = video
        .audio
        .as_ref()
        .map(|a| format!(", {:.2}s of {} Hz audio in {} channels", a.seconds(), a.sample_rate, a.channels.len()))
        .unwrap_or_default();
    let enc_opts = imaging::video::VideoEncodeOpts { audio: track, ..Default::default() };
    match imaging::video::encode_frames(&frames, std::path::Path::new(&out), video.fps as f64, &enc_opts)? {
        imaging::video::Encoded::Video(p) => {
            eprintln!("ltxv: wrote {} ({}x{}, {} frames at {} fps{audio_desc})", p.display(), video.width, video.height, frames.len(), video.fps);
        }
        imaging::video::Encoded::Frames { dir, command } => {
            eprintln!("ltxv: ffmpeg is not on PATH, so the {} frames are numbered PPMs in {}", frames.len(), dir.display());
            if video.audio.is_some() {
                eprintln!("ltxv: the generated sound is {}/audio.wav - it is NOT lost, the command below muxes it in", dir.display());
            }
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
    let mut context_frames = ltxv::longform::CONTEXT_FRAMES;
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
            "--context-frames" => context_frames = num(i, "--context-frames")?,
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
    o.context_latent_frames = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        vcfg.latent_frames(context_frames as u32).ok_or_else(|| format!("--context-frames {context_frames} is not of the form 1 + 8k"))? as usize
    };

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
    // `upscale` re-renders a finished clip's PICTURE. It never runs the
    // audio-visual DiT, so it has no sound track of its own to carry; any
    // audio the input file had stays in the input file.
    let clip = ltxv::pipeline::Video { width: w, height: h, fps, frames: decoded.into_iter().map(|f| f.px).collect(), audio: None };

    let plan = ltxv::pipeline::refine_plan(clip.frames.len(), (h as usize * o.factor) / 32, (w as usize * o.factor) / 32, o.context_latent_frames, o.max_refine_tokens)?;
    let ctx_desc = if paths.text_encoder.is_some() { "real Gemma-4 text encoder" } else { "stub text context (no real encoder)" };
    let dit_desc = if o.base.dit_config == "tiny" { "tiny random-weight DiT" } else { "REAL checkpoint DiT (int8 compute)" };
    let plan_desc = match plan.get(1).map(|s| s.context) {
        // The carried count is the whole story of a multi-pass upscale, so it
        // is in the one line the caller reads before committing an hour.
        Some(c) => format!(", {} refinement passes each carrying {c} latent frame(s) of the previous one", plan.len()),
        None => String::new(),
    };
    if prompt.is_empty() {
        eprintln!("ltxv upscale: no --prompt, so the refinement pass denoises against an empty context - pass the clip's original prompt for better detail");
    }
    eprintln!(
        "ltxv upscale ({dit_desc}, real VAE + real x2 latent spatial upscaler, {ctx_desc}): {} frames, {w}x{h} -> {}x{} at {fps} fps, {} refinement steps, guidance {}, seed {}{plan_desc}",
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
            "--scene",
            "--start-frame",
            "--end-frame",
            "--mid-frame",
            "--mid-frame-at",
            "--conditioning-strength",
            "--context-frames",
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
