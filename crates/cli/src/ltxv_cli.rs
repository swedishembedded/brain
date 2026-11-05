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
//! brain ltxv v2v --input stunt.mp4 --mask masks/ --prompt "a woman in a red coat" --output-path swapped.mp4
//! ```
//!
//! **`--dit-config` decides whether any of this is a quality claim.** The VAE
//! and the latent upscalers are always real (real weights, `--vae` resolved
//! from the model store - the only required role - and
//! `$BRAIN_LTXV_UPSAMPLER_SPATIAL`). The DiT is the tiny config with FRESH
//! RANDOM WEIGHTS by default, which makes every output a wiring proof and
//! nothing else; `--dit-config ltx25_22b` with a resolved `--dit` loads the
//! real checkpoint. The prompt likewise only folds into a deterministic
//! context stub unless a resolved `--text-encoder` names the real Gemma-4
//! encoder. dit/vae/audio_vae/text_encoder/tokenizer are resolved through
//! `brain_modelstore::resolve` (`ltxv::spec::LtxvSpec`) rather than
//! `BRAIN_LTXV_*` - see `resolve_ltxv`. See `ltxv::pipeline`'s module doc for
//! the full account of what is real and what is not.

use ltxv::pipeline::{DfrOpts, DfrPaths, GenOpts, LongOpts, MaskedOpts, Paths, Sampler, UpscaleOpts};
use ltxv::spec::LtxvSpec;

/// The guidance flags every generating subcommand takes ([`set_guidance`]).
macro_rules! guidance_help {
    () => {
        "  --guidance-preset <P>     none (one conditional forward per step - the
                            distilled checkpoint; the default) or reference (the
                            reference pipeline's non-distilled guidance: CFG 3
                            video / 7 audio, STG 1 on block 28, rescale 0.7,
                            modality guidance 3). The flags below override it.
  --guidance <G>            classifier-free guidance on both streams (1 = off)
  --audio-guidance <G>      classifier-free guidance on the audio stream
  --stg <S>                 spatio-temporal guidance: self-attention bypassed in
                            --stg-blocks (0 = off)
  --audio-stg <S>           spatio-temporal guidance on the audio stream
  --stg-blocks <L>          comma-separated blocks STG perturbs, e.g. 28
  --rescale <R>             scale the guided prediction back toward the
                            conditional one's standard deviation, 0..1
  --audio-rescale <R>       rescale on the audio stream
  --modality-guidance <M>   guidance against the pass with audio<->video
                            cross-attention cut, video stream (1 = off)
  --audio-modality-guidance <M>  the same for the audio stream
"
    };
}

const HELP: &str = concat!(r#"brain ltxv t2v - LTX-2.5 text to video (M4 smoke test: real VAE + tiny random-weight DiT, no real text encoder yet)

  brain ltxv t2v --prompt <text> --output-path <out.mp4> [options]

Required (one of):
  --prompt <text>          what the whole clip shows, with --frames for its
                           length - a single-scene clip
  --scene <frames>:<text>  ...or one repeat of this per scene, for a clip
                           that changes scene part way through (see Scenes).
                           --scene carries its own length and prompt, so it
                           cannot be combined with --prompt/--frames
  --output-path <file>     .mp4 / .mov / .mkv / .webm (AV1 video, Opus
                           sound) or .gif, written by brain itself

Sampling (defaults are a small smoke-test clip, see ltxv::pipeline::GenOpts):
  --frames <N>              video frames, must be 1 + 8k (default 9). Any
                            length is accepted: past what one denoising
                            window holds the clip is generated as several
                            windows with a rolling latent context (see
                            Long-form clips)
  --width <W> --height <H>  pixels, multiples of 32 (default 64x64)
  --steps <N>                denoise steps (default 4)
"#, guidance_help!(), r#"  --seed <S>                 initial-noise/weight/context seed (default: drawn at random and printed)
  --fps <N>                  frame rate written into the container (default 8)
  --base-shift <F>           LTX2Scheduler shift anchor at 1024 tokens (0.95)
  --max-shift <F>            LTX2Scheduler shift anchor at 4096 tokens (2.05)
  --no-stretch                skip the terminal-sigma stretch (on by default)
  --terminal <F>              stretch target sigma (default 0.1)
  --sampler <euler|res2s>     denoising loop (default euler): euler takes one
                             guided evaluation per step, ancestral when
                             --eta > 0 (the distilled pipeline's); res2s is
                             second-order, two guided evaluations per step
                             with SDE noise (the reference's high-quality
                             pipeline's) and also refines stage 2 with it
  --eta <F>                   sampler noise strength, 0..1: euler 0 is
                             deterministic, 1 fully ancestral (default 1.0);
                             res2s step-level SDE strength (default 0.5)
  --audio                     ALSO generate the clip's sound, from the same
                             prompt and the same forwards as the picture, and
                             mux it into the output container. LTX-2.5 is
                             natively audio-visual: without this flag only the
                             model's video half runs and the clip is silent.
                             Needs --dit-config ltx25_22b, a real
                             --text-encoder, and $BRAIN_LTXV_AUDIO_VAE. Any
                             clip length; the sound crosses a window seam
                             with the picture. Costs real time and memory -
                             see "Audio" below
  --audio-source <wav>        generate the video FOR this soundtrack
                             (audio-to-video): it is encoded through the audio
                             VAE, held frozen as the clip's audio stream while
                             the video denoises against it, and muxed as the
                             clip's sound. Needs --audio and a clip that fits
                             one denoising window
  --dit-config <name>         DiT config; "tiny" (default, fresh random
                             weights) or "ltx25_22b" (the real 22B
                             checkpoint, needs --dit/$BRAIN_LTXV_DIT)
  --start-frame <path>         PNG/JPEG still, encoded through the real VAE.
                             Given ALONE, it overwrites the clip's own
                             frame 0 outright (the generated video IS the
                             still there). Given alongside --mid-frame/
                             --end-frame, all of them switch to the OTHER
                             mechanism below - an appended guide, not a
                             replacement.
  --end-frame <path>           a clean image-guidance latent, appended to
                             the sequence at the clip's LAST pixel frame's
                             timestamp - not a replacement of that frame:
                             the generated token there still denoises
                             freely, strongly pulled toward the guide via
                             attention but not overwritten by it. Pass the
                             SAME path as --start-frame for a clip that
                             loops seamlessly (the generated content in
                             between connects the still to itself); a
                             different path for a clip that morphs between
                             two stills. Either flag works alone.
  --mid-frame <path>           same appended-guide mechanism, anchored at
                             ONE INTERIOR instant, so a single pass can be
                             anchored at its start, its middle and its end
                             at once - the way to keep a long or moving-
                             camera clip on course rather than only
                             pinning where it begins and ends. Works alone
                             or with either of the two above. Because this
                             is guidance rather than replacement, how
                             closely the clip actually reaches the mid
                             still's content depends on how much the scene
                             has to change to get there and how many
                             denoising steps it has to do it in - see
                             --conditioning-strength.
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
  a CFG scale other than 1, the conditional and unconditional forwards of every
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
  crates/ltxv/tests/it/cfg_parallel.rs, which times both placements and proves
  the two outputs identical. Set
  BRAIN_LTXV_CFG_PARALLEL=0 to force the old single-card behaviour; --device
  gpu0 also confines the run to one card, since placement never leaves the
  schedulable set --device names.

Weights (resolved from the models directory; a flag names a candidate
outright, an ambiguous or missing outcome prints every real candidate and
exits - --brain-data-dir/BRAIN_MODELS_DIR names where to look):
  --vae <path>              the causal 3D video VAE (the only required role)
  --dit <path>              real 22B DiT GGUF (only read when --dit-config
                             ltx25_22b)
  --text-encoder <path>     real Gemma-4 text encoder (optional; the
                             deterministic prompt stub runs without it)
  --spatial-upsampler <path> spatial x2 latent upscaler - REQUIRED above
                             6144 video tokens, unused below (see Stages);
                             $BRAIN_LTXV_UPSAMPLER_SPATIAL when the store
                             holds none

Stages:
  The distilled checkpoint's fixed 8-sigma schedule is only distilled to
  build a clip from noise at a modest token count - upstream never runs it
  at the requested resolution, only at half of it, and then refines. Past
  6144 video tokens (a 25-frame clip above ~1600x896) one stage measurably
  disintegrates the END of the clip while the start stays correct, so a
  request past that ceiling runs the reference's own two stages: the full
  schedule at half resolution (stage 1, where motion is decided), a real
  x2 latent upscale, then three deterministic refinement steps at the
  requested size (stage 2, where detail is refined).  That needs
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

  It is opt-in because of COST, not correctness - but it runs on the same
  streamed/quantized/device-resident path the video-only stream does: the
  audio-extended block is quantized to int8, cached per checkpoint in host
  RAM, and as many blocks as the card's budget allows stay resident on it
  between steps. A step therefore costs modestly more than the same clip
  without sound rather than several times more. The command refuses up front,
  with both numbers, if this machine cannot hold what that arm needs.

  BRAIN_LTXV_AV_FP32=1 denoises on the eager host-fp32 REFERENCE arm instead -
  the same math at higher precision, and what the quantized arm is measured
  and gated against. It expands the whole checkpoint to host fp32 and
  re-uploads it per forward, so expect most of a large machine's RAM and
  several times the wall clock. The banner names whichever arm is running.

  Output is 48 kHz stereo, coded as Opus in .mp4/.mov/.mkv/.webm. A .gif is
  written silent with a line on stderr, since the container holds no audio
  stream.

  Any --frames length works. Past one denoising window the audio latent
  crosses each seam exactly as the video latent does - the previous window's
  own last audio tokens, frozen at sigma 0 at the head of the next window -
  and the windows contribute to ONE audio latent, decoded once, so no vocoder
  boundary lands on a seam. The two streams do not share a time resolution
  (one video latent frame is 8 * 25 / fps audio tokens, 25/3 of one at 24
  fps), so an --audio plan places its seams only where both grids have a
  boundary: it advances in multiples of 3 latent frames at 24 and 30 fps and
  of 1 at 25 fps, which is a different window split from the same clip
  without --audio. A shape whose seams cannot be placed is refused, naming
  what would work - lower --context-frames, or pick a length this frame rate
  can tile. Silent drift is never the answer.

  --scene is the exception: a scene boundary deliberately carries nothing, so
  the sound would restart at every cut and the per-scene token counts would
  not sum to the clip's own. A multi-scene --audio request is refused.

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
  recorded real run at). --start-frame conditions the first window as
  usual; --end-frame conditions the windows that own their instants, and
  every window also receives the anchors still ahead of it as appended
  guides, so each window bends its motion toward the next keyframe instead
  of running on alone. --mid-frame is pinned INTO its owning window's
  latent slot - frozen there like a start frame, so the clip passes through
  the mid image instead of merely bending toward it; windows before it see
  it as an appended guide.

  Stage-major (two-stage, video-only, default): When two-stage generation
  is enabled, the clip is generated via two independent window plans:
  stage 1 at half resolution (where motion is decided) and stage 2 at full
  resolution (where refinement detail is carried).  Clip-global anchors
  (--end-frame, --mid-frame) are routed to their owning window via
  window_gen_opts.  Audio-visual clips keep the window-major path.
  BRAIN_LTXV_LONGFORM_STAGE_MAJOR=0 disables the stage-major path.

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

Adapters:
  --adapter <path>           a LoRA to apply, repeatable: the official LTX
                             trainer's / a published IC-LoRA's ComfyUI
                             .safetensors, or one brain trained. Runs beside
                             the quantized weights, never folded into them.
  --lora-scale <s>           strength of the --adapter it follows (default 1.0,
                             ComfyUI's strength_model)
  --reference-video <path>   an IC-LoRA's reference clip (pose, depth, edges,
                             a source to restyle...): its first --frames
                             frames, at the geometry the --adapter declares,
                             appended beside the clip as clean tokens. Needs
                             the IC-LoRA trained to read it; one window only.
  --reference-strength <s>   how strongly the reference is held, 0..1
                             (default 1.0: perfectly clean)
  --stage-2-ic-lora          keep the adapters and the reference on a two-stage
                             run's refinement too (default: stage 2 runs the
                             bare checkpoint, as the reference recipe does)

Devices:
  --device <cpu|gpu>         DiT + VAE (default: the ambient BRAIN_DEVICE)

Subcommands:
  brain ltxv t2v --help       text to video (this command)
  brain ltxv upscale --help   re-render a finished clip at 2x
  brain ltxv v2v --help       regenerate a masked region of an existing clip
  brain ltxv dfr --help       DFR (Diffusion Fidelity Rendering) smoke test"#);

const UPSCALE_HELP: &str = concat!(r#"brain ltxv upscale - re-render a finished clip at twice its resolution

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
  --input <file>             the clip to upscale (AV1 in .mp4/.mkv/.webm, or
                             anything an installed ffmpeg reads). Its
                             width and height must be multiples of 32 and its
                             frame count of the form 1 + 8k, which is what
                             this pipeline's own output always is
  --output-path <file>       .mp4 / .mov / .mkv / .webm (AV1) or .gif, written
                             by brain itself

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
"#, guidance_help!(), r#"  --seed <S>                 refinement noise seed (default: drawn at random and printed)
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

Weights (resolved from the models directory; a flag names a candidate
outright, an ambiguous or missing outcome prints every real candidate and
exits):
  --vae <path>               the causal 3D video VAE (the only required role)
  --upsampler-spatial <path> $BRAIN_LTXV_UPSAMPLER_SPATIAL  spatial x2 latent
                                                   upscaler - REQUIRED, it is
                                                   what this command runs
  --dit <path>               real 22B DiT GGUF (only read when --dit-config
                             ltx25_22b)
  --text-encoder <path>      real Gemma-4 text encoder (without it --prompt
                             reaches the model only as a stub and cannot
                             guide anything)

  $BRAIN_LTXV_AUDIO_VAE is NOT read here. Upscaling refines the VIDEO latent
  only: it runs no audio-visual DiT and decodes no audio, so setting that
  variable changes nothing about this command. Note the consequence - the
  output has NO sound track at all. Any audio --input carried stays in
  --input, and re-attaching it is a separate ffmpeg mux.

Adapters:
  --adapter <path>           a LoRA to apply, repeatable: the official LTX
                             trainer's / a published IC-LoRA's ComfyUI
                             .safetensors, or one brain trained. Runs beside
                             the quantized weights, never folded into them.
  --lora-scale <s>           strength of the --adapter it follows (default 1.0,
                             ComfyUI's strength_model)

Devices:
  --device <cpu|gpu>         DiT + VAE + upscaler (default: BRAIN_DEVICE)"#);

const DFR_HELP: &str = concat!(r#"brain ltxv dfr - LTX-2.5 DFR (Diffusion Fidelity Rendering) smoke test

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
"#, guidance_help!(), r#"  --seed <S>                   initial-noise/weight/context seed (default: drawn at random and printed)
  --fps <N>                    stage-1 frame rate; the written fps is this
                               times 2^rounds (default 8)
  --base-shift <F>             LTX2Scheduler shift anchor at 1024 tokens (0.95)
  --max-shift <F>              LTX2Scheduler shift anchor at 4096 tokens (2.05)
  --no-stretch                  skip the terminal-sigma stretch (on by default)
  --terminal <F>                stretch target sigma (default 0.1)
  --sampler <euler|res2s>       denoising loop (default euler; see t2v)
  --eta <F>                     sampler noise strength (default 1.0 euler,
                                0.5 res2s)
  --dit-config <name>           DiT size; only "tiny" is implemented (default)
  --temporal-upsample-rounds <N> 0, 1, or 2 real temporal x2 refine rounds
                                 (default 0; > 0 needs --temporal-upsampler)

Weights (flag wins over the environment variable):
  --vae <path>                 $BRAIN_LTXV_VAE       the causal 3D video VAE
  --spatial-upsampler <path>   $BRAIN_LTXV_UPSAMPLER_SPATIAL
  --temporal-upsampler <path>  $BRAIN_LTXV_UPSAMPLER_TEMPORAL

Adapters:
  --adapter <path>           a LoRA to apply, repeatable: the official LTX
                             trainer's / a published IC-LoRA's ComfyUI
                             .safetensors, or one brain trained. Runs beside
                             the quantized weights, never folded into them.
  --lora-scale <s>           strength of the --adapter it follows (default 1.0,
                             ComfyUI's strength_model)

Devices:
  --device <cpu|gpu>            DiT + VAE + upscalers (default: BRAIN_DEVICE)"#);

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
        "v2v" => {
            if let Err(e) = v2v(&args[1..]) {
                eprintln!("ltxv v2v: {e}");
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

const V2V_HELP: &str = concat!(r#"brain ltxv v2v - regenerate a masked region of an existing clip

Usage:
  brain ltxv v2v --input <clip.mp4> --mask <dir> --prompt <text> --output-path <out.mp4>

Every latent position the mask marks as CONDITIONING is handed the source
clip's own latent and excluded from denoising, so it leaves the sampler
bit-exactly unchanged; everything else is renoised and redrawn from the
prompt. That is LTX-2.5's own `VideoConditionByMask` (ported in
`ltxv::maskcond`), and it needs no adapter - which matters, because the only
IC-LoRA published for LTX-2.5 is a pixel spatial upscaler, so the adapter
route to this is closed.

The clip's size is the output size: this runs ONE stage at the source's own
resolution and never rescales. It must be VAE-representable - 1 + 8k frames on
a 32-pixel grid - and it is refused rather than trimmed, because trimming
would desync it from a mask sequence produced by a separate run.

Required:
  --input <file>            the clip to edit (AV1 in .mp4/.mkv/.webm, or
                            anything an installed ffmpeg reads)
  --mask <dir>              a `brain/sam2-maskseq/1` mask-sequence directory:
                            one 8-bit PNG per source frame at the source
                            resolution, plus a masks.json giving the file
                            pattern, the frame count, the resolution and the
                            POLARITY. brain's SAM 2 video tracker writes one.
                            The polarity is read from that manifest and never
                            inferred: the TRACKED OBJECT is what gets
                            regenerated, and reading it backwards would
                            preserve the subject and redraw the entire set.
                            A frame count or resolution that disagrees with
                            the clip is an error, never reconciled. A frame
                            the tracker marks occluded (object_score < 0) is
                            conditioned FULLY, so the edit simply does not
                            happen there rather than the frame dissolving.
  --window <start>:<end>    instead of --mask: a retake - regenerate the
                            seconds [start, end) of the clip and hold the rest
  --hold-video              regenerate no frame: with --audio this generates
                            the clip's sound for its picture (video-to-audio)
  --audio                   ALSO denoise the clip's sound, jointly with the
                            picture (needs the same checkpoints as t2v --audio)
  --audio-source <wav>      the clip's own soundtrack: held outside the
                            window and regenerated inside it (without one, the
                            whole soundtrack is generated)
  --hold-audio              keep --audio-source unchanged everywhere
  --prompt <text>           what the WHOLE frame shows - the model denoises
                            the regenerated region in the context of the
                            carried one. Identity comes from the prompt; no
                            face crop or per-subject embedding enters here.
  --output-path <file>      .mp4 / .mov / .mkv / .webm (AV1) or .gif, written
                            by brain itself

Sampling:
  --strength <S>            how hard the mask's conditioning region is pinned
                            (default 1.0 = bit-exact; lower lets it drift
                            toward the prompt)
  --steps <N>               denoise steps (default 4; IGNORED for
                            --dit-config ltx25_22b, whose distilled schedule
                            has a fixed shape)
"#, guidance_help!(), r#"  --seed <S>                noise seed (default: drawn at random and printed)
  --fps <N>                 frame rate for the output. Default: whatever
                            ffprobe reports for --input
  --dit-config <name>       "tiny" (default, fresh random weights - a wiring
                            proof, and enough to verify that the carried
                            region really is carried) or "ltx25_22b" for the
                            real checkpoint (needs --dit/$BRAIN_LTXV_DIT)

Sound:
  This pass conditions a VIDEO latent and never runs the audio stream, so the
  output has NO sound track. The source's own audio stays in --input, and
  re-attaching it is a separate ffmpeg mux.

Weights (resolved from the models directory; a flag names a candidate
outright, an ambiguous or missing outcome prints every real candidate and
exits):
  --vae <path>              the causal 3D video VAE (the only required role)
  --dit <path>              real 22B DiT GGUF (only read when --dit-config
                            names the real model)
  --text-encoder <path>     real Gemma-4 text encoder; without it the
                            prompt reaches a stub
                            $BRAIN_LTXV_UPSAMPLER_SPATIAL and the audio VAE
                            role are not read by this command - it never
                            upscales and never generates sound.

Adapters:
  --adapter <path>           a LoRA to apply, repeatable: the official LTX
                             trainer's / a published IC-LoRA's ComfyUI
                             .safetensors, or one brain trained. Runs beside
                             the quantized weights, never folded into them.
  --lora-scale <s>           strength of the --adapter it follows (default 1.0,
                             ComfyUI's strength_model)

Placement:
  --device <cpu|gpu>        DiT + VAE (default: the ambient BRAIN_DEVICE)"#);

/// Resolve `dit`/`vae`/`audio_vae`/`text_encoder`/`tokenizer` through the
/// model-store resolver (`brain_modelstore::resolve::resolve` +
/// [`LtxvSpec`]) instead of `BRAIN_LTXV_*` - `overrides` carries whatever
/// role flags [`crate::resolver_cli::extract_role_overrides`] already pulled
/// out of this command's own argv. `vae` is the only role `resolve_or_exit`
/// can fail to fill; every other role simply stays `None` in the returned
/// `Paths`, exactly as an unset env var/absent flag already did.
fn resolve_ltxv(overrides: &std::collections::BTreeMap<String, String>, spatial_upsampler: Option<&str>) -> Result<Paths, String> {
    let spec = LtxvSpec;
    let assembly = crate::resolver_cli::resolve_or_exit("ltxv", &spec, overrides);
    Paths::from_assembly(&assembly, spatial_upsampler)
}

/// What a [`GenOpts`] decides about a clip, for the run record beside it.
fn base_params(o: &GenOpts) -> serde_json::Value {
    serde_json::json!({
        "frames": o.frames, "width": o.width, "height": o.height, "steps": o.steps,
        "guidance": format!("{:?}", o.guidance), "fps": o.fps,
    })
}

fn t2v(args: &[String]) -> Result<(), String> {
    let (overrides, args) = crate::resolver_cli::extract_role_overrides(&LtxvSpec, args);
    let (seed, args) = crate::args::split_seed(&args, "ltxv t2v");
    let args = &args[..];
    let mut o = GenOpts { seed, ..GenOpts::default() };
    // `--eta`'s default depends on `--sampler`, which may come after it.
    let mut eta: Option<f64> = None;
    // How many `--adapter`s already carry their `--lora-scale`.
    let mut scaled = 0usize;
    let mut reference_path: Option<String> = None;
    let mut reference_strength = 1.0f32;
    let mut context_frames = ltxv::longform::CONTEXT_FRAMES;
    let mut prompt: Option<String> = None;
    let mut scene_specs: Vec<String> = Vec::new();
    // `--frames` has a default, so "was it given" cannot be read back off the
    // value - and refusing `--scene` alongside it depends on knowing.
    let mut frames_given = false;
    let mut out: Option<String> = None;
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
                out = Some(crate::args::strip_out_name_prefix(need(i)?, "video").to_string());
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
            f if guidance_flag(f) => set_guidance(&mut o.guidance, f, need(i)?)?,
            "--base-shift" => o.base_shift = dbl(i, "--base-shift")?,
            "--max-shift" => o.max_shift = dbl(i, "--max-shift")?,
            "--terminal" => o.terminal = dbl(i, "--terminal")?,
            "--sampler" => o.sampler = Sampler::parse(need(i)?)?,
            "--eta" => eta = Some(dbl(i, "--eta")?),
            "--dit-config" => o.dit_config = need(i)?.clone(),
            "--adapter" => o.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.adapters, &mut scaled, flt(i, "--lora-scale")?)?,
            "--reference-video" => reference_path = Some(need(i)?.clone()),
            "--reference-strength" => reference_strength = flt(i, "--reference-strength")?,
            "--stage-2-ic-lora" => {
                o.refine_in_context = true;
                i += 1;
                continue;
            }
            "--device" => {
                o.device = Some(need(i)?.clone());
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
            "--audio-source" => o.audio_source = Some(ltxv::audio::read_wav(need(i)?)?),
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{HELP}")),
        }
        i += 2;
    }
    o.eta = o.sampler.resolve_eta(eta)?;
    if let Some(path) = &reference_path {
        if !(0.0..=1.0).contains(&reference_strength) {
            return Err(format!("--reference-strength {reference_strength} is outside [0, 1]"));
        }
        o.reference = Some(ltxv::pipeline::VideoReference { clip: read_clip(path, o.fps)?, strength: reference_strength });
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
    let paths = resolve_ltxv(&overrides, None)?;

    let (lh, lw) = (o.height / 32, o.width / 32);
    let context_latents = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        vcfg.latent_frames(context_frames as u32).ok_or_else(|| format!("--context-frames {context_frames} is not of the form 1 + 8k"))? as usize
    };
    let long = LongOpts { context_latent_frames: context_latents, max_window_tokens: ltxv::longform::max_window_tokens_from_env(), max_refine_tokens: ltxv::pipeline::REFINE_MAX_TOKENS, base: o.clone() };
    // The plan decides what this run actually is, so it is resolved before
    // the preview line rather than after: a multi-window clip's per-forward
    // token count is one WINDOW's, and reporting the whole clip's would name
    // a number no forward in the run has.
    // The same quantum the generation will plan with, so this preview
    // describes the run that is about to happen rather than a different
    // window split: an audio-visual clip's seams have to land on whole audio
    // tokens (see `ltxv::audio`'s module doc).
    let align = if o.audio { ltxv::audio::window_latent_frame_quantum(o.fps) } else { 1 };
    let plan = ltxv::longform::scene_plan_aligned(&scenes, lh, lw, context_latents, long.max_window_tokens, align)
        .map_err(|e| if o.audio { format!("{e}. {}", ltxv::audio::quantum_note(o.fps)) } else { e })?;
    let windows: Vec<&ltxv::longform::Window> = plan.iter().flatten().collect();
    // A window seam carries both streams; a SCENE boundary carries neither,
    // by design - see `generate_scenes`, which refuses this too. Checked here
    // as well so a multi-scene audio request costs milliseconds rather than a
    // whole-checkpoint host expansion.
    if o.audio && scenes.len() > 1 {
        return Err("--audio is not supported for a multi-scene clip: every scene is generated with nothing carried from the one before it, so the sound would restart at each boundary".into());
    }
    // The audio stream's own layout over that plan, resolved now for the same
    // reason: a frame count and frame rate whose seams cannot land on whole
    // audio tokens is refused before any weight is read.
    if o.audio {
        for (windows, scene) in plan.iter().zip(&scenes) {
            ltxv::audio::audio_plan(windows, context_latents, scene.frames, o.fps)?;
        }
    }
    let tokens = windows.iter().map(|w| w.tokens(lh, lw)).max().unwrap_or(0);
    let window_desc = if scenes.len() > 1 {
        format!(" [{} scenes, {} windows, {context_frames}-frame rolling latent context within a scene and a reset at each scene boundary]", scenes.len(), windows.len())
    } else if windows.len() > 1 {
        format!(" [{} windows, {context_frames}-frame rolling latent context]", windows.len())
    } else {
        String::new()
    };
    let forwards = o.guidance.forwards_per_step(o.audio);
    // Not spelled out here: `av_arm_label` is the same pure function
    // `ltxv::pipeline::build_denoiser` routes on, so this line and the object
    // the run actually denoises with cannot disagree about which arm ran.
    // That matters because "it should be faster now" is not evidence and this
    // banner is what a reader checks instead.
    let dit_desc = ltxv::pipeline::av_arm_label(o.dit_config != "tiny", o.audio, ltxv::pipeline::av_fp32_reference());
    let ctx_desc = if paths.text_encoder.is_some() { "real Gemma-4 text encoder" } else { "stub text context (no real encoder)" };
    let audio_desc = if o.audio {
        format!(", +{} audio tokens ({:.2}s of 48 kHz stereo, denoised jointly)", ltxv::audio::latent_frames(o.frames, o.fps), o.frames as f32 / o.fps as f32)
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
        "ltxv ({dit_desc}, real VAE, {ctx_desc}): {} frames at {}x{}, {steps_desc} steps x {forwards} forward(s) of {tokens} tokens, eta {}, guidance {:?}, seed {}{img_desc}{mid_desc}{stage_desc}{window_desc}{audio_desc}",
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

    // The sound track first, then the frames MOVED rather than cloned: a
    // long clip's pixels are the largest buffer this command ever holds, and
    // copying all of them to hand them to the encoder is a copy nothing reads
    // twice.
    let audio_desc = video
        .audio
        .as_ref()
        .map(|a| format!(", {:.2}s of {} Hz audio in {} channels", a.seconds(), a.sample_rate, a.channels.len()))
        .unwrap_or_default();
    let track = video.audio.map(|a| imaging::video::AudioTrack { channels: a.channels, sample_rate: a.sample_rate });
    let (width, height) = (video.width, video.height);
    let frames: Vec<imaging::Rgb8> = video.frames.into_iter().map(|px| imaging::Rgb8::new(width, height, px)).collect::<Result<_, _>>()?;
    let enc_opts = imaging::video::VideoEncodeOpts { audio: track, ..Default::default() };
    let written = imaging::video::encode_frames(&frames, std::path::Path::new(&out), video.fps as f64, &enc_opts)?;
    eprintln!("ltxv: wrote {} ({}x{}, {} frames at {} fps{audio_desc})", written.display(), video.width, video.height, frames.len(), video.fps);
    crate::run_record::write_run_record(
        &out,
        "ltxv t2v",
        o.seed,
        serde_json::json!({
            "scenes": scenes.iter().map(|s| serde_json::json!({ "frames": s.frames, "prompt": s.prompt })).collect::<Vec<_>>(),
            "settings": base_params(&o),
        }),
    )?;
    Ok(())
}

/// Decode a whole video file into the pipeline's RGB8 clip, refusing one
/// whose frames change size.
fn read_clip(path: &str, fps: usize) -> Result<ltxv::pipeline::Video, String> {
    let decoded = imaging::video::decode_frames_rgb8(std::path::Path::new(path), &imaging::video::VideoDecodeOpts { fps: None, max_frames: 0, spread: 0 })?;
    let first = decoded.first().ok_or_else(|| format!("{path} has no frames"))?;
    let (w, h) = (first.w, first.h);
    if let Some(bad) = decoded.iter().position(|f| (f.w, f.h) != (w, h)) {
        return Err(format!("{path} changes size at frame {bad} ({}x{} after {w}x{h}) - a clip has one resolution", decoded[bad].w, decoded[bad].h));
    }
    Ok(ltxv::pipeline::Video { width: w, height: h, fps, frames: decoded.into_iter().map(|f| f.px).collect(), audio: None })
}

fn upscale(args: &[String]) -> Result<(), String> {
    let (overrides, args) = crate::resolver_cli::extract_role_overrides(&LtxvSpec, args);
    let (seed, args) = crate::args::split_seed(&args, "ltxv upscale");
    let args = &args[..];
    let mut o = UpscaleOpts::default();
    // How many `--adapter`s already carry their `--lora-scale`.
    let mut scaled = 0usize;
    o.base.seed = seed;
    let mut prompt: Option<String> = None;
    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut upsampler_spatial: Option<String> = None;
    let mut fps: Option<usize> = None;
    let mut context_frames = ltxv::longform::CONTEXT_FRAMES;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => {
                prompt = Some(need(i)?.clone());
            }
            "--input" | "--in" => {
                input = Some(need(i)?.clone());
            }
            "--output-path" | "--out" => {
                out = Some(crate::args::strip_out_name_prefix(need(i)?, "video").to_string());
            }
            "--factor" => o.factor = num(i, "--factor")?,
            "--refine-steps" => o.refine_steps = num(i, "--refine-steps")?,
            "--context-frames" => context_frames = num(i, "--context-frames")?,
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--fps" => fps = Some(num(i, "--fps")?),
            "--dit-config" => o.base.dit_config = need(i)?.clone(),
            "--adapter" => o.base.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.base.adapters, &mut scaled, need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?)?,
            "--device" => {
                o.base.device = Some(need(i)?.clone());
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
    let paths = resolve_ltxv(&overrides, upsampler_spatial.as_deref())?;
    let prompt = prompt.unwrap_or_default();
    o.context_latent_frames = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        vcfg.latent_frames(context_frames as u32).ok_or_else(|| format!("--context-frames {context_frames} is not of the form 1 + 8k"))? as usize
    };

    let in_path = std::path::Path::new(&input);
    let decoded = read_clip(&input, 0)?;
    let (w, h) = (decoded.width, decoded.height);
    // The source's own rate, so an upscale never silently changes playback
    // speed. `--fps` overrides; a missing `ffprobe` is not fatal (see
    // `imaging::video::probe_fps`), it just means the caller has to say.
    let fps = fps.unwrap_or_else(|| imaging::video::probe_fps(in_path).map(|f| f.round() as usize).filter(|&f| f > 0).unwrap_or(o.base.fps));
    o.base.fps = fps;
    // `upscale` re-renders a finished clip's PICTURE. It never runs the
    // audio-visual DiT, so it has no sound track of its own to carry; any
    // audio the input file had stays in the input file.
    let clip = ltxv::pipeline::Video { fps, ..decoded };

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
        "ltxv upscale ({dit_desc}, real VAE + real x2 latent spatial upscaler, {ctx_desc}): {} frames, {w}x{h} -> {}x{} at {fps} fps, {} refinement steps, guidance {:?}, seed {}{plan_desc}",
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

    // Moved, not cloned - an upscaled clip's pixels are four times the input's
    // and are the largest buffer this command holds; see `t2v`'s own note.
    let (out_w, out_h, out_fps) = (video.width, video.height, video.fps);
    let frames: Vec<imaging::Rgb8> = video.frames.into_iter().map(|px| imaging::Rgb8::new(out_w, out_h, px)).collect::<Result<_, _>>()?;
    let written = imaging::video::encode_frames(&frames, std::path::Path::new(&out), out_fps as f64, &Default::default())?;
    eprintln!("ltxv upscale: wrote {} ({out_w}x{out_h}, {} frames at {out_fps} fps)", written.display(), frames.len());
    crate::run_record::write_run_record(
        &out,
        "ltxv upscale",
        o.base.seed,
        serde_json::json!({
            "prompt": prompt, "input": input, "factor": o.factor, "refine_steps": o.refine_steps,
            "settings": base_params(&o.base),
        }),
    )?;
    Ok(())
}

/// Masked video-to-video: the clip and its mask sequence in, the same clip
/// with one region redrawn out.
///
/// Shaped like [`upscale`] - decode the input, hand the pipeline a
/// `Video`, encode what comes back - because the difference between the two is
/// entirely in what the pipeline does with the latent, not in the file
/// handling.
fn v2v(args: &[String]) -> Result<(), String> {
    let (overrides, args) = crate::resolver_cli::extract_role_overrides(&LtxvSpec, args);
    let (seed, args) = crate::args::split_seed(&args, "ltxv v2v");
    let args = &args[..];
    let mut o = MaskedOpts::default();
    // How many `--adapter`s already carry their `--lora-scale`.
    let mut scaled = 0usize;
    o.base.seed = seed;
    let mut prompt: Option<String> = None;
    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut fps: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let flt = |i: usize, what: &str| -> Result<f32, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--input" | "--in" => input = Some(need(i)?.clone()),
            "--mask" => o.mask_dir = need(i)?.clone(),
            "--window" => {
                let v = need(i)?;
                let (a, b) = v.split_once(':').ok_or_else(|| format!("--window: '{v}' is not <start>:<end> in seconds"))?;
                let sec = |x: &str| x.trim().parse::<f32>().map_err(|e| format!("--window: '{x}': {e}"));
                o.window = Some((sec(a)?, sec(b)?));
            }
            "--audio-source" => o.base.audio_source = Some(ltxv::audio::read_wav(need(i)?)?),
            f @ ("--hold-video" | "--hold-audio" | "--audio") => {
                match f {
                    "--hold-video" => o.regenerate_video = false,
                    "--hold-audio" => o.regenerate_audio = false,
                    _ => o.base.audio = true,
                }
                i += 1;
                continue;
            }
            "--output-path" | "--out" => out = Some(crate::args::strip_out_name_prefix(need(i)?, "video").to_string()),
            "--strength" => o.strength = flt(i, "--strength")?,
            "--steps" => o.base.steps = num(i, "--steps")?,
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--fps" => fps = Some(num(i, "--fps")?),
            "--dit-config" => o.base.dit_config = need(i)?.clone(),
            "--adapter" => o.base.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.base.adapters, &mut scaled, need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?)?,
            "--device" => o.base.device = Some(need(i)?.clone()),
            "--help" | "-h" => {
                println!("{V2V_HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{V2V_HELP}")),
        }
        i += 2;
    }
    let input = input.ok_or("--input is required")?;
    let out = out.ok_or("--output-path is required")?;
    let prompt = prompt.ok_or("--prompt is required: the model denoises the regenerated region against the text context, and an empty one costs exactly the detail this command exists to produce")?;
    if o.mask_dir.is_empty() && o.window.is_none() {
        return Err("--mask or --window is required: without one this command has nothing to regenerate and would return the input clip".into());
    }
    let paths = resolve_ltxv(&overrides, None)?;

    let in_path = std::path::Path::new(&input);
    let decoded = read_clip(&input, 0)?;
    let (w, h) = (decoded.width, decoded.height);
    // The source's own rate, so an edit never silently changes playback speed.
    let fps = fps.unwrap_or_else(|| imaging::video::probe_fps(in_path).map(|f| f.round() as usize).filter(|&f| f > 0).unwrap_or(o.base.fps));
    o.base.fps = fps;
    let clip = ltxv::pipeline::Video { fps, ..decoded };

    let ctx_desc = if paths.text_encoder.is_some() { "real Gemma-4 text encoder" } else { "stub text context (no real encoder)" };
    let dit_desc = if o.base.dit_config == "tiny" { "tiny random-weight DiT" } else { "REAL checkpoint DiT (int8 compute)" };
    eprintln!(
        "ltxv v2v ({dit_desc}, real VAE, {ctx_desc}): {} frames, {w}x{h} at {fps} fps, mask {}, strength {}, guidance {:?}, seed {}",
        clip.frames.len(),
        o.mask_dir,
        o.strength,
        o.base.guidance,
        o.base.seed
    );

    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it - see `t2v`'s
    // identical reasoning.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::generate_masked(&paths, &prompt, &clip, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv v2v [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "ltxv v2v: {wall:.1}s total  (build {:.2}s, text encode {:.1}s, denoise {:.1}s = {:.3}s/forward, vae {:.1}s, other {:.1}s)",
        timings.build_dit,
        timings.text_encode,
        timings.denoise,
        timings.secs_per_forward(),
        timings.decode,
        timings.unattributed(wall)
    );

    let (out_w, out_h, out_fps) = (video.width, video.height, video.fps);
    let frames: Vec<imaging::Rgb8> = video.frames.into_iter().map(|px| imaging::Rgb8::new(out_w, out_h, px)).collect::<Result<_, _>>()?;
    let written = imaging::video::encode_frames(&frames, std::path::Path::new(&out), out_fps as f64, &Default::default())?;
    eprintln!("ltxv v2v: wrote {} ({out_w}x{out_h}, {} frames at {out_fps} fps, no sound - see --help)", written.display(), frames.len());
    crate::run_record::write_run_record(
        &out,
        "ltxv v2v",
        o.base.seed,
        serde_json::json!({
            "prompt": prompt, "input": input, "mask": o.mask_dir, "strength": o.strength,
            "settings": base_params(&o.base),
        }),
    )?;
    Ok(())
}

fn dfr(args: &[String]) -> Result<(), String> {
    let (seed, rest) = crate::args::split_seed(args, "ltxv dfr");
    let args = &rest[..];
    let mut o = DfrOpts::default();
    let mut eta: Option<f64> = None;
    // How many `--adapter`s already carry their `--lora-scale`.
    let mut scaled = 0usize;
    o.base.seed = seed;
    let mut prompt: Option<String> = None;
    let mut out: Option<String> = None;
    let mut vae: Option<String> = None;
    let mut spatial_upsampler: Option<String> = None;
    let mut temporal_upsampler: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let dbl = |i: usize, what: &str| -> Result<f64, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => {
                prompt = Some(need(i)?.clone());
            }
            "--output-path" | "--out" => {
                out = Some(crate::args::strip_out_name_prefix(need(i)?, "video").to_string());
            }
            "--frames" => o.base.frames = num(i, "--frames")?,
            "--width" => o.base.width = num(i, "--width")?,
            "--height" => o.base.height = num(i, "--height")?,
            "--steps" => o.base.steps = num(i, "--steps")?,
            "--fps" => o.base.fps = num(i, "--fps")?,
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--base-shift" => o.base.base_shift = dbl(i, "--base-shift")?,
            "--max-shift" => o.base.max_shift = dbl(i, "--max-shift")?,
            "--terminal" => o.base.terminal = dbl(i, "--terminal")?,
            "--sampler" => o.base.sampler = Sampler::parse(need(i)?)?,
            "--eta" => eta = Some(dbl(i, "--eta")?),
            "--dit-config" => o.base.dit_config = need(i)?.clone(),
            "--adapter" => o.base.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.base.adapters, &mut scaled, need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?)?,
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
    o.base.eta = o.base.sampler.resolve_eta(eta)?;
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
    let written = imaging::video::encode_frames(&frames, std::path::Path::new(&out), video.fps as f64, &Default::default())?;
    eprintln!("ltxv dfr: wrote {} ({}x{}, {} frames at {} fps)", written.display(), video.width, video.height, frames.len(), video.fps);
    crate::run_record::write_run_record(
        &out,
        "ltxv dfr",
        o.base.seed,
        serde_json::json!({
            "prompt": prompt, "temporal_upsample_rounds": o.temporal_upsample_rounds,
            "settings": base_params(&o.base),
        }),
    )?;
    Ok(())
}

/// Whether `flag` is one of the guidance flags [`set_guidance`] parses.
fn guidance_flag(flag: &str) -> bool {
    flag == "--guidance-preset" || GUIDANCE_TERMS.contains(&flag.trim_start_matches("--").replace('-', "_").as_str())
}

/// The terms [`ltxv::guidance::Guidance::set`] names, as their flags spell them with `_`.
const GUIDANCE_TERMS: [&str; 9] = ["guidance", "audio_guidance", "stg", "audio_stg", "stg_blocks", "rescale", "audio_rescale", "modality_guidance", "audio_modality_guidance"];

/// Apply one guidance flag. `--guidance-preset` replaces the whole guidance,
/// so it must come before the flags that override its terms.
fn set_guidance(g: &mut ltxv::guidance::Guidance, flag: &str, value: &str) -> Result<(), String> {
    if flag == "--guidance-preset" {
        *g = ltxv::guidance::Guidance::preset(value)?;
        return Ok(());
    }
    g.set(&flag.trim_start_matches("--").replace('-', "_"), value).map(|_| ())
}

#[cfg(test)]
mod tests {
    use brain_modelstore::resolve::ArchSpec;

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
            "--sampler",
            "--dit-config",
            "--device",
            "--scene",
            "--start-frame",
            "--end-frame",
            "--mid-frame",
            "--mid-frame-at",
            "--conditioning-strength",
            "--context-frames",
        ] {
            assert!(super::HELP.contains(flag), "{flag} is parsed but not in --help");
            assert!(src.contains(&format!("\"{flag}\"")) || super::guidance_flag(flag), "{flag} is in --help but not parsed");
        }
        // `--vae`/`--dit`/`--text-encoder` are the model-store resolver's own
        // role-override flags now (`crate::resolver_cli::extract_role_
        // overrides`, matched against `LtxvSpec::roles()` - never a literal
        // match arm in this file), so they're checked against the resolver's
        // own role table instead of a quoted string in source.
        for role in ["vae", "dit", "text_encoder", "spatial_upsampler"] {
            let flag = format!("--{}", role.replace('_', "-"));
            assert!(super::HELP.contains(&flag), "{flag} is a resolver role flag but not in --help");
            assert!(ltxv::spec::LtxvSpec.roles().contains(&role), "{role} is documented as a role flag but LtxvSpec doesn't declare it");
        }
        // `BRAIN_LTXV_UPSAMPLER_SPATIAL` is the one `Paths` env var this
        // command still reads directly, as the fallback when the store holds
        // no spatial upscaler. The role flags are checked above via the
        // resolver's own role table instead: `Paths::from_env`'s env vars for
        // them are read only by the served path (`resident_ltxv.rs`).
        assert!(super::HELP.contains(ltxv::pipeline::OPTIONAL_PATH_VARS[2].0), "{} is read but not in --help", ltxv::pipeline::OPTIONAL_PATH_VARS[2].0);
    }

    /// Same self-check as [`every_flag_the_parser_accepts_is_documented`],
    /// scoped to `upscale`'s own flags and help text. `--upsampler-spatial`
    /// is a flag here rather than environment-only (as it is on `t2v`),
    /// because running that network is what this command is for.
    #[test]
    fn every_upscale_flag_the_parser_accepts_is_documented() {
        let src = include_str!("ltxv_cli.rs");
        for flag in ["--prompt", "--input", "--output-path", "--factor", "--refine-steps", "--guidance", "--seed", "--fps", "--dit-config", "--device", "--upsampler-spatial"] {
            assert!(super::UPSCALE_HELP.contains(flag), "{flag} is parsed but not in upscale --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in upscale --help but not parsed");
        }
        // See `every_flag_the_parser_accepts_is_documented`'s own comment:
        // these three are the resolver's role-override flags, not literal
        // match arms in this file.
        for role in ["vae", "dit", "text_encoder"] {
            let flag = format!("--{}", role.replace('_', "-"));
            assert!(super::UPSCALE_HELP.contains(&flag), "{flag} is a resolver role flag but not in upscale --help");
            assert!(ltxv::spec::LtxvSpec.roles().contains(&role), "{role} is documented as a role flag but LtxvSpec doesn't declare it");
        }
        // See `every_flag_the_parser_accepts_is_documented`'s own comment:
        // this is the one `Paths` env var this command still reads directly.
        assert!(super::UPSCALE_HELP.contains(ltxv::pipeline::OPTIONAL_PATH_VARS[2].0), "{} is read but not in upscale --help", ltxv::pipeline::OPTIONAL_PATH_VARS[2].0);
        // The one number in the help text that is a real constant rather than
        // prose: a ceiling the help promises and the pipeline enforces must
        // not drift apart.
        assert!(super::UPSCALE_HELP.contains(&ltxv::pipeline::REFINE_MAX_TOKENS.to_string()), "the refinement token ceiling in upscale --help is not REFINE_MAX_TOKENS");
    }

    /// Same self-check as [`every_flag_the_parser_accepts_is_documented`],
    /// scoped to `v2v`'s own flags and help text.
    #[test]
    fn every_v2v_flag_the_parser_accepts_is_documented() {
        let src = include_str!("ltxv_cli.rs");
        for flag in ["--prompt", "--input", "--mask", "--window", "--hold-video", "--audio", "--audio-source", "--hold-audio", "--output-path", "--strength", "--steps", "--guidance", "--seed", "--fps", "--dit-config", "--device"] {
            assert!(super::V2V_HELP.contains(flag), "{flag} is parsed but not in v2v --help");
            assert!(src.contains(&format!("\"{flag}\"")) || super::guidance_flag(flag), "{flag} is in v2v --help but not parsed");
        }
        // See `every_flag_the_parser_accepts_is_documented`'s own comment:
        // these three are the resolver's role-override flags, not literal
        // match arms in this file.
        for role in ["vae", "dit", "text_encoder"] {
            let flag = format!("--{}", role.replace('_', "-"));
            assert!(super::V2V_HELP.contains(&flag), "{flag} is a resolver role flag but not in v2v --help");
            assert!(ltxv::spec::LtxvSpec.roles().contains(&role), "{role} is documented as a role flag but LtxvSpec doesn't declare it");
        }
        // See `every_flag_the_parser_accepts_is_documented`'s own comment:
        // this is the one `Paths` env var this command still reads directly.
        assert!(super::V2V_HELP.contains(ltxv::pipeline::OPTIONAL_PATH_VARS[2].0), "{} is read but not in v2v --help", ltxv::pipeline::OPTIONAL_PATH_VARS[2].0);
        // The interchange format this command consumes is a contract with a
        // separate producer, so its version string is in the help verbatim
        // rather than paraphrased.
        assert!(super::V2V_HELP.contains(ltxv::maskcond::MASKSEQ_FORMAT), "the mask-sequence format v2v --help names is not MASKSEQ_FORMAT");
    }

    /// Every subcommand `run_ltxv` dispatches has to be listed in the help a
    /// bare `brain ltxv` prints, or it is reachable only by reading the
    /// source.
    #[test]
    fn every_subcommand_is_listed() {
        let src = include_str!("ltxv_cli.rs");
        for sub in ["t2v", "upscale", "v2v", "dfr"] {
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
            "--sampler",
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
