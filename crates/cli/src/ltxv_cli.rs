// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain ltxv {t2v,upscale,v2v,replace,dfr,t2a,extend}` - LTX-2.5 video
//! generation, post-hoc upscaling of a clip that already exists, masked
//! regeneration, character replacement (`crate::ltxv_replace`), the DFR smoke
//! test, text-to-audio, and the continuation of an existing clip.
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
//! from the model store - the only required role - like every other role). With the real checkpoint resolved the
//! DiT is the real 22B model and every unnamed setting takes the reference's
//! own default (`ltxv::pipeline::resolve_defaults`); without it, or with
//! `--dit-config tiny`, it is the tiny config with FRESH RANDOM WEIGHTS,
//! which makes every output a wiring proof and nothing else. The prompt likewise only folds into a deterministic
//! context stub unless a resolved `--text-encoder` names the real Gemma-4
//! encoder. dit/vae/audio_vae/text_encoder/tokenizer are resolved through
//! `brain_modelstore::resolve` (`ltxv::spec::LtxvSpec`) rather than
//! any environment variable - see `resolve_ltxv`. See `ltxv::pipeline`'s module doc for
//! the full account of what is real and what is not.

use ltxv::pipeline::{AudioOpts, DfrOpts, ExtendOpts, GenOpts, LongOpts, MaskedOpts, Named, Paths, Sampler, UpscaleOpts};
use ltxv::spec::LtxvSpec;

/// The guidance flags every generating subcommand takes ([`set_guidance`]).
macro_rules! guidance_help {
    () => {
        "  --guidance-preset <P>     none (one conditional forward per step - the
                            distilled checkpoint; the default), reference (the
                            reference pipeline's non-distilled guidance: CFG 3
                            video / 7 audio, STG 1 on block 28, rescale 0.7,
                            modality guidance 3) or hq (its high-quality
                            preset: no STG, rescale 0.45 video / 1.0 audio;
                            pair with --sampler res2s). The flags below
                            override it.
  --guidance <G>            classifier-free guidance on both streams (1 = off)
  --negative-prompt <text>  what CFG guides away from (default: the reference
                            pipelines' artifact list); read only when a CFG
                            scale is not 1
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

/// The execution flags every generating subcommand takes ([`set_runtime`]):
/// how a run is placed and computed, carried in `GenOpts::runtime`.
macro_rules! runtime_help {
    () => {
        "  --resident-blocks <N>     DiT blocks kept resident per card (default:
                            planned from free VRAM; 0 streams every block)
  --two-stage <on|off>      force the two-stage shape on or off (default:
                            decided from the size)
  --vae-tiling <on|off>     force the tiled VAE decode on or off (default:
                            tiled past the measured whole-decode ceiling)
  --text-precision <P>      fp32 or int8 text encoder arithmetic (default:
                            follows the checkpoint)
  --no-text-cache           neither read nor write the encoded-prompt cache
  --text-cache-max-bytes <N>  bytes the encoded-prompt cache is pruned to
  --latent-dump <path>      also write the latent being VAE-decoded to <path>
                            (ltxv_bench decode replays it)
  --av-fp32-reference       denoise an --audio run on the eager host-fp32
                            reference arm (an A/B, not a quality dial)
"
    };
}

const HELP: &str = concat!(r#"brain ltxv t2v - LTX-2.5 text to video

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

Sampling (with the real checkpoint resolved, unnamed settings take the
reference pipelines' defaults; without it, a tiny random-weight smoke test):
  --frames <N>              video frames, must be 1 + 8k (default: predicted
                            from the prompt by the duration head, 1-20 s,
                            when it is in the store; else 121, or 9 for the
                            smoke test). Any
                            length is accepted: past what one denoising
                            window holds the clip is generated as several
                            windows with a rolling latent context (see
                            Long-form clips)
  --width <W> --height <H>  pixels, multiples of 32 (default 768x512; 64x64
                            for the smoke test)
  --steps <N>                denoise steps (default 4)
"#, guidance_help!(), runtime_help!(), r#"  --seed <S>                 initial-noise/weight/context seed (default: drawn at random and printed)
  --fps <N>                  frame rate (default 24; 8 for the smoke test)
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
                             Needs the real checkpoint, a real
                             --text-encoder, and the audio VAE. Any
                             clip length; the sound crosses a window seam
                             with the picture. Costs real time and memory -
                             see "Audio" below
  --audio-source <wav>        generate the video FOR this soundtrack
                             (audio-to-video): it is encoded through the audio
                             VAE, held frozen as the clip's audio stream while
                             the video denoises against it, and muxed as the
                             clip's sound. Needs --audio; past one
                             denoising window each window holds its own
                             slice of it
  --dit-config <name>         DiT config; "ltx25_22b" (the real 22B
                             checkpoint - the default when it is resolved)
                             or "tiny" (fresh random weights, a smoke test -
                             the default without it)
  --start-frame <path>         PNG/JPEG still, encoded through the real VAE.
                             Given ALONE, it overwrites the clip's own
                             frame 0 outright (the generated video IS the
                             still there). Given alongside --keyframe/
                             --mid-frame/--end-frame, all of them switch to
                             the OTHER mechanism below - an appended guide,
                             not a replacement.
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
  --keyframe <F>:<path>        an interior keyframe, repeatable: the same
                             appended-guide mechanism anchored at pixel
                             frame F, strictly between 0 and --frames minus
                             one, so one pass is anchored at its start, at
                             every keyframe and at its end. Every token of
                             the clip still denoises freely, so the clip
                             moves THROUGH each still - reaching it at its
                             frame - rather than freezing on it or cutting
                             to it. F need not sit on the 1 + 8k boundary
                             the clip's LENGTH does: a guide carries its own
                             position, not a slot on the latent grid. Give
                             stills a plausible distance apart: the model
                             blends between them, so two very different
                             stills a few frames apart can only be a cut.
  --mid-frame <path>           one --keyframe at --mid-frame-at.
  --mid-frame-at <N>           which pixel frame --mid-frame anchors (the
                             default is the clip's own midpoint,
                             (frames - 1) / 2).
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
  request twice, once with --device gpu0, or run
  crates/ltxv/tests/it/cfg_parallel.rs, which times both placements and proves
  the two outputs identical. --device gpu0 confines the run to one card,
  since placement never leaves the schedulable set --device names.

Weights (resolved from the models directory; a flag names a candidate
outright, an ambiguous or missing outcome prints every real candidate and
exits - --brain-data-dir/BRAIN_MODELS_DIR names where to look):
  --vae <path>              the causal 3D video VAE (the only required role)
  --dit <path>              real 22B DiT GGUF (only read when --dit-config
                             ltx25_22b)
  --text-encoder <path>     real Gemma-4 text encoder (optional; the
                             deterministic prompt stub runs without it)
  --spatial-upsampler <path> spatial x2 latent upscaler - REQUIRED above
                             6144 video tokens, unused below (see Stages)
  --audio-vae <path>        the audio VAE + vocoder (read only with --audio)

Stages:
  The distilled checkpoint's fixed 8-sigma schedule is only distilled to
  build a clip from noise at a modest token count - upstream never runs it
  at the requested resolution, only at half of it, and then refines. Past
  6144 video tokens (a 25-frame clip above ~1600x896) one stage measurably
  disintegrates the END of the clip while the start stays correct, so a
  request past that ceiling runs the reference's own two stages: the full
  schedule at half resolution (stage 1, where motion is decided), a real
  x2 latent upscale, then three deterministic refinement steps at the
  requested size (stage 2, where detail is refined).  That needs the
  spatial x2 latent upscaler, and both --width and --height must be
  multiples of 64 (not just 32) so halving lands on the VAE's own stride.
  --two-stage on|off forces the choice either way.

Audio:
  LTX-2.5 is one model that denoises a video-latent stream and an audio-latent
  stream TOGETHER, coupled every block by cross-attention. --audio runs both;
  without it only the video half runs and the clip comes out silent. The sound
  is generated by the same forwards, from the same prompt, over the same time
  window as the frames - not synthesized separately and lined up afterwards.

  Needs the real checkpoint, a real --text-encoder (the audio stream is
  conditioned through its own text projection, which the stub context cannot
  stand in for) and the audio VAE
  (ltx-2.5-audio-vae-bf16.safetensors), which carries both the audio VAE decoder
  and the vocoder. All three are checked before any weight is read.

  It is opt-in because of COST, not correctness - but it runs on the same
  streamed/quantized/device-resident path the video-only stream does: the
  audio-extended block is quantized to int8, cached per checkpoint in host
  RAM, and as many blocks as the card's budget allows stay resident on it
  between steps. A step therefore costs modestly more than the same clip
  without sound rather than several times more. The command refuses up front,
  with both numbers, if this machine cannot hold what that arm needs.

  --av-fp32-reference denoises on the eager host-fp32 REFERENCE arm instead -
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
  --max-window-tokens <N> sets the per-window token ceiling (default 13200,
  the largest single-window generation this crate has a recorded real run
  at); --stage-major builds a video-only clip's motion stage-major instead
  of window by window (see below). --start-frame conditions the first window as
  usual; --end-frame conditions the windows that own their instants, and
  every window also receives the anchors still ahead of it as appended
  guides, so each window bends its motion toward the next keyframe instead
  of running on alone. Every --keyframe (and --mid-frame) guides the window
  that owns its instant, at that instant, the same way.

  Stage-major (--stage-major, two-stage, video-only): the clip is generated
  via two independent window plans: stage 1 at half resolution across the
  whole clip (where motion is decided) and stage 2 refining it window by
  window at full resolution. Off by default: each refinement window
  regenerates its slice seeing only its own part of the clip, and the
  slices do not stitch - measured, a refinement seam can cut to another
  scene. Audio-visual clips always use the window-major loop.

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
  --end-frame/--keyframe/--mid-frame are refused: a scene boundary is a cut
  that carries nothing across, so give a continuous shot with keyframes as
  one scene.

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
  --reference-attention-strength <s>
                             how strongly the reference and the clip attend
                             to each other, 0..1 (default 1.0; 0 cuts the
                             reference off)
  --reference-attention-mask <path>
                             a grayscale mask video setting that weight per
                             region and frame (white attends, black does not),
                             scaled by --reference-attention-strength
  --stage-2-ic-lora          keep the adapters and the reference on a two-stage
                             run's refinement too (default: stage 2 runs the
                             bare checkpoint, as the reference recipe does)

Devices:
  --device <cpu|gpu>         DiT + VAE (default: the ambient BRAIN_DEVICE)

Subcommands:
  brain ltxv t2v --help       text to video (this command)
  brain ltxv upscale --help   re-render a finished clip at 2x
  brain ltxv v2v --help       regenerate a masked region of an existing clip
  brain ltxv replace --help   replace a person in a clip with a character from a sheet
  brain ltxv dfr --help       DFR (Diffusion Fidelity Rendering) smoke test
  brain ltxv t2a --help       text to audio (the audio stream alone)
  brain ltxv extend --help    continue an existing clip"#);

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
"#, guidance_help!(), runtime_help!(), r#"  --seed <S>                 refinement noise seed (default: drawn at random and printed)
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
  --dit-config <name>        "ltx25_22b" (the real checkpoint - the default
                             when it is resolved) or "tiny" (fresh random
                             weights, a wiring smoke test - the default
                             without it)

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
  --upsampler-spatial <path> spatial x2 latent upscaler - REQUIRED, it is
                             what this command runs
  --dit <path>               real 22B DiT GGUF (only read when --dit-config
                             ltx25_22b)
  --text-encoder <path>      real Gemma-4 text encoder (without it --prompt
                             reaches the model only as a stub and cannot
                             guide anything)

  The audio VAE is NOT read here. Upscaling refines the VIDEO latent
  only: it runs no audio-visual DiT and decodes no audio. Note the consequence - the
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

const DFR_HELP: &str = concat!(r#"brain ltxv dfr - LTX-2.5 DFR (Diffusion Fidelity Rendering)

  brain ltxv dfr --prompt <text> --output-path <out.mp4> [options]

DFR lays the clip out on a keyframe-segment canvas, generates it at half
resolution together with a sharp keyframe at every segment border, then
carries both up with the spatial x2 upscaler and re-denoises at full
resolution under the Refine-Details IC-LoRA, the half-resolution video as
its reference. Runs the real LTX-2.5 checkpoint and text encoder.

Required:
  --prompt <text>            the clip's description
  --output-path <file>       .mp4 / .mkv / .webm / .gif

Sampling (--width/--height are the OUTPUT resolution; stage 1 halves it, so
both must be a multiple of 64):
  --frames <N>                video frames, 1 + 8k. Unset, the duration head
                              predicts the length from the prompt when it is
                              in the model store; otherwise 121
  --width <W> --height <H>    pixels (default 768x512)
  --fps <N>                   frame rate (default 24)
  --audio                     also generate the clip's sound
"#, guidance_help!(), runtime_help!(), r#"  --seed <S>                   initial-noise seed (default: drawn at random and printed)
  --sampler <euler|res2s>       denoising loop (default euler; see t2v)
  --eta <F>                     sampler noise strength (default 1.0 euler,
                                0.5 res2s)
  --temporal-upsample-rounds <N> 0, 1, or 2 temporal x2 rounds, each doubling
                                 the frame rate: N frames come back as
                                 (N - 1) * 2^rounds + 1 (default 0)
  --spatial-upscalings <N>       1 (default): stage 1 at half the output size;
                                 2: stage 1 at a quarter, the detailing pass
                                 and rounds at half, then a spatial epilogue
                                 at the output size (multiples of 128)

Weights (resolved from the models directory; a flag names one outright):
  --vae <path>                 the causal 3D video VAE
  --spatial-upsampler <path>   the spatial x2 latent upscaler
  --temporal-upsampler <path>  the temporal x2 latent upscaler
  --detailing-lora <path>      the Refine-Details IC-LoRA the detailing pass
                               applies at strength 0.5

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
        "t2a" => {
            if let Err(e) = t2a(&args[1..]) {
                eprintln!("ltxv t2a: {e}");
                std::process::exit(1);
            }
        }
        "replace" => {
            if let Err(e) = crate::ltxv_replace::replace(&args[1..]) {
                eprintln!("ltxv replace: {e}");
                std::process::exit(1);
            }
        }
        "extend" => {
            if let Err(e) = extend(&args[1..]) {
                eprintln!("ltxv extend: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("ltxv: unknown subcommand {other}\n\n{HELP}");
            std::process::exit(2);
        }
    }
}

const EXTEND_HELP: &str = concat!(r#"brain ltxv extend - continue an existing clip

  brain ltxv extend --input <clip.mp4> --prompt <text> --output-path <out.mp4> [options]

The clip's last --context-frames frames are held as a clean prefix while new
frames after them denoise from noise, the reference's forward-extension
recipe (video_extend_lora's prefix condition). The output is the input clip,
untouched, followed by the new frames.

Options:
  --input <clip>            the clip to continue (required)
  --prompt <text>           what the continuation shows (required)
  --output-path <file>      where to write the extended clip (required)
  --frames <N>              frames to add, rounded up to a multiple of 8
                            (default 64)
  --seconds <S>             ...or the time to add, at the clip's frame rate
  --context-frames <N>      how much of the clip's end the continuation sees
                            (default 57, the recipe's 8 latent frames)
  --audio                   extend the soundtrack too, jointly with the
                            picture (needs the same checkpoints as t2v
                            --audio, and --audio-source)
  --audio-source <wav>      the clip's own soundtrack
  --fps <N>                 the clip's frame rate (default: read from the file)
  --steps <N>               denoise steps (non-distilled checkpoints)
"#, guidance_help!(), runtime_help!(), r#"  --sampler <euler|res2s>   denoising loop (default euler; see t2v)
  --eta <F>                 sampler noise strength (default 1.0 euler, 0.5
                            res2s)
  --seed <S>                noise seed (default: drawn at random and printed)
  --dit-config <name>       ltx25_22b (default when the checkpoint is
                            resolved) or tiny (random-weight smoke test)
  --adapter <path>          a LoRA adapter (repeatable), e.g. an extension
                            LoRA trained with video_extend_lora
  --lora-scale <F>          strength of the preceding --adapter (default 1.0)
  --device <cpu|gpu>        DiT + VAE (default: BRAIN_DEVICE)"#);

const T2A_HELP: &str = concat!(r#"brain ltxv t2a - LTX-2.5 text to audio

  brain ltxv t2a --prompt <text> --output-path <out.wav> [options]

Generates sound alone: the checkpoint's audio stream denoises without a video
stream beside it, reading none of the video half's weights, then the audio
VAE and the vocoder render a 48 kHz stereo WAV. Needs the real checkpoint,
the real text encoder and the audio VAE (resolved like t2v --audio).

Options:
  --prompt <text>           what the clip should sound like (required)
  --output-path <out.wav>   where to write the clip (required)
  --seconds <S>             clip length (default: predicted from the prompt
                            by the duration head, 1-20 s; 5 s without one)
  --steps <N>               denoise steps for a non-distilled checkpoint; the
                            distilled one has a fixed schedule (default 4)
"#, guidance_help!(), runtime_help!(), r#"  --sampler <euler|res2s>   denoising loop (default euler; see t2v)
  --eta <F>                 sampler noise strength (default 1.0 euler, 0.5
                            res2s)
  --seed <S>                initial-noise seed (default: drawn at random and
                            printed)
  --adapter <path>          a LoRA adapter, applied beside the quantized
                            weights (repeatable)
  --lora-scale <F>          strength of the preceding --adapter (default 1.0)
  --device <cpu|gpu>        DiT + audio decode (default: BRAIN_DEVICE)"#);

const V2V_HELP: &str = concat!(r#"brain ltxv v2v - regenerate a masked region of an existing clip

Usage:
  brain ltxv v2v --input <clip.mp4> --mask <dir> --prompt <text> --output-path <out.mp4>

Every latent position the mask marks as CONDITIONING is handed the source
clip's own latent and excluded from denoising, so it leaves the sampler
bit-exactly unchanged; everything else is renoised and redrawn from the
prompt. That is LTX-2.5's own `VideoConditionByMask` (ported in
`ltxv::maskcond`), and it needs no adapter.

Keyframe guides compose with it: `--keyframe <frame>:<image>` appends an
edited still - a source frame with a new character painted in, say - as a
clean guiding block at that frame's instant. It pins nothing on the clip's
own grid; the regenerated region attends to it, which is how an identity
given in a few frames carries through motion the mask keeps.

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
  --keyframe <F>:<image>    a keyframe guide at pixel frame F (repeatable, any
                            order): the PNG/JPEG still at the clip's size,
                            encoded through the video VAE and appended as a
                            guiding block at F's instant
  --keyframe-strength <S>   how hard every keyframe guide is held, 0..1
                            (default 1.0)
  --steps <N>               denoise steps (default 4; IGNORED for
                            --dit-config ltx25_22b, whose distilled schedule
                            has a fixed shape)
"#, guidance_help!(), runtime_help!(), r#"  --seed <S>                noise seed (default: drawn at random and printed)
  --fps <N>                 frame rate for the output. Default: whatever
                            ffprobe reports for --input
  --dit-config <name>       "ltx25_22b" for the real checkpoint (the default
                            when it is resolved) or "tiny" (fresh random
                            weights - a wiring proof, and enough to verify
                            that the carried region really is carried; the
                            default without it)

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
                            The spatial upscaler and the audio VAE are not
                            read by this command - it never upscales and
                            never generates sound.

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
/// [`LtxvSpec`]) - `overrides` carries whatever
/// role flags [`crate::resolver_cli::extract_role_overrides`] already pulled
/// out of this command's own argv. `vae` is the only role `resolve_or_exit`
/// can fail to fill; every other role simply stays `None` in the returned
/// `Paths`.
pub(crate) fn resolve_ltxv(overrides: &std::collections::BTreeMap<String, String>, spatial_upsampler: Option<&str>) -> Result<Paths, String> {
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
    let mut reference_attention_strength = 1.0f32;
    let mut reference_attention_mask: Option<String> = None;
    let mut context_frames = ltxv::longform::CONTEXT_FRAMES;
    let mut prompt: Option<String> = None;
    let mut scene_specs: Vec<String> = Vec::new();
    // `--frames` has a default, so "was it given" cannot be read back off the
    // value - and refusing `--scene` alongside it depends on knowing.
    let mut frames_given = false;
    let (mut mid_frame, mut mid_frame_at) = (None, None);
    // The long-form plan's own settings ([`LongOpts`]).
    let mut max_window_tokens = ltxv::longform::LONGFORM_MAX_TOKENS;
    let mut stage_major = false;
    // Which settings the command line names; the rest come from what is
    // configured (`ltxv::pipeline::resolve_defaults`).
    let mut named = Named::default();
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
            "--width" => {
                o.width = num(i, "--width")?;
                named.size = true;
            }
            "--height" => {
                o.height = num(i, "--height")?;
                named.size = true;
            }
            "--steps" => o.steps = num(i, "--steps")?,
            "--fps" => {
                o.fps = num(i, "--fps")?;
                named.fps = true;
            }
            f if guidance_flag(f) => set_guidance(&mut o.guidance, f, need(i)?)?,
            "--negative-prompt" => o.negative_prompt = need(i)?.clone(),
            f if runtime_switch(f) => {
                set_runtime_switch(&mut o.runtime, f);
                i += 1;
                continue;
            }
            f if runtime_flag(f) => set_runtime(&mut o.runtime, f, need(i)?)?,
            "--max-window-tokens" => max_window_tokens = num(i, "--max-window-tokens")?.max(1),
            "--stage-major" => {
                stage_major = true;
                i += 1;
                continue;
            }
            "--base-shift" => o.base_shift = dbl(i, "--base-shift")?,
            "--max-shift" => o.max_shift = dbl(i, "--max-shift")?,
            "--terminal" => o.terminal = dbl(i, "--terminal")?,
            "--sampler" => o.sampler = Sampler::parse(need(i)?)?,
            "--eta" => eta = Some(dbl(i, "--eta")?),
            "--dit-config" => {
                o.dit_config = need(i)?.clone();
                named.dit_config = true;
            }
            "--adapter" => o.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.adapters, &mut scaled, flt(i, "--lora-scale")?)?,
            "--reference-video" => reference_path = Some(need(i)?.clone()),
            "--reference-strength" => reference_strength = flt(i, "--reference-strength")?,
            "--reference-attention-strength" => reference_attention_strength = flt(i, "--reference-attention-strength")?,
            "--reference-attention-mask" => reference_attention_mask = Some(need(i)?.clone()),
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
                mid_frame = Some(need(i)?.clone());
            }
            "--mid-frame-at" => mid_frame_at = Some(num(i, "--mid-frame-at")?),
            "--keyframe" => o.keyframes.push(ltxv::pipeline::parse_keyframe(need(i)?).map_err(|e| format!("--keyframe: {e}"))?),
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
    let out = out.ok_or("--output-path is required")?;
    let paths = resolve_ltxv(&overrides, None)?;
    ltxv::pipeline::resolve_defaults(&mut o, &paths, Named { frames: frames_given || !scene_specs.is_empty(), ..named });
    if let Some(path) = &reference_path {
        if !(0.0..=1.0).contains(&reference_strength) {
            return Err(format!("--reference-strength {reference_strength} is outside [0, 1]"));
        }
        if !(0.0..=1.0).contains(&reference_attention_strength) {
            return Err(format!("--reference-attention-strength {reference_attention_strength} is outside [0, 1]"));
        }
        let attention_mask = reference_attention_mask.as_deref().map(|m| read_clip(m, o.fps)).transpose()?;
        o.reference = Some(ltxv::pipeline::VideoReference { attention_strength: reference_attention_strength, attention_mask, ..ltxv::pipeline::VideoReference::new(read_clip(path, o.fps)?, reference_strength) });
    } else if reference_attention_mask.is_some() || reference_attention_strength != 1.0 {
        return Err("--reference-attention-strength and --reference-attention-mask weight a --reference-video, and there is none".into());
    }
    // One scene is what a `--prompt`/`--frames` run always was, so the whole
    // command below has exactly one shape to handle. Without `--frames` its
    // length is the duration head's prediction when the store has one.
    let scenes: Vec<ltxv::longform::Scene> = if scene_specs.is_empty() {
        let prompt = prompt.ok_or("--prompt is required (or one --scene <frames>:<prompt> per scene)")?;
        if !frames_given && ltxv::pipeline::can_predict_frames(&paths, &o) {
            o.frames = ltxv::pipeline::predict_frames(&paths, &prompt, &o, ltxv::pipeline::prediction_streams())?;
            eprintln!("ltxv: the duration head predicts {} frames ({:.2}s at {} fps) for this prompt", o.frames, o.frames as f64 / o.fps as f64, o.fps);
        }
        vec![ltxv::longform::Scene { frames: o.frames, prompt }]
    } else {
        if prompt.is_some() || frames_given {
            return Err("--scene already carries its own frame count and prompt, so it cannot be combined with --prompt/--frames - write every scene as its own --scene <frames>:<prompt>".into());
        }
        scene_specs.iter().map(|s| ltxv::longform::Scene::parse(s)).collect::<Result<_, _>>()?
    };
    o.frames = scenes.iter().map(|s| s.frames).sum();
    // `--mid-frame` is one more interior keyframe, at the instant
    // `--mid-frame-at` names or the clip's own midpoint - resolved now that
    // the clip's length is known.
    if let Some(path) = mid_frame {
        o.keyframes.push((ltxv::pipeline::mid_anchor_frame(o.frames, mid_frame_at)?, path));
    }
    o.keyframes.sort_by_key(|(at, _)| *at);
    ltxv::pipeline::check_keyframes(o.frames, &o.keyframes)?;

    let (lh, lw) = (o.height / 32, o.width / 32);
    let context_latents = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        vcfg.latent_frames(context_frames as u32).ok_or_else(|| format!("--context-frames {context_frames} is not of the form 1 + 8k"))? as usize
    };
    let long = LongOpts { context_latent_frames: context_latents, max_window_tokens, max_refine_tokens: ltxv::pipeline::REFINE_MAX_TOKENS, stage_major, base: o.clone() };
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
    let dit_desc = ltxv::pipeline::av_arm_label(o.dit_config != "tiny", o.audio, o.runtime.av_fp32_reference);
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
    let stage_desc = if ltxv::pipeline::should_two_stage(tokens, o.width, o.height, o.dit_config == "ltx25_22b", o.runtime.two_stage) {
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
    // The resolved pixel frames, not the flags: an unset --mid-frame-at is
    // the clip's own midpoint, and a caller should be told which frame that
    // is before the run rather than have to derive it.
    let mid_desc = if o.keyframes.is_empty() {
        String::new()
    } else {
        format!(", keyframes at frames {:?}", o.keyframes.iter().map(|(at, _)| *at).collect::<Vec<_>>())
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

    eprintln!("ltxv: wrote {}", write_clip(video, &out)?);
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

fn extend(args: &[String]) -> Result<(), String> {
    let (overrides, args) = crate::resolver_cli::extract_role_overrides(&LtxvSpec, args);
    let (seed, args) = crate::args::split_seed(&args, "ltxv extend");
    let args = &args[..];
    let mut o = ExtendOpts::default();
    o.base.seed = seed;
    let mut eta: Option<f64> = None;
    let mut scaled = 0usize;
    let mut seconds: Option<f64> = None;
    let mut fps: Option<usize> = None;
    let (mut prompt, mut input, mut out) = (None::<String>, None::<String>, None::<String>);
    // Whether --dit-config was given; unnamed, the real checkpoint runs when
    // it is configured (`ltxv::pipeline::resolve_defaults`).
    let mut dit_named = false;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let dbl = |i: usize, what: &str| -> Result<f64, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--input" | "--in" => input = Some(need(i)?.clone()),
            "--output-path" | "--out" => out = Some(crate::args::strip_out_name_prefix(need(i)?, "video").to_string()),
            "--frames" => o.frames = num(i, "--frames")?,
            "--seconds" => seconds = Some(dbl(i, "--seconds")?),
            "--context-frames" => o.context_frames = num(i, "--context-frames")?,
            "--audio" => {
                o.base.audio = true;
                i += 1;
                continue;
            }
            "--audio-source" => o.base.audio_source = Some(ltxv::audio::read_wav(need(i)?)?),
            "--fps" => fps = Some(num(i, "--fps")?),
            "--steps" => o.base.steps = num(i, "--steps")?,
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--negative-prompt" => o.base.negative_prompt = need(i)?.clone(),
            f if runtime_switch(f) => {
                set_runtime_switch(&mut o.base.runtime, f);
                i += 1;
                continue;
            }
            f if runtime_flag(f) => set_runtime(&mut o.base.runtime, f, need(i)?)?,
            "--sampler" => o.base.sampler = Sampler::parse(need(i)?)?,
            "--eta" => eta = Some(dbl(i, "--eta")?),
            "--dit-config" => {
                o.base.dit_config = need(i)?.clone();
                dit_named = true;
            }
            "--adapter" => o.base.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.base.adapters, &mut scaled, need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?)?,
            "--device" => o.base.device = Some(need(i)?.clone()),
            "--help" | "-h" => {
                println!("{EXTEND_HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{EXTEND_HELP}")),
        }
        i += 2;
    }
    o.base.eta = o.base.sampler.resolve_eta(eta)?;
    let input = input.ok_or("--input is required")?;
    let out = out.ok_or("--output-path is required")?;
    let prompt = prompt.ok_or("--prompt is required: the continuation is generated against it")?;
    let paths = resolve_ltxv(&overrides, None)?;
    ltxv::pipeline::resolve_defaults(&mut o.base, &paths, Named { dit_config: dit_named, frames: true, size: true, fps: true });
    let decoded = read_clip(&input, 0)?;
    let fps = source_fps(&input, fps)?;
    o.base.fps = fps;
    if let Some(s) = seconds {
        if !(s.is_finite() && s > 0.0) {
            return Err(format!("--seconds must be a positive number, got {s}"));
        }
        o.frames = (s * fps as f64).round().max(1.0) as usize;
    }
    let clip = ltxv::pipeline::Video { fps, ..decoded };
    eprintln!("ltxv extend: {} frames at {fps} fps, adding {} (in whole latent frames) against the last {} - guidance {:?}, seed {}", clip.frames.len(), o.frames, o.context_frames.min(clip.frames.len()), o.base.guidance, o.base.seed);
    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it - see `t2v`.
    let cancel = capability::CancelToken::default();
    let (video, timings) = ltxv::pipeline::extend(&paths, &prompt, &clip, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv extend [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    let wall = t0.elapsed().as_secs_f32();
    eprintln!("ltxv extend: {wall:.1}s total  (build {:.2}s, text encode {:.1}s, denoise {:.1}s, vae {:.1}s, other {:.1}s)", timings.build_dit, timings.text_encode, timings.denoise, timings.decode, timings.unattributed(wall));
    eprintln!("ltxv extend: wrote {}", write_clip(video, &out)?);
    crate::run_record::write_run_record(&out, "ltxv extend", o.base.seed, serde_json::json!({ "prompt": prompt, "input": input, "frames": o.frames, "context_frames": o.context_frames, "settings": base_params(&o.base) }))?;
    Ok(())
}

fn t2a(args: &[String]) -> Result<(), String> {
    let (overrides, args) = crate::resolver_cli::extract_role_overrides(&LtxvSpec, args);
    let (seed, args) = crate::args::split_seed(&args, "ltxv t2a");
    let args = &args[..];
    let mut o = AudioOpts::default();
    o.base.seed = seed;
    let mut eta: Option<f64> = None;
    let mut scaled = 0usize;
    let mut prompt: Option<String> = None;
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let dbl = |i: usize, what: &str| -> Result<f64, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--output-path" | "--out" => out = Some(crate::args::strip_out_name_prefix(need(i)?, "audio").to_string()),
            "--seconds" => o.seconds = Some(dbl(i, "--seconds")?),
            "--steps" => o.base.steps = need(i)?.parse().map_err(|e| format!("--steps: {e}"))?,
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--negative-prompt" => o.base.negative_prompt = need(i)?.clone(),
            f if runtime_switch(f) => {
                set_runtime_switch(&mut o.base.runtime, f);
                i += 1;
                continue;
            }
            f if runtime_flag(f) => set_runtime(&mut o.base.runtime, f, need(i)?)?,
            "--sampler" => o.base.sampler = Sampler::parse(need(i)?)?,
            "--eta" => eta = Some(dbl(i, "--eta")?),
            "--adapter" => o.base.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.base.adapters, &mut scaled, need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?)?,
            "--device" => o.base.device = Some(need(i)?.clone()),
            "--help" | "-h" => {
                println!("{T2A_HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{T2A_HELP}")),
        }
        i += 2;
    }
    o.base.eta = o.base.sampler.resolve_eta(eta)?;
    let prompt = prompt.ok_or("--prompt is required")?;
    let out = out.ok_or("--output-path is required")?;
    let paths = resolve_ltxv(&overrides, None)?;
    let length = match o.seconds {
        Some(s) => format!("{s:.2}s of audio"),
        None if ltxv::pipeline::can_predict_frames(&paths, &o.base) => "audio of the length the duration head predicts".to_string(),
        None => format!("{:.2}s of audio (no duration head to predict the length)", AudioOpts::DEFAULT_SECONDS),
    };
    eprintln!("ltxv t2a: {length}, sampler {:?} eta {}, guidance {:?}, seed {}", o.base.sampler, o.base.eta, o.base.guidance.audio, o.base.seed);
    let t0 = std::time::Instant::now();
    // A one-shot CLI run has no second party to cancel it - see `t2v`.
    let cancel = capability::CancelToken::default();
    let (clip, timings) = ltxv::pipeline::generate_audio(&paths, &prompt, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv t2a [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    let wall = t0.elapsed().as_secs_f32();
    eprintln!(
        "ltxv t2a: {wall:.1}s total  (build {:.2}s, text encode {:.1}s, denoise {:.1}s, audio decode {:.1}s, other {:.1}s)",
        timings.build_dit,
        timings.text_encode,
        timings.denoise,
        timings.audio_decode,
        timings.unattributed(wall)
    );
    let planes: Vec<&[f32]> = clip.channels.iter().map(Vec::as_slice).collect();
    std::fs::write(&out, audio::wav::encode_multi(&planes, clip.sample_rate)).map_err(|e| format!("writing {out}: {e}"))?;
    eprintln!("ltxv t2a: wrote {out} ({:.2}s of {} Hz audio in {} channels)", clip.seconds(), clip.sample_rate, clip.channels.len());
    crate::run_record::write_run_record(&out, "ltxv t2a", o.base.seed, serde_json::json!({ "prompt": prompt, "seconds": clip.seconds(), "steps": o.base.steps, "sampler": format!("{:?}", o.base.sampler), "eta": o.base.eta, "guidance": format!("{:?}", o.base.guidance.audio) }))?;
    Ok(())
}

/// Write `video` - its frames and, when it has one, its sound track - to
/// `out`, and say what was written.
///
/// The frames are MOVED rather than cloned: a long clip's pixels are the
/// largest buffer a command ever holds, and copying all of them to hand them
/// to the encoder is a copy nothing reads twice.
pub(crate) fn write_clip(video: ltxv::pipeline::Video, out: &str) -> Result<String, String> {
    let audio_desc = video
        .audio
        .as_ref()
        .map(|a| format!(", {:.2}s of {} Hz audio in {} channels", a.seconds(), a.sample_rate, a.channels.len()))
        .unwrap_or_else(|| ", no sound".to_string());
    let track = video.audio.map(|a| imaging::video::AudioTrack { channels: a.channels, sample_rate: a.sample_rate });
    let (width, height, fps) = (video.width, video.height, video.fps);
    let frames: Vec<imaging::Rgb8> = video.frames.into_iter().map(|px| imaging::Rgb8::new(width, height, px)).collect::<Result<_, _>>()?;
    let enc_opts = imaging::video::VideoEncodeOpts { audio: track, ..Default::default() };
    let written = imaging::video::encode_frames(&frames, std::path::Path::new(out), fps as f64, &enc_opts)?;
    Ok(format!("{} ({width}x{height}, {} frames at {fps} fps{audio_desc})", written.display(), frames.len()))
}

/// The rate a clip read from `path` plays at: `given` (`--fps`), else the
/// container's own, so an edit never changes playback speed. A container
/// whose rate cannot be read is an error naming the flag, never a default
/// that would play the result at the wrong speed.
pub(crate) fn source_fps(path: &str, given: Option<usize>) -> Result<usize, String> {
    match given {
        Some(0) => Err("--fps 0 is not a frame rate".into()),
        Some(f) => Ok(f),
        None => imaging::video::probe_fps(std::path::Path::new(path))
            .map(|f| f.round() as usize)
            .filter(|&f| f > 0)
            .ok_or_else(|| format!("{path}: its frame rate cannot be read from the container; give it with --fps")),
    }
}

/// Decode a whole video file into the pipeline's RGB8 clip, refusing one
/// whose frames change size.
pub(crate) fn read_clip(path: &str, fps: usize) -> Result<ltxv::pipeline::Video, String> {
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
    // Whether --dit-config was given; unnamed, the real checkpoint runs when
    // it is configured (`ltxv::pipeline::resolve_defaults`).
    let mut dit_named = false;
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
            "--negative-prompt" => o.base.negative_prompt = need(i)?.clone(),
            f if runtime_switch(f) => {
                set_runtime_switch(&mut o.base.runtime, f);
                i += 1;
                continue;
            }
            f if runtime_flag(f) => set_runtime(&mut o.base.runtime, f, need(i)?)?,
            "--fps" => fps = Some(num(i, "--fps")?),
            "--dit-config" => {
                o.base.dit_config = need(i)?.clone();
                dit_named = true;
            }
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
    ltxv::pipeline::resolve_defaults(&mut o.base, &paths, Named { dit_config: dit_named, frames: true, size: true, fps: true });
    let prompt = prompt.unwrap_or_default();
    o.context_latent_frames = {
        let vcfg = ltxv::LtxVaeConfig::conv25();
        vcfg.latent_frames(context_frames as u32).ok_or_else(|| format!("--context-frames {context_frames} is not of the form 1 + 8k"))? as usize
    };

    let decoded = read_clip(&input, 0)?;
    let (w, h) = (decoded.width, decoded.height);
    let fps = source_fps(&input, fps)?;
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
    // Whether --dit-config was given; unnamed, the real checkpoint runs when
    // it is configured (`ltxv::pipeline::resolve_defaults`).
    let mut dit_named = false;
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
            "--keyframe" => o.base.future_anchors.push(ltxv::pipeline::parse_keyframe(need(i)?).map_err(|e| format!("--keyframe: {e}"))?),
            "--keyframe-strength" => o.base.conditioning_strength = flt(i, "--keyframe-strength")?,
            "--steps" => o.base.steps = num(i, "--steps")?,
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--negative-prompt" => o.base.negative_prompt = need(i)?.clone(),
            f if runtime_switch(f) => {
                set_runtime_switch(&mut o.base.runtime, f);
                i += 1;
                continue;
            }
            f if runtime_flag(f) => set_runtime(&mut o.base.runtime, f, need(i)?)?,
            "--fps" => fps = Some(num(i, "--fps")?),
            "--dit-config" => {
                o.base.dit_config = need(i)?.clone();
                dit_named = true;
            }
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
    // The guides are appended in timeline order; the order they were typed in carries nothing.
    o.base.future_anchors.sort_by_key(|(at, _)| *at);
    let input = input.ok_or("--input is required")?;
    let out = out.ok_or("--output-path is required")?;
    let prompt = prompt.ok_or("--prompt is required: the model denoises the regenerated region against the text context, and an empty one costs exactly the detail this command exists to produce")?;
    if o.mask_dir.is_empty() && o.window.is_none() {
        return Err("--mask or --window is required: without one this command has nothing to regenerate and would return the input clip".into());
    }
    let paths = resolve_ltxv(&overrides, None)?;
    ltxv::pipeline::resolve_defaults(&mut o.base, &paths, Named { dit_config: dit_named, frames: true, size: true, fps: true });

    let decoded = read_clip(&input, 0)?;
    let (w, h) = (decoded.width, decoded.height);
    let fps = source_fps(&input, fps)?;
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

    eprintln!("ltxv v2v: wrote {}", write_clip(video, &out)?);
    crate::run_record::write_run_record(
        &out,
        "ltxv v2v",
        o.base.seed,
        serde_json::json!({
            "prompt": prompt, "input": input, "mask": o.mask_dir, "strength": o.strength,
            "keyframes": o.base.future_anchors.iter().map(|(at, path)| serde_json::json!({"frame": at, "image": path})).collect::<Vec<_>>(),
            "keyframe_strength": o.base.conditioning_strength,
            "settings": base_params(&o.base),
        }),
    )?;
    Ok(())
}

fn dfr(args: &[String]) -> Result<(), String> {
    let (seed, rest) = crate::args::split_seed(args, "ltxv dfr");
    let args = &rest[..];
    let mut o = DfrOpts::default();
    o.base.dit_config = "ltx25_22b".into();
    let mut eta: Option<f64> = None;
    // How many `--adapter`s already carry their `--lora-scale`.
    let mut scaled = 0usize;
    o.base.seed = seed;
    let mut named = Named { dit_config: true, ..Named::default() };
    let mut prompt: Option<String> = None;
    let mut out: Option<String> = None;
    let mut overrides = std::collections::BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> { args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i])) };
        let num = |i: usize, what: &str| -> Result<usize, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        let dbl = |i: usize, what: &str| -> Result<f64, String> { need(i)?.parse().map_err(|e| format!("{what}: {e}")) };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--output-path" | "--out" => out = Some(crate::args::strip_out_name_prefix(need(i)?, "video").to_string()),
            "--frames" => {
                o.base.frames = num(i, "--frames")?;
                named.frames = true;
            }
            "--width" => {
                o.base.width = num(i, "--width")?;
                named.size = true;
            }
            "--height" => {
                o.base.height = num(i, "--height")?;
                named.size = true;
            }
            "--fps" => {
                o.base.fps = num(i, "--fps")?;
                named.fps = true;
            }
            "--audio" => {
                o.base.audio = true;
                i += 1;
                continue;
            }
            f if guidance_flag(f) => set_guidance(&mut o.base.guidance, f, need(i)?)?,
            "--negative-prompt" => o.base.negative_prompt = need(i)?.clone(),
            f if runtime_switch(f) => {
                set_runtime_switch(&mut o.base.runtime, f);
                i += 1;
                continue;
            }
            f if runtime_flag(f) => set_runtime(&mut o.base.runtime, f, need(i)?)?,
            "--sampler" => o.base.sampler = Sampler::parse(need(i)?)?,
            "--eta" => eta = Some(dbl(i, "--eta")?),
            "--adapter" => o.base.adapters.push(model::adapter::AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => crate::args::attach_lora_scale(&mut o.base.adapters, &mut scaled, need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?)?,
            "--temporal-upsample-rounds" => o.temporal_upsample_rounds = num(i, "--temporal-upsample-rounds")?,
            "--spatial-upscalings" => o.spatial_upscalings = num(i, "--spatial-upscalings")?,
            "--device" => o.base.device = Some(need(i)?.clone()),
            f @ ("--vae" | "--spatial-upsampler" | "--temporal-upsampler" | "--detailing-lora") => {
                overrides.insert(f.trim_start_matches("--").replace('-', "_"), need(i)?.clone());
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
    let paths = resolve_ltxv(&overrides, None)?;
    ltxv::pipeline::resolve_defaults(&mut o.base, &paths, named);
    if !named.frames && ltxv::pipeline::can_predict_frames(&paths, &o.base) {
        o.base.frames = ltxv::pipeline::predict_frames(&paths, &prompt, &o.base, ltxv::pipeline::prediction_streams())?;
        eprintln!("ltxv dfr: the duration head predicts {} frames ({:.2}s at {} fps) for this prompt", o.base.frames, o.base.frames as f64 / o.base.fps as f64, o.base.fps);
    }
    eprintln!(
        "ltxv dfr: {} frames, {}x{}, {} temporal round(s), seed {}",
        o.base.frames, o.base.width, o.base.height, o.temporal_upsample_rounds, o.base.seed
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

/// Whether `flag` is one of the valued execution flags [`set_runtime`] parses.
fn runtime_flag(flag: &str) -> bool {
    matches!(flag, "--resident-blocks" | "--two-stage" | "--vae-tiling" | "--text-precision" | "--text-cache-max-bytes" | "--latent-dump")
}

/// Whether `flag` is one of the bare execution switches [`set_runtime_switch`] applies.
fn runtime_switch(flag: &str) -> bool {
    matches!(flag, "--no-text-cache" | "--av-fp32-reference")
}

/// Apply one valued execution flag to `rt`.
fn set_runtime(rt: &mut ltxv::pipeline::Runtime, flag: &str, value: &str) -> Result<(), String> {
    let on_off = |v: &str| match v {
        "on" | "1" | "true" => Ok(true),
        "off" | "0" | "false" => Ok(false),
        other => Err(format!("{flag}: '{other}' is not on or off")),
    };
    let count = |v: &str| v.parse::<u64>().map_err(|e| format!("{flag} '{v}': {e}"));
    match flag {
        "--resident-blocks" => rt.resident_blocks = Some(u32::try_from(count(value)?).map_err(|e| format!("{flag}: {e}"))?),
        "--two-stage" => rt.two_stage = Some(on_off(value)?),
        "--vae-tiling" => rt.vae_tiling = Some(on_off(value)?),
        "--text-precision" => {
            rt.text_precision = Some(match value {
                "fp32" | "f32" => ltxv::pipeline::TextPrecision::Fp32,
                "int8" | "i8" => ltxv::pipeline::TextPrecision::Int8,
                other => return Err(format!("{flag}: '{other}' is not fp32 or int8")),
            })
        }
        "--text-cache-max-bytes" => rt.text_cache.max_bytes = count(value)?.max(1),
        "--latent-dump" => rt.latent_dump = Some(value.to_string()),
        other => unreachable!("{other} is not a runtime flag"),
    }
    Ok(())
}

/// Apply one bare execution switch to `rt`.
fn set_runtime_switch(rt: &mut ltxv::pipeline::Runtime, flag: &str) {
    match flag {
        "--no-text-cache" => rt.text_cache.enabled = false,
        "--av-fp32-reference" => rt.av_fp32_reference = true,
        other => unreachable!("{other} is not a runtime switch"),
    }
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
            "--keyframe",
            "--conditioning-strength",
            "--context-frames",
            "--reference-video",
            "--reference-strength",
            "--reference-attention-strength",
            "--reference-attention-mask",
        ] {
            assert!(super::HELP.contains(flag), "{flag} is parsed but not in --help");
            assert!(src.contains(&format!("\"{flag}\"")) || super::guidance_flag(flag), "{flag} is in --help but not parsed");
        }
        // `--vae`/`--dit`/`--text-encoder` are the model-store resolver's own
        // role-override flags now (`crate::resolver_cli::extract_role_
        // overrides`, matched against `LtxvSpec::roles()` - never a literal
        // match arm in this file), so they're checked against the resolver's
        // own role table instead of a quoted string in source.
        for role in ["vae", "dit", "text_encoder", "spatial_upsampler", "audio_vae"] {
            let flag = format!("--{}", role.replace('_', "-"));
            assert!(super::HELP.contains(&flag), "{flag} is a resolver role flag but not in --help");
            assert!(ltxv::spec::LtxvSpec.roles().contains(&role), "{role} is documented as a role flag but LtxvSpec doesn't declare it");
        }
    }

    /// Same self-check as [`every_flag_the_parser_accepts_is_documented`],
    /// scoped to `upscale`'s own flags and help text. `--upsampler-spatial`
    /// is a flag of its own here because running that network is what this
    /// command is for.
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
        for flag in ["--prompt", "--input", "--mask", "--window", "--hold-video", "--audio", "--audio-source", "--hold-audio", "--output-path", "--strength", "--keyframe", "--keyframe-strength", "--steps", "--guidance", "--seed", "--fps", "--dit-config", "--device"] {
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
        for sub in ["t2v", "upscale", "v2v", "replace", "dfr", "t2a", "extend"] {
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
            "--fps",
            "--audio",
            "--guidance",
            "--negative-prompt",
            "--seed",
            "--eta",
            "--sampler",
            "--adapter",
            "--lora-scale",
            "--temporal-upsample-rounds",
            "--spatial-upscalings",
            "--device",
            "--vae",
            "--spatial-upsampler",
            "--temporal-upsampler",
            "--detailing-lora",
        ] {
            assert!(super::DFR_HELP.contains(flag), "{flag} is parsed but not in dfr --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in dfr --help but not parsed");
        }
    }

    /// The same self-check for `extend`.
    #[test]
    fn every_extend_flag_the_parser_accepts_is_documented() {
        let src = include_str!("ltxv_cli.rs");
        for flag in ["--input", "--prompt", "--output-path", "--frames", "--seconds", "--context-frames", "--audio", "--audio-source", "--fps", "--steps", "--guidance", "--sampler", "--eta", "--seed", "--dit-config", "--adapter", "--lora-scale", "--device"] {
            assert!(super::EXTEND_HELP.contains(flag), "{flag} is parsed but not in extend --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in extend --help but not parsed");
        }
    }

    /// The execution flags every generating subcommand parses through one
    /// helper are documented in every one of their helps.
    #[test]
    fn every_runtime_flag_is_documented_on_every_command() {
        let src = include_str!("ltxv_cli.rs");
        for flag in ["--resident-blocks", "--two-stage", "--vae-tiling", "--text-precision", "--text-cache-max-bytes", "--latent-dump", "--no-text-cache", "--av-fp32-reference"] {
            assert!(super::runtime_flag(flag) || super::runtime_switch(flag), "{flag} is documented but not parsed");
            assert!(src.contains(&format!("\"{flag}\"")));
            for (name, help) in [("t2v", super::HELP), ("upscale", super::UPSCALE_HELP), ("v2v", super::V2V_HELP), ("dfr", super::DFR_HELP), ("t2a", super::T2A_HELP), ("extend", super::EXTEND_HELP)] {
                assert!(help.contains(flag), "{flag} is parsed but not in {name} --help");
            }
        }
        assert!(super::HELP.contains("--max-window-tokens") && super::HELP.contains("--stage-major"));
    }

    /// The same self-check for `t2a`.
    #[test]
    fn every_t2a_flag_the_parser_accepts_is_documented() {
        let src = include_str!("ltxv_cli.rs");
        for flag in ["--prompt", "--output-path", "--seconds", "--steps", "--guidance", "--sampler", "--eta", "--seed", "--adapter", "--lora-scale", "--device"] {
            assert!(super::T2A_HELP.contains(flag), "{flag} is parsed but not in t2a --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in t2a --help but not parsed");
        }
    }
}
