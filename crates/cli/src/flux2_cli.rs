// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flux2 …` - FLUX.2 Klein text-to-image + image editing.
//!
//! `generate`'s weights come from [`resolve_flux2`] (the model-store
//! resolver over the models directory - see [`crate::model_dir::resolve`]),
//! never `BRAIN_FLUX2_*`; `finetune` still reads those variables directly.
//! Images in/out are binary PPM P6 (the CLI-wide convention).

use crate::args::strip_out_name_prefix;
use flux2::{AdapterSpec, Flux2Config, GenOpts, Paths, Pipeline};

const HELP: &str = "brain flux2 <cmd>
  generate --prompt <text> --out <out.ppm> [--width W] [--height H]
           [--steps N] [--seed S] [--guidance G] [--variant klein-4b|klein-9b|base-4b|base-9b]
                                    # on the distilled klein variants the sampler is FIXED (4
                                    # steps, guidance 1.0, no CFG - BFL ships both as fixed
                                    # params): --steps is ignored there, --guidance is a no-op
                                    # and warned about; --experimental-steps honours --steps
                                    # again for experiments that accept that
           [--precision fp32|int8]  # int8 = DP4A DiT (~4x smaller, GPU only);
                                    # .gguf defaults to int8 and rejects explicit fp32
           [--strength S]           # brain extension: img2img anchoring dial, 0..1, on the
                                    # first --ref (which untiled must be at the output size;
                                    # under --tile-size it is resampled to it).
                                    # 1.0 = free generation conditioned on the reference;
                                    # lower anchors progressively more of the source; 0 IS
                                    # the source (exact codec round trip, no denoise step).
                                    # The schedule is COMPRESSED into [0,S] - upstream
                                    # diffusers slices its timestep list instead, which a
                                    # distilled few-step sampler cannot survive - so the
                                    # same sampler runs at every value and 0.99 is a hair
                                    # from 1.0. The reference conditions at EVERY strength.
                                    # Under --tile-size with NO --ref it is the same dial on
                                    # the internally drafted anchor - how much of each window
                                    # starts from the draft - and defaults to 0.4 there.
           [--ref-resolution-scale S]
                                    # linear size of the conditioning copy of the FIRST --ref,
                                    # 0..1, the same at every strength (default 1.0 = the
                                    # reference's own size). 0 = do not condition on it at all
                                    # (cheapest: the reference then reaches the model only
                                    # through the init latent). Lower it to buy tokens back;
                                    # the conditioning sequence never changes with --strength.
                                    # REFUSED below 1.0 under a tiling that really windows: the
                                    # first reference is then the anchor and has to stay at the
                                    # canvas's own token grid to be one.
           [--ref <in.ppm>]...      # reference images => editing mode. Under --tile-size the
                                    # FIRST one is the run's anchor and is resampled to the
                                    # canvas (never bounded by --ref-size); the rest stay
                                    # generic guidance and every window sees all of them.
           [--ref-size N]           # long edge each --ref is encoded at, before the /16
                                    # crop, preserving aspect. Never upscales. DEFAULT 512:
                                    # a reference costs (w/16)*(h/16) tokens and attention is
                                    # quadratic, so an unscaled camera photo would cost more
                                    # than the generation itself. Pass 0 for no bound. The
                                    # --strength/--mask init reference is never bounded -- its
                                    # size is pinned by that role; use --ref-resolution-scale for it.
           [--tile-size N]          # denoise the canvas in overlapping NxN-pixel windows
                                    # (MultiDiffusion) instead of in one forward, so the DiT's
                                    # activation memory follows the WINDOW and not the canvas -
                                    # which is what makes a resolution past what fits in one
                                    # pass reachable at all. Multiple of 16. OFF by default,
                                    # and 0 also means off: a canvas that fits is cheaper in
                                    # one forward (every window repeats the text conditioning),
                                    # so this is a knob a caller who is buying resolution with
                                    # time turns on. Turning it on below the budget is free and
                                    # changes nothing: the plan is then a single window and the
                                    # run is bit-for-bit the untiled one.
                                    # A canvas that REALLY tiles is always TWO-STAGE. A window
                                    # sees a fraction of the picture, and windows given only the
                                    # prompt each compose their own whole version of it - a
                                    # named landmark comes out drawn once per window - so before
                                    # any window is denoised there is always a full-canvas
                                    # anchor, and each window is conditioned on its own region
                                    # of that one composition:
                                    #   no --ref: the composition is DRAFTED in one forward at
                                    #     the largest size this run's own per-forward budget
                                    #     allows (same prompt, seed and sampler), upscaled to
                                    #     the canvas, then refined window by window at
                                    #     --strength (default 0.4);
                                    #   with --ref: that reference IS the anchor, resampled to
                                    #     the canvas when it is not already there - so an
                                    #     upscale-this-photo run needs no pre-resize by hand.
                                    # Every window sees the same prompt and folded adapters and
                                    # carries the position ids its tokens have on the WHOLE
                                    # canvas, so they compose one scene. References past the
                                    # first are generic guidance and every window sees all of
                                    # them whole. The anchor roughly doubles the joint sequence
                                    # per window, which is what --tile-size is sized against;
                                    # --ref-resolution-scale below 1 is refused here, because
                                    # shrinking the anchor's conditioning copy is exactly what
                                    # stops it anchoring anything.
                                    # NOTE the VAE still decodes the canvas in one pass.
           [--tile-overlap N]       # how much adjacent windows share, in pixels (default: a
                                    # quarter of --tile-size). The blend across it is feathered,
                                    # so this is the width a seam is spread over. Multiple of
                                    # 16, smaller than --tile-size.
           [--mask <mask.png>]      # WHITE = regenerate, BLACK = preserve the first
                                    # --ref exactly (which must be at the output size, or
                                    # under --tile-size is resampled to it);
                                    # greys blend. Omit = regenerate everything.
           [--text-encoder <path>]  # state the text encoder outright: an HF directory, or a
                                    # single .safetensors/.gguf FILE, exactly as the models
                                    # directory scan already found it. Any checkpoint with the
                                    # stock tensor names/shapes drops in - a fine-tune, an
                                    # abliteration, a re-quantisation - `validate()` rejects a
                                    # mismatched size against the chosen DiT before any load.
           [--dit <path>]           # state the DiT weights outright, the same way.
           [--vae <path>]           # state the VAE outright, the same way.
           [--tokenizer <path>]     # state the tokenizer outright, the same way.
                                    # --dit/--vae/--text-encoder/--tokenizer/--variant are the
                                    # resolver's overrides, one flag per role: when they leave a
                                    # role genuinely ambiguous (more than one candidate, or
                                    # klein-vs-base unstated - never recoverable from a weight's
                                    # shape) every real candidate's selector flag prints and the
                                    # run exits rather than guessing.
           [--adapter <path>]...    # LoRA: brain's own `finetune` checkpoint, or a
                                    # third-party ai-toolkit/ComfyUI/LyCORIS
                                    # .safetensors in either recognised family -
                                    # LoRA (.lora_A/.lora_B, delta B*A) or LoKr
                                    # (.lokr_w1/.lokr_w2, delta W1 kron W2), read
                                    # from the file's own keys.
                                    # REPEATABLE - pass it once per adapter to stack
                                    # several in one generation (a face adapter plus a
                                    # style adapter, say). They fold in the order given,
                                    # each onto the result of the ones before it, and the
                                    # deltas ADD on any linear more than one of them
                                    # adapts: two adapters both at 1.0 move a shared
                                    # weight by the sum of two separately trained deltas,
                                    # which is usually stronger than either was validated
                                    # at. --lora-scale is the dial for that.
           [--lora-scale S]...      # LoRA strength (ComfyUI strength_model), default 1.0.
                                    # REPEATABLE and paired POSITIONALLY: the Nth
                                    # --lora-scale is the Nth --adapter's strength, and it
                                    # must come AFTER that --adapter on the command line.
                                    # A --lora-scale with no adapter of its own is an
                                    # error rather than a guess. Adapters with no
                                    # --lora-scale of their own fold at 1.0.
                                    #   --adapter face.brain --lora-scale 0.8 \
                                    #   --adapter style.safetensors --lora-scale 0.4
  finetune <data_dir> --out <adapter.brain> [--variant V] [--steps N] [--rank R] [--lr X]
           [--size S] [--seed K] [--ckpt-every N] [--resume] [--trainer device|host] [--cards N]
           [--text-encoder <path>] [--method lora|rslora] [--lr-ratio X] [--freeze-a]
           [--precision fp32|int8] [--warmup N] [--min-lr X]
           [--edit-weight A] [--ref-dropout P]
           # Train a LoRA on a folder of captioned images (see data::imageset for
           # the caption formats; `brain label` writes one). The adapter it writes
           # is what `generate --adapter` loads. Do NOT name it '.safetensors':
           # that extension is how --adapter recognises a THIRD-PARTY LoRA.
           #
           # PAIRED training: add a pairs.yaml to the folder (one
           # 'target.jpg: reference.jpg' per line) and each target trains CONDITIONED on
           # its reference, through the same joint token layout and t-axis RoPE
           # offset `generate --ref ... --strength 1.0` builds. That is the
           # adapter to train for a reference-image workflow; a folder without
           # pairs.yaml trains the caption-only concept LoRA it always did. The
           # manifest must cover every captioned image or none.
           #   --size S        square training size in px, multiple of 16 (default 512)
           #   --rank R        LoRA rank (default 16)
           #   --steps N       training steps (default 200)
           #   --lr X          PEAK learning rate (default 1e-4, which is what
           #                   every published FLUX LoRA recipe and BFL's own
           #                   klein guidance use). Held flat for the first 80%
           #                   of the run, then cosine-cooled to --min-lr.
           #   --min-lr X      the rate the cooldown lands on at the last step
           #                   (default --lr/10). Adam at a CONSTANT rate does
           #                   not converge on a stochastic objective, it
           #                   orbits the minimum at a radius set by the rate
           #                   times the gradient noise - at batch size 1 that
           #                   is wide. The cooldown closes it. Constant +
           #                   ~20% cooldown tracks a full cosine and does not
           #                   bake --steps into every step, so a run that is
           #                   later extended has not already annealed itself
           #                   against the shorter budget. `--min-lr <same as
           #                   --lr>` restores a flat rate.
           #   --edit-weight A region-aware flow loss for a PAIRED run
           #                   (default 0 = off; the published value is 2).
           #                   Weights each target token's squared error by
           #                   1 + A*(its |target-reference| change, normalised
           #                   by the largest one), rescaled to mean 1 so the
           #                   reported loss stays comparable. In a declutter
           #                   or removal pair the target latent equals the
           #                   reference latent almost everywhere, so a uniform
           #                   mean - and the gradient under it - is dominated
           #                   by tokens the model can hit by copying the
           #                   reference across, and a flat loss may just be a
           #                   competent copier. Only correct for SPATIALLY
           #                   ALIGNED pairs; every paired run prints the share
           #                   of tokens that actually differ, which is the
           #                   number that says whether yours are.
           #   --ref-dropout P probability a PAIRED step trains with its
           #                   reference tokens BLANKED (default 0; reach for
           #                   0.1). When a target and its reference differ
           #                   only locally, an adapter can drive the loss a
           #                   long way down by learning to copy the reference
           #                   across - a real solution, with a loss curve that
           #                   descends and then flattens, whose deployed
           #                   output reproduces the reference's own lighting,
           #                   grain and colour instead of transforming them.
           #                   Blanking the reference on some steps removes the
           #                   thing being copied, so the adapter has to learn
           #                   what a finished target looks like. It is not
           #                   free: klein is guidance-distilled, so unlike
           #                   InstructPix2Pix's 5% there is no null branch at
           #                   inference for those steps to be training, and
           #                   they come out of the same step budget.
           #   --warmup N      steps of linear LR warmup (default 0). A LoRA
           #                   starts at B=0, so its branch output and the
           #                   gradient through it start at zero and there is
           #                   no early instability for a warmup to protect
           #                   against; every mainstream FLUX LoRA trainer
           #                   defaults to none. The curve is a function of the
           #                   GLOBAL step, so --resume continues it.
           #   --ckpt-every N  checkpoint every N steps (default 100; 0 = final only).
           #                   Each write is atomic (temp file + rename), so an
           #                   interrupted write cannot damage the last good one.
           #   --resume        continue from the adapter already at --out, if one
           #                   is there, instead of starting over; with no file
           #                   there it starts fresh, so the SAME command is
           #                   correct whether or not it is the first run. The
           #                   step count rides in the checkpoint header, so the
           #                   sample cycle and sigma schedule continue too.
           #                   Adam moments are not stored and do restart.
           #   --trainer T     device (default, WGSL kernels, frozen base on the
           #                   card) or host (the FD-gradchecked reference the
           #                   device path is validated against - correct, and
           #                   minutes per step at klein scale)
           #   --cards N       GPUs the device trainer spreads the stack over
           #                   (default 1; klein-9b's fp32 base needs 2)
           #   --method M      lora (default, alpha/rank) or rslora
           #                   (alpha/sqrt(rank) - does not suppress the update
           #                   as rank grows)
           #   --lr-ratio X    LoRA+: B's effective lr is lr-ratio*lr (default
           #                   1.0, plain LoRA)
           #   --freeze-a      LoRA-FA: freeze A at its random init, train only B
           #   --precision P   the DiT tier this adapter will be GENERATED with
           #                   (default fp32; a .gguf DiT forces int8). It picks
           #                   the text-encoder tier the captions are embedded
           #                   through, so the conditioning trained against is
           #                   the conditioning the deployment produces.
           # Both trainers run the same op sequence; the device one keeps the
           # frozen base on the card and differentiates only the low-rank
           # factors. Which one ran is printed at the top of every run.
Weights: `generate` resolves dit/vae/text_encoder/tokenizer from the models
directory (--models-dir / BRAIN_MODELS_DIR) - --dit/--text-encoder/--variant
name a role outright, and an ambiguous or missing outcome prints every real
candidate and exits rather than guessing. `finetune` still reads
BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}.
Text-encoder placement (env): BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8] (truncated shard on that card)";

pub fn run_flux2(args: &[String]) {
    if args.is_empty() || args[0] == "--help" {
        eprintln!("{HELP}");
        return;
    }
    match args[0].as_str() {
        "generate" | "infer" => {
            if let Err(e) = generate(&args[1..]) {
                eprintln!("flux2 generate: {e}");
                std::process::exit(1);
            }
        }
        "finetune" => {
            if let Err(e) = finetune(&args[1..]) {
                eprintln!("flux2 finetune: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("flux2: unknown subcommand {other}\n{HELP}");
            std::process::exit(2);
        }
    }
}

/// The generation's output size.
///
/// `anchor` is the size of `refs[0]` when that reference seeds the init latent
/// (`--strength < 1` or `--mask`). That reference IS the canvas - it is
/// VAE-encoded into the starting latent - so an anchored run with no explicit
/// size takes the anchor's, per axis. An explicit `--width`/`--height` always
/// wins, and with no anchor the documented default stands.
fn output_size(w: Option<u32>, h: Option<u32>, anchor: Option<(u32, u32)>) -> (u32, u32) {
    let (dw, dh) = anchor.unwrap_or((512, 512));
    (w.unwrap_or(dw), h.unwrap_or(dh))
}

/// The tile plan `--tile-size`/`--tile-overlap` ask for, or `None` for the
/// untiled path.
///
/// Off unless asked for, and `--tile-size 0` is also off - the same shape
/// `--ref-size` gives its bound, so a script that computes the number can say
/// "off" in the field instead of having to drop the flag. An overlap with no
/// tile size is an error rather than a silently ignored word on the command
/// line, exactly as a `--lora-scale` with no adapter is.
fn tiling_from(size: Option<u32>, overlap: Option<u32>) -> Result<Option<flux2::Tiling>, String> {
    match (size.filter(|&n| n > 0), overlap) {
        (None, Some(_)) => Err("--tile-overlap needs --tile-size (tiling is off without it)".into()),
        (None, None) => Ok(None),
        (Some(n), ov) => {
            let t = flux2::Tiling { overlap: ov.unwrap_or(flux2::Tiling::new(n).overlap), size: n };
            t.check()?;
            Ok(Some(t))
        }
    }
}

/// Long edge, in pixels, a reference is encoded at when the caller does not
/// say. A reference costs `(w/16)*(h/16)` tokens and attention is quadratic in
/// the joint sequence, so an unscaled camera photograph costs several times
/// the generation it is conditioning. Bounding by default is the difference
/// between `--ref holiday.jpg` working and it quietly becoming the most
/// expensive part of the run.
pub const DEFAULT_REF_EDGE: u32 = 512;

/// `refs[0]` seeds the init latent - and so must be pinned to the output size,
/// never bounded - whenever a `--strength` was given at all, or under `--mask`.
/// `Some(1.0)` still counts: the option's own semantics treat 1.0 as "full
/// redraw from the anchor," not "no anchor" (`GenOpts::strength`'s documented
/// range is `(0, 1]`), so this must not use `s < 1.0` - that strict cutoff
/// silently drops anchoring at exactly 1.0, falling back to the default
/// `(512, 512)` canvas and an unbounded-but-uncropped `refs[0]` instead of one
/// pinned to the reference's own size, which breaks the reference's geometry
/// even though it still conditions the model as tokens.
fn is_anchored(strength: Option<f32>, has_mask: bool) -> bool {
    strength.is_some() || has_mask
}

/// The long-edge bound for reference `i`, or `None` to encode it at its own
/// resolution.
///
/// `pinned` is true when `refs[0]` is pinned to the canvas: it seeds the init
/// latent (see [`is_anchored`]), or it is a windowed run's anchor, which the
/// pipeline resamples to the canvas so every window can be conditioned on its
/// own region of it. Either way that reference must never be bounded on the way
/// in - the bound would decide the canvas's own resolution. A caller wanting
/// ITS conditioning cost down has `--ref-resolution-scale`, which exists for
/// exactly this asymmetry (and which a windowed run refuses, because shrinking
/// the anchor's conditioning copy is what un-registers it).
fn ref_bound(i: usize, pinned: bool, ref_size: Option<u32>) -> Option<u32> {
    if i == 0 && pinned {
        return None;
    }
    match ref_size {
        Some(0) => None, // explicit opt-out
        Some(m) => Some(m),
        None => Some(DEFAULT_REF_EDGE),
    }
}

/// Attach a `--lora-scale` to the `--adapter` it belongs to.
///
/// The two flags are both repeatable and pair POSITIONALLY: the Nth
/// `--lora-scale` is the Nth `--adapter`'s strength. `scaled` counts the
/// adapters whose strength has already been claimed, so the next one lands on
/// the next unscaled adapter - which means a strength must be written after
/// the adapter it belongs to.
///
/// A strength with no adapter of its own is REFUSED rather than guessed at. A
/// stacked run's whole point is that each adapter folds at its own strength,
/// and the failure mode of guessing (attaching a strength to the wrong
/// adapter, or to all of them) produces a plausible image from the wrong
/// weights - which no error message ever gets written about, because the run
/// succeeded.
fn attach_lora_scale(specs: &mut [AdapterSpec], scaled: &mut usize, value: f32) -> Result<(), String> {
    let Some(spec) = specs.get_mut(*scaled) else {
        return Err(format!(
            "--lora-scale {value} has no --adapter of its own ({} adapter(s) given, {} already \
             carry a strength). Each --lora-scale applies to the --adapter it FOLLOWS: write \
             `--adapter <path> --lora-scale <S>` once per adapter.",
            specs.len(),
            *scaled
        ));
    };
    spec.scale = value;
    *scaled += 1;
    Ok(())
}

/// Resolve FLUX.2's four weight roles through the model-store resolver
/// (`brain_modelstore::resolve::resolve` + [`flux2::spec::Flux2Spec`])
/// instead of `BRAIN_FLUX2_*` variables: `--dit`/`--text-encoder` name a
/// role's file outright (the resolver's own override contract - an exact
/// path already found by scanning the models directory), `--variant` states
/// the klein-vs-base family a weight's shape alone can never answer.
///
/// One flag per role, spelled as the role is: `dit` -> `--dit`, `vae` ->
/// `--vae`, and so on. That is not decoration - `describe_ambiguity` /
/// `describe_missing` print the ROLE NAME as the flag to pass when a role
/// cannot be resolved, so any role whose flag is spelled differently tells
/// the user to type something the parser rejects.
/// `crate::resolver_cli::resolve_or_exit` does the actual scan/resolve and
/// prints+exits on `Ambiguous`/`Missing` - shared with every other
/// architecture's own resolver-backed command, not flux2-specific.
fn resolve_flux2(dit: Option<&str>, vae: Option<&str>, text_encoder: Option<&str>, tokenizer: Option<&str>, variant: Option<&str>) -> Result<(Paths, capability::Assembly), String> {
    let mut overrides = std::collections::BTreeMap::new();
    if let Some(m) = dit {
        overrides.insert("dit".to_string(), m.to_string());
    }
    if let Some(v) = vae {
        overrides.insert("vae".to_string(), v.to_string());
    }
    if let Some(te) = text_encoder {
        overrides.insert("text_encoder".to_string(), te.to_string());
    }
    if let Some(t) = tokenizer {
        overrides.insert("tokenizer".to_string(), t.to_string());
    }
    if let Some(v) = variant {
        overrides.insert("variant".to_string(), v.to_string());
    }
    let spec = flux2::spec::Flux2Spec;
    let assembly = crate::resolver_cli::resolve_or_exit("flux2", &spec, &overrides);
    let paths = Paths::from_assembly(&assembly)?;
    Ok((paths, assembly))
}

fn generate(args: &[String]) -> Result<(), String> {
    let mut prompt = None;
    let mut out = None;
    let mut o = GenOpts { width: 512, height: 512, ..GenOpts::default() };
    let (mut want_w, mut want_h): (Option<u32>, Option<u32>) = (None, None);
    let mut variant_name = "klein-4b".to_string();
    let mut variant_explicit = false;
    let mut precision = flux2::Precision::F32;
    let mut precision_was_explicit = false;
    let mut refs: Vec<String> = Vec::new();
    let mut ref_size: Option<u32> = None;
    let mut tile_size: Option<u32> = None;
    let mut tile_overlap: Option<u32> = None;
    let mut mask_path: Option<String> = None;
    // Repeatable, like `--ref` above: one entry per `--adapter`, in the order
    // they were typed, which is the order they fold in.
    let mut adapters: Vec<AdapterSpec> = Vec::new();
    let mut scaled = 0usize;
    let mut text_encoder: Option<String> = None;
    let mut dit: Option<String> = None;
    let mut vae: Option<String> = None;
    let mut tokenizer: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> {
            args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--out" => out = Some(strip_out_name_prefix(need(i)?, "image").to_string()),
            "--width" => want_w = Some(need(i)?.parse().map_err(|e| format!("--width: {e}"))?),
            "--height" => want_h = Some(need(i)?.parse().map_err(|e| format!("--height: {e}"))?),
            "--steps" => o.steps = Some(need(i)?.parse().map_err(|e| format!("--steps: {e}"))?),
            // A boolean flag, so it consumes no value - `continue` past the
            // `i += 2` the value-taking arms use.
            "--experimental-steps" => {
                o.experimental_steps = true;
                i += 1;
                continue;
            }
            "--seed" => o.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--strength" => o.strength = Some(need(i)?.parse().map_err(|e| format!("--strength: {e}"))?),
            "--ref-resolution-scale" => {
                o.ref_resolution_scale = need(i)?.parse().map_err(|e| format!("--ref-resolution-scale: {e}"))?;
                if !(0.0..=1.0).contains(&o.ref_resolution_scale) {
                    return Err(format!("--ref-resolution-scale must be in 0..=1 (got {})", o.ref_resolution_scale));
                }
            }
            // A dedicated error rather than "unknown flag": a script carrying
            // the old spelling should be told where the dial went, not left to
            // find it in the help text.
            "--ref-cond-scale" => {
                return Err("--ref-cond-scale was renamed to --ref-resolution-scale (the same 0..=1 dial)".into())
            }
            "--guidance" => o.guidance = need(i)?.parse().map_err(|e| format!("--guidance: {e}"))?,
            "--variant" => {
                variant_name = need(i)?.clone();
                variant_explicit = true;
            }
            "--precision" => {
                precision = flux2::Precision::from_name(need(i)?)?;
                precision_was_explicit = true;
            },
            "--ref" => refs.push(need(i)?.clone()),
            "--ref-size" => {
                let n: u32 = need(i)?.parse().map_err(|e| format!("--ref-size: {e}"))?;
                if n != 0 && n < 16 {
                    return Err(format!("--ref-size must be 0 (unbounded) or at least 16 (got {n})"));
                }
                ref_size = Some(n);
            }
            // `--tile-size 0` is "no tiling", the same way `--ref-size 0` is
            // "no bound": a script that computes the number gets to say "off"
            // in the same field rather than having to drop the flag.
            "--tile-size" => {
                let n: u32 = need(i)?.parse().map_err(|e| format!("--tile-size: {e}"))?;
                tile_size = Some(n);
            }
            "--tile-overlap" => {
                tile_overlap = Some(need(i)?.parse().map_err(|e| format!("--tile-overlap: {e}"))?);
            }
            "--mask" => mask_path = Some(need(i)?.clone()),
            "--adapter" => adapters.push(AdapterSpec::new(need(i)?.clone())),
            "--lora-scale" => {
                let s: f32 = need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?;
                attach_lora_scale(&mut adapters, &mut scaled, s)?;
            }
            "--text-encoder" => text_encoder = Some(need(i)?.clone()),
            "--dit" => dit = Some(need(i)?.clone()),
            "--vae" => vae = Some(need(i)?.clone()),
            "--tokenizer" => tokenizer = Some(need(i)?.clone()),
            other => return Err(format!("unknown flag {other}\n{HELP}")),
        }
        i += 2;
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    let out = out.ok_or("--out is required")?;

    // Read every reference's pixels BEFORE any of them is bounded: the canvas
    // has to be settled first. A windowed run's first reference is the anchor
    // every window is conditioned on, the pipeline resamples it to the canvas,
    // and bounding it on the way in would hand that resample a thumbnail to
    // enlarge.
    let loaded: Vec<(Vec<f32>, u32, u32)> =
        refs.iter().map(|r| crate::image_io::load_image(r)).collect::<Result<_, String>>()?;
    let anchored = is_anchored(o.strength, mask_path.is_some());

    // The init reference is the canvas, so an anchored run inherits its size
    // rather than making the caller read the file and echo its dimensions.
    // It is the /16 crop the conversion below takes, not the file's own size.
    let anchor = anchored
        .then(|| loaded.first().map(|&(_, w, h)| flux2::pipeline::ref_crop_size(w, h)))
        .flatten();
    let (rw, rh) = output_size(want_w, want_h, anchor);
    if (rw, rh) != (o.width, o.height) && anchor.is_some() && (want_w.is_none() || want_h.is_none()) {
        eprintln!("flux2: output {rw}x{rh}, taken from the init reference");
    }
    o.width = rw;
    o.height = rh;
    o.tile = tiling_from(tile_size, tile_overlap)?;
    // Whether this canvas really denoises in more than one window, which is
    // what pins the first reference to the canvas and what makes a prompt-only
    // run draft an anchor of its own.
    let windowed = flux2::pipeline::runs_in_windows(&o);

    // Now convert: [-1,1] CHW, center-cropped to /16 (shared helper - the
    // capability provider uses the same one), optionally downscaled first so
    // one full-resolution photograph cannot outspend the whole generation.
    // `--ref-size` absent takes the bound-free path, unchanged.
    let mut ref_imgs: Vec<(Vec<f32>, u32, u32)> = Vec::new();
    for (i, ((hwc, w, h), r)) in loaded.iter().zip(&refs).enumerate() {
        let (w, h) = (*w, *h);
        let bound = ref_bound(i, anchored || windowed, ref_size);
        if let Some(m) = bound {
            let (tw, th) = flux2::pipeline::fit_long_edge(w, h, m);
            if (tw, th) != (w, h) {
                eprintln!("flux2: ref {r} {w}x{h} -> resampled to {tw}x{th} (--ref-size {m})");
            }
        } else if i == 0 && (anchored || windowed) {
            // Why it was NOT bounded, which is a decision and not the absence
            // of one: `--ref-size 0` reaches here too and says nothing.
            let why = if anchored { "it seeds the init latent" } else { "it anchors every window" };
            eprintln!("flux2: ref {r} {w}x{h} kept at full size - {why}");
        }
        ref_imgs.push(flux2::pipeline::ref_from_hwc_bounded(hwc, w, h, bound)?);
    }

    // The mask is over the OUTPUT canvas, so it is resampled to the latent grid
    // by the pipeline (area average, both axes independently) rather than being
    // required at any particular resolution here.
    if let Some(p) = &mask_path {
        let (hwc, w, h) = crate::image_io::load_image(p)?;
        let m = flux2::Mask::from_hwc(&hwc, w, h)?;
        eprintln!("flux2: mask {p} -> {m:?}");
        o.mask = Some(m);
    }

    // Weights come from the model-store resolver, never BRAIN_FLUX2_*: scan
    // the models directory for every candidate artifact, then let `--dit`/
    // `--text-encoder`/`--variant` (when the caller actually typed it) state
    // the roles nothing on disk can pick on its own. An ambiguous or missing
    // outcome prints and exits here - neither is recoverable within this
    // command, and resolve() never silently picks.
    let (paths, assembly) = resolve_flux2(dit.as_deref(), vae.as_deref(), text_encoder.as_deref(), tokenizer.as_deref(), variant_explicit.then_some(variant_name.as_str()))?;
    variant_name = assembly.variant.clone().ok_or("flux2: resolved assembly has no variant")?;
    flux2::caps::check_license(&variant_name)?; // 9B = FLUX Non-Commercial license
    let variant = Flux2Config::from_name(&variant_name)?;
    // Q8_0 GGUF is not an fp32 checkpoint with an optional output tier: the
    // FLUX.2 constructor consumes it through its packed DP4A representation.
    // Omitted `--precision` therefore follows the source; an explicit fp32
    // request is rejected rather than silently changing it.
    precision = flux2::pipeline::effective_dit_precision(&paths.dit, precision, precision_was_explicit)?;
    // Resolve the sampler the variant actually runs, and say so. BFL ships
    // the distilled klein models as fixed-param checkpoints (4 steps,
    // guidance 1.0, no CFG): a caller's --steps is ignored there unless
    // --experimental-steps opted in, and --guidance is a no-op the run
    // warns about rather than silently swallowing.
    let steps = flux2::pipeline::resolved_steps(&o, variant.distilled);
    if variant.distilled {
        if let Some(s) = o.steps.filter(|_| !o.experimental_steps) {
            eprintln!("flux2: {variant_name} is distilled (fixed 4-step sampler); --steps {s} ignored (pass --experimental-steps to override)");
        }
        if o.guidance > 1.0 {
            eprintln!("flux2: warning: --guidance {} is a no-op on distilled {variant_name} (guidance is fixed at 1.0, no CFG)", o.guidance);
        }
    }
    // Which model this run is actually about to load, before anything is
    // loaded.
    eprintln!("flux2: model {} ({}, {}, steps {steps})", assembly.id, variant_name, precision.name());
    let n_gen = (o.height / 16) * (o.width / 16);
    // Every supplied reference conditions the model; under `--strength` the
    // first one does so at `--ref-resolution-scale` of its own size *and* seeds the
    // init latent. Print the per-reference breakdown, not just the total:
    // reference tokens are what decides whether a run fits the card, and a
    // bare "N + M" does not say which reference spent them or at what size.
    let sizes = flux2::pipeline::cond_sizes(&ref_imgs, &o);
    let n_ref = flux2::pipeline::ref_tokens(&ref_imgs, &o);
    for (i, (size, (_, rh, rw))) in sizes.iter().zip(&ref_imgs).enumerate() {
        let role = if i == 0 && anchored { ", also the init latent" } else { "" };
        match size {
            Some((ch, cw)) => eprintln!(
                "flux2: ref {i} {rw}x{rh} -> conditions at {cw}x{ch} = {} tokens{role}",
                (ch / 16) * (cw / 16)
            ),
            None => eprintln!("flux2: ref {i} {rw}x{rh} -> no conditioning tokens (--ref-resolution-scale 0{role})"),
        }
    }
    // Say what the stack is before it is folded: with several adapters the
    // order and the per-adapter strength are what the result depends on, and
    // both are easy to get wrong on a long command line.
    for (n, a) in adapters.iter().enumerate() {
        eprintln!("flux2: adapter {n} {} at strength {}", a.path, a.scale);
    }
    // What ONE forward sees. Untiled that is the whole canvas, as it always
    // was; tiled it is the largest window, which is the entire point - the DiT
    // is sized for the window while the VAE still decodes the full canvas,
    // which is why `build_sized` takes the two ceilings separately.
    let n_fwd = flux2::pipeline::gen_tokens_per_forward(&o);
    // And the other half of it: a reference pinned to the output size is an
    // edit target, so a window is conditioned on its own region of it and the
    // reference cost follows the window too. Sizing for the whole reference
    // would reserve the canvas's worth on every forward - at 2048x1360 with
    // 512px tiles that is ten times the window's own tokens, and it is the
    // forward rather than the canvas that then fails to fit.
    let n_ref_fwd = flux2::pipeline::ref_tokens_per_forward(&ref_imgs, &o);
    if let Some(t) = o.tile {
        let tiles = flux2::pipeline::plan_tiles((o.height / 16) as usize, (o.width / 16) as usize, Some(t));
        eprintln!(
            "flux2: tiled generation: {} window(s) of {}x{} px (overlap {}), {n_fwd} generated tokens per forward instead of {n_gen}",
            tiles.len(),
            (tiles[0].tw * 16) as u32,
            (tiles[0].th * 16) as u32,
            t.overlap
        );
        // How this canvas gets the whole-canvas anchor its windows are
        // conditioned on. Never nothing: windows that see only the prompt each
        // compose their own version of it, which at this geometry is a
        // landmark drawn three times over.
        match flux2::pipeline::anchor_for(&ref_imgs, &o) {
            flux2::pipeline::Anchor::Whole => {
                eprintln!("flux2: the canvas fits one window - this run is the untiled one")
            }
            flux2::pipeline::Anchor::Given => eprintln!(
                "flux2: the reference is the canvas - each window is conditioned on its own {n_ref_fwd} of its {n_ref} tokens"
            ),
            flux2::pipeline::Anchor::Resampled { from: (rh, rw) } => eprintln!(
                "flux2: ref 0 {rw}x{rh} -> resampled to the {}x{} canvas, so each window can be conditioned on its own {n_ref_fwd} of its {n_ref} tokens",
                o.width, o.height
            ),
            flux2::pipeline::Anchor::Draft { size: (dw, dh), strength } => eprintln!(
                "flux2: no reference: drafting the composition at {dw}x{dh} in ONE forward, then refining it in {} windows at strength {strength} (each conditioned on its own {n_ref_fwd} tokens of the upscaled draft)",
                tiles.len()
            ),
        }
    }
    eprintln!("flux2: building pipeline ({n_fwd} generated + {n_ref_fwd} reference tokens per forward, {n_gen} decoded) ...");
    let pipe = Pipeline::build_sized(&variant, &paths, n_fwd + n_ref_fwd, n_gen, &adapters, precision, 1)?;
    let t0 = std::time::Instant::now();
    // Per-phase wall clock: the callback fires immediately BEFORE each phase,
    // so the gap between two calls is the previous phase's duration. Text
    // encode / denoise / VAE decode are the three costs a generation is made
    // of, and the split is what any perf claim has to be argued from.
    let mut phase = std::cell::RefCell::new((std::time::Instant::now(), String::new(), std::collections::BTreeMap::<String, f32>::new()));
    // The CLI has no cancel front-end - an unarmed Default token never fires.
    let (rgb, w, h) = pipe.generate(&prompt, &ref_imgs, &o, &Default::default(), |step, total, msg| {
        let mut p = phase.borrow_mut();
        let dt = p.0.elapsed().as_secs_f32();
        if !p.1.is_empty() {
            let key = p.1.clone();
            *p.2.entry(key).or_default() += dt;
        }
        p.0 = std::time::Instant::now();
        p.1 = msg.to_string();
        eprint!("\rflux2 [{step}/{total}] {msg}          ");
    })?;
    {
        let p = phase.get_mut();
        let last = p.1.clone();
        let dt = p.0.elapsed().as_secs_f32();
        *p.2.entry(last).or_default() += dt;
        eprintln!("\nflux2: {:.1}s total", t0.elapsed().as_secs_f32());
        for (k, v) in &p.2 {
            eprintln!("  {k:<20} {v:>7.2}s");
        }
    }
    imaging::save(&out, &imaging::Rgb8::new(w, h, rgb)?)?;
    eprintln!("flux2: wrote {out} ({w}x{h})");
    Ok(())
}

/// Refuse a `--out` that `--adapter` would later hand to the wrong parser.
///
/// `Pipeline::build_dit` distinguishes brain's own adapter container from a
/// third-party ai-toolkit/ComfyUI one **by file extension** - `.safetensors`
/// takes the external route, anything else takes `lora::load_adapter`. So an
/// adapter trained here and named `.safetensors` is written in one format and
/// read back as another. The failure would surface later, at generation time,
/// as a confusing parse error over a file that is not actually malformed.
fn check_adapter_out(path: &str) -> Result<(), String> {
    if path.to_ascii_lowercase().ends_with(".safetensors") {
        return Err(format!(
            "--out {path}: a trained adapter must not be named '.safetensors'. That extension is \
             how `--adapter` recognises a THIRD-PARTY (ai-toolkit/ComfyUI) LoRA, so this file \
             would be written in brain's own container and read back with the external parser. \
             Use '.brain' (or any other extension) instead."
        ));
    }
    Ok(())
}

/// `brain flux2 finetune <data_dir> --out <adapter>` - train a LoRA adapter on a
/// folder of captioned images.
///
/// The grammar follows `brain glm finetune <data_dir> ...`: the dataset is
/// positional, everything else is a flag. The training itself is
/// `flux2::finetune::run`, which is the same code the `lora_train` capability
/// action drives, so the CLI and the served path cannot drift on defaults.
fn finetune(args: &[String]) -> Result<(), String> {
    let mut data_dir: Option<String> = None;
    let mut variant_name = "klein-4b".to_string();
    let mut ft_text_encoder: Option<String> = None;
    let mut opts = flux2::finetune::TrainOpts {
        steps: 200,
        rank: 16,
        lr: 1e-4,
        // The device trainer is the default because the host one is the
        // reference, not a production path - but the choice is printed on
        // every run and `--trainer host` selects the oracle explicitly.
        trainer: flux2::finetune::Trainer::Device,
        cards: 1,
        size: 512,
        seed: 0,
        save_path: String::new(),
        ckpt_every: 100,
        resume: false,
        rank_stabilized: false,
        lr_ratio: 1.0,
        freeze_a: false,
        // Absent, not spelled out: `TrainOpts::lr_schedule` owns the default
        // curve so the CLI and the served `lora_train` action cannot drift on
        // it - and so it stays derived from whatever --steps turns out to be,
        // which is not known until the whole argument list is parsed.
        warmup: None,
        min_lr: None,
        // Off by default: the region-aware loss is only correct for pairs
        // whose two images are spatially registered, and the trainer cannot
        // tell that from a folder. Every paired run PRINTS the measurement
        // that decides it (the share of target tokens that differ from their
        // reference), so the choice is informed rather than guessed.
        edit_weight: 0.0,
        // Also off by default, and for a sharper reason: klein is
        // guidance-distilled, so there is no null branch at inference for
        // dropped steps to be training. They buy regularisation against the
        // copy shortcut and nothing else, which is worth paying for on a
        // dataset whose pairs are near-identical and not otherwise.
        ref_dropout: 0.0,
        // What `generate` would run this adapter at. fp32 is generate's own
        // default request; `effective_dit_precision` overrides it for a .gguf.
        precision: flux2::Precision::F32,
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> {
            args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" | "--save" => opts.save_path = strip_out_name_prefix(need(i)?, "adapter").to_string(),
            "--variant" => variant_name = need(i)?.clone(),
            "--steps" => opts.steps = need(i)?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--rank" => opts.rank = need(i)?.parse().map_err(|e| format!("--rank: {e}"))?,
            "--lr" => opts.lr = need(i)?.parse().map_err(|e| format!("--lr: {e}"))?,
            "--warmup" => opts.warmup = Some(need(i)?.parse().map_err(|e| format!("--warmup: {e}"))?),
            "--edit-weight" => opts.edit_weight = need(i)?.parse().map_err(|e| format!("--edit-weight: {e}"))?,
            "--ref-dropout" => opts.ref_dropout = need(i)?.parse().map_err(|e| format!("--ref-dropout: {e}"))?,
            "--min-lr" => opts.min_lr = Some(need(i)?.parse().map_err(|e| format!("--min-lr: {e}"))?),
            "--size" => opts.size = need(i)?.parse().map_err(|e| format!("--size: {e}"))?,
            "--seed" => opts.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--ckpt-every" => opts.ckpt_every = need(i)?.parse().map_err(|e| format!("--ckpt-every: {e}"))?,
            // A boolean flag, so it consumes no value - `continue` past the
            // `i += 2` the value-taking arms use.
            "--resume" => {
                opts.resume = true;
                i += 1;
                continue;
            }
            "--trainer" => opts.trainer = flux2::finetune::Trainer::from_name(need(i)?)?,
            "--text-encoder" => ft_text_encoder = Some(need(i)?.clone()),
            "--cards" => opts.cards = need(i)?.parse().map_err(|e| format!("--cards: {e}"))?,
            "--method" => {
                opts.rank_stabilized = match need(i)?.as_str() {
                    "lora" => false,
                    "rslora" => true,
                    other => return Err(format!("--method: {other} is not one of lora, rslora")),
                }
            }
            "--lr-ratio" => opts.lr_ratio = need(i)?.parse().map_err(|e| format!("--lr-ratio: {e}"))?,
            "--precision" => {
                opts.precision = match need(i)?.as_str() {
                    "fp32" | "f32" => flux2::Precision::F32,
                    "int8" | "i8" => flux2::Precision::Int8,
                    other => return Err(format!("--precision: {other} is not one of fp32, int8")),
                }
            }
            "--freeze-a" => {
                opts.freeze_a = true;
                i += 1;
                continue;
            }
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}\n{HELP}")),
            // The positional dataset directory, as in `brain glm finetune`.
            other => {
                if let Some(first) = &data_dir {
                    return Err(format!("unexpected argument {other} (the dataset directory is already {first})"));
                }
                data_dir = Some(other.to_string());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    let data_dir = data_dir.ok_or("the dataset directory is required (a positional argument)")?;
    if opts.save_path.is_empty() {
        return Err("--out is required".into());
    }
    check_adapter_out(&opts.save_path)?;
    // `encode_samples` enforces this too, but only after the whole dataset has
    // been decoded - which on a real folder is minutes spent to learn a typo.
    if !opts.size.is_multiple_of(16) {
        return Err(format!("--size must be a multiple of 16 (got {})", opts.size));
    }
    if opts.rank == 0 || opts.steps == 0 {
        return Err("--rank and --steps must both be at least 1".into());
    }
    let mut paths = Paths::from_env()?;
    // Training and generation must be able to name the SAME encoder: an
    // adapter learns against the conditioning it was shown, so training on one
    // encoder and generating on another silently degrades every result.
    if let Some(te) = ft_text_encoder {
        paths.te = te;
    }
    // Bound against the frozen base's own shapes, not trusted as `--variant`
    // stated it - see `bind_variant`'s doc. `variant_name` is reassigned to
    // the bound truth so the log line below names what is actually training.
    variant_name = flux2::caps::bind_variant(&paths.dit, &variant_name)?;
    flux2::caps::check_license(&variant_name)?; // 9B = FLUX Non-Commercial license
    let cfg = Flux2Config::from_name(&variant_name)?;

    let lr_curve = opts.lr_schedule();
    eprintln!(
        "flux2 finetune: {variant_name} {} trainer, rank {} ({}) steps {} size {} lr {:.3e} held to step {} then cooled to {:.3e} (x{} on B) seed {} ckpt-every {}{}{} -> {}",
        opts.trainer.name(),
        opts.rank,
        if opts.rank_stabilized { "rslora" } else { "lora" },
        opts.steps,
        opts.size,
        lr_curve.peak,
        lr_curve.decay_start(),
        lr_curve.floor,
        opts.lr_ratio,
        opts.seed,
        opts.ckpt_every,
        if opts.freeze_a { " freeze-a" } else { "" },
        if opts.resume { " resume" } else { "" },
        opts.save_path
    );
    // The CLI has no cancel front-end - an unarmed Default token never fires.
    let cancel = capability::CancelToken::default();
    let t0 = std::time::Instant::now();
    flux2::finetune::run(&cfg, &paths, std::path::Path::new(&data_dir), &opts, &cancel, |done, total, msg| {
        eprintln!("flux2 finetune [{done}/{total}] {msg}");
    })?;
    eprintln!("flux2 finetune: {:.1}s total -> {}", t0.elapsed().as_secs_f32(), opts.save_path);
    Ok(())
}

#[cfg(test)]
mod adapter_flag_tests {
    use super::{attach_lora_scale, AdapterSpec};

    /// Replay a `generate` argument list's `--adapter`/`--lora-scale` flags in
    /// the order they were typed, exactly as `generate`'s own loop does.
    fn parse(args: &[&str]) -> Result<Vec<AdapterSpec>, String> {
        let mut specs: Vec<AdapterSpec> = Vec::new();
        let mut scaled = 0usize;
        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--adapter" => specs.push(AdapterSpec::new(args[i + 1])),
                "--lora-scale" => attach_lora_scale(&mut specs, &mut scaled, args[i + 1].parse().unwrap())?,
                other => panic!("unexpected flag {other}"),
            }
            i += 2;
        }
        Ok(specs)
    }

    /// The single-adapter spelling this flag has always had must keep working
    /// unchanged, with and without a strength.
    #[test]
    fn one_adapter_is_unchanged() {
        let a = parse(&["--adapter", "a.brain"]).unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "a.brain");
        assert_eq!(a[0].scale, 1.0, "an adapter with no --lora-scale folds at the reference default");

        let a = parse(&["--adapter", "a.brain", "--lora-scale", "0.6"]).unwrap();
        assert_eq!(a[0].scale, 0.6);
    }

    /// The Nth `--lora-scale` belongs to the Nth `--adapter` - positional
    /// pairing, the way `--ref` is positional. Adapters past the last given
    /// strength take the default.
    #[test]
    fn scales_pair_positionally_with_adapters() {
        let a = parse(&["--adapter", "face.brain", "--lora-scale", "0.8", "--adapter", "style.safetensors", "--lora-scale", "0.4"]).unwrap();
        assert_eq!(a[0].path, "face.brain");
        assert_eq!(a[0].scale, 0.8);
        assert_eq!(a[1].path, "style.safetensors");
        assert_eq!(a[1].scale, 0.4);

        // Both adapters first, then one strength: it belongs to the FIRST
        // adapter, and the second keeps the default rather than inheriting it.
        let a = parse(&["--adapter", "face.brain", "--adapter", "style.safetensors", "--lora-scale", "0.8"]).unwrap();
        assert_eq!((a[0].scale, a[1].scale), (0.8, 1.0));
    }

    /// A strength with no adapter of its own is an error, not a silent
    /// mispairing: one written before any `--adapter`, and a second one for an
    /// adapter that already has its strength.
    #[test]
    fn an_unpaired_strength_is_an_error_that_says_how_to_fix_it() {
        let e = parse(&["--lora-scale", "0.5", "--adapter", "a.brain"]).unwrap_err();
        assert!(e.contains("--lora-scale") && e.contains("--adapter"), "{e}");
        let e = parse(&["--adapter", "a.brain", "--lora-scale", "0.5", "--lora-scale", "0.6"]).unwrap_err();
        assert!(e.contains("--lora-scale"), "{e}");
    }

    /// No adapters at all is the ordinary unadapted run: an empty list, not an
    /// error and not a phantom entry.
    #[test]
    fn no_adapter_is_an_empty_list() {
        assert!(parse(&[]).unwrap().is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // describe_ambiguity/describe_missing are now brain_modelstore::resolve's
    // own shared functions (imported above) - their behavior is tested there,
    // generically, not re-tested per architecture here.

    /// `--out` on this dedicated command has always taken a bare path
    /// (`--out out.png`), but `brain caps flux2` documents the generic
    /// capability-manifest convention `--out image=<path>` (what `brain do`/
    /// D-Bus actually use) - typing the documented form here silently wrote
    /// a file literally named `image=out.png` with no error at all.
    /// `crate::args::strip_out_name_prefix` (shared, tested there) is wired
    /// in at both this crate's `--out` sites; this just pins that the wiring
    /// itself did not regress.
    #[test]
    fn generate_out_accepts_the_documented_name_equals_path_form() {
        assert_eq!(strip_out_name_prefix("image=out.png", "image"), "out.png");
        assert_eq!(strip_out_name_prefix("adapter=my.brain", "adapter"), "my.brain");
    }

    /// `Pipeline::build_dit` tells brain's own adapter container apart from a
    /// third-party ai-toolkit/ComfyUI one **by file extension**: a
    /// `.safetensors` takes the external route. So a `finetune --out` ending in
    /// `.safetensors` would write brain's own container under a name that the
    /// `--adapter` flag then hands to the wrong parser. Refuse it at the point
    /// the name is chosen, where the message can still be acted on.
    #[test]
    fn a_trained_adapter_may_not_be_named_safetensors() {
        let err = check_adapter_out("out/my-lora.safetensors").unwrap_err();
        assert!(err.contains(".safetensors"), "{err}");
        assert!(err.contains("--adapter"), "the message must say why it matters: {err}");
        // The suggested spelling has to be one that actually round-trips.
        assert!(check_adapter_out("out/my-lora.brain").is_ok());
        assert!(check_adapter_out("out/my-lora").is_ok());
        // Case is not a loophole: the extension check downstream is exact, so
        // an uppercase spelling really would take the brain route - but naming
        // it that way is still a trap for a human reading the folder.
        assert!(check_adapter_out("a/b.SAFETENSORS").is_err());
    }
}

#[cfg(test)]
mod tile_flag_tests {
    use super::tiling_from;

    /// Tiling is **opt-in and off by default**, and `--tile-size 0` is the
    /// explicit off switch - the shape `--ref-size 0` already established, so a
    /// script that computes the number can say "off" in the field.
    #[test]
    fn tiling_is_off_unless_asked_for() {
        assert_eq!(tiling_from(None, None).unwrap(), None);
        assert_eq!(tiling_from(Some(0), None).unwrap(), None);
    }

    /// The overlap has a default (a quarter of the tile) and is honoured when
    /// stated, so `--tile-size` alone is a complete request.
    #[test]
    fn a_tile_size_alone_is_a_complete_request() {
        assert_eq!(tiling_from(Some(1024), None).unwrap().unwrap().overlap, 256);
        let t = tiling_from(Some(1024), Some(64)).unwrap().unwrap();
        assert_eq!((t.size, t.overlap), (1024, 64));
    }

    /// A plan that cannot be expressed in whole latent tokens, or whose windows
    /// would overlap entirely, is refused at the command line rather than deep
    /// inside the sampler - and an overlap with no tile size is a typo, not a
    /// word to swallow (the same rule `--lora-scale` with no `--adapter` gets).
    #[test]
    fn an_unworkable_tile_plan_is_refused_where_it_was_typed() {
        assert!(tiling_from(Some(1000), None).is_err(), "not a multiple of 16");
        assert!(tiling_from(Some(1024), Some(100)).is_err(), "overlap not a multiple of 16");
        assert!(tiling_from(Some(512), Some(512)).is_err(), "overlap swallows the tile");
        let err = tiling_from(None, Some(128)).unwrap_err();
        assert!(err.contains("--tile-size"), "{err}");
    }

    /// A run that really denoises in windows must not let `--ref-size` bound
    /// its first reference: that one is the ANCHOR, the pipeline resamples it
    /// to the canvas so every window can be conditioned on its own region of
    /// it, and bounding it on the way in would hand that resample a 512 px
    /// thumbnail to enlarge to a 2048 px canvas.
    ///
    /// The `--ref-size` default is the reason this has to be explicit: it
    /// applies unless something says otherwise, so "upscale this photo" would
    /// silently upscale a downscale of it.
    #[test]
    fn a_windowed_run_never_bounds_the_reference_that_anchors_it() {
        use super::{ref_bound, DEFAULT_REF_EDGE};
        let o = flux2::GenOpts {
            width: 2048,
            height: 1152,
            tile: tiling_from(Some(512), None).unwrap(),
            ..flux2::GenOpts::default()
        };
        let windowed = flux2::pipeline::runs_in_windows(&o);
        assert!(windowed, "a 2048x1152 canvas in 512 px windows really tiles");
        assert_eq!(ref_bound(0, windowed, Some(512)), None, "the anchor was bounded");
        assert_eq!(ref_bound(0, windowed, None), None, "the default bound reached the anchor");
        // Guidance references past the first are not anchors and keep the bound.
        assert_eq!(ref_bound(1, windowed, None), Some(DEFAULT_REF_EDGE));
        // And a tiling that plans to ONE window changes nothing at all.
        let small = flux2::GenOpts { width: 512, height: 512, ..o };
        assert!(!flux2::pipeline::runs_in_windows(&small));
        assert_eq!(ref_bound(0, false, None), Some(DEFAULT_REF_EDGE));
    }
}

#[cfg(test)]
mod ref_size_tests {
    use super::{is_anchored, ref_bound};

    /// `--ref-size` exists so one full-resolution photograph cannot outspend
    /// the whole generation. It must not touch the **init** reference.
    ///
    /// Under any `--strength` (and under `--mask`) `refs[0]` is not merely
    /// conditioning: it is VAE-encoded into the starting latent, and that role
    /// pins it to the output size. Shrinking it there is not a cost saving,
    /// it is a broken run - and the caller who wants that reference's
    /// CONDITIONING cost down already has `--ref-resolution-scale`, which is defined
    /// against exactly this asymmetry.
    ///
    /// Without this the two flags cannot be used together at all, which is why
    /// callers ended up pre-resizing references in a shell script instead.
    #[test]
    fn ref_size_spares_the_reference_that_seeds_the_latent() {
        assert_eq!(ref_bound(0, true, Some(384)), None);
        assert_eq!(ref_bound(1, true, Some(384)), Some(384));
        assert_eq!(ref_bound(2, true, Some(384)), Some(384));
        // not anchored: no reference seeds the latent, so every one binds.
        assert_eq!(ref_bound(0, false, Some(384)), Some(384));
    }

    /// Bounding a reference must be the DEFAULT, not something the caller has
    /// to know to ask for. A reference costs `(w/16)*(h/16)` tokens and
    /// attention is quadratic in the joint sequence, so one unscaled phone
    /// photograph costs more than the image being generated - and the caller
    /// who just passed `--ref holiday.jpg` has no way to know that. Every
    /// wrapper script that got this right did it by resampling the files
    /// itself first, which is work brain should not be delegating.
    #[test]
    fn references_are_bounded_by_default() {
        assert_eq!(ref_bound(0, false, None), Some(super::DEFAULT_REF_EDGE));
        assert_eq!(ref_bound(1, true, None), Some(super::DEFAULT_REF_EDGE));
        // the init reference is still spared: its size is pinned by its role.
        assert_eq!(ref_bound(0, true, None), None);
    }

    /// `0` is the explicit opt-out, for a caller who really does want a
    /// reference encoded at its own resolution and has counted the tokens.
    #[test]
    fn ref_size_zero_means_unbounded() {
        assert_eq!(ref_bound(1, true, Some(0)), None);
        assert_eq!(ref_bound(0, false, Some(0)), None);
    }

    /// `--strength 1.0` means "full redraw from the anchor," not "no anchor" -
    /// the option's own documented range is `(0, 1]`. A strict `s < 1.0` cutoff
    /// here used to fall out of anchoring at exactly 1.0, silently switching
    /// the output canvas to the free-generation default and unbounding
    /// `refs[0]` instead of pinning it to the reference's own size - breaking
    /// the reference's geometry even though it still conditions the model.
    #[test]
    fn strength_one_still_counts_as_anchored() {
        assert!(is_anchored(Some(1.0), false));
        assert!(is_anchored(Some(0.5), false));
        assert!(is_anchored(None, true)); // --mask alone
        assert!(!is_anchored(None, false));
    }
}

#[cfg(test)]
mod output_size_tests {
    use super::output_size;

    /// Under `--strength`/`--mask` the first reference IS the canvas: it is
    /// VAE-encoded into the init latent, so the generation has to be the size
    /// that reference already is. Requiring the caller to pass that size is
    /// asking them to read the file's dimensions and echo them back, which is
    /// why wrappers grew a resize step in Python just to make the two agree.
    ///
    /// So: no `--width`/`--height` and an anchored run takes its size from the
    /// anchor.
    #[test]
    fn an_anchored_run_takes_its_size_from_the_anchor() {
        assert_eq!(output_size(None, None, Some((768, 1024))), (768, 1024));
        assert_eq!(output_size(None, None, Some((512, 512))), (512, 512));
    }

    /// An explicit size always wins - including when only one axis is given,
    /// because a caller who says `--width 768` and nothing else means the
    /// other axis to follow the anchor, not to snap back to a default.
    #[test]
    fn an_explicit_size_wins_over_the_anchor() {
        assert_eq!(output_size(Some(640), Some(480), Some((768, 1024))), (640, 480));
        assert_eq!(output_size(Some(640), None, Some((768, 1024))), (640, 1024));
        assert_eq!(output_size(None, Some(480), Some((768, 1024))), (768, 480));
    }

    /// With no anchor there is nothing to inherit, so the documented default
    /// stands and free generation is unchanged.
    #[test]
    fn free_generation_keeps_its_default() {
        assert_eq!(output_size(None, None, None), (512, 512));
        assert_eq!(output_size(Some(768), None, None), (768, 512));
    }
}
