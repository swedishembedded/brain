// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flux2 …` - FLUX.2 Klein text-to-image + image editing.
//!
//! Weights via env (`BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}`) with the generic
//! `--model` flag overriding the DiT (see `model_flag`); images in/out are
//! binary PPM P6 (the CLI-wide convention).

use flux2::{AdapterSpec, Flux2Config, GenOpts, Paths, Pipeline};

const HELP: &str = "brain flux2 <cmd>
  generate --prompt <text> --out <out.ppm> [--width W] [--height H]
           [--steps N] [--seed S] [--guidance G] [--variant klein-4b|klein-9b|base-4b|base-9b]
           [--precision fp32|int8]  # int8 = DP4A DiT (~4x smaller, GPU only);
                                    # .gguf defaults to int8 and rejects explicit fp32
           [--strength S]           # img2img anchoring dial, 0..1, on the first --ref
                                    # (which must then be at the output size). 1.0 = free
                                    # generation conditioned on the reference; lower anchors
                                    # progressively more of the source; 0 returns it. The
                                    # same sampler runs at every value, so 0.99 is a hair
                                    # from 1.0. The reference conditions at EVERY strength.
           [--ref-resolution-scale S]
                                    # linear size of the conditioning copy of the --strength
                                    # init reference, 0..1 (default 0.75). 1.0 = full size
                                    # (same token cost as --strength 1.0); 0 = do not
                                    # condition on it at all (cheapest: the reference then
                                    # reaches the model only through the init latent)
           [--ref <in.ppm>]...      # reference images => editing mode
           [--ref-size N]           # long edge each --ref is encoded at, before the /16
                                    # crop, preserving aspect. Never upscales. DEFAULT 512:
                                    # a reference costs (w/16)*(h/16) tokens and attention is
                                    # quadratic, so an unscaled camera photo would cost more
                                    # than the generation itself. Pass 0 for no bound. The
                                    # --strength/--mask init reference is never bounded -- its
                                    # size is pinned by that role; use --ref-resolution-scale for it.
           [--mask <mask.png>]      # WHITE = regenerate, BLACK = preserve the first
                                    # --ref exactly (which must be at the output size);
                                    # greys blend. Omit = regenerate everything.
           [--text-encoder <path>]  # swap the text encoder: an HF directory, or a single
                                    # .safetensors/.gguf FILE. Overrides BRAIN_FLUX2_TE.
                                    # The shape is taken from --variant (klein-4b => Qwen3-4B,
                                    # klein-9b => Qwen3-8B), never from a config.json, so any
                                    # checkpoint with the stock tensor names and shapes drops
                                    # in - a fine-tune, an abliteration, a re-quantisation.
                                    # A checkpoint of a DIFFERENT shape is rejected at load.
           [--model <path>]         # the DiT weights. Overrides BRAIN_FLUX2_DIT. An
                                    # explicit .gguf/.safetensors extension is taken
                                    # literally (the file must exist); without one,
                                    # <name>.gguf then <name>.safetensors are probed beside
                                    # the path; a <vendor>/<repo> id resolves through the
                                    # model store and is DOWNLOADED when no local copy
                                    # exists. VAE/TE/tokenizer still come from env.
           [--adapter <path>]       # LoRA: brain's own `finetune` checkpoint, or a
                                    # third-party ai-toolkit/ComfyUI .safetensors
           [--lora-scale S]         # LoRA strength (ComfyUI strength_model), default 1.0
  finetune <data_dir> --out <adapter.brain> [--variant V] [--steps N] [--rank R] [--lr X]
           [--size S] [--seed K] [--ckpt-every N] [--resume] [--trainer device|host] [--cards N]
           [--text-encoder <path>]
           # Train a LoRA on a folder of captioned images (see data::imageset for
           # the caption formats; `brain label` writes one). The adapter it writes
           # is what `generate --adapter` loads. Do NOT name it '.safetensors':
           # that extension is how --adapter recognises a THIRD-PARTY LoRA.
           #   --size S        square training size in px, multiple of 16 (default 512)
           #   --rank R        LoRA rank (default 16)
           #   --steps N       training steps (default 200)
           #   --lr X          learning rate (default 1e-4)
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
           # Both trainers run the same op sequence; the device one keeps the
           # frozen base on the card and differentiates only the low-rank
           # factors. Which one ran is printed at the top of every run.
Weights (env): BRAIN_FLUX2_DIT, BRAIN_FLUX2_VAE, BRAIN_FLUX2_TE, BRAIN_FLUX2_TOKENIZER
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

/// Long edge, in pixels, a reference is encoded at when the caller does not
/// say. A reference costs `(w/16)*(h/16)` tokens and attention is quadratic in
/// the joint sequence, so an unscaled camera photograph costs several times
/// the generation it is conditioning. Bounding by default is the difference
/// between `--ref holiday.jpg` working and it quietly becoming the most
/// expensive part of the run.
pub const DEFAULT_REF_EDGE: u32 = 512;

/// The long-edge bound for reference `i`, or `None` to encode it at its own
/// resolution.
///
/// `anchored` is true when `refs[0]` seeds the init latent - under
/// `--strength < 1` or `--mask`. That reference is then pinned to the output
/// size and must never be bounded; a caller wanting ITS conditioning cost down
/// has `--ref-resolution-scale`, which exists for exactly this asymmetry.
fn ref_bound(i: usize, anchored: bool, ref_size: Option<u32>) -> Option<u32> {
    if i == 0 && anchored {
        return None;
    }
    match ref_size {
        Some(0) => None, // explicit opt-out
        Some(m) => Some(m),
        None => Some(DEFAULT_REF_EDGE),
    }
}

fn generate(args: &[String]) -> Result<(), String> {
    let mut prompt = None;
    let mut out = None;
    let mut o = GenOpts { width: 512, height: 512, ..GenOpts::default() };
    let (mut want_w, mut want_h): (Option<u32>, Option<u32>) = (None, None);
    let mut variant_name = "klein-4b".to_string();
    let mut precision = flux2::Precision::F32;
    let mut precision_was_explicit = false;
    let mut refs: Vec<String> = Vec::new();
    let mut ref_size: Option<u32> = None;
    let mut mask_path: Option<String> = None;
    let mut adapter: Option<String> = None;
    let mut lora_scale = 1.0f32;
    let mut text_encoder: Option<String> = None;
    let mut model: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> {
            args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--out" => out = Some(need(i)?.clone()),
            "--width" => want_w = Some(need(i)?.parse().map_err(|e| format!("--width: {e}"))?),
            "--height" => want_h = Some(need(i)?.parse().map_err(|e| format!("--height: {e}"))?),
            "--steps" => o.steps = Some(need(i)?.parse().map_err(|e| format!("--steps: {e}"))?),
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
            "--variant" => variant_name = need(i)?.clone(),
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
            "--mask" => mask_path = Some(need(i)?.clone()),
            "--adapter" => adapter = Some(need(i)?.clone()),
            "--lora-scale" => lora_scale = need(i)?.parse().map_err(|e| format!("--lora-scale: {e}"))?,
            "--text-encoder" => text_encoder = Some(need(i)?.clone()),
            "--model" => model = Some(need(i)?.clone()),
            other => return Err(format!("unknown flag {other}\n{HELP}")),
        }
        i += 2;
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    let out = out.ok_or("--out is required")?;
    let variant = Flux2Config::from_name(&variant_name)?;
    flux2::caps::check_license(&variant_name)?; // 9B = FLUX Non-Commercial license

    // load refs as [-1,1] CHW, center-cropped to /16 (shared helper - the
    // capability provider uses the same one), optionally downscaled first so
    // one full-resolution photograph cannot outspend the whole generation.
    // `--ref-size` absent takes the bound-free path, unchanged.
    let mut ref_imgs: Vec<(Vec<f32>, u32, u32)> = Vec::new();
    let anchored = o.strength.is_some_and(|s| s < 1.0) || mask_path.is_some();
    for (i, r) in refs.iter().enumerate() {
        let (hwc, w, h) = crate::image_io::load_image(r)?;
        let bound = ref_bound(i, anchored, ref_size);
        if let Some(m) = bound {
            let (tw, th) = flux2::pipeline::fit_long_edge(w, h, m);
            if (tw, th) != (w, h) {
                eprintln!("flux2: ref {r} {w}x{h} -> resampled to {tw}x{th} (--ref-size {m})");
            }
        } else if i == 0 && anchored {
            eprintln!("flux2: ref {r} {w}x{h} kept at full size - it seeds the init latent");
        }
        ref_imgs.push(flux2::pipeline::ref_from_hwc_bounded(&hwc, w, h, bound)?);
    }

    // The init reference is the canvas, so an anchored run inherits its size
    // rather than making the caller read the file and echo its dimensions.
    let anchor = if anchored { ref_imgs.first().map(|&(_, h, w)| (w, h)) } else { None };
    let (rw, rh) = output_size(want_w, want_h, anchor);
    if (rw, rh) != (o.width, o.height) && anchor.is_some() && (want_w.is_none() || want_h.is_none()) {
        eprintln!("flux2: output {rw}x{rh}, taken from the init reference");
    }
    o.width = rw;
    o.height = rh;

    // The mask is over the OUTPUT canvas, so it is resampled to the latent grid
    // by the pipeline (area average, both axes independently) rather than being
    // required at any particular resolution here.
    if let Some(p) = &mask_path {
        let (hwc, w, h) = crate::image_io::load_image(p)?;
        let m = flux2::Mask::from_hwc(&hwc, w, h)?;
        eprintln!("flux2: mask {p} -> {m:?}");
        o.mask = Some(m);
    }

    // A store id resolves to a canonical name worth printing; a plain path
    // (and the env-provided DiT) IS its own identity. The flag stands in
    // for BRAIN_FLUX2_DIT, so the variable is not required with it present.
    let mut model_name: Option<String> = None;
    let mut dit = None;
    if let Some(m) = &model {
        let resolved = crate::model_flag::resolve(m, "dit")?;
        dit = Some(resolved.path);
        model_name = Some(resolved.name);
    }
    let mut paths = Paths::from_env_with_dit(dit)?;
    if let Some(te) = text_encoder {
        paths.te = te;
    }
    // Q8_0 GGUF is not an fp32 checkpoint with an optional output tier: the
    // FLUX.2 constructor consumes it through its packed DP4A representation.
    // Omitted `--precision` therefore follows the source; an explicit fp32
    // request is rejected rather than silently changing it.
    precision = flux2::pipeline::effective_dit_precision(&paths.dit, precision, precision_was_explicit)?;
    // Which model this run is actually about to load, before anything is
    // loaded.
    eprintln!("flux2: model {} ({}, {})", model_name.as_deref().unwrap_or(&paths.dit), variant_name, precision.name());
    let n_gen = (o.height / 16) * (o.width / 16);
    // Every supplied reference conditions the model; under `--strength` the
    // first one does so at `--ref-resolution-scale` of its own size *and* seeds the
    // init latent. Print the per-reference breakdown, not just the total:
    // reference tokens are what decides whether a run fits the card, and a
    // bare "N + M" does not say which reference spent them or at what size.
    let sizes = flux2::pipeline::cond_sizes(&ref_imgs, &o);
    let n_ref = flux2::pipeline::ref_tokens(&ref_imgs, &o);
    for (i, (size, (_, rh, rw))) in sizes.iter().zip(&ref_imgs).enumerate() {
        let role = if i == 0 && o.strength.is_some_and(|s| s < 1.0) { ", also the init latent" } else { "" };
        match size {
            Some((ch, cw)) => eprintln!(
                "flux2: ref {i} {rw}x{rh} -> conditions at {cw}x{ch} = {} tokens{role}",
                (ch / 16) * (cw / 16)
            ),
            None => eprintln!("flux2: ref {i} {rw}x{rh} -> no conditioning tokens (--ref-resolution-scale 0{role})"),
        }
    }
    eprintln!("flux2: building pipeline ({n_gen} generated + {n_ref} reference tokens) ...");
    let spec = adapter.map(|path| AdapterSpec { path, scale: lora_scale });
    let pipe = Pipeline::build_sized(&variant, &paths, n_gen + n_ref, n_gen, spec.as_ref(), precision, 1)?;
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
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> {
            args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" | "--save" => opts.save_path = need(i)?.clone(),
            "--variant" => variant_name = need(i)?.clone(),
            "--steps" => opts.steps = need(i)?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--rank" => opts.rank = need(i)?.parse().map_err(|e| format!("--rank: {e}"))?,
            "--lr" => opts.lr = need(i)?.parse().map_err(|e| format!("--lr: {e}"))?,
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
    let cfg = Flux2Config::from_name(&variant_name)?;
    flux2::caps::check_license(&variant_name)?; // 9B = FLUX Non-Commercial license
    let mut paths = Paths::from_env()?;
    // Training and generation must be able to name the SAME encoder: an
    // adapter learns against the conditioning it was shown, so training on one
    // encoder and generating on another silently degrades every result.
    if let Some(te) = ft_text_encoder {
        paths.te = te;
    }

    eprintln!(
        "flux2 finetune: {variant_name} {} trainer, rank {} steps {} size {} lr {} seed {} ckpt-every {}{} -> {}",
        opts.trainer.name(), opts.rank, opts.steps, opts.size, opts.lr, opts.seed, opts.ckpt_every,
        if opts.resume { " resume" } else { "" }, opts.save_path
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
mod tests {
    use super::*;

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
mod ref_size_tests {
    use super::ref_bound;

    /// `--ref-size` exists so one full-resolution photograph cannot outspend
    /// the whole generation. It must not touch the **init** reference.
    ///
    /// Under `--strength < 1` (and under `--mask`) `refs[0]` is not merely
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
