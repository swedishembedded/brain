// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The three parts of the LoRA gate: **exact no-op at init**, **measured
//! descent**, and **fold-vs-apply bit-equality**. Plus the base staying frozen
//! and a save/load round trip.
//!
//! Bit-equality (not "close") is the right bar here because Wan fuses nothing:
//! every adapter pair covers a whole `[out, in]` tensor at offset 0, so
//! `apply` (add into the host training weights) and `fold_into_tensors` (add
//! into the inference tensor map) perform the identical additions in the
//! identical order. Any difference at all would mean the two walks disagree
//! about which tensor is which - the defect that trains `k` into `q` and still
//! produces plausible video.

use std::collections::HashMap;
use std::path::Path;

use wan::config::WanConfig;
use wan::import::dit_manifest;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::model::Tensors;
use wan::modelgrad::{grads, make_flow_batch, Batch, Cfg, ModelWeights};

fn tiny_wan(c: &Cfg) -> WanConfig {
    WanConfig {
        name: "tiny-lora",
        dim: c.dim,
        ffn_dim: c.ffn_dim,
        num_heads: c.n_heads,
        num_layers: c.n_layers,
        in_channels: c.in_channels,
        out_channels: c.out_channels,
        text_dim: c.text_dim,
        text_len: c.text_len,
        freq_dim: c.freq_dim,
        ..WanConfig::t2v_1_3b()
    }
}

fn synthetic_weights(cfg: &WanConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for (name, shape) in dit_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5));
        }
        if name.contains("norm_q") || name.contains("norm_k") || name.ends_with("norm3.weight") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn fixed_batch(cfg: &Cfg) -> Batch<f32> {
    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let rows = cfg.text_len - 1;
    let ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    make_flow_batch(cfg, &x0, &ctx, rows, 0.5, &noise)
}

#[test]
fn lora_is_a_no_op_at_init_then_descends_with_the_base_frozen() {
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&tiny_wan(&cfg));
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));

    // 1. exact no-op at init: B = 0, so W_eff is the base BIT for BIT.
    assert!(ad.apply(&base) == base, "a fresh adapter must not change a single weight");

    let b = fixed_batch(&cfg);
    let (l0, _) = grads(&cfg, &ad.apply(&base), &b);
    let mut last = l0;
    for step in 0..40 {
        let w_eff = ad.apply(&base);
        let (l, g) = grads(&cfg, &w_eff, &b);
        ad.step(&g, 3e-3);
        if step % 10 == 0 {
            println!("  lora step {step:>3}  loss {l:.6}");
        }
        last = l;
    }
    println!("lora: loss {l0:.6} -> {last:.6} over 40 steps (rank 4, lr 3e-3)");
    assert!(last < l0 * 0.9, "LoRA training must descend: {l0} -> {last}");

    // 2. the base is frozen: `apply` clones, so the original is untouched.
    let base_again = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    assert!(base == base_again, "the base weights must not move during LoRA training");
}

#[test]
fn folding_into_the_inference_tensors_equals_applying_to_the_training_weights() {
    let cfg = Cfg::tiny();
    let wc = tiny_wan(&cfg);
    let ts = synthetic_weights(&wc);
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");

    // Train a few steps so B is non-zero - folding zeros would prove nothing.
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));
    let b = fixed_batch(&cfg);
    for _ in 0..3 {
        let (_l, g) = grads(&cfg, &ad.apply(&base), &b);
        ad.step(&g, 5e-3);
    }

    let applied = ad.apply(&base);
    let mut folded_ts = ts.clone();
    ad.fold_into_tensors(&mut folded_ts).expect("fold");
    let folded = ModelWeights::from_tensors(&cfg, &folded_ts).expect("host weights");
    assert!(applied == folded, "fold-into-tensors and apply-to-weights must be bit-equal");
    // ...and the adapter really did move something (guards against a
    // vacuously-true comparison of two untouched copies).
    assert!(applied != base, "3 LoRA steps must have moved the effective weights");

    // A missing base tensor is an error BY NAME, not a silent skip.
    let mut broken = ts.clone();
    broken.remove("blocks.1.cross_attn.v.weight");
    let e = ad.fold_into_tensors(&mut broken).expect_err("a missing tensor must fail");
    assert!(e.contains("blocks.1.cross_attn.v.weight"), "{e}");
}

#[test]
fn an_adapter_round_trips_through_the_checkpoint_container() {
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&tiny_wan(&cfg));
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));
    let b = fixed_batch(&cfg);
    for _ in 0..2 {
        let (_l, g) = grads(&cfg, &ad.apply(&base), &b);
        ad.step(&g, 5e-3);
    }
    let dir = std::env::temp_dir().join(format!("wan-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("adapter.brain");
    let p = path.to_str().expect("utf-8 path");
    wan::lora::save_adapter(p, &ad).expect("save");
    let back = wan::lora::load_adapter(p, &cfg).expect("reload");
    assert_eq!(back.rank(), ad.rank());
    assert!(ad.apply(&base) == back.apply(&base), "a reloaded adapter must produce the same effective weights");
    let _ = std::fs::remove_dir_all(&dir);
}

// ================= G1: held-out loss (real weights, real data, SMOKE scale) =================
//
// Trains a real adapter on a HANDFUL of the procedural concept clips
// (`data::gen_clips`) and checks the flow-matching loss on (a) a few
// held-out concept clips (never trained on) and (b) a few distractor clips
// (a different shape/colour/motion). A concept-only LoRA should lower (a)
// more than (b) - it was never shown the distractor at all.
//
// This is a SMOKE-scale gate, not a statistically powered one: real umT5-XXL
// (CPU) + the real 1.3B DiT are both genuinely expensive per call, so
// everything here is sized to finish in low single-digit minutes on this
// host - ONE umT5 session (not two: the training loop is hand-rolled with
// the exact math `finetune::run` uses, `grads`+`adapter.step`, rather than
// calling `finetune::run` and paying a second umT5 load for its own internal
// encode), a handful of clips, and few steps. The full-power version of this
// same comparison, across many (prompt, seed) pairs with a matched-norm
// random-adapter control and a significance test, is G2
// (`tests/finetune_ab.rs`) - this test's job is just "does the real training
// path move the real held-out loss the right direction, fast".

/// `BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}` or a loud skip - see `pipeline::Paths`'s
/// doc on why the path is an env var, never a literal.
fn real_paths() -> Option<wan::Paths> {
    match wan::Paths::from_env() {
        Ok(p) => Some(p),
        Err(e) => {
            brain_testutil::skip(&format!("set BRAIN_WAN_{{DIT,VAE,T5,TOKENIZER}} to run the real-weight G1 gate: {e}"));
            None
        }
    }
}

fn read_pth(path: &str) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    Ok(checkpoint::torchpt::read(path)?.into_iter().map(|t| checkpoint::safetensors::StTensor { name: t.name, shape: t.shape, data: t.data }).collect())
}

/// Mean flow-matching loss of `adapter.apply(base)` over `(latent, ctx)`
/// pairs, at FIXED per-sample `(sigma, noise)` draws - so a before/after
/// comparison never confounds "the adapter changed" with "a different noise
/// draw landed".
fn mean_loss(tcfg: &Cfg, base: &ModelWeights<f32>, adapter: &LoraAdapter, samples: &[(Vec<f32>, Vec<f32>)], sigmas: &[f64], noises: &[Vec<f32>]) -> f64 {
    let w = adapter.apply(base);
    let total: f64 = samples
        .iter()
        .enumerate()
        .map(|(i, (latent, ctx))| {
            let rows = ctx.len() / tcfg.text_dim;
            let b = make_flow_batch(tcfg, latent, ctx, rows, sigmas[i], &noises[i]);
            grads(tcfg, &w, &b).0
        })
        .sum();
    total / samples.len() as f64
}

#[test]
fn a_concept_only_lora_lowers_held_out_concept_loss_more_than_distractor_loss() {
    let Some(paths) = real_paths() else { return };
    let cfg = WanConfig::t2v_1_3b();
    let (frames, size) = (5usize, 64u32);
    // Minimised hard against the umT5-XXL CPU floor: one import (~3 min,
    // fixed, dominated by converting the 11GB bf16 checkpoint to fp32) plus
    // roughly a minute per UNIQUE caption forward. Every extra sample here is
    // a real extra minute, so this stays as small as a before/after
    // comparison can be: 2 training windows, 1 held-out-concept eval window,
    // 1 distractor eval window - 4 captions total.
    let (n_train, n_eval) = (2usize, 1usize);

    let base_dir = std::env::temp_dir().join(format!("wan-g1-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base_dir);
    let concept = data::gen_clips::generate_concept_set(n_train + n_eval, frames, size, size, 101);
    let distractor = data::gen_clips::generate_distractor_set(n_eval, frames, size, size, 202);
    let (train_c, heldout_c) = concept.split_at(n_train);
    let (train_dir, heldout_dir, distractor_dir) = (base_dir.join("train"), base_dir.join("heldout"), base_dir.join("distractor"));
    data::videoset::write_clipset(&train_dir, train_c, size, size, 8).expect("train clips");
    data::videoset::write_clipset(&heldout_dir, heldout_c, size, size, 8).expect("heldout clips");
    data::videoset::write_clipset(&distractor_dir, &distractor, size, size, 8).expect("distractor clips");

    let train_set = wan::finetune::ClipSet::load_dir(&train_dir).expect("train set");
    let heldout_set = wan::finetune::ClipSet::load_dir(&heldout_dir).expect("heldout set");
    let distractor_set = wan::finetune::ClipSet::load_dir(&distractor_dir).expect("distractor set");
    let mut drng = data::rng::Rng::new(777);
    let train_clips: Vec<wan::finetune::Clip> = (0..n_train).map(|_| train_set.sample(&mut drng, frames).expect("sample")).collect();
    let heldout_clips: Vec<wan::finetune::Clip> = (0..n_eval).map(|_| heldout_set.sample(&mut drng, frames).expect("sample")).collect();
    let distractor_clips: Vec<wan::finetune::Clip> = (0..n_eval).map(|_| distractor_set.sample(&mut drng, frames).expect("sample")).collect();

    let (lf, lh, lw) = cfg.latent_shape(frames, size as usize, size as usize).expect("latent shape");
    let tcfg = Cfg::from_wan(&cfg, lf, lh, lw);
    let t0 = std::time::Instant::now();

    // ---- ONE umT5 session: train + held-out-concept + distractor captions ----
    let ctxs: Vec<Vec<f32>> = {
        let tok = if Path::new(&paths.tokenizer).is_dir() {
            data::unigram::UnigramTokenizer::from_dir(&paths.tokenizer)
        } else {
            data::unigram::UnigramTokenizer::from_file(&paths.tokenizer)
        }
        .expect("tokenizer");
        let t5cfg = t5encoder::config::T5Config::umt5_xxl();
        let imported = t5encoder::import::import_wan(read_pth(&paths.t5).expect("read t5"), &t5cfg).expect("import t5");
        let gpu = gpu_core::Gpu::new_cpu(t5encoder::model::PIPELINES);
        let enc = t5encoder::model::T5Encoder::new_on(gpu, t5cfg, 1, cfg.text_len as u32, &t5encoder::import::to_init(imported));
        train_clips
            .iter()
            .chain(&heldout_clips)
            .chain(&distractor_clips)
            .map(|c| {
                let (ids, mask) = tok.encode_padded(&c.caption, cfg.text_len);
                enc.set_tokens(&ids);
                enc.set_mask(&mask);
                enc.forward();
                enc.poll_wait();
                enc.read_context()
            })
            .collect()
    }; // umT5 dropped here - the ONE expensive load this test pays

    // ---- ONE Wan-VAE session: encode all clips to latents ----
    let latents: Vec<Vec<f32>> = {
        let vcfg = wan::vae3d::WanVaeConfig::wan21();
        let vweights = wan::import::import_vae(read_pth(&paths.vae).expect("read vae"), &vcfg).expect("import vae");
        let enc = wan::vae3d::WanVaeEncoder::build(&vcfg, &vweights, &vcfg.encode_chunks(frames as u32), size, size, None);
        train_clips.iter().chain(&heldout_clips).chain(&distractor_clips).map(|c| enc.encode(&c.video)).collect()
    };
    println!("G1: umT5 + VAE encode ({} clips) in {:.1}s", latents.len(), t0.elapsed().as_secs_f32());

    let samples: Vec<(Vec<f32>, Vec<f32>)> = latents.into_iter().zip(ctxs).collect();
    let (train_s, rest) = samples.split_at(n_train);
    let (heldout_s, distractor_s) = rest.split_at(n_eval);

    let mut ern = data::rng::Rng::new(555);
    let sigmas: Vec<f64> = (0..2 * n_eval).map(|_| (ern.next_f64()).clamp(1e-3, 1.0)).collect();
    let noises: Vec<Vec<f32>> = heldout_s.iter().chain(distractor_s).map(|(latent, _)| (0..latent.len()).map(|_| ern.next_gaussian() as f32).collect()).collect();
    let (c_sigmas, d_sigmas) = sigmas.split_at(n_eval);
    let (c_noises, d_noises) = noises.split_at(n_eval);

    let t1 = std::time::Instant::now();
    let raw = checkpoint::safetensors::read(&paths.dit).expect("read DiT safetensors");
    let tensors = wan::import::import_dit(raw, &cfg).expect("import DiT");
    let base = ModelWeights::from_tensors(&tcfg, &tensors).expect("host weights");
    println!("G1: DiT import in {:.1}s", t1.elapsed().as_secs_f32());

    let rank = 8;
    let mut adapter = LoraAdapter::new(&tcfg, LoraCfg::new(rank));
    let fresh = LoraAdapter::new(&tcfg, LoraCfg::new(rank));

    let before_concept = mean_loss(&tcfg, &base, &fresh, heldout_s, c_sigmas, c_noises);
    let before_distractor = mean_loss(&tcfg, &base, &fresh, distractor_s, d_sigmas, d_noises);

    // ---- hand-rolled training loop: the exact math `finetune::run` uses ----
    let steps = 15u32;
    let mut trng = data::rng::Rng::new(4242);
    let t2 = std::time::Instant::now();
    for _ in 0..steps {
        let idx = trng.gen_range_inclusive(0, train_s.len() as i64 - 1) as usize;
        let (latent, ctx) = &train_s[idx];
        let sigma = (trng.next_f64()).clamp(1e-3, 1.0);
        let noise: Vec<f32> = (0..latent.len()).map(|_| trng.next_gaussian() as f32).collect();
        let b = make_flow_batch(&tcfg, latent, ctx, cfg.text_len, sigma, &noise);
        let (_l, g) = grads(&tcfg, &adapter.apply(&base), &b);
        adapter.step(&g, 3e-3);
    }
    println!("G1: trained {steps} steps in {:.1}s ({:.2}s/step)", t2.elapsed().as_secs_f32(), t2.elapsed().as_secs_f32() / steps as f32);

    let after_concept = mean_loss(&tcfg, &base, &adapter, heldout_s, c_sigmas, c_noises);
    let after_distractor = mean_loss(&tcfg, &base, &adapter, distractor_s, d_sigmas, d_noises);

    let concept_drop = before_concept - after_concept;
    let distractor_drop = before_distractor - after_distractor;
    println!("G1: held-out concept loss {before_concept:.6} -> {after_concept:.6} (drop {concept_drop:.6})");
    println!("G1: distractor loss       {before_distractor:.6} -> {after_distractor:.6} (drop {distractor_drop:.6})");
    println!("G1: total wall time {:.1}s", t0.elapsed().as_secs_f32());

    assert!(concept_drop > 0.0, "held-out concept loss must fall during training: {before_concept} -> {after_concept}");
    assert!(
        concept_drop > distractor_drop,
        "held-out concept loss must fall MORE than the distractor's: concept drop {concept_drop:.6} vs distractor drop {distractor_drop:.6}"
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}
