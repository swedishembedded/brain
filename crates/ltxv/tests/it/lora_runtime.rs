// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA adapter reaches LTX inference: loaded from the reference's own
//! ComfyUI/PEFT layout (`diffusion_model.<module>.lora_{A,B}.weight`),
//! applied as `strength * B @ A`, folded exactly on the fp32 path and run
//! beside the packed weights on the quantized one.
//!
//! The quantized gate is the one that matters: the adapter's EFFECT (adapted
//! output minus base output) on the int8 path must point the same way as on
//! the fp32 path. A delta folded into int8 weights would be rounded onto the
//! base weight's grid and mostly lost; a runtime correction is not.

use ltxv::block::QTier;
use ltxv::dit::random_tiny_weights;
use ltxv::lora::runtime_adapter;
use ltxv::modelgrad::Cfg;
use ltxv::{LtxDit, LtxDitConfig};
use model::adapter::AdapterSpec;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn diff(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// A rank-`r` PEFT adapter over `targets` (`(base key, out, in)`), written the
/// way the official trainer exports one: `diffusion_model.` prefix, no alpha.
fn write_peft(path: &std::path::Path, targets: &[(&str, usize, usize)], r: usize, seed: u64) {
    let mut rng = data::rng::Lcg::new(seed);
    let mut tensors = Vec::new();
    for (key, out, inn) in targets {
        let stem = key.strip_suffix(".weight").unwrap();
        tensors.push((format!("diffusion_model.{stem}.lora_A.weight"), vec![r as u64, *inn as u64], rng.vec_scaled(r * inn, 0.5)));
        tensors.push((format!("diffusion_model.{stem}.lora_B.weight"), vec![*out as u64, r as u64], rng.vec_scaled(out * r, 0.5)));
    }
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
}

struct Run {
    latent: Vec<f32>,
    timesteps: Vec<f32>,
    positions: Vec<f32>,
    mask: Vec<f32>,
    context: Vec<f32>,
    context_len: usize,
    t: usize,
}

impl Run {
    fn new(cfg: &LtxDitConfig) -> Run {
        let (t, context_len) = (7usize, 5usize);
        let positions = Cfg::from_ltx(cfg, t, context_len).simple_positions();
        // Random per token: tokens that are near copies of each other make
        // attention weights irrelevant (every value row is the same), and a
        // query/key correction would then have nothing to change.
        let mut rng = data::rng::Lcg::new(0x5A17);
        let latent = rng.vec_scaled(t * cfg.in_channels as usize, 1.0);
        let context = rng.vec_scaled(context_len * cfg.cross_attention_dim as usize, 1.0);
        let timesteps = (0..t).map(|i| 0.2 + 0.05 * (i % 5) as f32).collect();
        let mut mask = vec![0f32; t];
        mask[0] = 1.0;
        Run { latent, timesteps, positions, mask, context, context_len, t }
    }

    fn fp32(&self, m: &LtxDit) -> Vec<f32> {
        m.forward(&self.latent, &self.timesteps, &self.positions, &self.mask, &self.context, self.context_len, self.t, &vec![1.0; self.context_len]).out
    }

    fn int8(&self, m: &LtxDit) -> Vec<f32> {
        m.forward_q(&self.latent, &self.timesteps, &self.positions, &self.mask, &self.context, self.context_len, self.t, &vec![1.0; self.context_len], QTier::Int8).out
    }
}

/// `random_tiny_weights` made to behave like a trained transformer rather
/// than an untrained one, where queries and keys would not matter.
///
/// Its N(0, 0.02) draws put the q/k norm gains near zero (every attention
/// logit ~0, a uniform softmax whatever the queries are) and let a random
/// adaLN shift swamp each token's own signal (every value row about the
/// same, so attention weights are irrelevant). Unit gains, an identity
/// modulation (zero shift and scale) and unit residual gates fix both - and
/// only then can a query or key correction change the output at all.
fn weights(cfg: &LtxDitConfig, seed: u64) -> vae::blocks::Tensors {
    let mut w = random_tiny_weights(cfg, seed);
    let dim = cfg.inner_dim as usize;
    for (k, (_, v)) in w.iter_mut() {
        if k.ends_with("q_norm.weight") || k.ends_with("k_norm.weight") {
            v.iter_mut().for_each(|x| *x = 2.0);
        } else if k == "adaln_single.linear.weight" {
            v.iter_mut().for_each(|x| *x = 0.0);
        } else if k.starts_with("transformer_blocks.") && k.ends_with(".scale_shift_table") {
            // `[9, dim]`: (shift, scale, gate) for attention, the FFN and the
            // text cross-attention.
            for (r, row) in v.chunks_mut(dim).enumerate() {
                row.iter_mut().for_each(|x| *x = if r % 3 == 2 { 1.0 } else { 0.0 });
            }
        }
    }
    w
}

fn targets(cfg: &LtxDitConfig) -> Vec<(&'static str, usize, usize)> {
    let d = cfg.inner_dim as usize;
    vec![
        ("transformer_blocks.0.attn1.to_q.weight", d, d),
        ("transformer_blocks.0.attn2.to_v.weight", d, d),
        ("transformer_blocks.1.attn1.to_out.0.weight", d, d),
        ("transformer_blocks.1.ff.net.2.weight", d, 4 * d),
        // A head target - folded, never quantized.
        ("patchify_proj.weight", d, cfg.in_channels as usize),
    ]
}

/// Every kind of target - self-attention, cross-attention, the output
/// projection, the FFN, and a head linear - changes the output on its own,
/// and the int8 path delivers the same change the exact fp32 fold does, with
/// the adapted int8 output tracking the adapted fp32 one as closely as the
/// unadapted pair. One target per run, so a linear whose correction never
/// reaches the int8 dispatch cannot hide behind another that does.
#[test]
fn a_runtime_adapter_delivers_the_same_effect_on_int8_as_the_exact_fp32_fold() {
    let cfg = LtxDitConfig::tiny();
    let w = weights(&cfg, 0x10A7);
    let run = Run::new(&cfg);
    let base = LtxDit::new(cfg, w.clone(), None);
    let (b32, b8) = (run.fp32(&base), run.int8(&base));
    let dir = tempdir("delivers");
    for (n, target) in targets(&cfg).into_iter().enumerate() {
        let path = dir.join(format!("adapter{n}.safetensors"));
        write_peft(&path, &[target], 4, 3 + n as u64);
        let model_at = |scale: f32| LtxDit::new(cfg, w.clone(), None).with_adapter(runtime_adapter(&[AdapterSpec { path: path.to_string_lossy().into_owned(), scale }]).unwrap()).unwrap();
        // The smallest strength whose fp32 effect is clearly above int8's
        // own noise: some linears (a query projection under a near-uniform
        // attention) move this random model's output very little per unit
        // of delta, and an effect smaller than the int8 rounding cannot be
        // compared between the paths at all.
        let (adapted, a32) = [0.1f32, 1.0, 10.0, 100.0, 1000.0]
            .into_iter()
            .map(|scale| {
                let m = model_at(scale);
                let out = run.fp32(&m);
                (m, out)
            })
            .find(|(_, a32)| rel(a32, &b32) > 0.02)
            .unwrap_or_else(|| panic!("{}: no strength up to 1000 moves the fp32 output by 2% - the adapter is not reaching this linear", target.0));
        let a8 = run.int8(&adapted);
        let size = rel(&a32, &b32);
        let agree = cosine(&diff(&a8, &b8), &diff(&a32, &b32));
        let close = cosine(&a8, &a32);
        println!("{}: |a-b|/|b| {size:.4}, int8-vs-fp32 effect cosine {agree:.6}, adapted int8-vs-fp32 cosine {close:.9}", target.0);
        assert!(size > 1e-3, "{}: the adapter must change the output measurably, rel {size}", target.0);
        assert!(agree > 0.99, "{}: the int8 path must deliver the fp32 fold's effect, cosine {agree}", target.0);
        assert!(close > 0.9999, "{}: the adapted int8 forward must track the adapted fp32 one, cosine {close}", target.0);
    }
}

/// Strength is a multiplier on the whole delta: at strength 0 the adapted
/// model IS the base model, on both paths.
#[test]
fn strength_zero_is_the_base_model() {
    let cfg = LtxDitConfig::tiny();
    let w = weights(&cfg, 0x5EED);
    let dir = tempdir("zero");
    let path = dir.join("adapter.safetensors");
    write_peft(&path, &targets(&cfg), 4, 9);
    let spec = AdapterSpec { path: path.to_string_lossy().into_owned(), scale: 0.0 };
    let run = Run::new(&cfg);
    let base = LtxDit::new(cfg, w.clone(), None);
    let zero = LtxDit::new(cfg, w, None).with_adapter(runtime_adapter(&[spec]).unwrap()).unwrap();
    assert_eq!(run.fp32(&zero), run.fp32(&base));
    assert_eq!(run.int8(&zero), run.int8(&base));
}

/// An adapter for a tensor this model does not have is refused before any
/// weight changes - never silently half-applied.
#[test]
fn an_adapter_for_a_tensor_the_model_lacks_is_refused() {
    let cfg = LtxDitConfig::tiny();
    let dir = tempdir("refused");
    let path = dir.join("adapter.safetensors");
    write_peft(&path, &[("transformer_blocks.0.not_a_module.weight", 8, 8)], 2, 1);
    let adapter = runtime_adapter(&[AdapterSpec::new(path.to_string_lossy())]).unwrap();
    let Err(e) = LtxDit::new(cfg, random_tiny_weights(&cfg, 1), None).with_adapter(adapter) else { panic!("an adapter for a missing tensor must be refused") };
    assert!(e.contains("transformer_blocks.0.not_a_module.weight"), "{e}");
}

/// A file brain's own trainer wrote records `alpha/rank` in its header and
/// it applies on top of the strength; the official layout has no alpha and
/// applies `strength * B @ A` alone.
#[test]
fn a_brain_trained_adapter_applies_its_alpha_over_rank() {
    let dir = tempdir("alpha");
    let mut rng = data::rng::Lcg::new(4);
    let (out, inn, r) = (6usize, 5usize, 2usize);
    let a = rng.vec_scaled(r * inn, 0.5);
    let b = rng.vec_scaled(out * r, 0.5);
    let tensors = vec![
        ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(), vec![r as u64, inn as u64], a.clone()),
        ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight".to_string(), vec![out as u64, r as u64], b.clone()),
    ];
    let official = dir.join("official.safetensors");
    checkpoint::st::save_safetensors(official.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
    let brain = dir.join("brain.safetensors");
    checkpoint::st::save_safetensors(brain.to_str().unwrap(), &tensors, &serde_json::json!({"model": "ltxv-lora", "rank": r, "alpha": 8.0}), None).unwrap();

    let delta = |path: &std::path::Path| -> Vec<f32> {
        let lora = runtime_adapter(&[AdapterSpec { path: path.to_string_lossy().into_owned(), scale: 0.5 }]).unwrap();
        let d = lora.claim("transformer_blocks.0.attn1.to_q.weight", inn, 0, out, 0, inn).expect("the adapter covers to_q");
        let mut w = vec![0f32; out * inn];
        d.fold_into(&mut w);
        w
    };
    let mut want = vec![0f32; out * inn];
    for o in 0..out {
        for i in 0..inn {
            want[o * inn + i] = 0.5 * (0..r).map(|k| b[o * r + k] * a[k * inn + i]).sum::<f32>();
        }
    }
    let got = delta(&official);
    assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-6), "official: strength * B @ A");
    let got = delta(&brain);
    assert!(got.iter().zip(&want).all(|(g, w)| (g - 4.0 * w).abs() < 1e-5), "brain-trained: alpha/rank (8/2) on top");
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ltxv-lora-runtime-{tag}-{}-{}", std::process::id(), data::rng::random_seed()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The real checkpoint's path: every residency shape a [`DitSession`] can
/// take - streaming (every block uploaded per forward), a full resident
/// window, and a window narrower than the model, where blocks are evicted and
/// rebuilt across forwards - applies the adapter identically, bit for bit,
/// across several forwards, and delivers the exact fp32 fold's effect.
#[test]
fn every_residency_shape_applies_the_adapter_and_agrees_bit_for_bit() {
    use ltxv::block::GenerationCache;
    use ltxv::devres::DitSession;
    use ltxv::dit::{forward_q_streamed_in, load_head_tensors_from_source};

    let cfg = LtxDitConfig { num_layers: 4, ..LtxDitConfig::tiny() };
    let w = weights(&cfg, 0xDE5);
    let head = load_head_tensors_from_source(&w, &cfg);
    let d = cfg.inner_dim as usize;
    let dir = tempdir("sessions");
    let path = dir.join("adapter.safetensors");
    write_peft(&path, &[("transformer_blocks.1.attn1.to_v.weight", d, d), ("transformer_blocks.3.ff.net.0.proj.weight", 4 * d, d)], 4, 21);
    let spec = AdapterSpec { path: path.to_string_lossy().into_owned(), scale: 0.3 };
    let shared = || -> ltxv::pipeline::SharedAdapter { std::sync::Arc::new(std::sync::Mutex::new(runtime_adapter(std::slice::from_ref(&spec)).unwrap())) };
    let run = Run::new(&cfg);
    let ctx_valid = vec![1.0f32; run.context_len];
    let forward = |s: &DitSession, cache: &GenerationCache, salt: f32| -> Vec<f32> {
        let latent: Vec<f32> = run.latent.iter().map(|v| v + salt).collect();
        forward_q_streamed_in(s, &cfg, &w, &head, QTier::Int8, &latent, &run.timesteps, &run.positions, &run.mask, &run.context, run.context_len, run.t, &ctx_valid, cache)
    };

    let arms = [
        ("streaming", DitSession::transient(None).with_adapter(Some(shared()))),
        ("full window", DitSession::resident_with_slots(None, cfg.num_layers).with_adapter(Some(shared()))),
        ("narrow window", DitSession::resident_with_slots(None, 2).with_adapter(Some(shared()))),
    ];
    let salts = [0.0f32, 0.07, -0.05];
    let outs: Vec<Vec<Vec<f32>>> = arms.iter().map(|(_, s)| {
        let cache = GenerationCache::default();
        salts.iter().map(|&salt| forward(s, &cache, salt)).collect()
    }).collect();
    for (a, (name, _)) in arms.iter().enumerate().skip(1) {
        assert_eq!(outs[a], outs[0], "{name}: must apply the adapter exactly as streaming does, every forward");
    }

    let base = forward(&DitSession::transient(None), &GenerationCache::default(), 0.0);
    let base32 = run.fp32(&LtxDit::new(cfg, w.clone(), None));
    let adapted32 = run.fp32(&LtxDit::new(cfg, w.clone(), None).with_adapter(runtime_adapter(std::slice::from_ref(&spec)).unwrap()).unwrap());
    let agree = cosine(&diff(&outs[0][0], &base), &diff(&adapted32, &base32));
    println!("session path: adapter effect rel {:.4}, int8-vs-fp32 effect cosine {agree:.6}", rel(&adapted32, &base32));
    assert!(rel(&adapted32, &base32) > 0.01, "the adapter must change the output measurably");
    assert!(agree > 0.99, "the session path must deliver the fp32 fold's effect, cosine {agree}");
}

/// The embeddings connectors run once per prompt (their routing is cached),
/// so the head a generation keeps resident leaves them on the checkpoint and
/// a forward reads them only when it routes a new context. An adapter that
/// corrects a connector linear must still reach that read: the streamed
/// path delivers the exact fp32 fold's effect.
#[test]
fn a_connector_correction_reaches_the_connector_read_on_demand() {
    use ltxv::block::GenerationCache;
    use ltxv::devres::DitSession;
    use ltxv::dit::{forward_q_streamed_in, load_head_tensors_from_source};

    // The int8 tiny width, with its connector on: registers tile the
    // context, so their count divides `Run`'s 5 tokens, and the connector
    // works at the context's own width.
    let cfg = LtxDitConfig {
        use_embeddings_connector: true,
        connector_num_attention_heads: 4,
        connector_attention_head_dim: 16,
        connector_num_learnable_registers: 5,
        ..LtxDitConfig::tiny()
    };
    let w = weights(&cfg, 0xC0C);
    let head = load_head_tensors_from_source(&w, &cfg);
    assert!(head.keys().all(|k| !k.contains("_embeddings_connector.")), "the resident head must leave the connectors on the checkpoint");

    let cd = cfg.connector_inner_dim() as usize;
    let dir = tempdir("connector");
    let path = dir.join("adapter.safetensors");
    write_peft(&path, &[("video_embeddings_connector.transformer_1d_blocks.0.attn1.to_v.weight", cd, cd)], 4, 33);
    // Strong enough that the effect stands clear of int8 rounding, the
    // threshold `a_runtime_adapter_delivers_the_same_effect_on_int8_as_the_
    // exact_fp32_fold` searches its strengths for.
    let spec = AdapterSpec { path: path.to_string_lossy().into_owned(), scale: 4.0 };
    let run = Run::new(&cfg);
    let ctx_valid = vec![1.0f32; run.context_len];
    let forward = |s: &DitSession| -> Vec<f32> {
        forward_q_streamed_in(s, &cfg, &w, &head, QTier::Int8, &run.latent, &run.timesteps, &run.positions, &run.mask, &run.context, run.context_len, run.t, &ctx_valid, &GenerationCache::default())
    };
    let base = forward(&DitSession::transient(None));
    let adapter: ltxv::pipeline::SharedAdapter = std::sync::Arc::new(std::sync::Mutex::new(runtime_adapter(std::slice::from_ref(&spec)).unwrap()));
    let adapted = forward(&DitSession::transient(None).with_adapter(Some(adapter)));

    let base32 = run.fp32(&LtxDit::new(cfg, w.clone(), None));
    let adapted32 = run.fp32(&LtxDit::new(cfg, w.clone(), None).with_adapter(runtime_adapter(std::slice::from_ref(&spec)).unwrap()).unwrap());
    let agree = cosine(&diff(&adapted, &base), &diff(&adapted32, &base32));
    assert!(rel(&adapted32, &base32) > 0.02, "the connector correction must change the output measurably, rel {}", rel(&adapted32, &base32));
    assert!(agree > 0.99, "the on-demand connector read must apply the correction, cosine {agree}");
}
