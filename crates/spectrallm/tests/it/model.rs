// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SpectralLM end-to-end gates, the ones the paper's correctness table
//! reports:
//!
//! - the FFT transform kernels compute the dense basis product;
//! - the whole model gives the same logits on either transform path;
//! - with the SwiGLU mixer the model IS brain's Qwen3 decoder (same weights,
//!   same logits), so the control the spectral mixer is compared against is a
//!   genuine Qwen3 architecture;
//! - incremental decoding reproduces the full-sequence forward;
//! - the CPU backend and the GPU agree;
//! - training drives a fixed small batch to near-zero loss.

use std::collections::HashMap;

use gpu_core::{Dispatch, Gpu};
use model::Model;
use spectrallm::{Mixer, SpectralConfig, SpectralLm, TransformPath};

fn gpu() -> Option<Gpu> {
    if brain_testutil::gpu_tests_skipped() {
        return None;
    }
    Some(gpu_core::testgpu::dev(spectrallm::PIPELINES))
}

fn fft_capable(g: &Gpu) -> bool {
    if g.caps().workgroup_reductions && g.caps().workgroup_mem_bytes >= spectrallm::mixer::FFT_WORKGROUP_BYTES {
        return true;
    }
    brain_testutil::skip_unavailable("the FFT transform kernels need workgroup barriers this device does not run");
    false
}

/// Assert `value <= tol` and print the measurement as one `GATE` line, which
/// the spectral-lm paper's correctness table is generated from.
fn gate(name: &str, metric: &str, value: f64, tol: f64) {
    println!("GATE {}", serde_json::json!({ "name": name, "metric": metric, "value": value, "tolerance": tol, "pass": value <= tol }));
    assert!(value <= tol, "{name}: {metric} {value} exceeds {tol}");
}

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.signed()).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn batch(rows: usize, vocab: u32, salt: u32) -> (Vec<u32>, Vec<u32>) {
    let x: Vec<u32> = (0..rows as u32).map(|i| (i * 7 + salt) % vocab).collect();
    let y: Vec<u32> = (0..rows as u32).map(|i| (i * 7 + salt + 3) % vocab).collect();
    (x, y)
}

/// `rdft_synth` and `rdft_analysis` against `S z` and `Sᵀ x` with the host
/// basis, at every regime the kernels branch on: an odd band (cosine without
/// its sine), the full basis with the Nyquist mode, the largest signal, and
/// more rows than threads in a workgroup.
#[test]
fn fft_kernels_compute_the_basis_products() {
    let Some(g) = gpu() else { return };
    if !fft_capable(&g) {
        return;
    }
    let synth = g.kernel_index("rdft_synth").unwrap();
    let analysis = g.kernel_index("rdft_analysis").unwrap();
    // Row counts that do and do not fill the last workgroup's 2048/D rows.
    for (d, m, rows) in [(32usize, 13usize, 5usize), (64, 64, 3), (512, 256, 301), (2048, 101, 2), (2, 2, 1), (128, 77, 33)] {
        let s = model::hostmath::real_fourier_basis(d, m);
        let z = rand(rows * m, (d * 31 + m) as u64);
        let x = rand(rows * d, (d * 17 + m) as u64);
        let want_x: Vec<f32> = (0..rows).flat_map(|r| (0..d).map(move |j| (r, j))).map(|(r, j)| (0..m).map(|i| s[j * m + i] as f64 * z[r * m + i] as f64).sum::<f64>() as f32).collect();
        let want_z: Vec<f32> = (0..rows).flat_map(|r| (0..m).map(move |i| (r, i))).map(|(r, i)| (0..d).map(|j| s[j * m + i] as f64 * x[r * d + j] as f64).sum::<f64>() as f32).collect();
        let (zb, xb) = (g.storage_init("z", &z), g.storage_init("x", &x));
        let (xo, zo) = (g.storage((rows * d) as u64), g.storage((rows * m) as u64));
        let params = [rows as u32, d as u32, m as u32, (d as u32).ilog2()];
        let wg = Dispatch::Workgroups(spectrallm::mixer::fft_workgroups(rows as u32, d as u32));
        g.submit(&[], &[g.dispatch(synth, &[&zb, &xo], &params, wg), g.dispatch(analysis, &[&xb, &zo], &params, wg)]);
        let (got_x, got_z) = (g.read(&xo, rows * d), g.read(&zo, rows * m));
        gate(&format!("FFT synthesis D={d} m={m}"), "max abs diff vs basis product", max_diff(&got_x, &want_x) as f64, 2e-5);
        gate(&format!("FFT analysis D={d} m={m}"), "max abs diff vs basis product", max_diff(&got_z, &want_z) as f64, 5e-5);
    }
}

/// The same weights give the same logits whichever path computes the
/// transforms.
#[test]
fn fft_and_dense_paths_give_the_same_model() {
    let Some(g) = gpu() else { return };
    if !fft_capable(&g) {
        return;
    }
    let base = SpectralConfig { mixer: Mixer::Spectral { signal_len: 64, channels: 4 }, d_model: 24, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&base, 9);
    let (x, _) = batch(2 * 12, base.vocab, 1);
    let logits = |path| {
        let m = SpectralLm::new_on(g.share(), SpectralConfig { transform: path, ..base.clone() }, 2, 12, &init, false);
        m.logits_all(&x).unwrap()
    };
    let (dense, fft) = (logits(TransformPath::Dense), logits(TransformPath::Fft));
    gate("model logits, FFT vs dense transforms", "max abs diff", max_diff(&dense, &fft) as f64, 1e-4);
}

/// The SwiGLU-mixer model and `qwen3::Qwen` on one weight map.
#[test]
fn swiglu_control_is_the_qwen3_decoder() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = SpectralConfig::tiny_swiglu();
    let Mixer::SwiGlu { d_ff } = cfg.mixer else { unreachable!() };
    let qcfg = qwen3::QwenConfig {
        vocab: cfg.vocab,
        block_size: cfg.block_size,
        n_layers: cfg.n_layers,
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        d_ff,
        rope_theta: cfg.rope_theta,
        rms_eps: cfg.rms_eps,
        max_position_embeddings: cfg.block_size,
        tie_embeddings: true,
        qk_norm: true,
        attn_bias: false,
        lora: None,
        rope_scaling: None,
    };
    let init: HashMap<String, Vec<f32>> = spectrallm::init_weights(&cfg, 4);
    let names: Vec<String> = qcfg.param_list().into_iter().map(|(n, _)| n).collect();
    let mut ours: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
    let mut theirs = names.clone();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs, "the SwiGLU control must have exactly Qwen3's parameters");
    // `Qwen::logits_all` takes one sequence.
    let (x, _) = batch(12, cfg.vocab, 2);
    let ours = SpectralLm::new_inference(cfg, 1, 12, &init).logits_all(&x).unwrap();
    let theirs = qwen3::Qwen::new(qcfg, 1, 12, &init).logits_all(&x);
    gate("SwiGLU control vs brain's Qwen3", "max abs logit diff", max_diff(&ours, &theirs) as f64, 1e-4);
}

/// Token-by-token decoding gives the full-sequence forward's logits at every
/// position, for both mixers.
#[test]
fn incremental_decode_matches_the_full_forward() {
    let Some(g) = gpu() else { return };
    for cfg in [SpectralConfig::tiny(), SpectralConfig::tiny_swiglu()] {
        let init = spectrallm::init_weights(&cfg, 6);
        let t = cfg.block_size;
        let (x, _) = batch(t as usize, cfg.vocab, 5);
        let m = SpectralLm::new_on(g.share(), cfg.clone(), 1, t, &init, false);
        let full = m.logits_all(&x).unwrap();
        let v = cfg.vocab as usize;
        let mut worst = 0.0f32;
        for (pos, tok) in x.iter().enumerate() {
            let step = m.step(*tok);
            worst = worst.max(max_diff(&step, &full[pos * v..(pos + 1) * v]));
        }
        let kind = if matches!(cfg.mixer, Mixer::Spectral { .. }) { "spectral" } else { "SwiGLU" };
        gate(&format!("incremental decode vs full forward ({kind})"), "max abs logit diff", worst as f64, 1e-4);
        assert_eq!(m.cache_pos(), t);
        m.reset_cache();
        assert!(max_diff(&m.step(x[0]), &full[..v]) < 1e-4, "decode after reset restarts at position 0");
    }
}

/// The CPU backend (dense transform, per-element kernels) and the GPU
/// (whatever it selects) compute the same loss and gradients.
#[test]
fn cpu_and_gpu_backends_agree() {
    let Some(g) = gpu() else { return };
    let cfg = SpectralConfig::tiny();
    let init = spectrallm::init_weights(&cfg, 8);
    let (x, y) = batch(2 * 6, cfg.vocab, 3);
    let run = |gpu: Gpu| {
        let m = SpectralLm::new_on(gpu, cfg.clone(), 2, 6, &init, true);
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        m.zero_grads();
        let loss = m.forward();
        m.backward();
        (loss, m.read_grad("blocks.0.mlp.gate_filter"), m.read_grad("tok.weight"))
    };
    let (lc, gc, ec) = run(Gpu::new_cpu(spectrallm::PIPELINES));
    let (lg, gg, eg) = run(g.share());
    gate("CPU vs GPU backend, loss", "abs diff", (lc - lg).abs() as f64, 1e-5);
    gate("CPU vs GPU backend, gradients", "max abs diff", max_diff(&gc, &gg).max(max_diff(&ec, &eg)) as f64, 1e-5);
}

/// A fixed batch is memorized: the optimizer path, every gradient and the
/// loss together make training work, not just agree with finite
/// differences.
#[test]
fn training_memorizes_a_fixed_batch() {
    let Some(g) = gpu() else { return };
    let cfg = SpectralConfig { mixer: Mixer::Spectral { signal_len: 64, channels: 8 }, d_model: 32, n_heads: 4, n_kv_heads: 2, head_dim: 8, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&cfg, 12);
    let m = SpectralLm::new_on(g.share(), cfg.clone(), 4, 12, &init, true);
    let x: Vec<u32> = (0..48u32).map(|i| (i * 13 + i / 12) % cfg.vocab).collect();
    let y: Vec<u32> = (0..48u32).map(|i| (i * 11 + 5) % cfg.vocab).collect();
    m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    let first = m.forward();
    let mut last = first;
    for step in 1..=400 {
        m.zero_grads();
        last = m.forward();
        m.backward();
        m.adamw_step(step, 3e-3, 0.0, model::Adam::default(), Some(1.0), 1.0);
    }
    last = last.min(m.forward());
    assert!(first > 2.5, "an untrained model should be near uniform over 23 tokens, got {first}");
    gate("fixed batch memorized in 400 steps", "final loss (nats)", last as f64, 0.05);
}

/// At a training-length sequence the GPU takes flash attention and the
/// row-cooperative cross-entropy; the CPU backend runs the materialized
/// reference path. Same loss, same gradients.
#[test]
fn flash_attention_path_matches_the_materialized_reference() {
    let Some(g) = gpu() else { return };
    if !fft_capable(&g) {
        return;
    }
    let cfg = SpectralConfig { block_size: 256, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&cfg, 21);
    let (x, y) = batch(256, cfg.vocab, 4);
    let run = |gpu: Gpu| {
        let m = SpectralLm::new_on(gpu, cfg.clone(), 1, 256, &init, true);
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        m.zero_grads();
        let loss = m.forward();
        m.backward();
        (loss, m.read_grad("blocks.1.attn.wq.weight"), m.read_grad("blocks.0.attn.wv.weight"), m.read_grad("tok.weight"))
    };
    let (lc, qc, vc, ec) = run(Gpu::new_cpu(spectrallm::PIPELINES));
    let (lg, qg, vg, eg) = run(g.share());
    gate("flash vs materialized attention, loss", "abs diff", (lc - lg).abs() as f64, 1e-4);
    let mut worst = 0.0f32;
    for (a, b) in [(&qc, &qg), (&vc, &vg), (&ec, &eg)] {
        let scale = a.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        worst = worst.max(max_diff(a, b) / scale);
    }
    gate("flash vs materialized attention, gradients", "max diff / max magnitude", worst as f64, 1e-3);
}

/// Scoring documents in length buckets, each on a model sized for its
/// bucket, gives the same bits per byte as scoring them all at the longest
/// length: every document is scored alone in its row.
#[test]
fn bucketed_scoring_matches_scoring_at_full_length() {
    let Some(g) = gpu() else { return };
    let cfg = SpectralConfig::tiny();
    let init = spectrallm::init_weights(&cfg, 31);
    let docs: Vec<model::score::ByteDoc> =
        [3usize, 5, 12, 7, 2, 11, 9].iter().enumerate().map(|(i, &n)| model::score::ByteDoc { tokens: (0..n as u32).map(|t| (t * 5 + i as u32) % cfg.vocab).collect(), bytes: 3 * n }).collect();
    let full = model::score::score_bytes(&SpectralLm::new_on(g.share(), cfg.clone(), 3, 12, &init, false), &docs, 3, 12).unwrap();
    let bucketed = model::score::score_bytes_bucketed(&docs, 3, &[4, 8, 12], |len| SpectralLm::new_on(g.share(), cfg.clone(), 3, len, &init, false)).unwrap();
    assert_eq!((full.tokens, full.bytes, full.documents), (bucketed.tokens, bucketed.bytes, bucketed.documents));
    gate("length-bucketed scoring vs full-length scoring", "abs diff in bits per byte", (full.bits_per_byte() - bucketed.bits_per_byte()).abs(), 1e-5);
}
