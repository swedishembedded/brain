// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device gate for wan's runtime LoRA correction: what the RECORDED GRAPH
//! actually computes on an INT8 build, against the fp32 forward of the same
//! adapted model.
//!
//! Swedish Embedded AB implements quantized inference and low-rank adapter
//! deployment for its clients. If your team needs expertise in INT8 weight
//! quantization or LoRA serving then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! `tests/lora_int8_delivery.rs` proves the arithmetic on the host: a fold
//! into an INT8-grid weight rounds the delta onto that grid and loses most of
//! it, and a correction applied beside the weight does not. This file proves
//! that the engine that ships is the second one, end to end - the kernels, the
//! upload, the routing, the scratch - by comparing the ADAPTER'S OWN EFFECT
//! (adapted forward minus base forward) between the fp32 reference and the
//! INT8 engine.
//!
//! The comparison is deliberately in that difference rather than in the raw
//! outputs: INT8 storage has its own error against fp32, which has nothing to
//! do with the adapter and would swamp the thing under test. Subtracting each
//! path's own unadapted forward removes it.

use std::collections::HashMap;

use model::adapter::{BaseStorage, Delivery};
use model::lora::RuntimeLora;
use model::int8::{dequantize_weight, quantize_weight};
use wan::config::WanConfig;
use wan::dev::{WanDitDev, WanDtype};
use wan::import::dit_manifest;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::model::{Tensors, WanDit};
use wan::modelgrad::{grads, make_flow_batch, Cfg, ModelWeights};

const LEAVES: [&str; 10] =
    ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"];

/// INT8 quantizes in groups of 32 along the reduction axis, so both widths
/// must be multiples of one. `head_dim` stays 8, which the inherited
/// `rope_axes` are sized for.
fn tiny_cfg() -> Cfg {
    Cfg { dim: 32, ffn_dim: 64, n_heads: 4, ..Cfg::tiny() }
}

fn tiny_wan(c: &Cfg) -> WanConfig {
    WanConfig {
        name: "tiny-lora-runtime",
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

/// Block weights already on the INT8 grid, so the base tensors requantize
/// bit-exactly and every difference measured below comes from the adapter.
fn q8_grid_weights(cfg: &WanConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x1234_5678_9abc_def0;
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
        if shape.len() == 2 && LEAVES.iter().any(|l| name.ends_with(&format!("{l}.weight"))) {
            let (rows, k) = (shape[0], shape[1]);
            let (p, sw) = quantize_weight(&v, rows, k);
            v = dequantize_weight(&p, &sw, rows, k);
        }
        t.insert(name, (shape, v));
    }
    t
}

fn trained_adapter(cfg: &Cfg, ts: &Tensors, rank: usize, steps: usize) -> LoraAdapter {
    let base = ModelWeights::from_tensors(cfg, ts).expect("host weights");
    let mut ad = LoraAdapter::new(cfg, LoraCfg::new(rank));
    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let rows = cfg.text_len - 1;
    let ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let b = make_flow_batch(cfg, &x0, &ctx, rows, 0.5, &noise);
    for _ in 0..steps {
        let (_l, g) = grads(cfg, &ad.apply(&base), &b);
        ad.step(&g, 5e-3);
    }
    ad
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let nb = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn diff(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// This engine's INT8 tier is DP4A (`matmul_i8_dyn`) and the correction's
/// second GEMM stages through workgroup memory, so both need a device with
/// workgroup barriers. `Sel` gates the whole fast tier on the same cap.
fn gpu_tier_available() -> bool {
    gpu_core::Gpu::open(None, &wan::block::KERNELS).caps().workgroup_reductions
}

struct Fixture {
    cfg: Cfg,
    wc: WanConfig,
    ts: Tensors,
    ad: LoraAdapter,
    latent: Vec<f32>,
    ctx: Vec<f32>,
    ctx_rows: usize,
}

fn fixture(rank: usize, steps: usize) -> Fixture {
    let cfg = tiny_cfg();
    let wc = tiny_wan(&cfg);
    let ts = q8_grid_weights(&wc);
    let ad = trained_adapter(&cfg, &ts, rank, steps);
    let (lf, lh, lw) = cfg.latent;
    let latent: Vec<f32> = (0..cfg.in_channels * lf * lh * lw).map(|i| ((i % 29) as f32 / 29.0 - 0.5) * 1.3).collect();
    let ctx_rows = cfg.text_len - 1;
    let ctx: Vec<f32> = (0..ctx_rows * cfg.text_dim).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 1.2).collect();
    Fixture { cfg, wc, ts, ad, latent, ctx, ctx_rows }
}

impl Fixture {
    /// The base weights with the adapter folded in at `strength` - the dense
    /// tier's delivery, and the fp32 reference's model.
    fn folded(&self, strength: f32) -> Tensors {
        let mut out = self.ts.clone();
        match self.ad.delivery(BaseStorage::Dense, strength).expect("a dense base always has a delivery") {
            Delivery::Fold(f) => f.into_tensors(&mut out).expect("fold"),
            Delivery::Runtime(_) => panic!("a dense base must fold"),
        }
        out
    }

    /// The same adapter at the same strength, as the quantized tier takes it.
    fn runtime(&self, strength: f32) -> RuntimeLora {
        match self.ad.delivery(BaseStorage::Quantized, strength).expect("lora has a runtime form") {
            Delivery::Runtime(d) => d,
            Delivery::Fold(_) => panic!("a quantized base must not fold"),
        }
    }

    fn host_forward(&self, w: &Tensors) -> Vec<f32> {
        let (lf, lh, lw) = self.cfg.latent;
        WanDit::new(self.wc.clone(), w.clone(), None)
            .forward(&self.latent, lf as u32, lh as u32, lw as u32, &self.ctx, self.ctx_rows, 500.0)
    }

    fn dev_forward(&self, w: &Tensors, lora: Option<&RuntimeLora>) -> Vec<f32> {
        let (lf, lh, lw) = self.cfg.latent;
        let d = WanDitDev::build_adapted(&self.wc, w, lf as u32, lh as u32, lw as u32, None, &[], WanDtype::Int8, lora)
            .expect("build");
        d.set_context(&self.ctx, self.ctx_rows);
        d.forward(&self.latent, 500.0)
    }
}

/// **The gate.** The INT8 engine's adapted forward must carry the adapter's
/// own effect, and the folded build must not - measured as each path's
/// adapted-minus-base output difference against the fp32 reference's.
#[test]
fn an_int8_build_delivers_the_adapter_at_runtime_where_a_fold_loses_it() {
    if !gpu_tier_available() {
        brain_testutil::skip_unavailable("wan's int8 tier (DP4A) and the LoRA correction kernel both need workgroup barriers");
        return;
    }
    let fx = fixture(4, 3);
    // A realistic trained-adapter magnitude. `strength` is the inference dial,
    // and it sets the one ratio that decides whether a fold survives: the
    // delta's size against one INT8 step. A real adapter's delta is a fraction
    // of a percent of the weight magnitude, i.e. well under half a step - so
    // the realistic setting here is a SMALL strength, not 1.0
    // (`tests/lora_int8_delivery.rs` sweeps the whole range).
    let strength = 0.05;
    let folded = fx.folded(strength);

    // The intended effect, in fp32: adapted minus base.
    let want = diff(&fx.host_forward(&folded), &fx.host_forward(&fx.ts));

    // The INT8 base - the same weights both adapted builds start from.
    let base_i8 = fx.dev_forward(&fx.ts, None);
    // What ships now: the correction beside untouched weights.
    let got_runtime = diff(&fx.dev_forward(&fx.ts, Some(&fx.runtime(strength))), &base_i8);
    // What shipped before: the delta folded in, then quantized.
    let got_fold = diff(&fx.dev_forward(&folded, None), &base_i8);

    let (c_rt, c_fold) = (cosine(&got_runtime, &want), cosine(&got_fold, &want));
    let norm = |v: &[f32]| (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64).sqrt();
    let (n_rt, n_fold, n_want) = (norm(&got_runtime), norm(&got_fold), norm(&want));
    eprintln!(
        "adapter effect on an int8 wan forward: runtime cosine {c_rt:.6} (rms {n_rt:.3e}), folded cosine {c_fold:.6} (rms {n_fold:.3e}), intended rms {n_want:.3e}"
    );

    // The runtime path delivers the adapter in both direction and size.
    //
    // The residual against 1.0 is NOT a defect in the correction, which is
    // computed in fp32 from the same `A`/`B` the reference folds: it is that
    // the two forwards evaluate the same delta at two different operating
    // points. INT8 weight storage moves the base forward a little, and a
    // network with attention and GELU in it responds to the same perturbation
    // slightly differently there. At a realistic adapter magnitude that
    // second-order difference is a visible fraction of a first-order effect
    // this small, which is why the bar is 0.9 and not 0.999 - the number that
    // carries the claim is the GAP to the folded path.
    assert!(c_rt > 0.9, "the int8 engine did not deliver the adapter's effect (cosine {c_rt:.6})");
    assert!((n_rt / n_want - 1.0).abs() < 0.2, "the delivered effect is the wrong SIZE: rms {n_rt:.3e} against an intended {n_want:.3e}");

    // The fold, on the same weights and the same adapter, delivers almost
    // none of it - most of the delta never reached a weight at all.
    assert!(c_fold < 0.5, "the fold no longer reproduces the defect (cosine {c_fold:.6}) - fixture is not in the sub-step regime");
    assert!(n_fold < 0.5 * n_want, "the fold kept {n_fold:.3e} of an intended {n_want:.3e} - fixture is not in the sub-step regime");
}

/// **The no-regression guarantee, at the bit.** A runtime correction of
/// strength zero contributes exactly nothing, so an adapted build must
/// reproduce the unadapted build's output EXACTLY. That is only true if the
/// base weights were uploaded untouched - which is the property the whole
/// mechanism rests on, and the one a fold destroys.
#[test]
fn a_zero_strength_correction_reproduces_the_unadapted_int8_build_bit_for_bit() {
    if !gpu_tier_available() {
        brain_testutil::skip_unavailable("wan's int8 tier (DP4A) needs workgroup barriers");
        return;
    }
    let fx = fixture(4, 3);
    let zero = fx.runtime(0.0);
    assert_eq!(zero.len(), fx.cfg.n_layers * LEAVES.len(), "a zero-strength delivery still names every target");

    let plain = fx.dev_forward(&fx.ts, None);
    let adapted = fx.dev_forward(&fx.ts, Some(&zero));
    for (i, (a, b)) in plain.iter().zip(&adapted).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "output {i} moved on a zero-strength adapter: {a} vs {b}");
    }
}

/// A dense build has no runtime path here - it folds, exactly. Offering it a
/// runtime correction is a caller bug, and must be refused by name rather than
/// silently ignored (which would serve base-model video from an adapted run).
#[test]
fn a_dense_build_refuses_a_runtime_correction() {
    let fx = fixture(2, 1);
    let (lf, lh, lw) = fx.cfg.latent;
    let Err(err) = WanDitDev::build_adapted(&fx.wc, &fx.ts, lf as u32, lh as u32, lw as u32, None, &[], WanDtype::F32, Some(&fx.runtime(1.0)))
    else {
        panic!("a dense build must refuse a runtime correction")
    };
    assert!(err.contains("fold"), "the refusal must say what to do instead, got {err:?}");
}
