// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host trainer forward vs the parity-proven device forward, at **cosine 1.0** -
//! the gate that stops the training path and the serving path from being two
//! different models.
//!
//! The f32 instantiation of [`wan::modelgrad::forward`] (the trainer's own math)
//! must reproduce [`wan::WanDit::forward`], which is itself at cosine
//! 1.000000000 against the reference on the real 1.3B weights
//! (`tests/dit_parity.rs`). That transitively pins the training path's
//! conventions - patch ordering, the modulation fold, RoPE tables, QK-RMSNorm
//! across all heads, the text pad, the head's `e` (not `e0`) - against the
//! checked-in device graph, with no goldens of its own.
//!
//! Both attention implementations are covered: the default device takes flash
//! where it can, and the `cpu` leg takes the query-chunked materialised trio.

use std::collections::HashMap;

use wan::config::WanConfig;
use wan::import::dit_manifest;
use wan::model::{Tensors, WanDit};
use wan::modelgrad::{forward, make_flow_batch, Cfg, ModelWeights};

/// Deliberately non-coincidental: 18 latent tokens against 5 text rows, dim 24
/// against ffn 10, 3 heads of 8, a (3, 2, 3) patch grid. See
/// [`wan::modelgrad::Cfg::tiny`].
fn tiny_cfg() -> WanConfig {
    let c = Cfg::tiny();
    WanConfig {
        name: "tiny-train",
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
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for (name, shape) in dit_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5));
        }
        // Norm gains near 1: a gain of ~0 zeroes everything downstream and
        // hides a real difference behind an all-zero compare.
        if name.contains("norm_q") || name.contains("norm_k") || name.ends_with("norm3.weight") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn compare(label: &str, host: &[f32], dev: &[f32]) {
    assert_eq!(host.len(), dev.len(), "{label}: {} values vs {}", host.len(), dev.len());
    let (mut dot, mut na, mut nb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&a, &b) in host.iter().zip(dev) {
        dot += a as f64 * b as f64;
        na += a as f64 * a as f64;
        nb += b as f64 * b as f64;
        max_abs = max_abs.max((a - b).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
    let rel = (host.iter().zip(dev).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / nb.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("{label}: cosine={cos:.9}  rel_l2={rel:.3e}  max_abs={max_abs:.3e}");
    // Same bar as `tests/dit_parity.rs` reports for the device path itself:
    // cosine 1.0 to nine digits. The residual is fp32 reassociation between a
    // serial host reduction and a blocked device one, nothing else.
    assert!(cos >= 0.999999999, "{label}: cosine {cos:.9} < 1.0");
    assert!(rel <= 1e-5, "{label}: rel_l2 {rel:.3e}");
}

fn run(device: Option<&str>) {
    let wc = tiny_cfg();
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&wc);

    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.3).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 17) as f32 / 17.0 - 0.5) * 0.9).collect();
    let rows = cfg.text_len - 2;
    let ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 1.7).collect();
    let b = make_flow_batch(&cfg, &x0, &ctx, rows, 0.45, &noise);

    let w = ModelWeights::from_tensors(&cfg, &ts).expect("host weights from the manifest");
    let (host, _) = forward(&cfg, &w, &b.latent, &b.ctx, b.t, &b.cos, &b.sin);

    let (f, h, wd) = cfg.latent;
    let m = WanDit::new(wc, ts, device);
    let dev = m.forward(&b.latent, f as u32, h as u32, wd as u32, &ctx, rows, b.t as f32);

    compare(&format!("host f32 trainer vs device ({})", device.unwrap_or("default")), &host, &dev);
}

#[test]
fn host_f32_forward_matches_the_device_forward() {
    run(None);
}

/// The CPU JIT cannot run the flash kernel's barriers, so this leg takes the
/// query-chunked materialised attention instead - the A/B that proves the host
/// reference agrees with BOTH device implementations.
#[test]
fn host_f32_forward_matches_the_chunked_device_forward() {
    run(Some("cpu"));
}
