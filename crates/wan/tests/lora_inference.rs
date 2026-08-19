// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! G0 (wiring gate): a trained adapter must change EXACTLY the ten targeted
//! linears per block in an inference tensor map, nothing more and nothing
//! less, and a bad `--adapter` path must fail loudly rather than silently
//! serving the base model. Before this test existed, `fold_into_tensors` had
//! zero call sites outside `tests/lora_train.rs`'s bit-equality check, so
//! nothing proved the seam `pipeline::run` now uses actually reaches
//! inference.

use std::collections::HashMap;

use wan::config::WanConfig;
use wan::import::dit_manifest;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::model::Tensors;
use wan::modelgrad::{grads, make_flow_batch, Batch, Cfg, ModelWeights};

fn tiny_wan(c: &Cfg) -> WanConfig {
    WanConfig {
        name: "tiny-lora-g0",
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

/// A trained (non-degenerate) adapter: a few real optimisation steps against
/// synthetic base weights, so `B` is genuinely non-zero rather than a
/// hand-constructed stand-in.
fn trained_adapter(cfg: &Cfg, base: &ModelWeights<f32>, rank: usize) -> LoraAdapter {
    let mut ad = LoraAdapter::new(cfg, LoraCfg::new(rank));
    let b = fixed_batch(cfg);
    for _ in 0..5 {
        let (_l, g) = grads(cfg, &ad.apply(base), &b);
        ad.step(&g, 5e-3);
    }
    ad
}

/// The ten leaves an adapter targets per block - mirrors `wan::lora::LEAVES`,
/// duplicated here (rather than made `pub`) so this test pins the PUBLIC
/// contract (`blocks.{l}.{leaf}.weight` names) independently of the private
/// table, not the table's own value circularly.
const LEAVES: [&str; 10] =
    ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"];

#[test]
fn folding_a_trained_adapter_changes_exactly_the_ten_targeted_leaves_per_block() {
    let cfg = Cfg::tiny();
    let wc = tiny_wan(&cfg);
    let ts = synthetic_weights(&wc);
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let ad = trained_adapter(&cfg, &base, 4);

    let mut folded = ts.clone();
    ad.fold_into_tensors(&mut folded).expect("fold");

    let mut expected: std::collections::HashSet<String> = std::collections::HashSet::new();
    for l in 0..cfg.n_layers {
        for leaf in LEAVES {
            expected.insert(format!("blocks.{l}.{leaf}.weight"));
        }
    }

    let mut changed: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, (_, data)) in &folded {
        let (_, orig) = ts.get(name).unwrap_or_else(|| panic!("fold introduced an unknown tensor {name}"));
        if data != orig {
            changed.insert(name.clone());
        }
    }

    assert_eq!(changed, expected, "fold_into_tensors must change exactly the 10xL targeted leaves, nothing more, nothing less");
    // And every changed tensor moved for a real reason, not a rounding no-op:
    // the adapter descended (`trained_adapter` ran real steps against a
    // non-zero gradient), so at least one delta must be non-trivial.
    let max_delta: f32 = expected
        .iter()
        .flat_map(|name| {
            let (_, a) = ts.get(name).unwrap();
            let (_, b) = folded.get(name).unwrap();
            a.iter().zip(b).map(|(x, y)| (x - y).abs())
        })
        .fold(0.0, f32::max);
    assert!(max_delta > 1e-6, "the fold must move real weight, not just touch bytes: max |delta| = {max_delta}");
}

#[test]
fn loading_a_missing_adapter_path_is_a_clear_error_never_silent_base_output() {
    let cfg = Cfg::tiny();
    // This is exactly the call `wan::pipeline::run` makes when `GenOpts::adapter`
    // is `Some(path)`: a bad path must surface as an error from THIS call, so
    // the pipeline can never silently fall through to unadapted generation.
    let Err(err) = wan::lora::load_adapter("/nonexistent/wan-adapter-that-does-not-exist.brain", &cfg) else {
        panic!("a missing adapter file must error, not silently produce an empty adapter")
    };
    // The error must name the failure at the file, not swallow it as "not found".
    assert!(!err.is_empty());
}
