// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA fine-tuning for the RVQ depth decoder's 7 per-layer linear
//! projections (`attn.{to_q,to_k,to_v,to_out}`, `{gate,up,down}_proj`).
//! Reuses `crate::lora`'s adapter math (`LoraW`/`delta`/`apply`/`backward`)
//! unchanged - that math only ever needed a flat `[rows, cols]` weight and
//! never referenced the vocoder's own types, so it is exactly as
//! applicable here. Fold-then-run, same as the vocoder: `W_eff = W_base +
//! (alpha/r)*B@A` is composed on the host each step and handed to the
//! ordinary `depth_decoder::forward`/`backward`, which run completely
//! unaware an adapter exists.

use crate::config::DepthDecoderConfig;
use crate::depth_decoder::{backward as dd_backward, forward as dd_forward, BlockW, DepthDecoderWeights};
use crate::lora::LoraW;
use std::collections::HashMap;

/// `(rows, cols)` of one per-layer linear weight's own storage layout.
pub fn linear_shape(cfg: &DepthDecoderConfig, name: &str) -> (usize, usize) {
    let d = cfg.hidden_size as usize;
    let inter = cfg.intermediate_size as usize;
    let suffix = name.rsplit('.').next().unwrap();
    match suffix {
        "to_q" | "to_k" | "to_v" | "to_out" => (d, d),
        "gate_proj" | "up_proj" => (inter, d),
        "down_proj" => (d, inter),
        _ => panic!("depth_lora::linear_shape: unknown linear name {name:?}"),
    }
}

/// `base` with every adapter in `adapters` applied.
pub fn effective_weights(base: &DepthDecoderWeights, adapters: &HashMap<String, LoraW>, scale: f32) -> DepthDecoderWeights {
    let mut eff = base.clone();
    for (name, w) in adapters {
        let slot = eff.linear_mut(name).unwrap_or_else(|| panic!("depth_lora: {name:?} is not a linear weight"));
        crate::lora::apply(slot, w, scale);
    }
    eff
}

/// The gradient of one named linear weight out of `backward`'s per-layer
/// `d_layers`, matching `DepthDecoderWeights::linear_mut`'s naming.
fn read_layer_grad<'a>(d_layers: &'a [BlockW], name: &str) -> &'a Vec<f32> {
    let rest = name.strip_prefix("layers.").expect("depth_lora: name must start with layers.");
    let (i, rest) = rest.split_once('.').expect("depth_lora: malformed name");
    let layer = &d_layers[i.parse::<usize>().expect("depth_lora: bad layer index")];
    match rest {
        "attn.to_q" => &layer.attn.wq,
        "attn.to_k" => &layer.attn.wk,
        "attn.to_v" => &layer.attn.wv,
        "attn.to_out" => &layer.attn.wo,
        "gate_proj" => &layer.mlp.gate,
        "up_proj" => &layer.mlp.up,
        "down_proj" => &layer.mlp.down,
        _ => panic!("depth_lora: unknown linear name {name:?}"),
    }
}

/// `(dA, dB)` per named adapter.
pub type LoraGrads = HashMap<String, (Vec<f32>, Vec<f32>)>;

/// One LoRA-only forward+backward step: `loss` (an MSE-against-`target`
/// reconstruction loss, matching `train::Trainer`'s own gradcheck-only
/// loss - not a real training objective) and `(dA, dB)` for every adapter.
pub fn step(cfg: &DepthDecoderConfig, base: &DepthDecoderWeights, adapters: &HashMap<String, LoraW>, scale: f32, inputs_embeds: &[f32], s: usize, target: &[f32]) -> (f32, LoraGrads) {
    let eff = effective_weights(base, adapters, scale);
    let (out, cache) = dd_forward(&eff, cfg, inputs_embeds, s);
    let n = out.len() as f32;
    let loss = out.iter().zip(target).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (2.0 * n);
    let dy: Vec<f32> = out.iter().zip(target).map(|(a, b)| (a - b) / n).collect();
    let (_, d_layers, _, _) = dd_backward(&eff, cfg, &cache, &dy);

    let grads = adapters
        .iter()
        .map(|(name, w)| {
            let d_w_eff = read_layer_grad(&d_layers, name);
            (name.clone(), crate::lora::backward(w, d_w_eff, scale))
        })
        .collect();
    (loss, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth_decoder::AttnW;
    use crate::depth_decoder::MlpW;
    use data::rng::Lcg;

    fn random_weights(cfg: &DepthDecoderConfig, seed: u64) -> DepthDecoderWeights {
        let mut r = Lcg::new(seed);
        let d = cfg.hidden_size as usize;
        let inter = cfg.intermediate_size as usize;
        let lin = |out: usize, inn: usize, r: &mut Lcg| r.vec_scaled(out * inn, 0.2);
        let layers = (0..cfg.num_layers as usize)
            .map(|_| BlockW {
                ln1: vec![1.0; d],
                attn: AttnW { wq: lin(d, d, &mut r), wk: lin(d, d, &mut r), wv: lin(d, d, &mut r), wo: lin(d, d, &mut r) },
                ln2: vec![1.0; d],
                mlp: MlpW { gate: lin(inter, d, &mut r), up: lin(inter, d, &mut r), down: lin(d, inter, &mut r) },
            })
            .collect();
        DepthDecoderWeights {
            audio_embeddings: lin((cfg.audio_vocab_size * (cfg.num_codebooks - 1)) as usize, d, &mut r),
            projection: lin(d, d, &mut r),
            pos_embedding: lin(cfg.max_position_embeddings as usize, d, &mut r),
            layers,
            norm: vec![1.0; d],
            audio_heads: (0..cfg.num_codebooks as usize - 1).map(|_| lin(cfg.audio_vocab_size as usize, d, &mut r)).collect(),
        }
    }

    fn rank2_adapters(cfg: &DepthDecoderConfig, base: &DepthDecoderWeights, seed: u64) -> HashMap<String, LoraW> {
        let mut r = Lcg::new(seed);
        base.linear_names()
            .into_iter()
            .map(|name| {
                let (rows, cols) = linear_shape(cfg, &name);
                (name, LoraW::zero_init(rows, cols, 2, |_| r.signed() * 0.1))
            })
            .collect()
    }

    #[test]
    fn zero_b_is_an_exact_no_op() {
        let cfg = DepthDecoderConfig::tiny();
        let base = random_weights(&cfg, 1);
        let adapters = rank2_adapters(&cfg, &base, 2);
        let eff = effective_weights(&base, &adapters, 1.0);
        for name in base.linear_names() {
            let want = base.clone().linear_mut(&name).unwrap().clone();
            let got = eff.clone().linear_mut(&name).unwrap().clone();
            assert_eq!(want, got, "{name}: B=0 must leave the weight untouched");
        }
    }

    #[test]
    fn lora_grads_match_finite_differences() {
        let cfg = DepthDecoderConfig::tiny();
        let base = random_weights(&cfg, 3);
        let mut adapters = rank2_adapters(&cfg, &base, 4);
        for w in adapters.values_mut() {
            for b in w.b.iter_mut() {
                *b = 0.05;
            }
        }
        let scale = 0.7f32;
        let s = cfg.num_codebooks as usize;
        let mut r = Lcg::new(5);
        let inputs_embeds = r.vec_scaled(s * cfg.hidden_size as usize, 0.4);
        let target = r.vec_scaled(s * cfg.hidden_size as usize, 0.4);

        let (_, grads) = step(&cfg, &base, &adapters, scale, &inputs_embeds, s, &target);
        let loss_at = |adapters: &HashMap<String, LoraW>| -> f32 { step(&cfg, &base, adapters, scale, &inputs_embeds, s, &target).0 };

        let eps = 5e-3f32;
        let mut checked = 0;
        for (name, (da, db)) in &grads {
            let mut pa = adapters.clone();
            let base_a0 = pa[name].a[0];
            pa.get_mut(name).unwrap().a[0] = base_a0 + eps;
            let lp = loss_at(&pa);
            pa.get_mut(name).unwrap().a[0] = base_a0 - eps;
            let lm = loss_at(&pa);
            let num_a = (lp - lm) / (2.0 * eps);
            assert!((num_a - da[0]).abs() < 2e-2 + 2e-2 * num_a.abs().max(da[0].abs()), "{name}.a[0]: numeric={num_a} analytic={}", da[0]);

            let mut pb = adapters.clone();
            let base_b0 = pb[name].b[0];
            pb.get_mut(name).unwrap().b[0] = base_b0 + eps;
            let lp = loss_at(&pb);
            pb.get_mut(name).unwrap().b[0] = base_b0 - eps;
            let lm = loss_at(&pb);
            let num_b = (lp - lm) / (2.0 * eps);
            assert!((num_b - db[0]).abs() < 2e-2 + 2e-2 * num_b.abs().max(db[0].abs()), "{name}.b[0]: numeric={num_b} analytic={}", db[0]);
            checked += 1;
        }
        // ::tiny() has 2 layers * 7 linear weights = 14.
        assert_eq!(checked, 14, "expected every one of ::tiny()'s 14 linear weights to be checked, got {checked}");
    }

    #[test]
    fn lora_only_overfits_with_base_frozen() {
        let cfg = DepthDecoderConfig::tiny();
        let base = random_weights(&cfg, 6);
        let base_snapshot: Vec<(String, Vec<f32>)> = base.linear_names().into_iter().map(|n| (n.clone(), base.clone().linear_mut(&n).unwrap().clone())).collect();
        let mut adapters = rank2_adapters(&cfg, &base, 7);
        let scale = 1.0f32;
        let s = cfg.num_codebooks as usize;
        let mut r = Lcg::new(8);
        let inputs_embeds = r.vec_scaled(s * cfg.hidden_size as usize, 0.4);
        let target = r.vec_scaled(s * cfg.hidden_size as usize, 0.4);
        let lr = 0.2f32;

        let loss0 = step(&cfg, &base, &adapters, scale, &inputs_embeds, s, &target).0;
        let mut loss = loss0;
        for _ in 0..1500 {
            let (l, grads) = step(&cfg, &base, &adapters, scale, &inputs_embeds, s, &target);
            loss = l;
            for (name, (da, db)) in &grads {
                let w = adapters.get_mut(name).unwrap();
                for (ai, dai) in w.a.iter_mut().zip(da) {
                    *ai -= lr * dai;
                }
                for (bi, dbi) in w.b.iter_mut().zip(db) {
                    *bi -= lr * dbi;
                }
            }
        }
        assert!(loss < loss0 * 0.6, "LoRA-only training did not reduce loss enough: start={loss0} end={loss}");
        for (name, snapshot) in &base_snapshot {
            let now = base.clone().linear_mut(name).unwrap().clone();
            assert_eq!(snapshot, &now, "{name}: base weight must stay frozen during LoRA-only training");
        }
    }
}
