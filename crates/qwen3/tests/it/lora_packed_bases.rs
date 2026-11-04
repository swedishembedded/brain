// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A Qwen3 LoRA model whose frozen base is held in 4-bit NormalFloat (NF4) or
//! block-scaled FP8 reads it through the shared quantized-execution contract
//! (`model::ops::tiled`). The only approximation is the weight rounding, so an
//! fp32 build over the values the packed base decodes to computes the same
//! loss and adapter gradients, and the packed base still learns. The base is
//! held in the encoding that was asked for.

use std::collections::HashMap;

use data::rng::Rng;
use model::fp8::{dequantize_weight_fp8, quantize_weight_fp8, Fp8Format};
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use model::adapter::DeviceKind;
use model::ops::FP8_BLOCK;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

const BLOCK: u32 = 12;

fn batch() -> (Vec<u32>, Vec<u32>) {
    let n = BLOCK as usize;
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 22 + 1).collect();
    let targets = (0..n).map(|i| tokens[(i + 1) % n]).collect();
    (tokens, targets)
}

/// Every contraction width a multiple of the NF4 group (32).
fn config() -> QwenConfig {
    config_of(DeviceKind::Lora)
}

/// [`config`] with the adapter kind chosen.
fn config_of(kind: DeviceKind) -> QwenConfig {
    QwenConfig { block_size: BLOCK, d_model: 64, n_heads: 4, n_kv_heads: 2, head_dim: 16, d_ff: 128, lora: Some(LoraCfg::attn(2, 4.0).with_kind(kind)), ..QwenConfig::tiny() }
}

/// One packed encoding: its dtype and how the weights are rounded through it.
struct Encoding {
    name: &'static str,
    dtype: Dtype,
    round: fn(&[f32], usize, usize) -> Vec<f32>,
}

const ENCODINGS: [Encoding; 2] = [
    Encoding {
        name: "nf4",
        dtype: Dtype::NF4,
        round: |w, n, k| {
            let (packed, scales) = quantize_weight_lut4(w, n, k, &NF4_LUT);
            dequantize_weight_lut4(&packed, &scales, n, k, &NF4_LUT)
        },
    },
    Encoding {
        name: "fp8 e4m3",
        dtype: Dtype::F8E4M3,
        round: |w, n, k| {
            let (packed, scales) = quantize_weight_fp8(w, n, k, Fp8Format::E4M3, FP8_BLOCK as usize);
            dequantize_weight_fp8(&packed, &scales, n, k, Fp8Format::E4M3, FP8_BLOCK as usize)
        },
    },
];

/// Weights that survive `enc`'s round trip unchanged, adapters with a non-zero `B`.
fn weights(cfg: &QwenConfig, enc: &Encoding, adapter_b: f32) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 31);
    // seed: only the adapter values; the agreement bounds below do not depend on them
    let mut rng = Rng::new(5);
    for (name, v) in w.iter_mut() {
        if name.ends_with(".lora_b") || name.ends_with(".dora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * adapter_b);
        } else if qwen3::q8::Q8::is_i8_linear(name) {
            let k = if name.ends_with("down.weight") { cfg.d_ff } else if name.ends_with("wo.weight") { cfg.q_dim() } else { cfg.d_model } as usize;
            *v = (enc.round)(v, v.len() / k, k);
        }
    }
    w
}

fn loss_and_grads(m: &Qwen) -> (f32, Vec<(String, Vec<f32>)>) {
    let (tokens, targets) = batch();
    m.set_batch(&tokens, &targets);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

#[test]
fn a_packed_base_trains_the_adapters_an_fp32_build_over_its_decoded_values_does() {
    for enc in &ENCODINGS {
        let cfg = config();
        let init = weights(&cfg, enc, 0.3);
        let (want_loss, want) = loss_and_grads(&Qwen::new(cfg.clone(), 1, BLOCK, &init));
        let packed = Qwen::new_lora_dt(cfg, 1, BLOCK, &init, enc.dtype);
        assert_eq!(packed.linear_dtype(), Some(enc.dtype), "{}: the base is held as asked, not widened", enc.name);
        let (loss, got) = loss_and_grads(&packed);
        assert!((loss - want_loss).abs() < 1e-3 * want_loss.abs().max(1.0), "{}: loss {loss} vs {want_loss}", enc.name);
        let mut nonzero = 0;
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            if w.iter().all(|x| *x == 0.0) {
                continue;
            }
            nonzero += 1;
            let c = cosine(g, w);
            assert!(c > 0.999, "{}: {name}: adapter gradient cosine {c}", enc.name);
        }
        assert!(nonzero >= 4, "{}: the adapters received gradient ({nonzero} tensors)", enc.name);
    }
}

#[test]
fn a_packed_base_still_overfits_a_fixed_batch() {
    overfits(DeviceKind::Lora);
}

/// DoRA reads its frozen base's row norms through the tile decode, so it trains over the packed bases too.
#[test]
fn a_dora_adapter_over_a_packed_base_matches_the_decoded_fp32_build_and_overfits() {
    for enc in &ENCODINGS {
        let cfg = config_of(DeviceKind::Dora);
        let init = weights(&cfg, enc, 0.3);
        let (want_loss, want) = loss_and_grads(&Qwen::new(cfg.clone(), 1, BLOCK, &init));
        let (loss, got) = loss_and_grads(&Qwen::new_lora_dt(cfg, 1, BLOCK, &init, enc.dtype));
        assert!((loss - want_loss).abs() < 1e-3 * want_loss.abs().max(1.0), "{}: loss {loss} vs {want_loss}", enc.name);
        let mut checked = 0;
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            if w.iter().all(|x| *x == 0.0) {
                continue;
            }
            checked += 1;
            let c = cosine(g, w);
            assert!(c > 0.999, "{}: {name}: adapter gradient cosine {c}", enc.name);
        }
        assert!(checked >= 12, "{}: the dora factors and magnitudes all received gradient ({checked} tensors)", enc.name);
    }
    overfits(DeviceKind::Dora);
}

fn overfits(kind: DeviceKind) {
    for enc in &ENCODINGS {
        let cfg = config_of(kind);
        let init = weights(&cfg, enc, 0.0);
        let m = Qwen::new_lora_dt(cfg, 1, BLOCK, &init, enc.dtype);
        let (tokens, targets) = batch();
        m.set_batch(&tokens, &targets);
        let before = m.forward();
        for step in 1..=60 {
            m.zero_grads();
            m.forward();
            m.backward();
            model::Model::adamw_step(&m, step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
            m.poll_wait();
        }
        let after = m.forward();
        assert!(after < before * 0.8, "{} {}: the packed base did not learn: {before} -> {after}", kind.name(), enc.name);
    }
}

#[test]
fn a_tier_without_a_training_path_is_refused_by_name() {
    let cfg = config();
    let init = qwen3::init_weights(&cfg, 31);
    let refused = std::panic::catch_unwind(|| Qwen::new_lora_dt(cfg, 1, BLOCK, &init, Dtype::F16)).err().expect("an f16 frozen base is refused");
    let why = refused.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(why.contains("inference-only"), "{why}");
}
