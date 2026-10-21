// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of quantized
// language models on a single GPU for its clients. If your team needs
// expertise in training adapters through a packed FP8 or 4-bit base without
// expanding any layer to full precision, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The streaming LoRA trainer over a PACKED frozen base.
//!
//! Spec: training through layers resident as FP8 (E4M3, E5M2) or NF4 gives the loss and the adapter
//! gradients that training through the fp32 DECODING of the same stored values gives - the base's forward
//! and input gradient go through `Ops` over the packed tier, with no fp32 layer anywhere. The oracle is the
//! same trainer over `Weight::F32` weights holding the host decode of the quantized bytes, so what is left
//! is only the arithmetic of the device path (exact fp32 on the CPU backend). And a checkpoint directory
//! loads every layer matrix in its stored encoding (`PackedBase`), equal to the eager importer's values.

use std::collections::HashMap;

use checkpoint::mmap::MmapSafetensors;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::fp8::{dequantize_weight_fp8, quantize_weight_fp8, Fp8Format};
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use model::ops::{training_kernel_list, Ops, Weight, FP8_BLOCK};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::fp8_source::PackedBase;
use qwen35::import::import_layer;
use qwen35::model::pipelines;
use qwen35::stream_train::StreamTrainer;

mod common;
use common::hf_fixture::{write_checkpoint, BLOCK};

const N: u32 = 6;

fn lora_config(base: Qwen35Config) -> Qwen35Config {
    Qwen35Config { lora: Some(lora_cfg(2, 4.0)), ..base }
}

/// `(name, n, k)` of every matrix a streamed layer holds.
fn matrices(cfg: &Qwen35Config) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    for (l, ty) in cfg.layer_types().iter().enumerate() {
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            out.push((format!("blocks.{l}.{suffix}"), n as usize, k as usize));
        }
    }
    out
}

/// `init` with every matrix replaced by the host decode of its quantization at `tier`.
fn decoded(cfg: &Qwen35Config, init: &HashMap<String, Vec<f32>>, tier: Dtype) -> HashMap<String, Vec<f32>> {
    let mut out = init.clone();
    for (name, n, k) in matrices(cfg) {
        let raw = &init[&name];
        let back = match tier {
            Dtype::F8E4M3 | Dtype::F8E5M2 => {
                let fmt = if tier == Dtype::F8E4M3 { Fp8Format::E4M3 } else { Fp8Format::E5M2 };
                let (packed, scales) = quantize_weight_fp8(raw, n, k, fmt, FP8_BLOCK as usize);
                dequantize_weight_fp8(&packed, &scales, n, k, fmt, FP8_BLOCK as usize)
            }
            Dtype::NF4 => {
                let (packed, scales) = quantize_weight_lut4(raw, n, k, &NF4_LUT);
                dequantize_weight_lut4(&packed, &scales, n, k, &NF4_LUT)
            }
            other => panic!("no host decode for {other:?}"),
        };
        out.insert(name, back);
    }
    out
}

fn gradients(trainer: &StreamTrainer, cfg: &Qwen35Config, loader: &(dyn Fn(usize) -> qwen35::stream::OwnedStreamedLayer + Sync), x0: &[f32], targets: &[u32]) -> (f32, Vec<(String, Vec<f32>)>) {
    trainer.lora.zero_grads(&trainer.gpu);
    let loss = trainer.forward_backward(cfg, loader, x0, targets);
    let mut grads = Vec::new();
    for (name, _) in cfg.param_list() {
        if name.ends_with(".lora_a") || name.ends_with(".lora_b") {
            grads.push((name.clone(), trainer.lora.ps.read_grad(&trainer.gpu, &name)));
        }
    }
    (loss, grads)
}

fn check_tier(base: Qwen35Config, tier: Dtype) {
    let cfg = lora_config(base);
    let init = qwen35::init::init_weights(&cfg, data::rng::random_seed());
    let reference_weights = decoded(&cfg, &init, tier);
    let tokens: Vec<u32> = (0..N).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..N).map(|i| (i * 5 + 2) % cfg.vocab).collect();

    let reference = StreamTrainer::new_synthetic(Gpu::new_cpu(pipelines()), &cfg, N, 2, &reference_weights);
    let x0 = StreamTrainer::embed_synthetic(&reference_weights, &tokens, cfg.d_model as usize);
    let ref_loader = reference.synthetic_loader(&cfg, &reference_weights);
    let (want_loss, want) = gradients(&reference, &cfg, &ref_loader, &x0, &targets);

    let packed = StreamTrainer::new_synthetic(Gpu::new_cpu(pipelines()), &cfg, N, 2, &reference_weights);
    // The packed loader quantizes the ORIGINAL f32 matrices: the same bytes the reference decoded.
    let packed_loader = packed.synthetic_loader_at(&cfg, &init, tier);
    let (got_loss, got) = gradients(&packed, &cfg, &packed_loader, &x0, &targets);

    assert!((want_loss - got_loss).abs() < 1e-4, "{tier:?}: loss {got_loss} (packed) vs {want_loss} (decoded fp32)");
    assert_eq!(want.len(), got.len());
    let mut worst = 0f32;
    for ((name, w), (_, g)) in want.iter().zip(&got) {
        let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let diff = w.iter().zip(g).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(diff / scale);
        assert!(diff <= 1e-4 * scale + 1e-6, "{tier:?}: {name}: max grad difference {diff} (grad scale {scale})");
    }
    println!("{tier:?}: loss {got_loss:.6} vs {want_loss:.6}, worst adapter-gradient difference {worst:.2e} of the gradient scale");
}

#[test]
fn an_fp8_e4m3_base_trains_like_its_fp32_decoding() {
    check_tier(Qwen35Config::tiny(), Dtype::F8E4M3);
}

#[test]
fn an_fp8_e5m2_base_trains_like_its_fp32_decoding() {
    check_tier(Qwen35Config::tiny(), Dtype::F8E5M2);
}

#[test]
fn an_nf4_base_trains_like_its_fp32_decoding() {
    check_tier(Qwen35Config::tiny_i8(), Dtype::NF4);
}

/// A checkpoint directory loads each layer matrix in its stored encoding (FP8 stays FP8 with its block
/// scales, bf16 stays bf16) and every small tensor as the eager importer's f32, bit for bit.
#[test]
fn a_checkpoint_layer_loads_packed_and_equals_the_eager_importer() {
    let cfg = Qwen35Config { d_model: 96, intermediate_size: 112, vocab: 64, n_layers: 4, ..Qwen35Config::tiny() };
    let dir = write_checkpoint(&cfg, "stream_packed", None);
    let base = PackedBase::open(&dir, &cfg, BLOCK).expect("open the packed base");
    let ops = Ops::new(Gpu::new_cpu(training_kernel_list())).expect("Ops::new");
    let mut fp8 = 0;
    for (l, ty) in cfg.layer_types().iter().enumerate() {
        let shard = MmapSafetensors::open(dir.join(format!("layers-{l}.safetensors"))).unwrap();
        let eager = import_layer(&shard, &cfg, l, BLOCK).unwrap();
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            let name = format!("blocks.{l}.{suffix}");
            let w = base.weight(&ops, &name, n as usize, k as usize).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((w.n() as usize, w.k() as usize), (n as usize, k as usize), "{name}");
            fp8 += matches!(w, Weight::FP8 { .. }) as usize;
            if let Some(f32_ref) = eager.get(&name) {
                let g = ops.gpu();
                let out = g.storage((n * k) as u64);
                let mut steps = Vec::new();
                ops.decode_tile(&mut steps, &w, 0, n, &out);
                g.submit(&[], &steps);
                let got = g.read(&out, (n * k) as usize);
                assert!(got.iter().zip(f32_ref).all(|(a, b)| a.to_bits() == b.to_bits() || (a - b).abs() <= 1e-6 * b.abs()), "{name}: the resident tile differs from the eager import");
            }
        }
        for small in ["ln1.weight", "ln2.weight"] {
            let name = format!("blocks.{l}.{small}");
            assert_eq!(base.small_f32(&name).unwrap(), eager[&name], "{name}");
        }
    }
    assert!(fp8 > 0, "the fixture's linears are FP8 and must stay FP8 on the device");
    std::fs::remove_dir_all(&dir).ok();
}

/// The per-layer scratch is pooled (one replay arena per layer kind and direction): a second pass over the same
/// trainer draws its buffers from the arenas the first filled, holding the first pass's stale contents, and
/// still gives the loss and every adapter gradient bit for bit.
#[test]
fn a_pass_over_pooled_scratch_is_bit_identical_to_the_first_pass() {
    let cfg = lora_config(Qwen35Config::tiny());
    let init = qwen35::init::init_weights(&cfg, data::rng::random_seed());
    let tokens: Vec<u32> = (0..N).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..N).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let trainer = StreamTrainer::new_synthetic(Gpu::new_cpu(pipelines()), &cfg, N, 2, &init);
    let x0 = StreamTrainer::embed_synthetic(&init, &tokens, cfg.d_model as usize);
    let loader = trainer.synthetic_loader_at(&cfg, &init, Dtype::F8E4M3);
    let (first_loss, first) = gradients(&trainer, &cfg, &loader, &x0, &targets);
    let (second_loss, second) = gradients(&trainer, &cfg, &loader, &x0, &targets);
    assert_eq!(first_loss, second_loss);
    assert_eq!(first, second, "adapter gradients differ between a fresh pass and a replayed one");
    assert!(first.iter().any(|(_, g)| g.iter().any(|v| *v != 0.0)), "the pass produced gradient to compare");
}
