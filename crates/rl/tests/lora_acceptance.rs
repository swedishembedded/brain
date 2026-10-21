// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning models of any
// architecture over quantized frozen bases through one training stack, for its
// clients. If your team needs expertise in wiring a new model family into
// quantized LoRA training then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The four-architecture acceptance test: `model::acceptance::run` is the one
//! LoRA training workflow; each architecture below only says which frozen-base
//! encodings it trains over and how to build a tiny random instance. qwen3,
//! qwen35 and qwen35moe are `Model`s and take their step from
//! `model::acceptance::ModelStep`; flux2 is an image-diffusion trainer, not a
//! `Model`, and implements `LoraStep` itself.

mod arch_builds;

use flux2::devgrad::BasePrecision;
use flux2::devtrain::{DeviceTrainer, TrainerSpec};
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{self, Batch, Cfg, ModelWeights};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::acceptance::{run, LoraArch, LoraStep, ModelStep};
use model::conformance::{BATCH, SEQ};

// seed: fixed so a failure reproduces
const SEED: u64 = 41;

struct Qwen3;
impl LoraArch for Qwen3 {
    const NAME: &'static str = "qwen3";
    fn encodings() -> Vec<Dtype> {
        vec![Dtype::F32, Dtype::BF16, Dtype::I8]
    }
    fn build(encoding: Dtype, seed: u64) -> Box<dyn LoraStep> {
        let (m, held) = arch_builds::qwen3(BATCH, SEQ, seed, encoding);
        Box::new(ModelStep::new(m, held))
    }
}

struct Qwen35;
impl LoraArch for Qwen35 {
    const NAME: &'static str = "qwen35";
    fn encodings() -> Vec<Dtype> {
        vec![Dtype::F32, Dtype::BF16, Dtype::NF4, Dtype::F8E4M3]
    }
    fn build(encoding: Dtype, seed: u64) -> Box<dyn LoraStep> {
        let (m, held) = arch_builds::qwen35(BATCH, SEQ, seed, encoding);
        Box::new(ModelStep::new(m, held))
    }
}

/// The mixture-of-experts sibling trains over an fp32 base only.
fn qwen35moe(rows: u32, seq: u32, seed: u64) -> (qwen35moe::model::Qwen35, Dtype) {
    let cfg = qwen35moe::config::Qwen35Config {
        n_layers: 4,
        block_size: seq,
        lora: Some(qwen35moe::config::lora_cfg(arch_builds::RANK, arch_builds::RANK as f32)),
        ..qwen35moe::config::Qwen35Config::tiny()
    };
    let init = <qwen35moe::model::Qwen35 as model::Model>::init_weights(&cfg, seed);
    (qwen35moe::model::Qwen35::new_train(cfg, rows, seq, &init), Dtype::F32)
}

struct Qwen35Moe;
impl LoraArch for Qwen35Moe {
    const NAME: &'static str = "qwen35moe";
    fn learning_rate() -> f32 {
        1e-1
    }
    fn steps() -> u32 {
        100
    }
    /// The adapters sit on the mixers only; the experts and the head stay frozen.
    fn overfit_ratio() -> f64 {
        0.97
    }
    fn encodings() -> Vec<Dtype> {
        vec![Dtype::F32]
    }
    fn build(_encoding: Dtype, seed: u64) -> Box<dyn LoraStep> {
        let (m, held) = qwen35moe(BATCH, SEQ, seed);
        Box::new(ModelStep::new(m, held))
    }
}

/// A tiny klein-topology config every int8 path accepts (whole 32-element scale
/// groups, text rows a whole number of 64-row activation blocks).
fn flux2_cfg() -> Cfg {
    Cfg {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 64,
        n_heads: 4,
        depth_double: 1,
        depth_single: 1,
        mlp: 192,
        txt_len: 64,
        lh: 4,
        lw: 4,
        axes_dim: [4, 4, 4, 4],
        rope_theta: 2000.0,
        refs: Vec::new(),
    }
}

struct Flux2;

struct Flux2Step {
    held: Dtype,
    trainer: DeviceTrainer,
    adapter: LoraAdapter,
    batch: Batch<f32>,
    base: ModelWeights<f32>,
}

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn digest(values: impl Iterator<Item = f32>) -> u64 {
    values.fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3))
}

impl LoraArch for Flux2 {
    const NAME: &'static str = "flux2";
    fn encodings() -> Vec<Dtype> {
        let mut encodings = vec![Dtype::F32];
        // The int8 base needs a device with the native int8-weight input gradient.
        if Gpu::new_gpu(flux2::devgrad::KERNELS).has_fused(gpu_core::Fused::I8wDx) {
            encodings.push(Dtype::I8);
        }
        encodings
    }
    fn build(encoding: Dtype, seed: u64) -> Box<dyn LoraStep> {
        let c = flux2_cfg();
        let base = modelgrad::init_model::<f32>(&c, seed);
        let trainer = match encoding {
            Dtype::F32 => DeviceTrainer::with_gpu(Gpu::new_cpu(flux2::devgrad::KERNELS), c.clone(), 4, &base),
            Dtype::I8 => DeviceTrainer::build(&TrainerSpec { card: None, cards: 1, base: BasePrecision::Int8 }, c.clone(), 4, &base).unwrap_or_else(|e| panic!("{e}")),
            other => panic!("flux2 trains over an f32 or int8 base, not {other:?}"),
        };
        let x0 = noise(seed ^ 1, c.n_gen() * c.in_channels);
        let ctx = noise(seed ^ 2, c.txt_len * c.context_in_dim);
        let eps = noise(seed ^ 3, x0.len());
        let batch = modelgrad::make_flow_batch_paired(&c, &x0, &[], &ctx, 0.37, &eps);
        let adapter = LoraAdapter::new(&c, LoraCfg::new(4, seed));
        Box::new(Flux2Step { held: encoding, trainer, adapter, batch, base })
    }
}

impl LoraStep for Flux2Step {
    fn held_encoding(&self) -> Dtype {
        self.held
    }
    fn loss(&mut self) -> f64 {
        self.trainer.grads(&self.adapter, &self.batch).0
    }
    fn step(&mut self, lr: f32) -> f64 {
        self.trainer.step(&mut self.adapter, &self.batch, lr)
    }
    fn adapter_digest(&self) -> u64 {
        digest(self.adapter.pairs().into_iter().flat_map(|p| p.a.iter().chain(&p.b).copied()))
    }
    /// The trainer holds the base on the device and never exposes it; the
    /// host-side original it was uploaded from is what a step must not touch.
    fn frozen_digest(&self) -> Option<u64> {
        Some(digest(self.base.dbl.iter().flat_map(|b| b.img.wq.iter().chain(&b.txt.wq).copied())))
    }
}

#[test]
fn qwen3_trains_lora_over_every_base_encoding() {
    run::<Qwen3>(SEED).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn qwen35_trains_lora_over_every_base_encoding() {
    run::<Qwen35>(SEED).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn qwen35moe_trains_lora_over_every_base_encoding() {
    run::<Qwen35Moe>(SEED).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn flux2_trains_lora_over_every_base_encoding() {
    run::<Flux2>(SEED).unwrap_or_else(|e| panic!("{e}"));
}
