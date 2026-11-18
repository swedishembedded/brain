// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory admission for fine-tuning jobs that share accelerators with serving, for
// its clients. If your team needs a training scheduler that refuses what cannot fit and fits what it admits, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The LoRA training footprint (`qwen3::footprint::lora_training_demands`) against what a training build holds: the
//! device's own allocation accounting (`Gpu::memory_usage`, the peak of live bytes over the build and one whole step:
//! forward, backward, optimiser update) must not exceed the estimate, and the estimate must not exceed it by more than
//! the slack below, at each block length and saved-activation policy. The shapes keep Qwen3-4B's ratios (query width
//! 1.6 d, key/value width 0.4 d, MLP width 3.8 d, a vocabulary of 59 d, 128-wide heads) at a width the CPU backend
//! trains in seconds, with every projection adapted as served training adapts them.
//!
//! On a card the same buffers are allocated; what a card adds beside them (cached freed blocks, library workspace)
//! is the caller's term (`brain::ChatFineTune::footprint`), checked on a card by the ignored test at the end.

use gpu_core::provider::library::GemmPrecision;
use model::saved::{SavedFormat, SavedPolicy};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

/// How far above the measured peak the estimate may be: the head chunk is sized at the largest binding budget any
/// device reports, and a few `[n]`-sized buffers are not itemised.
const SLACK: f64 = 1.15;

fn cfg(layers: u32, t: u32, d: u32) -> QwenConfig {
    let targets = qwen3::finetune::LORA_TARGETS.iter().map(|s| s.to_string()).collect();
    let (heads, kv_heads) = ((d * 8 / 5) / 128, (d * 2 / 5) / 128);
    QwenConfig { vocab: d * 59, block_size: t, max_position_embeddings: t, n_layers: layers, d_model: d, n_heads: heads, n_kv_heads: kv_heads.max(1), head_dim: 128, d_ff: d * 19 / 5, lora: Some(LoraCfg::new(16, 32.0, targets)), ..QwenConfig::tiny() }
}

/// The peak device bytes of building `cfg` and running one optimiser step over a `t`-token row, and whether the
/// device ran the flash attention kernels.
fn measured_peak(cfg: &QwenConfig, t: u32, saved: SavedPolicy) -> (u64, bool) {
    let init = qwen3::init_weights(cfg, 5);
    let m = Qwen::new_lora_dt_saved(cfg.clone(), 1, t, &init, Dtype::F32, GemmPrecision::Auto, saved);
    let tokens: Vec<u32> = (0..t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..t).map(|i| (i * 11 + 3) % cfg.vocab).collect();
    m.set_batch(&tokens, &targets);
    m.zero_grads();
    let loss = m.forward();
    assert!(loss.is_finite());
    m.backward();
    m.adamw_step(1, 1e-3, 0.0, model::Adam::default(), Some(1.0), 1.0);
    m.poll_wait();
    let usage = m.gpu.memory_usage().expect("the test device accounts for its allocations");
    (usage.peak_live, model::block::FlashGqaIds::supported(&m.gpu, cfg.head_dim))
}

#[test]
fn a_lora_step_holds_no_more_than_its_footprint_and_the_footprint_is_tight() {
    for (layers, t, d, saved) in [
        (2u32, 128u32, 320u32, SavedPolicy::Recompute),
        (4, 128, 320, SavedPolicy::Recompute),
        (2, 512, 320, SavedPolicy::Recompute),
        (4, 512, 320, SavedPolicy::Quantized(SavedFormat::Bf16)),
        (6, 256, 640, SavedPolicy::Quantized(SavedFormat::Fp8)),
    ] {
        let c = cfg(layers, t, d);
        let (peak, flash) = measured_peak(&c, t, saved);
        let demands = qwen3::footprint::lora_training_demands(&c, Dtype::F32, 1, t, saved, flash).unwrap();
        let estimate = model::memplan::device_total(&demands);
        let ratio = estimate as f64 / peak as f64;
        assert!(
            (1.0..=SLACK).contains(&ratio),
            "{layers} layers, d {d}, {t} tokens, {}: measured peak {peak} B, estimate {estimate} B ({ratio:.3}x): {}",
            saved.label(),
            model::memplan::summary(&demands)
        );
    }
}

/// A longer block costs what the estimate says it costs: the difference between two block lengths is the
/// activations and logits the estimate attributes to the extra tokens.
#[test]
fn the_footprint_grows_with_the_block_as_the_trainer_does() {
    let (short, long) = (128u32, 512u32);
    let (c_short, c_long) = (cfg(3, short, 320), cfg(3, long, 320));
    let ((peak_short, flash), (peak_long, _)) = (measured_peak(&c_short, short, SavedPolicy::Recompute), measured_peak(&c_long, long, SavedPolicy::Recompute));
    let est = |c: &QwenConfig, t| model::memplan::device_total(&qwen3::footprint::lora_training_demands(c, Dtype::F32, 1, t, SavedPolicy::Recompute, flash).unwrap());
    let (grew, modelled) = (peak_long - peak_short, est(&c_long, long) - est(&c_short, short));
    let ratio = modelled as f64 / grew as f64;
    // Within a word or two a token of what a longer block adds (the total, above, is held to at least the peak).
    assert!((0.995..=SLACK).contains(&ratio), "the block's own bytes: measured {grew} B, modelled {modelled} B ({ratio:.3}x)");
}

/// On a card: a Qwen3-4B-shaped LoRA build (random weights, a bf16 base, every projection at rank 16) through one
/// step at 4,096 and 16,384 tokens takes no more of the card than the reservation served training makes for it
/// (`model::memplan::train_reservation` over these demands: what `brain::ChatFineTune::footprint` reserves), measured
/// as the driver reports the card's free memory, the device context excluded. Run on a card that is yours:
/// `CUDA_VISIBLE_DEVICES=0,1,2,3 BRAIN_DEVICE=gpuN BRAIN_BACKEND=cuda make test/one P=brain-qwen3 T=it
/// ARGS="--ignored on_a_card --nocapture"`.
#[test]
#[ignore = "needs a card of at least 48 GB: BRAIN_DEVICE=gpuN BRAIN_BACKEND=cuda"]
fn on_a_card_a_qwen3_4b_lora_step_takes_no_more_than_its_reservation() {
    let Some(card) = gpu_core::devices::current_gpu() else {
        brain_testutil::skip_unavailable("BRAIN_DEVICE names no card");
        return;
    };
    let total = gpu_core::capacity::probe_gpus().into_iter().find(|g| g.index == card && g.measured).map(|g| g.total).expect("the driver reports the card's memory");
    let targets: Vec<String> = qwen3::finetune::LORA_TARGETS.iter().map(|s| s.to_string()).collect();
    for t in [4096u32, 16_384] {
        let c = QwenConfig { vocab: 151_936, block_size: t, max_position_embeddings: t, n_layers: 36, d_model: 2560, n_heads: 32, n_kv_heads: 8, head_dim: 128, d_ff: 9728, lora: Some(LoraCfg::new(16, 32.0, targets.clone())), ..QwenConfig::tiny() };
        let init = qwen3::init_weights(&c, 5);
        // The context exists before the baseline, so it is not charged to the run.
        let context = gpu_core::Gpu::new(&[]);
        let baseline = gpu_core::capacity::free_on(card).expect("probed card");
        let (flash, taken) = gpu_core::capacity::peak_taken_while(card, baseline, || {
            let m = Qwen::new_lora_dt_gemm(c.clone(), 1, t, &init, Dtype::BF16, GemmPrecision::Auto);
            let tokens: Vec<u32> = (0..t).map(|i| (i * 7 + 1) % c.vocab).collect();
            m.set_batch(&tokens, &tokens);
            m.zero_grads();
            assert!(m.forward().is_finite());
            m.backward();
            m.adamw_step(1, 1e-4, 0.0, model::Adam::default(), Some(1.0), 1.0);
            m.poll_wait();
            model::block::FlashGqaIds::supported(&m.gpu, c.head_dim)
        });
        drop(context);
        let demands = qwen3::footprint::lora_training_demands(&c, Dtype::BF16, 1, t, SavedPolicy::Recompute, flash).unwrap();
        let reserved = model::memplan::train_reservation(model::memplan::device_total(&demands), Some(total));
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!("{t} tokens: measured {:.2} GiB taken, reserved {:.2} GiB ({})", gib(taken), gib(reserved), model::memplan::summary(&demands));
        assert!(taken <= reserved, "{t} tokens: the step took {:.2} GiB of the card, more than the {:.2} GiB reserved for it", gib(taken), gib(reserved));
    }
}
