// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-GPU fine-tuning of large language
// models for its clients. If your team needs expertise in fitting training
// runs into the cards you have then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A LoRA fine-tune is laid out by what the cards can hold, never by a flag:
//! the whole model on one card when one has room, otherwise the fewest
//! pipeline stages the cards can hold between them, otherwise a refusal that
//! names the bytes. Decided here against a stand-in for the machine's
//! placer, so it needs no device.

use gpu_core::devices::{Home, Need};
use qwen3::finetune::plan_lora_layout;
use qwen3::{Dtype, QwenConfig};

const GIB: u64 = 1 << 30;

/// A 7B-class decoder's dimensions (28 layers, GQA 28/4, a 152k vocabulary,
/// untied head).
fn seven_b() -> QwenConfig {
    QwenConfig {
        vocab: 152_064,
        block_size: 4096,
        n_layers: 28,
        d_model: 3584,
        n_heads: 28,
        n_kv_heads: 4,
        head_dim: 128,
        d_ff: 18_944,
        rope_theta: 1_000_000.0,
        rms_eps: 1e-6,
        max_position_embeddings: 4096,
        tie_embeddings: false,
        qk_norm: false,
        attn_bias: true,
        lora: Some(qwen3::LoraCfg::attn(8, 16.0)),
        rope_scaling: None,
    }
}

/// Cards that each hold `free` bytes; a plan is refused when any part is larger
/// than one card or the parts together exceed what the cards hold (first fit).
fn machine(free: &[u64]) -> impl Fn(&[Need]) -> Result<Vec<Home>, String> + '_ {
    move |needs| {
        let mut left = free.to_vec();
        needs
            .iter()
            .map(|n| match left.iter().position(|&f| f >= n.vram) {
                Some(i) => {
                    left[i] -= n.vram;
                    Ok(Home::Gpu(i as u32))
                }
                None => Err(format!("{} needs {} B, no card has it", n.name, n.vram)),
            })
            .collect()
    }
}

#[test]
fn a_model_that_fits_one_card_stays_on_it() {
    let cfg = QwenConfig { block_size: 512, ..seven_b() };
    let shards = plan_lora_layout(&cfg, 1, 512, Dtype::BF16, 2, machine(&[40 * GIB, 40 * GIB])).unwrap();
    assert_eq!(shards.len(), 1);
    assert!(shards[0].is_whole(cfg.n_layers as usize));
    assert_eq!(shards[0].gpu_index, 0);
}

#[test]
fn a_model_too_big_for_one_card_is_cut_into_the_fewest_stages_that_fit() {
    let cfg = seven_b();
    // A bf16 base at 4k tokens is about 25 GiB: not one 20 GiB card, but two.
    let shards = plan_lora_layout(&cfg, 1, 4096, Dtype::BF16, 2, machine(&[20 * GIB, 20 * GIB])).unwrap();
    assert_eq!(shards.len(), 2, "two cards, two stages");
    assert_eq!((shards[0].start, shards[0].end), (0, shards[1].start), "contiguous stages");
    assert_eq!(shards[1].end, cfg.n_layers as usize);
    assert!(shards[0].embed && !shards[0].head && shards[1].head && !shards[1].embed);
    assert_eq!((shards[0].gpu_index, shards[1].gpu_index), (0, 1), "each stage on the card the placer gave it");
}

#[test]
fn a_model_no_cards_can_hold_is_refused_by_name() {
    let cfg = seven_b();
    let err = plan_lora_layout(&cfg, 1, 4096, Dtype::BF16, 2, machine(&[8 * GIB, 8 * GIB])).unwrap_err();
    assert!(err.contains("2 card"), "names how many cards it tried: {err}");
    assert!(err.contains("GiB"), "names the bytes: {err}");
}

#[test]
fn a_machine_with_one_card_does_not_invent_a_pipeline() {
    let cfg = seven_b();
    let err = plan_lora_layout(&cfg, 1, 4096, Dtype::BF16, 1, machine(&[16 * GIB])).unwrap_err();
    assert!(err.contains("1 card"), "{err}");
}
