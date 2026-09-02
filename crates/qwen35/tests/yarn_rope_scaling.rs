// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen35Config::rope_scaling` end-to-end wiring: a checkpoint that opts
//! into YaRN (`rope_scaling: {"type": "yarn", ...}`) must build and decode
//! without panicking, and must diverge from the unscaled config once
//! decoding passes `original_max_position_embeddings` - the concrete,
//! verifiable proof that `max_position_embeddings`/YaRN is no longer a dead
//! config field for this model.
//!
//! The regression half of the gate - an UNCONFIGURED `Qwen35Config` (no
//! `rope_scaling`, today's default for every existing checkpoint) is
//! byte-identical to before this feature existed - is proven at the table
//! level by `qwen3vl::mrope`'s own
//! `scaled_none_matches_plain_mrope_tables_bit_for_bit` test (the exact
//! function this model's forward/decode now call), and re-confirmed here at
//! the whole-model level by `unconfigured_forward_is_unaffected_by_the_yarn_plumbing_existing`.

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

/// A `tiny()`-shaped config with YaRN scaling turned on at a deliberately
/// small `original_max_position_embeddings` (6, well under `tiny()`'s
/// `block_size = 24`), so a 24-token decode run genuinely crosses the
/// "beyond the pretrained context" boundary partway through - not just at
/// the very last position.
fn yarn_cfg() -> Qwen35Config {
    Qwen35Config { rope_scaling: Some(model::yarn::YarnConfig::new(3.0, 6)), ..Qwen35Config::tiny() }
}

/// Building the model and driving `step()` across the whole `block_size`
/// (i.e. past `original_max_position_embeddings`) must not panic and must
/// keep producing finite hidden states - the "wiring didn't break anything"
/// half of the integration gate.
#[test]
fn yarn_scaled_config_builds_and_decodes_without_panicking() {
    let cfg = yarn_cfg();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 11);
    let m = Qwen35::new_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, t, &init);

    m.reset_decode_cache();
    for pos in 0..t {
        let tok = (pos * 5 + 3) % cfg.vocab;
        let hidden = m.step(tok);
        assert!(hidden.iter().all(|x| x.is_finite()), "position {pos}: YaRN-scaled step() produced a non-finite hidden state");
    }
}

/// The actual scaling proof: two models, IDENTICAL weights (same seed) and
/// IDENTICAL token stream, differing only in `rope_scaling` (unset vs. the
/// YaRN config above), must produce a DIFFERENT hidden state once decoding
/// passes `original_max_position_embeddings = 6` - the position at which
/// the per-channel frequency correction actually starts pulling the
/// rotation away from plain RoPE's angle. Below that boundary the two are
/// not asserted equal (YaRN's ramp already blends some channels below the
/// boundary too - see `model::yarn`'s ramp-mask test), only that they
/// eventually diverge, which is the property that makes long-context
/// actually usable.
#[test]
fn yarn_scaled_decode_diverges_from_unscaled_beyond_original_context() {
    let base_cfg = Qwen35Config::tiny();
    let yarn_cfg = yarn_cfg();
    assert!(base_cfg.rope_scaling.is_none(), "baseline must be the unconfigured default");
    let t = base_cfg.block_size;

    let init = qwen35::init::init_weights(&base_cfg, 11); // same seed for both models
    let m_base = Qwen35::new_on(Gpu::new_cpu(pipelines()), base_cfg.clone(), 1, t, &init);
    let m_yarn = Qwen35::new_on(Gpu::new_cpu(pipelines()), yarn_cfg, 1, t, &init);

    m_base.reset_decode_cache();
    m_yarn.reset_decode_cache();

    let mut diverged_beyond_boundary = false;
    for pos in 0..t {
        let tok = (pos * 5 + 3) % base_cfg.vocab;
        let h_base = m_base.step(tok);
        let h_yarn = m_yarn.step(tok);
        let maxabs = h_base.iter().zip(&h_yarn).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        if pos >= 6 && maxabs > 1e-4 {
            diverged_beyond_boundary = true;
        }
    }
    assert!(diverged_beyond_boundary, "YaRN-scaled decode must diverge from the unscaled baseline once past original_max_position_embeddings=6");
}

/// Regression proof at the whole-model level: a config that carries
/// `rope_scaling: Some(YarnConfig::new(1.0, ..))` - i.e. the YaRN code path
/// IS exercised (it is not simply `None`) but requests no real extension
/// (`factor = 1.0`) - must produce logits BYTE-IDENTICAL to
/// `rope_scaling: None`. This is `model::yarn::scaled_inv_freq`'s own
/// identity gate (unit-tested in isolation in `crates/model/src/yarn.rs`)
/// re-proven through the full prefill forward pass, not just the table
/// builder.
#[test]
fn unconfigured_forward_is_unaffected_by_the_yarn_plumbing_existing() {
    let none_cfg = Qwen35Config::tiny();
    let identity_cfg = Qwen35Config { rope_scaling: Some(model::yarn::YarnConfig::new(1.0, 32768)), ..Qwen35Config::tiny() };
    let t = none_cfg.block_size;
    let init = qwen35::init::init_weights(&none_cfg, 5);

    let m_none = Qwen35::new_on(Gpu::new_cpu(pipelines()), none_cfg.clone(), 1, t, &init);
    let m_identity = Qwen35::new_on(Gpu::new_cpu(pipelines()), identity_cfg, 1, t, &init);

    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 3) % none_cfg.vocab).collect();
    let logits_none = m_none.logits_all(&tokens);
    let logits_identity = m_identity.logits_all(&tokens);
    assert_eq!(logits_none, logits_identity, "factor=1.0 YaRN config must be bit-for-bit identical to rope_scaling: None");
}
