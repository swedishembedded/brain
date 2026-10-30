// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching serving of FP8 LLM releases at their stored precision for its
// clients. If your team needs expertise in LLM serving throughput on FP8 checkpoints then you can procure our services
// by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! The paged resident engine served from a HuggingFace FP8 directory (E4M3 linears with 128 x 128 block scales held as
//! stored, a bf16 head and embedding, the GDN gate projections in bf16) through the generic
//! [`model::serve::conformance`] suite: continuous batching, no head-of-line blocking, prefix reuse. The same engine type
//! the released-GGUF suite runs (`gguf_conformance`), built from the other checkpoint format.
//!
//! Run with `-- --device gpu1 --backend cuda`; without `--device` the binary reports the test skipped.

use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use model::serve::conformance::{self, Conformance};
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident, SpecServing, SpeculativeMode};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

/// Per-sequence capacity: the prefix test's 152-token prompt plus its 8-token extension and decode.
const CAP: u32 = 192;
const PREFIX_PROMPT_TOKENS: usize = 152;

fn engine() -> Qwen35GgufEngine {
    engine_speculating(SpeculativeMode::Off)
}

/// The engine of the suite with `mode` speculation, drafting from the directory's own `mtp.*` tensors for `mtp`.
fn engine_speculating(mode: SpeculativeMode) -> Qwen35GgufEngine {
    // The verify kernels take the 27B's recurrent geometry (key and value heads of 128, a width-4 conv).
    let cfg = Qwen35Config { linear_num_key_heads: 1, linear_num_value_heads: 2, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, ..fp8_resident_cfg() };
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = write_checkpoint_conditioned(&cfg, &format!("conformance{n}"), 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    assert_eq!(tier.default_dtype(), Dtype::F8E4M3, "the linears are held as FP8");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, CAP, tier).with_max_batch(4).with_prefix_slots(4).with_kv_block_rows(16).with_kv_tier(KvTier::F32).with_speculation(SpecServing { mode, ..SpecServing::default() });
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the tiny FP8 directory loads");

    std::fs::remove_dir_all(dir).ok();
    engine
}

fn separate_lanes(mut engine: Qwen35GgufEngine) -> Qwen35GgufEngine {
    engine.set_mixed_step(false);
    engine
}

fn the_hf_fp8_engine_conforms() {
    conformance::run_all(&Conformance { make: engine, prefix_reuse: true, prefix_prompt_tokens: PREFIX_PROMPT_TOKENS, load_adapter: None });
}


/// The MTP head read from the directory's `mtp.safetensors` (FP8 layer, bf16 projection) drafts, the verify pass
/// scores its proposals, and the suite's speculation property (tokens equal to plain greedy decode under a drafter
/// that is right, half right and always wrong) holds on the FP8 tier.
///
/// The suite compares token streams exactly, and the fixture's random weights leave logits within 0.01 of a tie
/// where its periodic prompts repeat: which side a tie falls on depends on whether the prompt was read in alone or in
/// one forward with the others, so the engine reads prompts in on the separate lane here (the mixed forward's rows
/// are compared in `hf_fp8_speculative`).
fn the_hf_fp8_engine_conforms_speculating_with_its_own_mtp_head() {
    assert!(engine_speculating(SpeculativeMode::Mtp).instance().has_mtp(), "the FP8 directory's MTP head is loaded");
    conformance::run_all(&Conformance { make: || separate_lanes(engine_speculating(SpeculativeMode::Mtp)), prefix_reuse: true, prefix_prompt_tokens: PREFIX_PROMPT_TOKENS, load_adapter: None });
}

gpu_core::card_tests!(the_hf_fp8_engine_conforms, the_hf_fp8_engine_conforms_speculating_with_its_own_mtp_head);
