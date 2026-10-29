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
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

/// Per-sequence capacity: the prefix test's 152-token prompt plus its 8-token extension and decode.
const CAP: u32 = 192;
const PREFIX_PROMPT_TOKENS: usize = 152;

fn engine() -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = write_checkpoint_conditioned(&fp8_resident_cfg(), &format!("conformance{n}"), 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    assert_eq!(tier.default_dtype(), Dtype::F8E4M3, "the linears are held as FP8");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, CAP, tier).with_max_batch(4).with_prefix_slots(4).with_kv_block_rows(16).with_kv_tier(KvTier::F32);
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the tiny FP8 directory loads");
    std::fs::remove_dir_all(dir).ok();
    engine
}

fn the_hf_fp8_engine_conforms() {
    conformance::run_all(&Conformance { make: engine, prefix_reuse: true, prefix_prompt_tokens: PREFIX_PROMPT_TOKENS, load_adapter: None });
}

gpu_core::card_tests!(the_hf_fp8_engine_conforms);
