// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching LLM serving for clients
// whose agents resend long conversation histories. If your team needs
// expertise in LLM serving throughput then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The released-GGUF engine (the 27B's serving path: int8 weights loaded from
//! a GGUF, hybrid Gated-DeltaNet and attention, prompt-prefix cache of KV rows
//! plus recurrent-state snapshots) through the generic
//! [`model::serve::conformance`] suite, on a 2-layer random-weight GGUF that
//! has the real tensor names and layer kinds.

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use model::serve::conformance::{self, Conformance};
use qwen35::int8_gguf_resident::{budgeted_gpus, Qwen35GgufEngine, Qwen35GgufResident};

/// Per-sequence capacity: the prefix test's 72-token prompt (the engine caches
/// prompts of at least 64 tokens) plus its 8-token extension and decode.
const CAP: u32 = 96;
const PREFIX_PROMPT_TOKENS: usize = 72;

fn engine() -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("brain-gguf-conformance-{}-{n}.gguf", std::process::id())).to_string_lossy().into_owned();
    qwen35::gguf_import::testing::write_random_servable_gguf(&path, 7);
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let resident = Qwen35GgufResident::new(path.clone(), devices, CAP, TierPolicy::uniform(Dtype::I8)).with_max_batch(4).with_prefix_slots(4);
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the tiny GGUF loads");
    std::fs::remove_file(path).ok();
    engine
}

/// Run with `-- --device gpu1 --backend cuda` (the int8 serving path is a CUDA path);
/// without `--device` the binary reports the test skipped.
fn the_released_gguf_engine_conforms() {
    conformance::run_all(&Conformance { make: engine, prefix_reuse: true, prefix_prompt_tokens: PREFIX_PROMPT_TOKENS, load_adapter: None });
}

gpu_core::card_tests!(the_released_gguf_engine_conforms);
