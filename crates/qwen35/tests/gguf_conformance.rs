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
use model::kv_tier::KvTier;
use model::ops::TierPolicy;
use model::adapter::bank::{AdapterBankRequest, AdapterTargets, VerifiedAdapter};
use model::adapter::sites::LoraSites;
use model::lora::device_adapter::{AdapterSite, DeviceAdapter};
use model::serve::conformance::{self, Conformance};
use model::serve::{AdapterId, Request, Scheduler};
use qwen35::int8_gguf_resident::{budgeted_gpus, Qwen35GgufEngine, Qwen35GgufResident};

/// Per-sequence capacity: the prefix test's 72-token prompt (the engine caches
/// prompts of at least 64 tokens) plus its 8-token extension and decode.
const CAP: u32 = 96;
const PREFIX_PROMPT_TOKENS: usize = 72;

fn engine() -> Qwen35GgufEngine {
    engine_with(0)
}

/// The same engine with its KV addressed through tables of 16-row blocks.
fn paged_engine() -> Qwen35GgufEngine {
    engine_with(16)
}

fn engine_with(kv_block_rows: u32) -> Qwen35GgufEngine {
    engine_with_kv(kv_block_rows, KvTier::F32)
}

fn engine_with_kv(kv_block_rows: u32, kv_tier: KvTier) -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("brain-gguf-conformance-{}-{n}.gguf", std::process::id())).to_string_lossy().into_owned();
    qwen35::gguf_import::testing::write_random_servable_gguf(&path, 7);
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let bank = AdapterBankRequest { slots: 2, rank_cap: 4, targets: AdapterTargets::ALL, max_registered: 4 };
    let resident = Qwen35GgufResident::new(path.clone(), devices, CAP, TierPolicy::uniform(Dtype::I8)).with_max_batch(4).with_prefix_slots(4).with_kv_block_rows(kv_block_rows).with_kv_tier(kv_tier).with_adapter_bank(bank);
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the tiny GGUF loads");
    std::fs::remove_file(path).ok();
    engine
}

/// Run with `-- --device gpu1 --backend cuda` (the int8 serving path is a CUDA path);
/// without `--device` the binary reports the test skipped.
fn the_released_gguf_engine_conforms() {
    conformance::run_all(&Conformance { make: engine, prefix_reuse: true, prefix_prompt_tokens: PREFIX_PROMPT_TOKENS, load_adapter: Some(load_adapter) });
}

/// The same suite with every sequence's KV addressed through block tables (16 rows per block, six blocks
/// per sequence): batching, prefix reuse and adapter isolation are unchanged by the layout.
fn the_released_gguf_engine_conforms_with_its_kv_paged() {
    conformance::run_all(&Conformance { make: paged_engine, prefix_reuse: true, prefix_prompt_tokens: PREFIX_PROMPT_TOKENS, load_adapter: Some(load_adapter) });
}

/// A distinct rank-2 adapter over every linear of the engine, with factors large enough to move the
/// logits (so an applied adapter and an ignored one cannot look alike).
fn load_adapter(e: &mut Qwen35GgufEngine) -> AdapterId {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(100);
    let seed = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut rng = seed | 1;
    let mut next = move |amp: f32| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
    };
    let rank = 2usize;
    let sites = e
        .instance()
        .config()
        .lora_sites()
        .iter()
        .map(|s| AdapterSite {
            base: s.name.clone(),
            a: (0..rank * s.spec.inn).map(|_| next(1.0 / (s.spec.inn as f32).sqrt())).collect(),
            b: (0..s.spec.out * rank).map(|_| next(0.8)).collect(),
        })
        .collect();
    let adapter = DeviceAdapter { rank: rank as u32, alpha: 4.0, sites };
    e.load_adapter(VerifiedAdapter { label: format!("adapter-{seed}"), digest: format!("sha256:{seed:064x}"), adapter }).expect("the adapter fits the tiny GGUF's sites")
}

/// The tokens the engine greedily produces for fixed prompts, served together and then through the
/// prefix cache, recorded from the engine before its KV layout changed: any change of layout
/// (blocks, tables, slots) must leave them as they are.
fn greedy_tokens_are_the_recorded_ones() {
    greedy_tokens_are_the_recorded_ones_on(engine);
}

/// Paging the KV through block tables must not move a single token.
fn greedy_tokens_are_the_recorded_ones_with_its_kv_paged() {
    greedy_tokens_are_the_recorded_ones_on(paged_engine);
}

fn greedy_tokens_are_the_recorded_ones_on(make: fn() -> Qwen35GgufEngine) {
    let long: Vec<u32> = (0..72u32).map(|i| 3 + (i * 7 + i * i) % 26).collect();
    let mut extended = long.clone();
    extended.extend([5, 9, 11, 4, 8, 20, 13, 6]);
    let prompts = [(1..=20u32).map(|i| 3 + i % 26).collect::<Vec<u32>>(), long, extended];
    let mut sched = Scheduler::new(make(), 4);
    let first: Vec<u64> = prompts[..2].iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: 12, stop: Vec::new() })).collect();
    let together = sched.run();
    let warm = sched.submit(Request { prompt: prompts[2].clone(), max_new: 12, stop: Vec::new() });
    let warm = sched.run().remove(&warm).expect("the extended prompt completes");
    let got = [together[&first[0]].clone(), together[&first[1]].clone(), warm];
    assert_eq!(sched.prefix_stats().0, 72, "the extended prompt resumed from the 72-token entry");
    assert_eq!(got.each_ref().map(|t| t.as_slice()), RECORDED.each_ref().map(|t| t.as_slice()), "the engine's greedy tokens changed");
}

/// The compact KV tiers store a scale (int8) or packed words (bf16) beside the rows: paging must address
/// those exactly as the rows, so each tier's tokens equal its own one-block tokens.
fn paging_does_not_move_the_tokens_of_any_kv_tier() {
    let long: Vec<u32> = (0..72u32).map(|i| 3 + (i * 7 + i * i) % 26).collect();
    let mut extended = long.clone();
    extended.extend([5, 9, 11, 4, 8, 20, 13, 6]);
    for tier in KvTier::ALL {
        let run = |rows: u32| {
            let mut sched = Scheduler::new(engine_with_kv(rows, tier), 4);
            let a = sched.submit(Request { prompt: long.clone(), max_new: 12, stop: Vec::new() });
            let b = sched.submit(Request { prompt: (1..=20u32).map(|i| 3 + i % 26).collect(), max_new: 12, stop: Vec::new() });
            let mut together = sched.run();
            let warm = sched.submit(Request { prompt: extended.clone(), max_new: 12, stop: Vec::new() });
            vec![together.remove(&a).unwrap(), together.remove(&b).unwrap(), sched.run().remove(&warm).unwrap()]
        };
        assert_eq!(run(16), run(0), "{tier}: paged tokens differ from the one-block layout's");
    }
}

const RECORDED: [[u32; 12]; 3] = [[10, 9, 26, 17, 9, 10, 1, 8, 21, 12, 2, 4], [11, 9, 1, 19, 2, 13, 4, 26, 11, 18, 17, 22], [22, 12, 2, 12, 7, 17, 22, 14, 9, 1, 28, 16]];

gpu_core::card_tests!(
    the_released_gguf_engine_conforms,
    the_released_gguf_engine_conforms_with_its_kv_paged,
    greedy_tokens_are_the_recorded_ones,
    greedy_tokens_are_the_recorded_ones_with_its_kv_paged,
    paging_does_not_move_the_tokens_of_any_kv_tier
);
