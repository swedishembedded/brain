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
use model::paged::BlockTable;
use model::serve::{AdapterId, PagedDecoder, Request, Scheduler};
use qwen35::int8_gguf_resident::{budgeted_gpus, PrefillPolicy, Qwen35GgufEngine, Qwen35GgufResident};

/// Per-sequence capacity: the prefix test's 152-token prompt (the engine caches a checkpoint on a 64-token boundary
/// at least 16 tokens short of the prompt's end) plus its 8-token extension and decode.
const CAP: u32 = 192;
const PREFIX_PROMPT_TOKENS: usize = 152;

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
    engine_with_pool(kv_block_rows, kv_tier, 0)
}

/// `kv_pool_rows` 0 sizes the paged pool as the one-block layout's memory.
fn engine_with_pool(kv_block_rows: u32, kv_tier: KvTier, kv_pool_rows: u32) -> Qwen35GgufEngine {
    engine_with_cap(kv_block_rows, kv_tier, kv_pool_rows, CAP)
}

/// [`engine_with_pool`] with an explicit per-sequence capacity (a pool smaller than one `CAP` sequence needs a smaller one).
fn engine_with_cap(kv_block_rows: u32, kv_tier: KvTier, kv_pool_rows: u32, cap: u32) -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("brain-gguf-conformance-{}-{n}.gguf", std::process::id())).to_string_lossy().into_owned();
    qwen35::gguf_import::testing::write_random_servable_gguf(&path, 7);
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let bank = AdapterBankRequest { slots: 2, rank_cap: 4, targets: AdapterTargets::ALL, max_registered: 4 };
    let resident = Qwen35GgufResident::new(path.clone(), devices, cap, TierPolicy::uniform(Dtype::I8)).with_max_batch(4).with_prefix_slots(4).with_kv_block_rows(kv_block_rows).with_kv_pool_rows(kv_pool_rows).with_kv_tier(kv_tier).with_adapter_bank(bank);
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
    assert_eq!(sched.prefix_stats().0, 0, "a 72-token prompt is too short for a checkpoint (a whole 64-token chunk, 16 tokens short of its end), so nothing is resumed");
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

/// A paged pool smaller than `max_batch * cap` holds the sequences at their real lengths: four
/// 32-token sequences fit eight 16-row blocks, which the one-block layout (four 96-row windows)
/// could never hold in that memory; six blocks cannot hold all four, so the scheduler must preempt
/// and recompute. In both cases the tokens are the one-block layout's, and every block and state
/// row is back in the pool afterwards.
fn a_small_pool_serves_what_fits_and_preempts_what_does_not() {
    let prompts: Vec<Vec<u32>> = (0..4u32).map(|k| (1..=20u32).map(|i| 3 + (i * (k + 2) + k) % 26).collect()).collect();
    let run = |mut sched: Scheduler<Qwen35GgufEngine>| {
        let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: 12, stop: Vec::new() })).collect();
        let mut done = sched.run();
        let tokens: Vec<Vec<u32>> = ids.iter().map(|id| done.remove(id).expect("every request completes")).collect();
        (tokens, sched)
    };
    let (want, _) = run(Scheduler::new(engine_with(0), 4));
    for pool_blocks in [8u32, 6] {
        let engine = engine_with_cap(16, KvTier::F32, pool_blocks * 16, 96);
        assert_eq!(engine.kv_pool_capacity_tokens(), pool_blocks as u64 * 16);
        let (got, mut sched) = run(Scheduler::new(engine, 4));
        assert_eq!(got, want, "{pool_blocks} blocks: tokens differ from the one-block layout's");
        let e = sched.decoder_mut();
        assert_eq!(e.free_blocks(), pool_blocks, "{pool_blocks} blocks: blocks leaked");
    }
}

/// A prompt of `n` tokens, distinct per `seed` (the engine caches prompts of at least 64).
fn prompt_of(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 3 + (i * (seed + 5) + i * i + seed) % 26).collect()
}

/// Prefill `prompt` on its own and release it, leaving it in the engine's prefix cache.
fn cache_prompt(e: &mut Qwen35GgufEngine, prompt: &[u32]) {
    let mut table = BlockTable::new();
    e.prefill(&mut table, prompt);
    e.release_table(&mut table);
}

/// A prompt resuming a cached one does not copy the cached KV: it holds the same physical blocks. The entry of a
/// 152-token prompt is its 128-token checkpoint: eight 16-row blocks; a 160-token prompt extending the prompt holds
/// the eight as the entry's own and takes two blocks of its own for the rest, so the pool is short of ten blocks, not
/// eighteen.
fn a_prefix_hit_shares_the_cached_blocks() {
    let mut e = paged_engine();
    let total = e.free_blocks();
    let first = prompt_of(1, 152);
    cache_prompt(&mut e, &first);
    assert_eq!(e.free_blocks(), total - 8, "the entry retains its eight blocks");
    let mut extended = first.clone();
    extended.extend(prompt_of(2, 8));
    let mut table = BlockTable::new();
    e.prefill(&mut table, &extended);
    assert_eq!(e.prefix_stats().0, 128, "the extended prompt resumed from the entry");
    assert_eq!(e.free_blocks(), total - 10, "eight blocks shared, two of its own");
    e.release_table(&mut table);
}

/// Admission reclaims cached prompts for a waiting request but spares the one that request resumes from, even when
/// it is the least recently used: reclaiming in recency order would free exactly the rows the request came to reuse.
fn reclaim_spares_the_prefix_the_waiting_prompt_resumes_from() {
    let mut e = paged_engine();
    let (a, b) = (prompt_of(1, 152), prompt_of(3, 152));
    cache_prompt(&mut e, &a);
    cache_prompt(&mut e, &b);
    let free = e.free_blocks();
    let mut a_next = a.clone();
    a_next.extend(prompt_of(2, 8));
    assert_eq!(e.reclaim_prefix_keeping(8, &a_next, None), 8, "the other conversation's eight blocks");
    assert_eq!(e.free_blocks(), free + 8);
    let mut table = BlockTable::new();
    e.prefill(&mut table, &a_next);
    assert_eq!(e.prefix_stats().0, 128, "the spared entry is still there to resume from");
    e.release_table(&mut table);
}

/// A chat prompt ends with the generation prompt, which the next turn re-renders as the finished assistant message:
/// the next turn shares the whole cached prompt except its last few tokens. It must still resume from the cached
/// prompt, short of those tokens, and answer exactly as a cold prefill does (an entry for the whole prompt matched
/// only a prompt that extended it, so a thinking-mode conversation prefilled its full history every turn).
fn a_prompt_diverging_in_its_last_tokens_resumes_from_the_cached_prompts_checkpoint() {
    let first = prompt_of(1, 152);
    let mut next = first[..148].to_vec();
    next.extend(prompt_of(2, 6));
    let mut sched = Scheduler::new(paged_engine(), 4);
    let id = sched.submit(Request { prompt: first.clone(), max_new: 4, stop: Vec::new() });
    sched.run().remove(&id).expect("the first turn completes");
    let before = sched.prefix_stats().0;
    let id = sched.submit(Request { prompt: next.clone(), max_new: 8, stop: Vec::new() });
    let warm = sched.run().remove(&id).expect("the next turn completes");
    let hit = sched.prefix_stats().0 - before;
    assert_eq!(hit, 128, "the next turn resumed from {hit} cached tokens: expected the first prompt's checkpoint, a whole 64-token chunk short of its end");
    let mut cold = Scheduler::new(paged_engine(), 4);
    let id = cold.submit(Request { prompt: next, max_new: 8, stop: Vec::new() });
    assert_eq!(warm, cold.run().remove(&id).unwrap(), "output after resuming from a checkpoint must equal a cold prefill's");
}

/// Single-stream greedy decoding is reproducible: the same prompt through two fresh engines yields the same tokens, and
/// prefilling a prompt twice on one engine yields the same last hidden row bit for bit. (A served 27B repeated 6 times
/// per prompt, thinking on and off, cache on and off, returned the same logprob bits at every step; a batch of
/// several sequences is not batch-invariant and is not claimed here.)
fn single_stream_greedy_decoding_is_reproducible() {
    let prompt = prompt_of(4, 70);
    let run = || {
        let mut sched = Scheduler::new(paged_engine(), 4);
        let id = sched.submit(Request { prompt: prompt.clone(), max_new: 24, stop: Vec::new() });
        sched.run().remove(&id).expect("the request completes")
    };
    assert_eq!(run(), run(), "two fresh engines decoded the same prompt differently");
    let mut e = paged_engine();
    let short = prompt_of(5, 40); // below the cache minimum, so both prefills compute
    let hidden = |e: &mut Qwen35GgufEngine| {
        let mut table = BlockTable::new();
        let h = e.prefill(&mut table, &short);
        e.release_table(&mut table);
        h
    };
    let (a, b) = (hidden(&mut e), hidden(&mut e));
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "the same prompt prefilled twice gave different hidden rows");
}

/// A prompt resumed from a cached checkpoint prefills to the very hidden row a cold prefill of it does: the checkpoint
/// sits on a whole recurrent-state chunk, so both walk the same chunks, and the cached state and KV rows are copies.
fn a_resumed_prefill_is_bitwise_a_cold_one() {
    let first = prompt_of(1, 152);
    let mut extended = first.clone();
    extended.extend(prompt_of(2, 8));
    let mut warm_engine = paged_engine();
    cache_prompt(&mut warm_engine, &first);
    let mut table = BlockTable::new();
    let warm = warm_engine.prefill(&mut table, &extended);
    assert_eq!(warm_engine.prefix_stats().0, 128, "the extended prompt resumed from the checkpoint");
    warm_engine.release_table(&mut table);
    let mut cold_engine = paged_engine();
    let mut table = BlockTable::new();
    let cold = cold_engine.prefill(&mut table, &extended);
    assert_eq!(cold_engine.prefix_stats().0, 0);
    cold_engine.release_table(&mut table);
    assert!(warm.iter().zip(&cold).all(|(w, c)| w.to_bits() == c.to_bits()), "a resumed prefill's hidden row differs from a cold prefill's");
}

/// Prompts that share a document and then diverge: the first one's checkpoint ends past the document, so the second
/// cannot resume and files the state where it leaves the first (a branch point on a whole recurrent chunk); every
/// later prompt sharing the document resumes from it, to the hidden row of a cold prefill.
fn prompts_sharing_a_document_resume_from_where_they_part() {
    const CAP: u32 = 320;
    let doc = prompt_of(1, 128);
    let ask = |seed: u32| doc.iter().copied().chain(prompt_of(seed, 100)).collect::<Vec<u32>>();
    let mut e = engine_with_cap(16, KvTier::F32, 0, CAP);
    cache_prompt(&mut e, &ask(2));
    cache_prompt(&mut e, &ask(3));
    assert_eq!(e.prefix_stats().0, 0, "the first entry ends 64 tokens past the document: the second prompt cannot resume");
    let third = ask(4);
    let mut table = BlockTable::new();
    let warm = e.prefill(&mut table, &third);
    assert_eq!(e.prefix_stats().0, 128, "the third prompt resumes from the document");
    e.release_table(&mut table);
    let mut cold_engine = engine_with_cap(16, KvTier::F32, 0, CAP);
    let mut table = BlockTable::new();
    let cold = cold_engine.prefill(&mut table, &third);
    cold_engine.release_table(&mut table);
    assert!(warm.iter().zip(&cold).all(|(w, c)| w.to_bits() == c.to_bits()), "a prefill resumed from a branch point differs from a cold one");
}

/// Questions about one document submitted together: none of them is cached when they are admitted, so the scheduler
/// holds the later ones back while the first files the document as it passes it, and they resume from it. Each still
/// answers exactly as it does alone.
fn questions_arriving_together_prefill_their_document_once() {
    const CAP: u32 = 320;
    let doc = prompt_of(1, 128);
    let ask = |seed: u32| doc.iter().copied().chain(prompt_of(seed, 100)).collect::<Vec<u32>>();
    let solo = |prompt: Vec<u32>| {
        let mut sched = Scheduler::new(engine_with_cap(16, KvTier::F32, 0, CAP), 1);
        let id = sched.submit(Request { prompt, max_new: 6, stop: Vec::new() });
        sched.run().remove(&id).expect("finished")
    };
    let mut e = engine_with_cap(16, KvTier::F32, 0, CAP);
    // Slices of 32 tokens: the 128-token document is four of them, long enough to be worth waiting for.
    e.set_prefill_policy(PrefillPolicy { slice_tokens: 32, ..PrefillPolicy::default() });
    let mut sched = Scheduler::new(e, 4);
    let ids: Vec<u64> = (2..5).map(|q| sched.submit(Request { prompt: ask(q), max_new: 6, stop: Vec::new() })).collect();
    let mut got = sched.run();
    assert_eq!(sched.prefix_stats().0, 2 * 128, "the second and third question resume from the document");
    for (id, q) in ids.into_iter().zip(2..5) {
        assert_eq!(got.remove(&id).expect("finished"), solo(ask(q)), "question {q} answers as it does alone");
    }
}

const RECORDED: [[u32; 12]; 3] = [[10, 9, 26, 17, 9, 10, 1, 8, 21, 12, 2, 4], [11, 9, 1, 19, 2, 13, 4, 26, 11, 18, 17, 22], [22, 12, 2, 12, 7, 17, 22, 14, 9, 1, 28, 16]];

gpu_core::card_tests!(
    the_released_gguf_engine_conforms,
    the_released_gguf_engine_conforms_with_its_kv_paged,
    greedy_tokens_are_the_recorded_ones,
    greedy_tokens_are_the_recorded_ones_with_its_kv_paged,
    paging_does_not_move_the_tokens_of_any_kv_tier,
    a_small_pool_serves_what_fits_and_preempts_what_does_not,
    a_prefix_hit_shares_the_cached_blocks,
    reclaim_spares_the_prefix_the_waiting_prompt_resumes_from,
    single_stream_greedy_decoding_is_reproducible,
    a_resumed_prefill_is_bitwise_a_cold_one,
    a_prompt_diverging_in_its_last_tokens_resumes_from_the_cached_prompts_checkpoint,
    prompts_sharing_a_document_resume_from_where_they_part,
    questions_arriving_together_prefill_their_document_once
);
