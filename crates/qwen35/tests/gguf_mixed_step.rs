// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching LLM serving that reads
// every weight once for the decoding conversations and the prompts arriving
// beside them. If your team needs expertise in LLM serving throughput then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The released-GGUF engine's mixed iteration (decode rows and prompt slices of several sequences as ONE forward,
//! [`model::serve::PagedDecoder::forward_mixed`]) against the separate passes, on a 2-layer random-weight GGUF with
//! the real tensor names and layer kinds. Run with `-- --device gpu<i> --backend cuda`.

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillJob, PrefillProgress, Request, Scheduler};
use qwen35::int8_gguf_resident::{budgeted_gpus, PrefillPolicy, Qwen35GgufEngine, Qwen35GgufResident};

const CAP: u32 = 512;

fn engine(mixed: bool) -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("brain-gguf-mixed-{}-{n}.gguf", std::process::id())).to_string_lossy().into_owned();
    qwen35::gguf_import::testing::write_random_servable_gguf(&path, 7);
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the suite needs a GPU").0;
    let resident = Qwen35GgufResident::new(path.clone(), devices, CAP, TierPolicy::uniform(Dtype::I8)).with_max_batch(8).with_prefix_slots(4).with_kv_block_rows(16);
    let (mut engine, _, _) = resident.activate_engine(&[device]).expect("the tiny GGUF loads");
    std::fs::remove_file(path).ok();
    engine.set_mixed_step(mixed);
    engine
}

fn prompt_of(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 3 + (seed * 5 + i * 7 + i * i) % 26).collect()
}

/// Staggered requests through the scheduler: the long prompts arrive while the short ones decode, so their slices
/// ride in the decode iterations.
fn tokens(mixed: bool) -> Vec<Vec<u32>> {
    let mut sched = Scheduler::new(engine(mixed), 8);
    let mut ids = Vec::new();
    for (seed, len, max_new) in [(1u32, 20u32, 24usize), (2, 33, 24)] {
        ids.push(sched.submit(Request { prompt: prompt_of(seed, len), max_new, stop: Vec::new() }));
    }
    for _ in 0..4 {
        sched.step();
    }
    for (seed, len, max_new) in [(3u32, 200u32, 10usize), (4, 150, 10), (5, 90, 10)] {
        ids.push(sched.submit(Request { prompt: prompt_of(seed, len), max_new, stop: Vec::new() }));
    }
    let mut done = sched.run();
    ids.iter().map(|id| done.remove(id).expect("the request completes")).collect()
}

/// A mixed iteration gives the tokens separate passes give. A decode row's projections run as rows of a wide GEMM,
/// not of the decode GEMV, so the arithmetic differs in rounding and a near-tie of the tiny random model's flat
/// logits may flip late in a sequence: every request must start with the same tokens (a prompt read in wrongly
/// would not), and the logits of one forward are compared at the numerics floor in the test below.
fn mixed_iterations_give_the_tokens_separate_passes_give() {
    const AGREE: usize = 12;
    for (n, (m, s)) in tokens(true).into_iter().zip(tokens(false)).enumerate() {
        assert_eq!(m.len(), s.len(), "request {n}");
        let agree = m.iter().zip(&s).take_while(|(a, b)| a == b).count();
        assert!(agree >= AGREE.min(m.len()), "request {n} diverges after {agree} tokens: {m:?} against {s:?}");
    }
}

/// One mixed forward against the two passes on twin engines: the decode rows' candidates agree (same top token,
/// logits within the numerics of a different GEMM row count) and the prompts' final hidden rows are close.
fn one_mixed_forward_matches_a_prefill_round_and_a_decode_step() {
    let (mut a, mut b) = (engine(true), engine(false));
    let running: Vec<Vec<u32>> = (0..3).map(|i| prompt_of(10 + i, 20 + 7 * i)).collect();
    let fresh = [prompt_of(20, 100), prompt_of(21, 60)];
    let start = |e: &mut Qwen35GgufEngine| -> (Vec<BlockTable>, Vec<u32>) {
        let mut tables = Vec::new();
        let mut inputs = Vec::new();
        for p in &running {
            let mut t = BlockTable::new();
            let h = e.prefill(&mut t, p);
            inputs.push(e.admit_greedy(&h));
            tables.push(t);
        }
        (tables, inputs)
    };
    let (mut ta, ia) = start(&mut a);
    let (mut tb, ib) = start(&mut b);
    assert_eq!(ia, ib);
    let (mut na, mut nb) = ([BlockTable::new(), BlockTable::new()], [BlockTable::new(), BlockTable::new()]);
    let mut refs_a: Vec<&mut BlockTable> = ta.iter_mut().collect();
    let [n0, n1] = &mut na;
    let mut jobs_a = [PrefillJob { table: n0, prompt: &fresh[0], budget: 100, embeds: None }, PrefillJob { table: n1, prompt: &fresh[1], budget: 60, embeds: None }];
    let mixed = a.forward_mixed(&mut refs_a, &ia, 4, &mut jobs_a);
    let [m0, m1] = &mut nb;
    let mut jobs_b = [PrefillJob { table: m0, prompt: &fresh[0], budget: 100, embeds: None }, PrefillJob { table: m1, prompt: &fresh[1], budget: 60, embeds: None }];
    let prefilled = b.prefill_chunks(&mut jobs_b);
    let mut refs_b: Vec<&mut BlockTable> = tb.iter_mut().collect();
    let decoded = b.forward_batched_topk(&mut refs_b, &ib, 4);
    for (row, (x, y)) in mixed.decoded.iter().zip(&decoded).enumerate() {
        assert_eq!(x[0].0, y[0].0, "decode row {row}: the top token differs");
        for (cx, cy) in x.iter().zip(y) {
            assert!((cx.1 - cy.1).abs() <= 5e-2 * cy.1.abs().max(1.0), "decode row {row}: logit {} against {}", cx.1, cy.1);
        }
    }
    for (job, (x, y)) in mixed.prefilled.iter().zip(&prefilled).enumerate() {
        let (hx, hy) = match (x, y) {
            (PrefillProgress::Done(hx), PrefillProgress::Done(hy)) => (hx, hy),
            // The engine ends a slice at a prompt's checkpoint: both ways must stop at the same place.
            (PrefillProgress::Partial { consumed: cx }, PrefillProgress::Partial { consumed: cy }) => {
                assert_eq!(cx, cy, "job {job}: slices differ");
                continue;
            }
            _ => panic!("job {job}: one way finished the prompt and the other did not"),
        };
        let (num, den) = hx.iter().zip(hy).fold((0f64, 0f64), |(n, d), (&p, &q)| (n + ((p - q) as f64).powi(2), d + (q as f64).powi(2)));
        assert!((num / den).sqrt() < 5e-2, "job {job}: hidden rel-L2 {}", (num / den).sqrt());
    }
}

/// Questions about one document arriving beside a decoding conversation still prefill the document once.
fn mixed_iterations_keep_the_prefix_cache_and_in_flight_sharing() {
    let doc = prompt_of(1, 128);
    let ask = |seed: u32| doc.iter().copied().chain(prompt_of(seed, 100)).collect::<Vec<u32>>();
    let mut e = engine(true);
    // Slices of 32 tokens: the 128-token document is four of them, long enough to be worth waiting for.
    e.set_prefill_policy(PrefillPolicy { slice_tokens: 32, ..PrefillPolicy::default() });
    let mut sched = Scheduler::new(e, 8);
    let busy = sched.submit(Request { prompt: prompt_of(9, 20), max_new: 40, stop: Vec::new() });
    for _ in 0..3 {
        sched.step();
    }
    let ids: Vec<u64> = (2..5).map(|q| sched.submit(Request { prompt: ask(q), max_new: 6, stop: Vec::new() })).collect();
    let done = sched.run();
    assert_eq!(done[&busy].len(), 40);
    assert!(ids.iter().all(|id| done[id].len() == 6));
    assert_eq!(sched.prefix_stats().0, 2 * 128, "the second and third question resume from the document");
}

gpu_core::card_tests!(mixed_iterations_keep_the_prefix_cache_and_in_flight_sharing, mixed_iterations_give_the_tokens_separate_passes_give, one_mixed_forward_matches_a_prefill_round_and_a_decode_step);
