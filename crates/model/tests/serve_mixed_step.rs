// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving that runs the next
// token of every conversation and the prompts still being read in as one
// forward under a token-interval objective, for its clients. If your team
// needs expertise in token-budgeted continuous batching, then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec of the mixed iteration: a decoder that carries the decode rows of every
//! running sequence and slices of the waiting prompts in ONE forward
//! ([`PagedDecoder::mixed_step`] / [`PagedDecoder::forward_mixed`]).
//!
//! * the tokens are the ones separate passes give;
//! * a forward never carries more rows than the decoder takes;
//! * the prompt part is sized so the iteration is expected to fit the decoder's
//!   latency target, from the cost the scheduler measured (virtual time here:
//!   a forward costs a fixed part plus a part per row);
//! * a prompt always advances, however busy decode is;
//! * with nothing decoding the forward is filled.
//!
//! The toy keeps real histories: a token depends on every earlier token of its
//! own sequence, so a row mixed up with another sequence's changes the output.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{MixedLimits, MixedOutcome, PagedDecoder, PrefillJob, PrefillProgress, Request, Scheduler};

const VOCAB: usize = 61;
const BLOCK: u32 = 8;
/// Virtual seconds of a forward: the weights read once, plus a part per row.
const FIXED_SECS: f64 = 0.020;
const PER_ROW_SECS: f64 = 0.0001;
const TARGET: std::time::Duration = std::time::Duration::from_millis(45);
const LIMITS: MixedLimits = MixedLimits { max_rows: 512, min_prompt_rows: 32 };

#[derive(Default)]
struct Log {
    /// Per forward: (decode rows, prompt rows).
    forwards: Vec<(usize, usize)>,
    /// Virtual time at the end of each forward.
    ends: Vec<f64>,
}

struct Toy {
    alloc: BlockAllocator,
    history: HashMap<u32, Vec<u32>>,
    mixed: bool,
    log: Arc<Mutex<Log>>,
    now: Arc<Mutex<f64>>,
}

impl Toy {
    fn new(mixed: bool) -> (Toy, Arc<Mutex<Log>>, Arc<Mutex<f64>>) {
        let (log, now) = (Arc::new(Mutex::new(Log::default())), Arc::new(Mutex::new(0.0)));
        (Toy { alloc: BlockAllocator::new(4096, BLOCK), history: HashMap::new(), mixed, log: log.clone(), now: now.clone() }, log, now)
    }

    fn digest(history: &[u32]) -> u32 {
        (history.iter().fold(7u64, |h, &t| h.wrapping_mul(1_000_003) ^ (t as u64 + 1)) >> 5) as u32 % VOCAB as u32
    }

    fn slice(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        let from = table.len() as usize;
        let to = (from + budget.max(1)).min(prompt.len());
        table.reserve((to - from) as u32, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        self.history.entry(table.blocks()[0]).or_default().extend_from_slice(&prompt[from..to]);
        if to == prompt.len() {
            PrefillProgress::Done(vec![Self::digest(&self.history[&table.blocks()[0]]) as f32])
        } else {
            PrefillProgress::Partial { consumed: to - from }
        }
    }

    fn decode(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &input)| {
                t.append(&mut self.alloc).unwrap();
                let h = self.history.get_mut(&t.blocks()[0]).unwrap();
                h.push(input);
                Self::digest(h)
            })
            .collect()
    }
}

impl PagedDecoder for Toy {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        64
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        8192
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(BLOCK)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(b) = t.blocks().first() {
            self.history.remove(b);
        }
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        match self.slice(table, prompt, prompt.len()) {
            PrefillProgress::Done(h) => h,
            PrefillProgress::Partial { .. } => unreachable!("the whole prompt"),
        }
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        self.slice(table, prompt, budget)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[hidden[0] as usize] = 1.0;
        l
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.decode(tables, inputs)
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        4096 * BLOCK as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("greedy only")
    }
    fn topk_capacity(&self) -> usize {
        1
    }
    fn step_latency_target(&self) -> Option<std::time::Duration> {
        Some(TARGET)
    }
    fn mixed_step(&self) -> Option<MixedLimits> {
        self.mixed.then_some(LIMITS)
    }
    fn forward_mixed(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], cands: usize, jobs: &mut [PrefillJob<'_>]) -> MixedOutcome {
        assert_eq!(cands, 0, "greedy only");
        let prompt_rows: usize = jobs.iter().map(|j| j.budget).sum();
        let rows = tables.len() + prompt_rows;
        assert!(rows <= LIMITS.max_rows, "a forward of {rows} rows exceeds what the decoder takes");
        let prefilled = jobs.iter_mut().map(|j| self.slice(j.table, j.prompt, j.budget)).collect();
        let decoded = self.decode(tables, inputs).into_iter().map(|t| vec![(t, 0.0)]).collect();
        let mut now = self.now.lock().unwrap();
        *now += FIXED_SECS + PER_ROW_SECS * rows as f64;
        let mut log = self.log.lock().unwrap();
        log.forwards.push((tables.len(), prompt_rows));
        log.ends.push(*now);
        MixedOutcome { decoded, prefilled }
    }
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (seed * 13 + i * 5 + i / 7) % VOCAB as u32).collect()
}

fn scheduler(mixed: bool) -> (Scheduler<Toy>, Arc<Mutex<Log>>) {
    let (toy, log, now) = Toy::new(mixed);
    let mut s = Scheduler::new(toy, 64);
    s.set_clock(Box::new(move || *now.lock().unwrap()));
    (s, log)
}

/// Requests: (arrival iteration, prompt seed, prompt length, max_new).
type Load = [(usize, u32, usize, usize)];

fn run(s: &mut Scheduler<Toy>, load: &Load) -> HashMap<usize, Vec<u32>> {
    let mut ids: HashMap<u64, usize> = HashMap::new();
    let mut out = HashMap::new();
    for iteration in 0..10_000 {
        for (n, &(at, seed, len, max_new)) in load.iter().enumerate() {
            if at == iteration {
                ids.insert(s.submit(Request { prompt: prompt(seed, len), max_new, stop: vec![] }), n);
            }
        }
        for (id, toks) in s.step() {
            out.insert(ids[&id], toks);
        }
        if out.len() == load.len() {
            return out;
        }
    }
    panic!("the load did not finish");
}

const STAGGERED: &Load = &[(0, 1, 40, 30), (0, 2, 300, 12), (3, 3, 1200, 8), (5, 4, 17, 40), (9, 5, 700, 20), (9, 6, 90, 6), (30, 7, 2000, 4)];

#[test]
fn mixed_iterations_give_the_tokens_separate_passes_give() {
    let (mut mixed, _) = scheduler(true);
    let (mut separate, _) = scheduler(false);
    assert_eq!(run(&mut mixed, STAGGERED), run(&mut separate, STAGGERED));
}

#[test]
fn the_default_mixed_forward_is_the_two_separate_passes() {
    let (mut a, _, _) = Toy::new(false);
    let (mut b, _, _) = Toy::new(false);
    let (p0, p1) = (prompt(1, 20), prompt(2, 50));
    // A running sequence on each decoder, and two prompts to read in.
    let (mut t_a, mut t_b) = (BlockTable::new(), BlockTable::new());
    a.prefill(&mut t_a, &p0);
    b.prefill(&mut t_b, &p0);
    let (mut new_a, mut new_b) = (BlockTable::new(), BlockTable::new());
    let mixed = a.forward_mixed(&mut [&mut t_a], &[3], 0, &mut [PrefillJob { table: &mut new_a, prompt: &p1, budget: 50, embeds: None }]);
    let prefilled = b.prefill_chunks(&mut [PrefillJob { table: &mut new_b, prompt: &p1, budget: 50, embeds: None }]);
    let decoded = b.forward_batched_greedy(&mut [&mut t_b], &[3]);
    assert_eq!(mixed.decoded, vec![vec![(decoded[0], 0.0)]]);
    match (&mixed.prefilled[0], &prefilled[0]) {
        (PrefillProgress::Done(x), PrefillProgress::Done(y)) => assert_eq!(x, y),
        _ => panic!("the prompt fits one slice"),
    }
}

#[test]
fn a_forward_is_sized_to_the_latency_target_once_the_step_cost_is_known() {
    let (mut s, log) = scheduler(true);
    // Eight sequences decoding for a long time while long prompts keep arriving.
    let mut load: Vec<(usize, u32, usize, usize)> = (0..8).map(|i| (0, i as u32, 16, 400)).collect();
    load.extend((0..6).map(|i| (10 + i * 5, 20 + i as u32, 3000, 2)));
    run(&mut s, &load);
    let log = log.lock().unwrap();
    // 45 ms holds (45 - 20) / 0.1 = 250 rows. After the cost is learnt no forward that carries decode rows and a
    // slice larger than the minimum is expected to overrun it (a small margin for the fit).
    let warm = &log.forwards[12..];
    let mixed: Vec<_> = warm.iter().filter(|&&(d, p)| d > 0 && p > LIMITS.min_prompt_rows).collect();
    assert!(!mixed.is_empty(), "no forward mixed decode rows and a real slice of prompt: {:?}", log.forwards);
    for &&(d, p) in &mixed {
        let secs = FIXED_SECS + PER_ROW_SECS * (d + p) as f64;
        assert!(secs <= TARGET.as_secs_f64() * 1.05, "a forward of {d} decode and {p} prompt rows takes {secs} s, over the {TARGET:?} target");
    }
    // ... and the target is used, not left idle: the slices are far larger than the minimum.
    assert!(mixed.iter().any(|&&(d, p)| d + p >= 200), "the slices stayed small: {mixed:?}");
}

#[test]
fn a_prompt_advances_every_iteration_however_busy_decode_is() {
    let (mut s, log) = scheduler(true);
    // 60 sequences decoding: their rows alone cost 26 ms of the 45, which leaves less than the minimum slice.
    let mut load: Vec<(usize, u32, usize, usize)> = (0..60).map(|i| (0, i as u32, 8, 200)).collect();
    load.push((5, 99, 640, 2));
    run(&mut s, &load);
    let log = log.lock().unwrap();
    // Every forward from the prompt's first slice carries at least the minimum of it (or what is left), until it is in.
    let mut left = 640usize;
    for &(_, p) in log.forwards.iter().skip_while(|&&(_, p)| p == 0) {
        if left == 0 {
            break;
        }
        assert!(p >= LIMITS.min_prompt_rows.min(left), "a forward carried {p} prompt rows with {left} still to read: {:?}", log.forwards);
        left -= p.min(left);
    }
}

#[test]
fn with_nothing_decoding_the_forward_is_filled() {
    let (mut s, log) = scheduler(true);
    run(&mut s, &[(0, 1, 3000, 2)]);
    let log = log.lock().unwrap();
    assert_eq!(log.forwards[0], (0, LIMITS.max_rows), "{:?}", log.forwards);
}

#[test]
fn a_running_sequence_gets_a_token_every_forward() {
    let (mut s, log) = scheduler(true);
    let load = [(0, 1, 8, 60), (4, 2, 2500, 2), (6, 3, 2500, 2)];
    run(&mut s, &load);
    let log = log.lock().unwrap();
    // The short request decodes from its second forward until it ends: every forward in between carries its row.
    let decoding: Vec<_> = log.forwards.iter().skip(1).take(55).collect();
    assert!(decoding.iter().all(|&&(d, _)| d >= 1), "{decoding:?}");
}
