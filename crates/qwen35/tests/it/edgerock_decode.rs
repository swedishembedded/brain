// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! EdgeRock inference state at tiny dims: the window's ring KV and the anchor bank make every sequence's state a
//! fixed size, and every way of reaching a position (one token at a time, chunked prefill at awkward chunk sizes,
//! a batch of sequences at different positions, a snapshot restored) must give the logits the whole-sequence forward
//! gives. The reference is the goldens of `tools/goldens/edgerock_dump_reference.py` (T = 48 against a window of 11
//! and commit blocks of 4: the ring wraps four times and the 6-slot bank evicts at every block).
//!
//! Each scenario runs on every backend this machine has: the CPU JIT, the default device, and CUDA pinned to the
//! fp32 reference tier. A device whose arithmetic is TF32 is held to the tolerance `lora_gemm_precision` states.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::safetensors::StTensor;
use gpu_core::provider::library::{GemmPath, GemmPrecision};
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn to_map(tensors: Vec<StTensor>) -> HashMap<String, Vec<f32>> {
    tensors.into_iter().map(|t| (t.name, t.data)).collect()
}

struct Golden {
    weights: HashMap<String, Vec<f32>>,
    tokens: Vec<u32>,
    /// `[T, vocab]`
    logits: Vec<f32>,
}

fn load(variant: &str) -> Option<Golden> {
    let dir = brain_testutil::testdata(&format!("golden/edgerock/{variant}"));
    let (w_path, t_path) = (format!("{dir}/edgerock_tiny_weights.safetensors"), format!("{dir}/edgerock_tiny.safetensors"));
    if !Path::new(&w_path).exists() || !Path::new(&t_path).exists() {
        brain_testutil::skip(&format!("fixture {w_path} absent - run tools/goldens/edgerock_dump_reference.py --seed 7"));
        return None;
    }
    let mut weights = to_map(checkpoint::safetensors::read(&w_path).expect("read golden weights"));
    let mut taps = to_map(checkpoint::safetensors::read(&t_path).expect("read golden taps"));
    qwen35::import::fold_plain_rmsnorm_weights(&mut weights);
    let tokens = taps["tokens"].iter().map(|&v| v.round() as u32).collect();
    Some(Golden { weights, tokens, logits: taps.remove("logits").unwrap() })
}

/// How closely a device must reproduce the whole-sequence logits.
#[derive(Clone, Copy)]
enum Tol {
    /// fp32 arithmetic: worst logit difference.
    Exact(f32),
    /// TF32 tensor-core products: relative L2 error of a row.
    Tf32(f32),
}

impl Tol {
    fn check(self, what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        match self {
            Tol::Exact(max) => {
                let worst = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(worst < max, "{what}: worst logit difference {worst} (limit {max})");
            }
            Tol::Tf32(max) => {
                let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
                let den: f32 = want.iter().map(|b| b * b).sum();
                let rel = (num / den).sqrt();
                assert!(rel < max, "{what}: relative L2 error {rel} (limit {max})");
            }
        }
    }
}

/// Every device this machine offers, with a model built on it: `(name, model, tolerance)`. `t` is the model's decode
/// capacity (positions it may reach); the ring and the bank do not depend on it.
fn models(golden: &Golden, t: u32) -> Vec<(&'static str, Qwen35, Tol)> {
    let cfg = Qwen35Config::edgerock_tiny();
    let mut out = Vec::new();
    let on = |gpu: Gpu| Qwen35::new_on(gpu, cfg.clone(), 1, t, &golden.weights);
    out.push(("cpu", on(Gpu::new_cpu(pipelines())), Tol::Exact(2e-3)));
    let default = on(Gpu::new(pipelines()));
    let tol = match default.gemm_path() {
        GemmPath::Bf16TensorCore => Tol::Tf32(2e-2),
        GemmPath::Fp32Kernels { .. } => Tol::Exact(2e-3),
    };
    out.push(("default", default, tol));
    match Gpu::try_new_cuda(pipelines()) {
        Ok(gpu) => {
            let opts = TrainOpts { gemm_precision: GemmPrecision::Fp32Reference, ..TrainOpts::default() };
            out.push(("cuda-fp32", Qwen35::new_train_opts_on(gpu, cfg.clone(), 1, t, &golden.weights, &opts), Tol::Exact(2e-3)));
        }
        Err(e) => brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}")),
    }
    for (name, _, _) in &out {
        println!("{name}: {}", gpu_core::backend_name());
    }
    out
}

/// Logits of a `[d_model]` final-norm hidden state through the tied head.
fn logits_of(m: &Qwen35, hidden: &[f32]) -> Vec<f32> {
    let head = m.read_weight("tok.weight");
    let d = hidden.len();
    head.chunks(d).map(|row| row.iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
}

fn want_row(golden: &Golden, vocab: usize, i: usize) -> &[f32] {
    &golden.logits[i * vocab..(i + 1) * vocab]
}

const VARIANTS: [&str; 2] = ["tiny_text", "tiny_ties"];

#[test]
fn step_by_step_decode_matches_the_whole_sequence_at_every_position() {
    for variant in VARIANTS {
        let Some(golden) = load(variant) else { return };
        let cfg = Qwen35Config::edgerock_tiny();
        let vocab = cfg.vocab as usize;
        for (name, m, tol) in models(&golden, cfg.block_size) {
            m.reset_decode_cache();
            for (i, &tok) in golden.tokens.iter().enumerate() {
                let hidden = m.step(tok);
                tol.check(&format!("{variant}/{name} position {i}"), &logits_of(&m, &hidden), want_row(&golden, vocab, i));
            }
            assert_eq!(m.decode_pos(), golden.tokens.len() as u32);
        }
    }
}

#[test]
fn chunked_prefill_at_awkward_sizes_then_decode_matches_the_whole_sequence() {
    for variant in VARIANTS {
        let Some(golden) = load(variant) else { return };
        let cfg = Qwen35Config::edgerock_tiny();
        let vocab = cfg.vocab as usize;
        let prompt = 29usize; // past the window (11) and across seven commit blocks (4)
        for (name, m, tol) in models(&golden, cfg.block_size) {
            // 1, 3, 5, 13 and 48 are not multiples of the commit block (4) or the window (11); 48 is the whole
            // sequence in one round.
            for chunk in [1u32, 3, 5, 13, 48] {
                m.reset_decode_cache();
                let hidden = m.prefill_chunked(&golden.tokens[..prompt], chunk);
                tol.check(&format!("{variant}/{name} chunk {chunk} prompt end"), &logits_of(&m, &hidden), want_row(&golden, vocab, prompt - 1));
                for (i, &tok) in golden.tokens.iter().enumerate().skip(prompt) {
                    let hidden = m.step(tok);
                    tol.check(&format!("{variant}/{name} chunk {chunk} decode position {i}"), &logits_of(&m, &hidden), want_row(&golden, vocab, i));
                }
            }
            m.reset_decode_cache();
            let hidden = m.prefill_chunked(&golden.tokens, 48);
            tol.check(&format!("{variant}/{name} whole prompt in one round"), &logits_of(&m, &hidden), want_row(&golden, vocab, golden.tokens.len() - 1));
        }
    }
}

#[test]
fn a_batch_of_sequences_at_different_positions_equals_each_run_alone() {
    let Some(golden) = load("tiny_text") else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let vocab = cfg.vocab as usize;
    // Three sequences prefilled to 7, 19 and 30 tokens of the golden stream (in the ring's first fill, past its
    // wrap, and mid-block), then stepped together through the next 12 tokens. A fourth, idle slot stays untouched.
    let starts = [7usize, 19, 30];
    for (name, m, tol) in models(&golden, cfg.block_size) {
        // Each sequence alone, in a pool of its own: the rows the batch must reproduce.
        let alone: Vec<Vec<Vec<f32>>> = starts
            .iter()
            .map(|&start| {
                let solo = m.new_slot_pool(1, qwen35::model::SlotTiers::EXACT);
                m.slot_prefill(&solo, 0, &golden.tokens[..start], 13);
                (0..12).map(|step| m.slot_decode(&solo, &[(0, golden.tokens[start + step])]).remove(0)).collect()
            })
            .collect();
        let pool = m.new_slot_pool(4, qwen35::model::SlotTiers::EXACT);
        for (slot, &start) in starts.iter().enumerate() {
            m.slot_prefill(&pool, slot as u32, &golden.tokens[..start], 5);
        }
        // `step` indexes the golden tokens of every sequence and its recorded rows alike.
        #[allow(clippy::needless_range_loop)]
        for step in 0..12 {
            // Slot order in the batch is not slot order in the pool.
            let items: Vec<(u32, u32)> = [2usize, 0, 1].iter().map(|&slot| (slot as u32, golden.tokens[starts[slot] + step])).collect();
            let rows = m.slot_decode(&pool, &items);
            for (&(slot, _), row) in items.iter().zip(&rows) {
                let i = starts[slot as usize] + step;
                tol.check(&format!("{name} slot {slot} position {i} batched vs the whole sequence"), &logits_of(&m, row), want_row(&golden, vocab, i));
                tol.check(&format!("{name} slot {slot} position {i} batched vs alone"), &logits_of(&m, row), &logits_of(&m, &alone[slot as usize][step]));
            }
        }
        assert_eq!(pool.position(3), 0, "the idle slot did not move");
    }
}

#[test]
fn the_state_does_not_grow_over_ten_windows_of_tokens() {
    let Some(golden) = load("tiny_text") else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let window = cfg.swa_window.unwrap();
    let total = 12 * window + 1; // more than ten windows
    for (name, m, _) in models(&golden, total.next_multiple_of(8)) {
        let bytes = m.decode_state_bytes();
        let before = m.decode_state_allocations();
        assert!(bytes > 0 && !before.is_empty());
        m.reset_decode_cache();
        let mut worst = 0f32;
        for i in 0..total {
            let hidden = m.step((i * 7 + 3) % cfg.vocab);
            worst = hidden.iter().fold(worst, |w, x| w.max(x.abs()));
            assert!(hidden.iter().all(|x| x.is_finite()), "{name}: position {i} is not finite");
        }
        assert!(worst > 0.0);
        assert_eq!(m.decode_pos(), total);
        // The same buffers, the same sizes, after more than ten windows: nothing in the per-sequence state grew.
        assert_eq!(m.decode_state_allocations(), before, "{name}: the persistent state changed identity or size while decoding");
        assert_eq!(m.decode_state_bytes(), bytes);
        // What a cache that grows with the sequence would hold for the same run, for scale.
        let grown = u64::from(total) * u64::from(cfg.kv_dim()) * 4 * 2 * 2;
        println!("{name}: {bytes} state bytes after {total} tokens (a KV cache alone would be {grown} at this length, {} per 1000 tokens)", 1000u64 * u64::from(cfg.kv_dim()) * 4 * 2 * 2);
    }
}

#[test]
fn a_snapshot_restored_continues_identically() {
    let Some(golden) = load("tiny_text") else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    for (name, m, _) in models(&golden, cfg.block_size) {
        m.reset_decode_cache();
        m.prefill_chunked(&golden.tokens[..20], 7);
        let snap = m.decode_snapshot();
        let first: Vec<Vec<f32>> = golden.tokens[20..40].iter().map(|&t| m.step(t)).collect();
        // Restore, and take a different continuation in between: nothing of it may survive the restore.
        m.decode_restore(&snap);
        assert_eq!(m.decode_pos(), 20);
        let detour: Vec<Vec<f32>> = golden.tokens[..6].iter().map(|&t| m.step(t)).collect();
        assert!(detour.iter().zip(&first).any(|(a, b)| a != b), "the detour must differ from the original continuation");
        m.decode_restore(&snap);
        let again: Vec<Vec<f32>> = golden.tokens[20..40].iter().map(|&t| m.step(t)).collect();
        assert_eq!(first, again, "{name}: continuing from a restored snapshot must reproduce the original continuation bit for bit");
    }
}

#[test]
fn a_reused_slot_starts_from_an_empty_bank_and_ring() {
    let Some(golden) = load("tiny_text") else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let other: Vec<u32> = (0..40).map(|i| (i * 11 + 5) % cfg.vocab).collect();
    for (name, m, _) in models(&golden, cfg.block_size) {
        let pool = m.new_slot_pool(2, qwen35::model::SlotTiers::EXACT);
        let fresh = m.new_slot_pool(1, qwen35::model::SlotTiers::EXACT);
        let reference = m.slot_prefill(&fresh, 0, &golden.tokens[..30], 7);

        // Slot 1 serves a different sequence far past its window, is released, and serves the golden prompt.
        m.slot_prefill(&pool, 1, &other, 9);
        assert_eq!(pool.position(1), 40);
        let leaked = m.slot_prefill(&pool, 1, &golden.tokens[..30], 7);
        assert_ne!(leaked, reference, "{name}: without a reset the old sequence's state shows through (the control)");

        m.slot_reset(&pool, 1);
        assert_eq!(pool.position(1), 0);
        let reused = m.slot_prefill(&pool, 1, &golden.tokens[..30], 7);
        assert_eq!(reused, reference, "{name}: a reset slot must behave exactly like a fresh one");

        // The same through the decode path, and slot 0 beside it was never disturbed.
        let more = m.slot_decode(&pool, &[(1, golden.tokens[30])]);
        let more_fresh = m.slot_decode(&fresh, &[(0, golden.tokens[30])]);
        assert_eq!(more, more_fresh);
        assert_eq!(pool.position(0), 0);

        // A copy is a snapshot: the destination continues as the source does.
        m.slot_copy(&pool, 1, 0);
        assert_eq!(pool.position(0), pool.position(1));
        let a = m.slot_decode(&pool, &[(0, golden.tokens[31])]);
        let b = m.slot_decode(&pool, &[(1, golden.tokens[31])]);
        assert_eq!(a, b, "{name}: a copied slot continues identically");
    }
}

/// Decode `plan` (a list of batches of `(slot, token index into the slot's own stream)`) on `pool`, returning every
/// row's hidden state.
fn run_batches(m: &Qwen35, pool: &qwen35::model::SlotPool, golden: &Golden, starts: &[usize], plan: &[Vec<u32>]) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for slots in plan {
        let items: Vec<(u32, u32)> = slots.iter().map(|&slot| (slot, golden.tokens[pool.position(slot) as usize % golden.tokens.len()])).collect();
        assert!(items.iter().all(|&(slot, _)| pool.position(slot) as usize >= starts[slot as usize]));
        out.extend(m.slot_decode(pool, &items));
    }
    out
}

#[test]
fn a_recorded_decode_step_equals_the_unrecorded_one_bit_for_bit() {
    let Some(golden) = load("tiny_text") else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    // One sequence alone for the whole golden stream (the ring wraps four times, twelve blocks commit), and three at
    // 7, 19 and 30 tokens whose batch loses a member and regains it: a batch size that holds for a few steps is
    // recorded once and replayed for every composition of that size (a size seen for two steps is built the ordinary way),
    // and a commit lands in some steps and not in others.
    let b1_starts = [0usize];
    let b1_plan: Vec<Vec<u32>> = (0..golden.tokens.len()).map(|_| vec![0]).collect();
    let b3_starts = [7usize, 19, 30];
    let rep = |slots: &[u32], n: usize| std::iter::repeat_n(slots.to_vec(), n);
    let b3_plan: Vec<Vec<u32>> = rep(&[2, 0, 1], 5).chain(rep(&[0, 2], 2)).chain(rep(&[1, 0, 2], 4)).chain(rep(&[2, 1], 4)).collect();
    for (name, m, _) in models(&golden, cfg.block_size) {
        for (label, starts, plan, sizes) in [("b1", &b1_starts[..], &b1_plan, 1usize), ("b3", &b3_starts[..], &b3_plan, 3)] {
            let prepared = |m: &Qwen35| {
                let pool = m.new_slot_pool(starts.len() as u32, qwen35::model::SlotTiers::EXACT);
                for (slot, &start) in starts.iter().enumerate() {
                    if start > 0 {
                        m.slot_prefill(&pool, slot as u32, &golden.tokens[..start], 5);
                    }
                }
                pool
            };
            m.set_slot_decode_tape(false);
            let reference_pool = prepared(&m);
            let reference = run_batches(&m, &reference_pool, &golden, starts, plan);
            assert_eq!(reference_pool.recorded_steps(), 0, "{name}/{label}: with the tape off nothing is recorded");
            m.set_slot_decode_tape(true);
            let pool = prepared(&m);
            let recorded = run_batches(&m, &pool, &golden, starts, plan);
            assert_eq!(recorded, reference, "{name}/{label}: a replayed step must give the unrecorded step's hidden states bit for bit");
            // The sizes that held for `RECORD_AFTER_STEPS` steps in a row.
            let mut held = std::collections::BTreeSet::new();
            let mut run = 0;
            for (i, batch) in plan.iter().enumerate() {
                run = if i > 0 && plan[i - 1].len() == batch.len() { run + 1 } else { 1 };
                if run >= qwen35::decode_tape::RECORD_AFTER_STEPS {
                    held.insert(batch.len());
                }
            }
            assert_eq!(pool.recorded_steps(), held.len(), "{name}/{label}: one recording per batch size that held ({held:?}), replayed for every composition");
            assert!(held.contains(&sizes) && (label == "b1" || held.len() == 2));
            // The recorded state goes on exactly as the reference's: the pools hold the same sequences afterwards.
            for slot in 0..starts.len() as u32 {
                assert_eq!(pool.position(slot), reference_pool.position(slot));
            }
        }
    }
}

/// A pool that stores the anchor banks' keys and values and the Gated-DeltaNet state and conv history as bf16 (f32
/// arithmetic wherever they are read) decodes what the f32 pool decodes, to bf16 storage error: the hidden states of
/// three sequences at different positions, through ring wraps and bank commits, stay within a relative L2 error of 2e-2
/// of the f32 pool's at every step, and the compact pool is smaller by exactly the elements it narrows.
#[test]
fn bf16_bank_and_recurrent_state_track_the_f32_pool() {
    use qwen35::model::{slot_state_bytes, SlotTiers};
    let Some(golden) = load("tiny_text") else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let starts = [7usize, 19, 30];
    let compact = SlotTiers { ring: model::kv_tier::KvTier::F32, bank: model::kv_tier::KvTier::BF16, recurrent: model::kv_tier::KvTier::BF16 };
    let full = SlotTiers::EXACT;
    assert!(slot_state_bytes(&cfg, compact) < slot_state_bytes(&cfg, SlotTiers { bank: model::kv_tier::KvTier::F32, ..compact }));
    assert!(slot_state_bytes(&cfg, compact) < slot_state_bytes(&cfg, full));
    for (name, m, _) in models(&golden, cfg.block_size) {
        m.set_slot_decode_tape(false);
        let prepared = |tiers: SlotTiers| {
            let pool = m.new_slot_pool(3, tiers);
            for (slot, &start) in starts.iter().enumerate() {
                m.slot_prefill(&pool, slot as u32, &golden.tokens[..start], 5);
            }
            pool
        };
        let reference_pool = prepared(full);
        let plan: Vec<Vec<u32>> = (0..12).map(|_| vec![2, 0, 1]).collect();
        let want = run_batches(&m, &reference_pool, &golden, &starts, &plan);
        for (label, tiers) in [("bank", SlotTiers { recurrent: model::kv_tier::KvTier::F32, ..compact }), ("recurrent", SlotTiers { bank: model::kv_tier::KvTier::F32, ..compact }), ("both", compact)] {
            let pool = prepared(tiers);
            assert_eq!(pool.bytes(), slot_state_bytes(&cfg, tiers) * 3, "{name}: the allocated bytes are the analytic ones");
            assert!(pool.bytes() < reference_pool.bytes());
            let got = run_batches(&m, &pool, &golden, &starts, &plan);
            let errs: Vec<f32> = got.iter().zip(&want).map(|(g, w)| (g.iter().zip(w).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / w.iter().map(|b| b * b).sum::<f32>()).sqrt()).collect();
            println!("{name}/{label}: relative L2 error per row {:?}", errs.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>());
        }
    }
}
