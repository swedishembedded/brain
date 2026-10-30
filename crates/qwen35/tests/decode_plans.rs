// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency LLM serving that keeps the host out of the token loop for its clients.
// If your team needs expertise in recording a decode step once and replaying it as one submission then you can
// procure our services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! A decode step replayed from its recording (layers and head as one submission, [`qwen35::int8_gguf_resident`]
//! `decode_planned`) gives bit-for-bit the candidates and tokens the same step built the ordinary way gives: across a
//! context bucket boundary, a batch that shrinks and a head that changes between greedy and top-k.
//!
//! Run with `-- --device gpu<i> --backend cuda`.

use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

/// Per-sequence capacity: past the 128-key bucket the recorded steps are keyed on.
const CAP: u32 = 192;

fn engine(plans: bool) -> Qwen35GgufEngine {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = write_checkpoint_conditioned(&fp8_resident_cfg(), &format!("plans{n}"), 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    assert_eq!(tier.default_dtype(), Dtype::F8E4M3, "the linears are held as FP8");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the test needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, CAP, tier).with_max_batch(4).with_prefix_slots(4).with_kv_block_rows(16).with_kv_tier(KvTier::F32);
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the tiny FP8 directory loads");
    std::fs::remove_dir_all(dir).ok();
    engine.instance().set_decode_plans(plans);
    engine
}

fn prompt_of(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 3 + (seed * 5 + i * 7 + i * i) % 50).collect()
}

/// Everything a run observed: per step, each live row's candidates as bit patterns.
type Trace = Vec<Vec<Vec<(u32, u32)>>>;

/// Three sequences decode from short prompts until the longest passes 150 positions (the 128-key bucket boundary
/// inside); one finishes early so the batch shrinks; ten steps in turn ask for greedy picks and for 4
/// candidates. Each sequence continues from its own best candidate.
fn run(plans: bool) -> (Trace, usize) {
    let mut e = engine(plans);
    let lens = [30u32, 41, 52];
    let mut tables: Vec<BlockTable> = lens.iter().map(|_| BlockTable::new()).collect();
    let mut next: Vec<u32> = Vec::new();
    for (i, t) in tables.iter_mut().enumerate() {
        let h = e.prefill(t, &prompt_of(i as u32 + 1, lens[i]));
        next.push(e.admit_greedy(&h));
    }
    let mut trace = Trace::new();
    for step in 0..110u32 {
        if step == 40 {
            // The shortest sequence leaves: the batch shrinks and a recording for the smaller count is made.
            let mut gone = tables.remove(0);
            e.release_table(&mut gone);
            next.remove(0);
        }
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let rows: Vec<Vec<(u32, f32)>> = if (step / 10) % 2 == 0 {
            e.forward_batched_greedy(&mut refs, &next).into_iter().map(|t| vec![(t, 0.0)]).collect()
        } else {
            e.forward_batched_topk(&mut refs, &next, 4)
        };
        next = rows.iter().map(|r| r[0].0).collect();
        trace.push(rows.iter().map(|r| r.iter().map(|&(t, v)| (t, v.to_bits())).collect()).collect());
    }
    (trace, e.instance().decode_plans_held())
}

fn a_replayed_decode_step_gives_the_candidates_of_the_step_built_the_ordinary_way() {
    let (base, held_off) = run(false);
    let (planned, held_on) = run(true);
    assert_eq!(held_off, 0, "plans off records nothing");
    assert!(held_on >= 2, "steady steps were recorded ({held_on} held)");
    for (n, (a, b)) in base.iter().zip(&planned).enumerate() {
        assert_eq!(a, b, "step {n} differs between the replay and the ordinary step");
    }
}

gpu_core::card_tests!(a_replayed_decode_step_gives_the_candidates_of_the_step_built_the_ordinary_way);
