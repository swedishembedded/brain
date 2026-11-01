// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory management for LLM serving on hybrid recurrent models for clients whose
// concurrency is bound by the cache of the tokens. If your team needs expertise in sharing one device memory budget
// between the KV cache and the recurrent state, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The gates for drawing the recurrent-state rows and the KV blocks of the served Qwen3.8-27B from one pool
//! (`Qwen35GgufResident::with_unified_pool`), on the REAL FP8 checkpoint and a CUDA card with sparse buffers.
//!
//! * **exactness**: a unified pool serves the greedy tokens of a pool that reserves a state row for every slot, at the same
//!   block size, through slots that are reused (more prompts than slots) - the memory behind a row changes, its arithmetic
//!   does not;
//! * **one budget**: a running sequence takes the memory of its state row from the block pool and gives it back, in the blocks
//!   the geometry says (3 for the first f32 row, 5 for two, 9 for four: neighbouring rows share a granule), and an idle server
//!   holds none.
//!
//! Self-skips when the checkpoint is not in the model store (`brain fetch Qwen/Qwen3.8-27B-FP8`) or the card has no sparse
//! buffers.

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::policy::GdnState;
use model::serve::{PagedDecoder, Request, Scheduler};
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};

const STORE: &str = "Qwen/Qwen3.8-27B-FP8";
/// Rows of a bf16 KV block that is one granule of a plane: 2 MiB / (1024 elements x 2 bytes).
const BLOCK_ROWS: u32 = 1024;
const CAP: u32 = 8192;
const RESERVE: u64 = 2 << 30;

/// The resident and the card it is placed on.
fn resident(path: &str, slots: u32, unified: bool, pool_blocks: u32) -> Option<(Qwen35GgufResident, residency::Device)> {
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE);
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM");
        return None;
    }
    let tier = held_tiers_at(path).expect("the directory is a servable checkpoint");
    let devices = devices[..1].to_vec();
    let device = devices[0].0;
    let r = Qwen35GgufResident::new(path.to_string(), devices, CAP, tier).with_kv_tier(KvTier::BF16).with_gdn_state(GdnState::F32).with_max_batch(slots).with_prefix_slots(0).with_kv_block_rows(BLOCK_ROWS);
    let r = if unified { r.with_unified_pool(true) } else { r };
    let pool = pool_blocks * BLOCK_ROWS + r.sequence_state_rows().expect("the shape of the model");
    Some((r.with_kv_pool_rows(pool), device))
}

fn engine(r: &(Qwen35GgufResident, residency::Device)) -> Qwen35GgufEngine {
    r.0.activate_engine(&[r.1]).unwrap_or_else(|e| panic!("activate: {e}")).0
}

fn checkpoint() -> Option<String> {
    let found = brain_testutil::model_dir(STORE).filter(|d| std::path::Path::new(d).join("config.json").exists());
    if found.is_none() {
        brain_testutil::skip(&format!("{STORE} is not in the model store (brain fetch {STORE})"));
    }
    found
}

fn prompts(engine: &Qwen35GgufEngine, n: usize) -> Vec<Vec<u32>> {
    include_str!("fixtures/real_prompts.txt")
        .split("\n=====\n")
        .take(n)
        .map(|body| engine.instance().tokenize(&format!("<|im_start|>user\n{}\n\nSummarize the text above in detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", body.trim())))
        .collect()
}

fn serve(engine: Qwen35GgufEngine, prompts: &[Vec<u32>], slots: usize, new_tokens: usize) -> (Qwen35GgufEngine, Vec<Vec<u32>>) {
    let mut sched = Scheduler::new(engine, slots);
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: new_tokens, stop: Vec::new() })).collect();
    let mut done = sched.run();
    let streams = ids.iter().map(|id| done.remove(id).expect("the request completes")).collect();
    (sched.into_decoder(), streams)
}

#[test]
fn a_unified_pool_serves_the_tokens_of_pools_that_reserve_a_state_row_per_slot() {
    let Some(path) = checkpoint() else { return };
    let Some(unified) = resident(&path, 2, true, 12) else { return };
    if !unified.0.unified() {
        brain_testutil::skip_unavailable(&format!("no unified pool here: {}", unified.0.unified_pool_notice().unwrap_or_default()));
        return;
    }
    let separate = resident(&path, 2, false, 12).expect("the same card");
    let (sep_engine, prompts) = {
        let e = engine(&separate);
        let p = prompts(&e, 4);
        (e, p)
    };
    // Four prompts through two slots: every slot's row is released and given to a later sequence.
    let (sep_engine, reference) = serve(sep_engine, &prompts, 2, 64);
    drop(sep_engine);
    assert!(reference.iter().all(|s| s.len() == 64), "every sequence ran the whole budget");

    let (uni_engine, streams) = serve(engine(&unified), &prompts, 2, 64);
    assert_eq!(streams, reference, "the unified pool's tokens are the separate pools' tokens");
    assert_eq!(uni_engine.free_blocks(), uni_engine.alloc_total(), "every block, lent or not, is back in the pool");
    let (_, state_granules, lent) = uni_engine.unified_report().expect("a unified engine reports");
    assert_eq!((state_granules, lent), (0, 0), "an idle server holds no state row");
}

#[test]
fn a_running_sequence_takes_its_state_row_from_the_block_pool_and_gives_it_back() {
    let Some(path) = checkpoint() else { return };
    let Some(r) = resident(&path, 4, true, 16) else { return };
    if !r.0.unified() {
        brain_testutil::skip_unavailable(&format!("no unified pool here: {}", r.0.unified_pool_notice().unwrap_or_default()));
        return;
    }
    let mut engine = engine(&r);
    let prompt = engine.instance().tokenize("The recurrent state of a sequence takes memory from the cache while it runs.");
    let total = engine.free_blocks();
    let mut tables: Vec<BlockTable> = Vec::new();
    // (sequences running, blocks their f32 state rows take: 48 layers x the granules the rows touch / 32 per block)
    for (running, state_blocks) in [(1usize, 3u32), (2, 5), (3, 8), (4, 9)] {
        let mut table = BlockTable::new();
        engine.prefill(&mut table, &prompt);
        tables.push(table);
        assert_eq!(total - engine.free_blocks(), running as u32 + state_blocks, "{running} sequences: one KV block each beside {state_blocks} blocks of state");
        let (_, granules, lent) = engine.unified_report().expect("a unified engine reports");
        assert_eq!(lent as u32, state_blocks);
        assert_eq!(granules, [2u64, 3, 5, 6][running - 1], "neighbouring rows share the granule at their edge");
    }
    for mut table in tables {
        engine.release_table(&mut table);
    }
    assert_eq!(engine.free_blocks(), total, "an idle server holds no state");
    assert_eq!(engine.unified_report().map(|(_, g, l)| (g, l)), Some((0, 0)));
}
