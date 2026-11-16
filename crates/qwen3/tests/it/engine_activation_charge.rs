// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for sizing the device memory of served language models so that a full
// card is planned for, not discovered, for its clients. If your team needs expertise in GPU inference servers then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The device memory the serving engine takes beyond its weights and its KV pool, against what a plan charges for it
//! ([`qwen3::serve::activation_bytes`]). A plan sizes the KV pool from the card's bytes less the weights and that
//! charge: a charge below what a prefill round and the recorded decode steps really hold leaves a pool the card
//! cannot hold beside them, and a 32K-row Qwen3-4B round at a pool of 32 whole contexts ran the card out while the
//! weights were still loading.

use model::ops::TierPolicy;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen3::serve::Engine;
use qwen3::Dtype;

/// What one engine of the released Qwen3-4B holds on its own handle beyond its KV pool and its fp32 tables (the embedding
/// and the norms; the linears live on the upload side's handle), at the peak of a prefill of two widest rounds and a
/// decode step of every row the engine was built for (its decode steps recorded first, as a server records them before
/// its first request), and what a plan charges for that sizing.
struct Census {
    held: u64,
    charged: u64,
}

fn census(gpu: &gpu_core::Gpu, path: &str, max_batch: u32, rows: u32) -> Census {
    let (cfg, src) = qwen3::open_checkpoint(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let block = 16u32;
    let per_seq = (2 * rows + 64).div_ceil(block);
    let num_blocks = per_seq + max_batch * 2;
    let tiers = TierPolicy::uniform(Dtype::BF16);
    let mut e = Engine::from_source_tiers_on(gpu, cfg.clone(), &*src, block, num_blocks, max_batch, per_seq, rows, true, &tiers);
    let charged = qwen3::serve::activation_bytes(&cfg, block, per_seq, max_batch, rows, true, e.fused_attention(), qwen3::serve::writes_fused_products(&cfg, &tiers, false, false));
    let tables_fp32: u64 = cfg.param_list().into_iter().filter(|(name, _)| !qwen3::q8::Q8::is_i8_linear(name) && name != "lm_head.weight").map(|(_, elems)| elems as u64 * 4).sum();
    e.gpu().reset_memory_peak();
    PagedDecoder::prepare_decode(&mut e, max_batch as usize);
    let mut long = BlockTable::new();
    let prompt: Vec<u32> = (0..2 * rows).map(|i| (i * 7919 + 13) % cfg.vocab).collect();
    e.prefill_for_perf(&mut long, &prompt);
    e.release_for_perf(&mut long);
    let mut tables: Vec<BlockTable> = (0..max_batch).map(|_| BlockTable::new()).collect();
    for (i, t) in tables.iter_mut().enumerate() {
        e.prefill_for_perf(t, &[(i as u32 * 31 + 5) % cfg.vocab, 7, 11]);
    }
    let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
    let inputs: Vec<u32> = (0..max_batch).map(|i| (i * 13 + 1) % cfg.vocab).collect();
    PagedDecoder::forward_batched_greedy(&mut e, &mut refs, &inputs);
    let peak = e.gpu().memory_usage().expect("the CUDA backend accounts for its allocations").peak_live;
    Census { held: peak - e.kv_pool_bytes() - tables_fp32, charged }
}

/// The charge holds what the engine allocates beyond its weights and KV pool at the widest round a server or a
/// measurement builds it for, and is not far above it (an over-charge is KV a plan does not hand out).
#[test]
fn the_engines_activations_at_its_widest_round_fit_their_charge() {
    let Some(path) = brain_testutil::model_dir("Qwen/Qwen3-4B").filter(|d| std::path::Path::new(d).join("config.json").is_file()) else {
        return brain_testutil::skip_unavailable("Qwen/Qwen3-4B is not in the model store");
    };
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let sizings = [(16u32, 512u32), (16, 4096), (32, 16384)];
    let censuses: Vec<Census> = sizings.iter().map(|&(max_batch, rows)| census(&gpu, &path, max_batch, rows)).collect();
    for (&(max_batch, rows), c) in sizings.iter().zip(&censuses) {
        println!("Qwen3-4B, {max_batch} sequences, {rows}-row rounds: held {} B ({:.3} GiB), charged {} B ({:.3} GiB)", c.held, gib(c.held), c.charged, gib(c.charged));
    }
    for (&(max_batch, rows), c) in sizings.iter().zip(&censuses) {
        assert!(c.held <= c.charged, "{max_batch} sequences, {rows}-row rounds: the engine held {:.3} GiB beyond its weights and KV pool, the plan charges {:.3} GiB", gib(c.held), gib(c.charged));
        assert!(c.charged <= c.held + c.held / 4, "{max_batch} sequences, {rows}-row rounds: the plan charges {:.3} GiB for {:.3} GiB the engine held", gib(c.charged), gib(c.held));
    }
}
