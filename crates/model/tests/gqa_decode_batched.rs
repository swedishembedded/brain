// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::block::gqa_decode_batched_step` - the CROSS-SEQUENCE decode
//! primitive - against `model::block::gqa_decode_step`, the per-sequence one
//! it replaces.
//!
//! `crates/model/src/paged.rs`'s own `batched_paged_matches_per_sequence`
//! already proves the three batched KERNELS are numerically right over one
//! shared pool. What it does NOT cover, and this does, is the whole step a
//! serving engine actually needs: the batched KV APPEND that puts each
//! sequence's new token into its own physical block first (the batched twin of
//! `gqa_decode_step`'s two `kv_append` dispatches), and the param/thread wiring
//! of all five dispatches as one builder. The reference side here is the real
//! single-sequence decode primitive on its own dedicated flat cache, which is
//! exactly what `qwen35::serve::Engine` ran one sequence at a time before this
//! builder existed - so this test IS the "batched N sequences == N independent
//! single-sequence runs" equivalence that the batched serving path rests on.
//!
//! Swedish Embedded AB implements paged-attention serving kernels for
//! inference engines for its clients. If your team needs expertise in
//! continuous batching and paged KV caches then you can procure our services
//! by sending an email to info@swedishembedded.com.

use data::rng::Rng;
use gpu_core::Gpu;
use model::block::{gqa_decode_batched_step, gqa_decode_step, GqaDecodeBatchedIds, GqaDecodeIds};

const PIPES: &[(&str, &str)] = &[
    ("kv_append", kernels::KV_APPEND),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Three sequences at DIFFERENT context lengths, living in NON-adjacent
/// physical blocks of one shared pool, decoded in a single batched dispatch,
/// must reproduce each sequence's own `gqa_decode_step` on a dedicated flat
/// cache - including the new token each one appends this step.
///
/// The physical blocks are deliberately scrambled (sequence `b` gets block
/// `2*b+1`, so block 0 is never used and the occupied blocks are not in
/// sequence order) - a builder that quietly assumed "row `b` of the batch lives
/// in block `b`" would pass at any in-order assignment.
#[test]
fn batched_decode_matches_per_sequence_decode() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (nh, nkv, hd) = (4u32, 2u32, 8u32);
    let (hkv, hq) = (nkv * hd, nh * hd);
    // Context length ALREADY cached per sequence; the step below appends one
    // more token each, so sequence b ends up attending `ctx[b]+1` positions.
    let ctx_len = [5u32, 12, 20];
    let batch = ctx_len.len() as u32;
    // One physical block backs a whole sequence (the hybrid engines' own
    // `block_size == max_seq_len` shape), so every sequence's block table is
    // one entry wide.
    let block_size = 24u32;
    let max_bt = 1u32;
    let num_blocks = 6u32;
    // Strictly greater than every sequence's real length, so a kernel that
    // mistook the scratch stride for a compute bound would be caught.
    let cap = block_size;

    let mut rng = Rng::new(7);
    let qs: Vec<Vec<f32>> = (0..batch).map(|_| (0..hq).map(|_| rng.next_gaussian() as f32).collect()).collect();
    let knew: Vec<Vec<f32>> = (0..batch).map(|_| (0..hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
    let vnew: Vec<Vec<f32>> = (0..batch).map(|_| (0..hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
    let ks: Vec<Vec<f32>> = ctx_len.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();
    let vs: Vec<Vec<f32>> = ctx_len.iter().map(|&t| (0..t * hkv).map(|_| rng.next_gaussian() as f32).collect()).collect();

    // Scrambled physical assignment - see this test's own doc.
    let phys: Vec<u32> = (0..batch).map(|b| 2 * b + 1).collect();

    // ---- batched: one shared pool, pre-seeded with each sequence's context --
    let mut pk = vec![0f32; (num_blocks * block_size * hkv) as usize];
    let mut pv = vec![0f32; (num_blocks * block_size * hkv) as usize];
    for b in 0..batch as usize {
        let base = (phys[b] * block_size * hkv) as usize;
        pk[base..base + ks[b].len()].copy_from_slice(&ks[b]);
        pv[base..base + vs[b].len()].copy_from_slice(&vs[b]);
    }
    let pool_k = g.storage_init("pool_k", &pk);
    let pool_v = g.storage_init("pool_v", &pv);

    let qflat: Vec<f32> = qs.iter().flatten().copied().collect();
    let kflat: Vec<f32> = knew.iter().flatten().copied().collect();
    let vflat: Vec<f32> = vnew.iter().flatten().copied().collect();
    let qb = g.storage_init("q", &qflat);
    let kb = g.storage_init("k", &kflat);
    let vb = g.storage_init("v", &vflat);

    let blocks = g.storage(batch as u64);
    g.write(&blocks, &phys);
    let offsets = g.storage(batch as u64);
    g.write(&offsets, &ctx_len);
    let block_tables = g.storage((batch * max_bt) as u64);
    g.write(&block_tables, &phys);
    let seq_lens = g.storage(batch as u64);
    g.write(&seq_lens, &ctx_len.iter().map(|&t| t + 1).collect::<Vec<u32>>());

    let scores = g.storage((batch * nh * cap) as u64);
    let probs = g.storage((batch * nh * cap) as u64);
    let ctxb = g.storage((batch * hq) as u64);

    let ids = GqaDecodeBatchedIds {
        kv_append_batched: idx(&g, "paged_kv_append_batched"),
        scores_batched: idx(&g, "paged_decode_scores_batched"),
        softmax_batched: idx(&g, "decode_softmax_batched"),
        apply_batched: idx(&g, "paged_decode_apply_batched"),
    };
    let steps = gqa_decode_batched_step(
        &g,
        &ids,
        nh,
        nkv,
        hd,
        batch,
        block_size,
        max_bt,
        cap,
        &qb,
        &kb,
        &vb,
        &pool_k,
        &pool_v,
        &blocks,
        &offsets,
        &block_tables,
        &seq_lens,
        &scores,
        &probs,
        &ctxb,
    );
    g.submit(&[], &steps);
    let got = g.read(&ctxb, (batch * hq) as usize);

    // ---- reference: each sequence's own flat cache, one at a time ----------
    let dec_ids = GqaDecodeIds {
        kv_append: idx(&g, "kv_append"),
        attn_decode_scores: idx(&g, "attn_decode_scores"),
        decode_softmax: idx(&g, "decode_softmax"),
        attn_decode_apply: idx(&g, "attn_decode_apply"),
    };
    let mut worst = 0f32;
    for b in 0..batch as usize {
        let pos = ctx_len[b];
        let mut flat = vec![0f32; (block_size * hkv) as usize];
        flat[..ks[b].len()].copy_from_slice(&ks[b]);
        let kcache = g.storage_init("kc", &flat);
        let mut flatv = vec![0f32; (block_size * hkv) as usize];
        flatv[..vs[b].len()].copy_from_slice(&vs[b]);
        let vcache = g.storage_init("vc", &flatv);
        let qc = g.storage_init("qc", &qs[b]);
        let kc = g.storage_init("knew", &knew[b]);
        let vc = g.storage_init("vnew", &vnew[b]);
        let sc = g.storage((nh * block_size) as u64);
        let pr = g.storage((nh * block_size) as u64);
        let cx = g.storage(hq as u64);
        let rs = gqa_decode_step(&g, &dec_ids, nh, nkv, hd, pos, block_size, &qc, &kc, &vc, &kcache, &vcache, &sc, &pr, &cx);
        g.submit(&[], &rs);
        let want = g.read(&cx, hq as usize);
        let mine = &got[b * hq as usize..(b + 1) * hq as usize];
        let e = want.iter().zip(mine).fold(0f32, |m, (a, c)| m.max((a - c).abs()));
        worst = worst.max(e);
    }
    println!("gqa_decode_batched_step vs per-sequence gqa_decode_step: worst maxabs = {worst:e}");
    assert!(worst < 1e-6, "batched decode vs per-sequence decode maxabs={worst}");

    // The batched append must have landed in each sequence's OWN block, at its
    // own offset - a scatter that collided would still let the attention above
    // agree for the sequence that wrote last.
    let pool_after = g.read(&pool_k, (num_blocks * block_size * hkv) as usize);
    for b in 0..batch as usize {
        let row = ((phys[b] * block_size + ctx_len[b]) * hkv) as usize;
        let e = pool_after[row..row + hkv as usize].iter().zip(&knew[b]).fold(0f32, |m, (a, c)| m.max((a - c).abs()));
        assert_eq!(e, 0.0, "batched KV append: sequence {b}'s new K row is not at block {} offset {}", phys[b], ctx_len[b]);
    }
}
