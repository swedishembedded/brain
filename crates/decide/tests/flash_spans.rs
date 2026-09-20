// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ragged-span flash kernel must compute what the per-span one does.
//!
//! `flash_attn_bidir_spans` is `flash_attn_bidir_reg2` with its sequence
//! length and batch base read from a host-built work table instead of from a
//! uniform, so that one dispatch can cover spans of different lengths. That is
//! an ADDRESSING change to a kernel whose arithmetic is delicate - an online
//! softmax over a software-pipelined tile - and the failure mode of getting it
//! wrong is not a crash: it is one span reading another span's keys, which
//! produces plausible numbers for every row that is not near a boundary.
//!
//! So the two are run on the same data and compared, at deliberately awkward
//! span lengths: shorter than a key tile, exactly a query tile, one past one,
//! and a long one, in an order that puts a short span after a long one.

use gpu_core::Gpu;
use model::block;

/// Awkward on purpose. BC = 16 keys per shared tile and BR = 128 query rows
/// per workgroup, so: under a key tile, exactly a key tile, one over, exactly
/// a query tile, one over, and one long enough to need three.
const SPANS: &[u32] = &[13, 16, 17, 128, 129, 300, 7, 64];

const HEADS: u32 = 12;
const HEAD_DIM: u32 = 32;
const D: u32 = HEADS * HEAD_DIM;

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8192.0 - 0.125
        })
        .collect()
}

#[test]
fn one_dispatch_over_ragged_spans_matches_one_dispatch_per_span() {
    let gpu = Gpu::new(decide::kern::PIPELINES);
    let caps = gpu.caps();
    if !block::flash_spans_supported(&caps) {
        brain_testutil::skip("this device cannot run the ragged-span kernel");
        return;
    }
    let k = decide::kern::Ids::resolve(&gpu);

    let mut spans: Vec<(u32, u32)> = Vec::new();
    let mut at = 0u32;
    for &len in SPANS {
        spans.push((at, len));
        at += len;
    }
    let rows = at;
    let qkv = gpu.storage_init("qkv", &rand((rows * 3 * D) as usize, 99));
    let per_span = gpu.storage((rows * D) as u64);
    let ragged = gpu.storage((rows * D) as u64);
    let table = block::flash_spans_table(HEADS, &spans);
    let work = gpu.storage(table.len() as u64);
    gpu.write(&work, &table);

    let mut steps = Vec::new();
    block::flash_bidir_fwd(
        &gpu,
        block::FlashIds { bidir: k.flash_bidir, split: None, reg: None, reg2: Some(k.flash_bidir_reg2) },
        HEADS,
        HEAD_DIM,
        D,
        &qkv,
        3 * D,
        0,
        D,
        2 * D,
        &per_span,
        &spans,
        &mut steps,
    );
    steps.push(block::flash_bidir_spans_step(
        &gpu,
        k.flash_bidir_spans,
        HEADS,
        HEAD_DIM,
        D,
        &qkv,
        3 * D,
        0,
        D,
        2 * D,
        &ragged,
        &work,
        &spans,
    ));
    gpu.submit(&[], &steps);
    gpu.poll_wait();

    let n = (rows * D) as usize;
    let (a, b) = (gpu.read(&per_span, n), gpu.read(&ragged, n));
    let mut worst = 0.0f32;
    let mut at_i = 0usize;
    for (i, (&p, &q)) in a.iter().zip(&b).enumerate() {
        let d = (p - q).abs();
        if d > worst {
            worst = d;
            at_i = i;
        }
    }
    // Both run the same accumulation in the same order, so this is bit-equal
    // in practice; the tolerance exists so a future retiling that reorders the
    // online softmax fails loudly rather than by a hair.
    let row = at_i / D as usize;
    let span = spans.iter().position(|&(r0, l)| row as u32 >= r0 && (row as u32) < r0 + l);
    assert!(
        worst < 1e-5,
        "ragged and per-span attention disagree by {worst} at row {row} (span {span:?}) \
         channel {} - {} vs {}",
        at_i % D as usize,
        a[at_i],
        b[at_i]
    );
}
