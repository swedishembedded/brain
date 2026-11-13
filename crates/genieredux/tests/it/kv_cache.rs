// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for cutting autoregressive
// generation cost for its clients. If your team needs expertise in
// transformer inference caching then you can procure our services by sending
// an email to info@swedishembedded.com.

//! The ST-transformer key/value cache: extending a cache frame by frame must
//! reproduce the full recompute exactly, because every layer is causal in
//! time (spatial attention and the feed-forwards are per frame, the PEG
//! convolutions and the temporal attention look only backwards). Bitwise
//! equality is asserted: the cached path runs the same kernels on the same
//! rows and the host attention mirrors the kernels' accumulation order.

use crate::tiny;
use gpu_core::Gpu;
use genieredux::{
    cached_forward_flops, decode_extend, decode_indices, dynamics_forward, dynamics_forward_extend,
    dynamics_spatial_bias, kernel_sources, StCache,
};

const HW: usize = tiny::HW;

fn ids(frames: usize) -> Vec<u32> {
    (0..(frames * HW) as u32).map(|i| (i * 7 + 3) % 16).collect()
}
fn acts(frames: usize) -> Vec<f32> {
    tiny::one_hot(&(0..frames as u32).map(|a| (a * 2 + 1) % 3).collect::<Vec<_>>(), 3)
}

fn uncached(gpu: &Gpu, frames: usize) -> Vec<f32> {
    let dw = tiny::dynamics(&tiny::tokenizer());
    let dc = tiny::dynamics_config();
    dynamics_forward(
        gpu, &ids(frames), &acts(frames), &dw, 1, frames as u32, 2, 2, dc.dim, dc.heads, dc.head_dim,
        dc.n_codes, dc.code_dim, dc.num_actions,
    )
}

/// Extend a fresh cache through `chunks` (frames per call) and concatenate.
fn cached(gpu: &Gpu, chunks: &[usize]) -> Vec<f32> {
    let dw = tiny::dynamics(&tiny::tokenizer());
    let dc = tiny::dynamics_config();
    let sb = dynamics_spatial_bias(&dw, &dc, 2, 2);
    let (all_ids, all_acts) = (ids(chunks.iter().sum()), acts(chunks.iter().sum()));
    let mut cache = StCache::new();
    let (mut out, mut at) = (Vec::new(), 0usize);
    for &c in chunks {
        out.extend(dynamics_forward_extend(
            gpu, &mut cache, &all_ids[at * HW..(at + c) * HW], &all_acts[at * 3..(at + c) * 3],
            &dw, &dc, 2, 2, &sb,
        ));
        at += c;
    }
    assert_eq!(cache.frames(), at);
    out
}

#[test]
fn cached_dynamics_equals_full_recompute_whatever_the_chunking() {
    let gpu = Gpu::new_cpu(&kernel_sources());
    let full = uncached(&gpu, 5);
    for chunks in [&[1usize, 1, 1, 1, 1][..], &[2, 3], &[1, 4], &[5]] {
        assert_eq!(cached(&gpu, chunks), full, "chunks {chunks:?}");
    }
}

#[test]
fn rollback_restores_the_cache_exactly() {
    let gpu = Gpu::new_cpu(&kernel_sources());
    let dw = tiny::dynamics(&tiny::tokenizer());
    let dc = tiny::dynamics_config();
    let sb = dynamics_spatial_bias(&dw, &dc, 2, 2);
    let (all_ids, all_acts) = (ids(5), acts(5));
    let mut cache = StCache::new();
    dynamics_forward_extend(&gpu, &mut cache, &all_ids[..3 * HW], &all_acts[..9], &dw, &dc, 2, 2, &sb);
    let mark = cache.mark();
    let speculative = dynamics_forward_extend(&gpu, &mut cache, &all_ids[3 * HW..4 * HW], &all_acts[9..12], &dw, &dc, 2, 2, &sb);
    cache.rollback(&mark);
    assert_eq!(cache.frames(), 3);
    // A different frame in the same slot, then the original again.
    let other = dynamics_forward_extend(&gpu, &mut cache, &ids(6)[5 * HW..], &acts(6)[15..], &dw, &dc, 2, 2, &sb);
    assert_ne!(other, speculative, "the scratch frame must differ for this check to mean anything");
    cache.rollback(&mark);
    let again = dynamics_forward_extend(&gpu, &mut cache, &all_ids[3 * HW..4 * HW], &all_acts[9..12], &dw, &dc, 2, 2, &sb);
    assert_eq!(again, speculative);
}

#[test]
fn cached_decoder_equals_decoding_the_whole_video() {
    let gpu = Gpu::new_cpu(&kernel_sources());
    let tw = tiny::tokenizer();
    let tc = tiny::tokenizer_config();
    let sb = genieredux::bias::cpb_bias(&tw.cpb_net, 2, 2, tc.heads as usize);
    let frames = 4usize;
    let all = ids(frames);
    let video = decode_indices(&gpu, &all, &tw, 1, frames as u32, 2, 2, tc.dim, tc.heads, tc.head_dim, tc.code_dim);
    let mut cache = StCache::new();
    let mut got = Vec::new(); // frame-major
    for f in 0..frames {
        got.extend(decode_extend(&gpu, &mut cache, &all[f * HW..(f + 1) * HW], &tw, &tc, 2, 2, &sb));
    }
    assert_eq!(tiny::to_video(&got, frames), video);
}

#[test]
fn flop_counts_add_up_and_the_cache_saves_the_context() {
    let dc = tiny::dynamics_config();
    let full = |t: usize| cached_forward_flops(&dc, HW, 0, t);
    // Computing frames one at a time against the cache does exactly the work of
    // the single full pass: nothing is recomputed and nothing is skipped.
    let incremental: u64 = (0..6).map(|g| cached_forward_flops(&dc, HW, g, 1)).sum();
    assert_eq!(incremental, full(6));
    // The next frame costs one frame, not a pass over the whole context.
    assert!(cached_forward_flops(&dc, HW, 5, 1) * 3 < full(6));
    assert!(cached_forward_flops(&dc, HW, 5, 1) < cached_forward_flops(&dc, HW, 9, 1));
}
