// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The splat backward's gradient record is a fixed-width struct written by
//! `splat_bwd_tile_reduce.wgsl`, read by `splat_bwd_keys.wgsl` and
//! `splat_grad_reduce.wgsl`, and SIZED by the
//! host (`splat::renderer::RECORD_WORDS`), which is also what decides whether
//! a scene fits inside one storage binding. Three places, one stride.
//!
//! Widening the record in the kernels without widening it on the host does not
//! fail loudly: the host allocates a buffer a fraction of the size the emit
//! kernel strides through, and the pass writes past the records it owns into
//! the next one's - producing gradients that are wrong rather than absent.
//! This gate reads both kernels and requires them to agree with each other and
//! with the host constant.
//!
//! Swedish Embedded AB implements differentiable GPU rasterizers whose host
//! and shader sides cannot drift apart. If your team needs expertise in
//! compute-shader ABI design then you can procure our services by sending an
//! email to info@swedishembedded.com.

/// Highest `recs[r + <n>u]` (or `recs[r]`) subscript in a kernel body.
fn highest_recs_slot(src: &str) -> Option<usize> {
    let mut best = None;
    let mut rest = src;
    while let Some(i) = rest.find("recs[") {
        rest = &rest[i + "recs[".len()..];
        let end = match rest.find(']') {
            Some(e) => e,
            None => break,
        };
        let expr = rest[..end].trim();
        // `recs[r]` is slot 0; `recs[r + 7u]` is slot 7. `recs[r + k]` (a loop
        // variable, as grad_reduce uses) carries no literal and is skipped.
        let slot = match expr.split_once('+') {
            None => Some(0),
            Some((_, n)) => n.trim().trim_end_matches('u').parse::<usize>().ok(),
        };
        if let Some(s) = slot {
            best = Some(best.map_or(s, |b: usize| b.max(s)));
        }
    }
    best
}

#[test]
fn the_gradient_record_is_the_same_width_on_both_sides_of_the_dispatch() {
    let emit = highest_recs_slot(kernels::SPLAT_BWD_TILE_REDUCE)
        .expect("splat_bwd_tile_reduce.wgsl no longer subscripts recs[..] by a literal - this gate has gone blind");
    assert_eq!(
        emit + 1,
        splat::renderer::RECORD_WORDS,
        "splat_bwd_tile_reduce.wgsl writes {} words per record but splat::renderer::RECORD_WORDS is {}; \
         the host sizes `recs` from that constant, so the pass would stride past its own buffer",
        emit + 1,
        splat::renderer::RECORD_WORDS,
    );
}

/// The readers walk the record with a loop bound, not a literal, so what is
/// checkable there is the declared stride in each binding comment.
#[test]
fn the_reading_side_declares_the_same_stride() {
    for (name, src) in [("splat_grad_reduce", kernels::SPLAT_GRAD_REDUCE), ("splat_bwd_keys", kernels::SPLAT_BWD_KEYS)] {
        let want = format!("n*{}", splat::renderer::RECORD_WORDS);
        assert!(
            src.contains(&want),
            "{name}.wgsl does not declare `recs: array<f32>; // {want}` - either the stride changed or \
             the declaration did, and the two sides can no longer be checked against each other"
        );
    }
}
