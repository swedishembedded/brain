// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The splat backward's gradient record is written channel by channel by a
//! slots kernel (`splat_bwd_slots.wgsl` for the EWA renderer,
//! `splat_ray_bwd_slots.wgsl` for the ray renderer), reduced into records of
//! `channels + 1` words by `splat_bwd_tile_reduce.wgsl`, read by
//! `splat_bwd_keys.wgsl` and `splat_grad_reduce.wgsl`, and SIZED by the host
//! (`splat::renderer::{SLOT_CHANNELS, RAY_SLOT_CHANNELS}` and the record
//! widths), which is also what decides whether a scene fits inside one
//! storage binding. The strides travel to the shared kernels as parameters;
//! what is fixed in WGSL is how many channels each slots kernel writes and
//! how many the shared kernels have room for.
//!
//! Widening a record in a slots kernel without widening it on the host does
//! not fail loudly: the host allocates a buffer a fraction of the size the
//! kernel strides through, and the pass writes past the records it owns into
//! the next one's - producing gradients that are wrong rather than absent.
//! This gate reads the kernels and requires them to agree with each other and
//! with the host constants.
//!
//! Swedish Embedded AB implements differentiable GPU rasterizers whose host
//! and shader sides cannot drift apart. If your team needs expertise in
//! compute-shader ABI design then you can procure our services by sending an
//! email to info@swedishembedded.com.

/// The size in a `<decl>: array<f32, N>` declaration.
fn array_len(src: &str, decl: &str) -> usize {
    let at = src.find(&format!("{decl}: array<f32, ")).unwrap_or_else(|| panic!("no `{decl}: array<f32, N>` - this gate has gone blind"));
    let rest = &src[at + decl.len() + ": array<f32, ".len()..];
    rest[..rest.find('>').expect("closing >")].trim().parse().expect("a literal length")
}

#[test]
fn every_slots_kernel_writes_the_channels_the_host_sizes_for() {
    use splat::renderer::{RAY_RECORD_WORDS, RAY_SLOT_CHANNELS, RECORD_WORDS, SLOT_CHANNELS};
    assert_eq!(array_len(kernels::SPLAT_BWD_SLOTS, "var part"), SLOT_CHANNELS, "splat_bwd_slots.wgsl vs SLOT_CHANNELS");
    let ch = kernels::SPLAT_RAY_BWD_SLOTS
        .split("const CH: u32 = ")
        .nth(1)
        .and_then(|r| r.split('u').next())
        .and_then(|n| n.parse::<usize>().ok())
        .expect("splat_ray_bwd_slots.wgsl declares `const CH: u32 = <n>u;`");
    assert_eq!(ch, RAY_SLOT_CHANNELS, "splat_ray_bwd_slots.wgsl vs RAY_SLOT_CHANNELS");
    assert_eq!(array_len(kernels::SPLAT_RAY_BWD_SLOTS, "var part"), RAY_SLOT_CHANNELS);
    assert_eq!(RECORD_WORDS, SLOT_CHANNELS + 1, "a record is its channels and the gaussian id");
    assert_eq!(RAY_RECORD_WORDS, RAY_SLOT_CHANNELS + 1, "a record is its channels and the gaussian id");
}

#[test]
fn the_shared_stages_have_room_for_the_widest_record() {
    use splat::renderer::{RAY_SLOT_CHANNELS, SLOT_CHANNELS};
    let widest = SLOT_CHANNELS.max(RAY_SLOT_CHANNELS);
    let partial = array_len(kernels::SPLAT_BWD_TILE_REDUCE, "var<workgroup> partial");
    assert!(partial >= 64 * widest, "splat_bwd_tile_reduce.wgsl reduces {} floats per workgroup, {widest} channels x 64 threads need {}", partial, 64 * widest);
    let acc = array_len(kernels::SPLAT_GRAD_REDUCE, "var acc");
    assert!(acc >= widest, "splat_grad_reduce.wgsl sums {acc} channels, the widest record has {widest}");
}
