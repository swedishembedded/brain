// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The splat backward's gradient record is computed channel by channel -
//! for the EWA renderer by `splat_bwd_slots.wgsl`, for the ray renderer by
//! the pair library `lib/splat_ray_pair.wgsl` that both
//! `splat_ray_bwd_slots.wgsl` and `splat_ray_bwd_tile.wgsl` import - stored
//! at `channels` words per instance by `splat_bwd_tile_reduce.wgsl` or the
//! tile kernel, read by `splat_grad_reduce.wgsl`, and SIZED by the host
//! (`splat::renderer::{RECORD_WORDS, RAY_RECORD_WORDS}`), which is also what
//! decides whether a scene fits inside one storage binding. The strides
//! travel to the shared kernels as parameters; what is fixed in WGSL is how
//! many channels each kernel writes and how many the shared kernels have
//! room for.
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

/// The `N` of a `const <name>: u32 = <N>u;` declaration.
fn const_u32(src: &str, name: &str) -> usize {
    src.split(&format!("const {name}: u32 = "))
        .nth(1)
        .and_then(|r| r.split('u').next())
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or_else(|| panic!("no `const {name}: u32 = <n>u;` - this gate has gone blind"))
}

#[test]
fn every_backward_kernel_writes_the_channels_the_host_sizes_for() {
    use splat::renderer::{RAY_RECORD_WORDS, RECORD_WORDS};
    assert_eq!(array_len(kernels::SPLAT_BWD_SLOTS, "var part"), RECORD_WORDS, "splat_bwd_slots.wgsl vs RECORD_WORDS");
    for (name, src) in [("splat_ray_bwd_slots", kernels::SPLAT_RAY_BWD_SLOTS), ("splat_ray_bwd_tile", kernels::SPLAT_RAY_BWD_TILE)] {
        assert_eq!(const_u32(src, "CH"), RAY_RECORD_WORDS, "{name}.wgsl's CH vs RAY_RECORD_WORDS");
        assert_eq!(array_len(src, "part"), RAY_RECORD_WORDS, "{name}.wgsl's pair channels vs RAY_RECORD_WORDS");
    }
}

#[test]
fn the_shared_stages_have_room_for_the_widest_record() {
    use splat::renderer::{RAY_RECORD_WORDS, RECORD_WORDS};
    let widest = RECORD_WORDS.max(RAY_RECORD_WORDS);
    let partial = array_len(kernels::SPLAT_BWD_TILE_REDUCE, "var<workgroup> partial");
    assert!(partial >= 64 * widest, "splat_bwd_tile_reduce.wgsl reduces {} floats per workgroup, {widest} channels x 64 threads need {}", partial, 64 * widest);
    let acc = array_len(kernels::SPLAT_GRAD_REDUCE, "var acc");
    assert!(acc >= widest, "splat_grad_reduce.wgsl sums {acc} channels, the widest record has {widest}");
}
