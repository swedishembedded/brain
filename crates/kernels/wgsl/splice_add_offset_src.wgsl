// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Residual DeepStack add with an independent SOURCE offset: dst[dst_base+i] += src[src_base+i]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// `splice_add.wgsl`'s sibling for the opposite offset direction. That kernel's
// `base` lands on `dst` only (`dst[base+i] += src[i]`) -- exactly right for
// PREFILL's DeepStack add (a compact `[n_rows,d]` source written whole into the
// big sequence residual at the image rows' offset), but wrong for DECODE's
// add: there `dst` (the current step's own `[d]`-sized residual row) needs NO
// offset, while `src` (`deepstack_bufs[level]`, `[n_rows,d]` compact) needs to
// be read starting at THIS step's own image row (`local_row * d`), which
// `splice_add.wgsl`'s uniform-only interface cannot express w.r.t. `src`.
//
// Do not use `splice_add.wgsl`'s `step_sliced` bind-group offset for this
// instead: a bind-group buffer offset must be a multiple of
// `min_storage_buffer_offset_alignment` (256B), and `local_row * d * 4` has no
// such guarantee for an arbitrary row/`d_model` -- this was tried and failed
// on real hardware enforcing the full 256B limit. A uniform-parameter offset
// has no such constraint.
//
// New kernel rather than adding `src_base` to `splice_add.wgsl` itself:
// `splice_add` is used elsewhere (`model::gdn::gdn_chunk_bwd`'s accumulator
// commits) with its existing 2-field `Params`; changing its layout would
// touch every call site for a need only this one has.

struct Params {
    n:        u32,
    src_base: u32,
    dst_base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    dst[p.dst_base + idx] = dst[p.dst_base + idx] + src[p.src_base + idx];
}
