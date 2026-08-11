// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Elementwise subtract, with an independent flat offset into each input
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Elementwise subtract:  out[i] = a[a_off + i] - b[b_off + i],  i in [0,total).
// `a_off`/`b_off` let a caller subtract a CONTIGUOUS sub-range of a larger
// buffer without binding a byte-offset slice (the 256-byte slice-alignment
// rule that applies to those, which a `Params` offset sidesteps — the same
// convention `splice_add.wgsl`/`bmm.wgsl` use). Gated DeltaNet's per-chunk
// `v_new = v_c - v_prime` (`torch_chunk_gated_delta_rule`) needs exactly
// this: `v_c` is one chunk's slice of the full `value` tensor
// (`a_off = c * chunk_stride`), `v_prime` is a small dense per-chunk scratch
// recomputed every chunk (`b_off = 0`). `out` carries no offset (every
// current caller writes a dedicated destination); add one if a future caller
// needs it. `add2.wgsl` is the zero-offset, no-callers-to-break `+` sibling;
// this generalises `sub` with offsets from its first day in the tree, since
// (unlike `add2`) it has no prior callers whose contract would break.

struct Params { total: u32, a_off: u32, b_off: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    out[idx] = a[p.a_off + idx] - b[p.b_off + idx];
}
