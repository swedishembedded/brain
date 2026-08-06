// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Reduce the partial maxima from max_abs_part into the int8 quantization scale
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
//
// Reduce the partial maxima from max_abs_part into the int8 quantization scale:
// sx = max(part) / 127. Single thread (P is small, e.g. 256). Output is one f32.

struct Params { p: u32 };

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> sx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // One active thread. `num_workgroups` is declared and the index reconstructed
    // the standard way even though this is always dispatched as a single
    // workgroup: the CPU (Cranelift) JIT requires every kernel to take the
    // argument, and omitting it made this the ONE kernel of 346 that failed to
    // compile there — i.e. int8 quantization had no working CPU-backend path.
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t != 0u) { return; }
    var m = 0.0;
    for (var i: u32 = 0u; i < pr.p; i = i + 1u) {
        m = max(m, part[i]);
    }
    // Guard the all-zero case so the scale is finite (dequant multiplies by it).
    sx[0] = max(m, 1e-8) / 127.0;
}
