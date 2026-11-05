// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Second int8 level of an activation: quantize r = x - clamp(round(x / sx), -127, 127) * sx straight into packed rq [M, K/4] u32 and its per-(row, 32-group) scales rs
// @how   one thread per (row, 32-group); the residual is recomputed, never stored
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The second half of a two-level int8 activation x ≈ sx·q₁ + rs·q₂.
// `quant_pack` rounds x onto its group's int8 grid; this kernel quantizes the
// rounding error that grid drops, with the same per-group layout
// `max_abs_row` + `quant_pack` would give it:
//
//   r        = x - clamp(round(x / sx), -127, 127) · sx
//   rs[m, g] = max(max|r[m, 32g .. 32g+32]|, 1e-8) / 127
//   rq       = clamp(round(r / rs), -127, 127), packed 4 per u32 (LE along K)
//
// Multiplied by the same int8 weight, the second level restores the activation
// to ~1/(127·254) of its group scale. The first-level rounding must be
// quant_pack's exactly, or the residual would describe a different first
// level than the one the GEMM reads.
//
// Fusing the three steps keeps the residual in registers: an f32 [M, K]
// residual costs 4·M·K bytes, 770 MiB for a 12320-token LTX-2.5 FF
// down-projection, which is what a stage-2 forward cannot spare.

struct Params { m: u32, k: u32 };  // k = full K (multiple of 32)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       sx: array<f32>;
@group(0) @binding(3) var<storage, read_write> rq: array<u32>;
@group(0) @binding(4) var<storage, read_write> rs: array<f32>;

fn residual(i: u32, inv: f32, s: f32) -> f32 {
    let q = clamp(round(x[i] * inv), -127.0, 127.0);
    return x[i] - q * s;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let groups = p.k / 32u;
    if (idx >= p.m * groups) { return; }
    let base = (idx / groups) * p.k + (idx % groups) * 32u;
    let s = sx[idx];
    let inv = 1.0 / s;
    var a = 0.0;
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
        a = max(a, abs(residual(base + c, inv, s)));
    }
    let scale = max(a, 1e-8) / 127.0;
    rs[idx] = scale;
    let inv2 = 1.0 / scale;
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        var word: u32 = 0u;
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
            let q = clamp(round(residual(base + w * 4u + b, inv, s) * inv2), -127.0, 127.0);
            word = word | (u32(i32(q) & 0xff) << (8u * b));
        }
        rq[base / 4u + w] = word;
    }
}
