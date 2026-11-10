// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training rotary embedding: rotate the pairs (j, j + hd/2) of every head by a per-row angle (sign -1 inverts)
// @how   one thread per rotated pair; assigns or accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// cs holds (cos, sin) per row and pair: cs[2 (row pairs + j)], cs[2 (row pairs + j) + 1]. y = R(sign) x, so the backward
// is the same kernel with sign -1 and acc 1. Params: total (rows * heads * pairs), heads, hd, sign (f32 bits), acc.
// Dispatch: total threads.

struct Params { total: u32, heads: u32, hd: u32, sign: f32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read>  cs: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let pairs = p.hd / 2u;
    let j = idx % pairs;
    let head = (idx / pairs) % p.heads;
    let row = idx / (pairs * p.heads);
    let base = (row * p.heads + head) * p.hd;
    let c = cs[2u * (row * pairs + j)];
    let s = cs[2u * (row * pairs + j) + 1u] * p.sign;
    let a = x[base + j];
    let b = x[base + j + pairs];
    let lo = a * c - b * s;
    let hi = b * c + a * s;
    if (p.acc != 0u) {
        y[base + j] = y[base + j] + lo;
        y[base + j + pairs] = y[base + j + pairs] + hi;
    } else {
        y[base + j] = lo;
        y[base + j + pairs] = hi;
    }
}
