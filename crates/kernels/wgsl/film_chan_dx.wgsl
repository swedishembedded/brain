// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  FiLM per-channel modulation, input gradient — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// FiLM per-channel modulation, input gradient — spec:
// docs/world-models/specs/P1.film.md §4.2. One invocation per element
// (N*C*H*W threads):
//   dx[n,c,h,w] = dy[n,c,h,w] * (1 + s[n,c])
// sb[N,2C] packed scale-first (s[n,c] = sb[n*2C + c]); reads ONLY the scale
// half. Indexing as film_chan. OVERWRITES dx (=, SSA fresh buffer).
//

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       sb: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.N * p.C * p.H * p.W) { return; }
    let hw = p.H * p.W;
    let c = (idx / hw) % p.C;
    let n = idx / (p.C * hw);
    dx[idx] = dy[idx] * (1.0 + sb[n * 2u * p.C + c]);
}
