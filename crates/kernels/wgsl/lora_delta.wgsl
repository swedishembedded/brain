// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Low-rank adapter epilogue: out += t @ bt, t [m,r] the already-computed A·x, bt [r,n] the transposed scaled B
// @how   one thread per output element, serial reduction over the rank axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Low-rank adapter epilogue: `out += t @ bt`, where `t` is `[m, r]` (the
// already-computed `A·x` of a LoRA correction) and `bt` is `[r, n]` - the
// adapter's `B [n, r]` stored TRANSPOSED, with its scale already folded in.
//
// This is the second half of applying a LoRA at RUNTIME instead of folding it
// into the base weight: `y = W·x + B·(A·x)`. The first half (`A·x`) is a plain
// GEMM with `n = r`, so it needs no kernel of its own; this half has to
// ACCUMULATE into an output another kernel already wrote, which a plain matmul
// cannot do.
//
// Why a serial reduction and not a tile: `r` is the adapter rank, 8–32 against
// hidden dims of 3072–12288, so the whole reduction is a handful of FMAs and
// the kernel is bound by the read-modify-write of `out` (2 floats of traffic
// per r FLOPs). Tiling would buy nothing there; what DOES matter is that both
// large operands are coalesced, and they are: consecutive threads take
// consecutive `j`, so `out[idx]` is contiguous and `bt[k*n + j]` is contiguous.
// That is the whole reason `B` is stored transposed rather than in its natural
// `[n, r]` layout, where the same access would be a stride-`r` gather.
//
// `t` is read broadcast (all `n` threads of one output row read the same `r`
// values), and `bt` is small - `r·n` floats, L2-resident - so both ride the
// cache instead of the DRAM path.

struct Params {
    m: u32,
    n: u32,
    r: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read>       t:   array<f32>;
@group(0) @binding(3) var<storage, read>       bt:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.m * p.n) { return; }
    let i = idx / p.n;
    let j = idx - i * p.n;
    var acc = 0.0;
    for (var k = 0u; k < p.r; k = k + 1u) {
        acc = acc + t[i * p.r + k] * bt[k * p.n + j];
    }
    out[idx] = out[idx] + acc;
}
