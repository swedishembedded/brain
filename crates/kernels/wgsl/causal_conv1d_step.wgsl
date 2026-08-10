// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal depthwise Conv1d, single-token streaming step (ring-buffer state)
// @how   one thread per (n,c) output element, serial reduction over K taps
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// One decode step of a causal depthwise Conv1d (`groups = C`, i.e. `Cin/G=1`,
// matching `conv1d.wgsl`'s NCL layout specialised to `L=1`): given the current
// token's per-channel value `x[n,c]` and a persistent per-(n,c) sliding window
// of the last `K-1` input values (`hist`, the "conv state"), computes
//
//   y[n,c] = sum_{j=0}^{K-2} hist[n,c,j]*w[c,j]  +  x[n,c]*w[c,K-1]
//
// then shifts the window for the next call (drop the oldest tap, append this
// token): `hist[n,c,0..K-2] <- hist[n,c,1..K-1]`, `hist[n,c,K-2] <- x[n,c]`.
// Running this once per token, threading the SAME `hist` buffer, reproduces
// `conv1d.wgsl`'s whole-sequence causal conv (`pad = K-1`, `stride = 1`,
// `dilation = 1`, `groups = Cin`) bit-for-bit -- gated by
// `crates/model/tests/causal_conv1d_step.rs`, which runs both and compares.
// `hist` must start ZEROED for a fresh sequence: that is the same implicit
// left zero-pad `conv1d.wgsl` applies via its own `pad` parameter for the
// first `K-1` tokens.
//
// `x`/`y`: `[N,C]` (`conv1d.wgsl`'s `[N,Cin-or-Cout,L]` layout specialised to
// `L=1`, so `idx = n*C+c` matches that kernel's own addressing exactly).
// `w`: `[C,K]` (`conv1d.wgsl`'s depthwise weight layout `[Cout,Cin/G=1,K]`
// with `Cout=C`, `Cin/G=1` collapsed away). `hist`: `[N,C,K-1]`, `read_write`
// -- both this call's window AND the write-back the next call reads.
//
// No existing elementwise/reduction primitive fit this without an extra pass
// (`docs/kernel-checklist.md` sec A): `row_dot.wgsl` is the nearest sibling in
// SHAPE (one thread, small serial reduction) but its two operands are
// addressed IDENTICALLY per row, whereas here the weight varies only by
// CHANNEL while `hist`/`x` vary by both `n` and `c` -- using `row_dot` would
// need `w` pre-broadcast to `N` copies, a real `N`x memory cost paid on every
// decode step. This kernel is that same "one thread, small serial reduction"
// shape (`@opt 2`), plus the ring-buffer shift fused into the same
// invocation: each thread owns its own `(n,c)` row of `hist` exclusively, so
// reading then overwriting it within one invocation has no cross-thread
// hazard (the usual `assert_no_output_alias` host-side rule is about two
// SEPARATE bindings of one buffer within a single dispatch's bind group, not
// about one binding read then written by the same invocation).

struct Params { n: u32, c: u32, k: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       w:    array<f32>;
@group(0) @binding(3) var<storage, read_write> hist: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n * p.c;
    if (idx >= total) { return; }

    let ci = idx % p.c;
    let wbase = ci * p.k;
    let hbase = idx * (p.k - 1u);
    let xi = x[idx];

    var acc = 0.0;
    for (var j: u32 = 0u; j < p.k - 1u; j = j + 1u) {
        acc = acc + hist[hbase + j] * w[wbase + j];
    }
    acc = acc + xi * w[wbase + p.k - 1u];
    y[idx] = acc;

    // Shift the window: drop tap 0, append this token at the end. Guarded on
    // `k >= 2u` so `p.k - 2u` (u32) never underflows when `k == 1` (a
    // pointwise "conv", no history at all -- legal, just untested by any
    // current caller).
    if (p.k >= 2u) {
        for (var j: u32 = 0u; j < p.k - 2u; j = j + 1u) {
            hist[hbase + j] = hist[hbase + j + 1u];
        }
        hist[hbase + p.k - 2u] = xi;
    }
}
