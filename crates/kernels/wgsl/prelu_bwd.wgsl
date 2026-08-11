// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  PReLU backward, NCHW — produces BOTH gradients in one pass
// @how   one thread per channel, 2 nested serial loops (N x H*W) — the deliberately-naive CPU fallback of prelu_bwd_wg
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// PReLU backward, NCHW — produces BOTH gradients in one pass.
// The PORTABLE reference: barrier-free, so it runs on every backend including
// the CPU Cranelift JIT. prelu_bwd_wg.wgsl is the cooperative, coalesced
// variant with the SAME bindings and the SAME Params; pick between them on the
// QUERIED `DeviceCaps::workgroup_reductions` (see below).
//
//   x  : [N, C, H, W]  read       idx = ((n*C + c)*H + h)*W + w
//   a  : [nslope]      read       ai  = select(0, c, nslope > 1)
//   dy : [N, C, H, W]  read       same idx as x
//   dx : [N, C, H, W]  read_write same idx as x        (OVERWRITES)
//   da : [C]           read_write one partial per channel (ACCUMULATES)
//
// One INVOCATION per CHANNEL. Dispatch: C invocations — NOT the element count.
// Passing N*C*H*W is the `silu_mul` failure mode: N*H*W redundant invocations
// would each re-accumulate the same channel, `da` would come out inflated by
// N*H*W, `dx` would still be correct, and nothing would crash.
//
// The math (positive test `x > 0`, identical to prelu.wgsl — see its header):
//   dx[i]  = dy[i]                          where x[i] >  0
//   dx[i]  = a[ai] * dy[i]                  where x[i] <= 0
//   da[c] += sum_{n,h,w} dy[i] * x[i]       over the x[i] <= 0 elements only
// (the positive elements contribute nothing to da because y = x there, with no
// dependence on a at all).
//
// WHY THIS SHAPE EXISTS EVEN THOUGH IT IS THE SLOW ONE.
// One invocation per channel is a known double fault — C threads total, each
// walking a contiguous run alone so a warp's 32 lanes sit H*W floats apart
// — and on a GPU it is the
// wrong kernel; prelu_bwd_wg.wgsl is. But `da` is a reduction over N*H*W into
// one scalar per channel, and this engine has no atomics, so the only
// barrier-free way to produce it is to give the whole reduction to a single
// invocation. That matters because `DeviceCaps::workgroup_reductions` is FALSE
// on the CPU backend: its split-at-barrier JIT mis-executes `var<workgroup>` +
// `workgroupBarrier()` kernels (measured: `da` comes back ALL ZEROS while `dx`
// is correct — a PReLU whose slopes never move, training to a plausible loss).
// Without this kernel the family would have no correct CPU path at all, and
// `BRAIN_DEVICE=cpu` would train PReLU silently wrong. It is the same
// reference/cooperative pairing as gn_stats/gn_stats_wg, rmsnorm/rmsnorm_rows,
// max_abs_row/max_abs_rows and gradnorm_sq/gradnorm_part.
//
// On the CPU backend this shape is also not the trap it is on a GPU: the C
// invocations are spread over the rayon pool, and each walks CONTIGUOUS memory,
// which is what a cache prefetcher wants. Coalescing is a GPU property.
//
// Iteration order: plane by plane, ascending (n, h, w) — contiguous within each
// (n, c) plane. A single ascending fp32 accumulation, so `da` here and `da`
// from prelu_bwd_wg.wgsl agree to reassociation, not bit-exactly (`+`
// reassociates).
//
// `da` IS ALWAYS [C], one entry per channel, EVEN WHEN nslope == 1. With a
// single shared slope the true gradient is the sum of all C entries, and this
// kernel deliberately does not compute it — so that the two backward variants
// have byte-identical output contracts. The caller folds the C values (a
// C-length sum). Getting this wrong is silent: the shapes still match and the
// numbers are still plausible, they are just per-channel instead of summed.
//
// `da` ACCUMULATES into a pre-zeroed buffer (`da[c] = da[c] + acc`), like
// conv2d_gd_dw / convtr1d_dw / bn_dgamma, so the pass composes with a prior
// grad buffer. Callers zero it via `submit`'s clear list:
// `g.submit(&[&dab], &[step])`.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    nslope: u32,  // C (per-channel) or 1 (single shared slope)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       a:  array<f32>;
@group(0) @binding(3) var<storage, read>       dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> da: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.C) { return; }

    let hw = p.H * p.W;
    let slope = a[select(0u, c, p.nslope > 1u)];

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        let base = (n * p.C + c) * hw;
        for (var i: u32 = 0u; i < hw; i = i + 1u) {
            let idx = base + i;
            let v = x[idx];
            let g = dy[idx];
            if (v > 0.0) {
                dx[idx] = g;
            } else {
                dx[idx] = slope * g;
                acc = acc + g * v;
            }
        }
    }
    da[c] = da[c] + acc;
}
