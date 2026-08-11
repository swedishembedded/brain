// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  PReLU backward, one WORKGROUP per CHANNEL — the cooperative, COALESCED twin of prelu_bwd.wgsl
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// PReLU backward, one WORKGROUP per CHANNEL — the cooperative, COALESCED twin
// of prelu_bwd.wgsl. Identical bindings, identical Params, identical outputs;
// only the dispatch differs (C*64 invocations here, C there), so a call site
// picks between the two with one `if` and no other change.
//
//   x  : [N, C, H, W]  read       idx = ((n*C + c)*H + h)*W + w
//   a  : [nslope]      read       ai  = select(0, c, nslope > 1)
//   dy : [N, C, H, W]  read       same idx as x
//   dx : [N, C, H, W]  read_write same idx as x        (OVERWRITES)
//   da : [C]           read_write one partial per channel (ACCUMULATES)
//
// Dispatch: C * 64 invocations (C workgroups of 64). NOT the element count —
// passing N*C*H*W here is the `silu_mul` failure mode: N*H*W redundant
// workgroups would each re-accumulate the same channel and `da` would come out
// inflated by N*H*W with `dx` still correct. Plausible numbers, no crash.
//
// **GPU-ONLY — this kernel needs `DeviceCaps::workgroup_reductions`.** It uses
// `var<workgroup>` + `workgroupBarrier()`, and the CPU backend's Cranelift JIT
// mis-executes exactly that construct (`backend_cpu::caps` reports
// `workgroup_reductions: false` for this reason; measured here: `da` comes back
// ALL ZEROS while `dx` is correct — silent, not a crash, and a PReLU whose
// slopes never move still trains to a plausible-looking loss). The fallback is
// prelu_bwd.wgsl, which is exactly what this replaces; the gate is a
// CORRECTNESS gate, not a preference, the same one `Op::GradNorm` /
// `Op::MaxAbsRow` apply in `backend_api::select`.
//
// The math (positive test `x > 0`, identical to prelu.wgsl — see its header):
//   dx[i]  = dy[i]                          where x[i] >  0
//   dx[i]  = a[ai] * dy[i]                  where x[i] <= 0
//   da[c] += sum_{n,h,w} dy[i] * x[i]       over the x[i] <= 0 elements only
// (the positive elements contribute nothing to da because y = x there, with no
// dependence on a at all).
//
// WHY A WORKGROUP PER CHANNEL, not a thread per channel.
// `da` is a reduction over N*H*W into a single scalar per channel. The obvious
// shape — one invocation per channel, as bn_dgamma.wgsl / gn_dgamma.wgsl still
// do, and as the prelu_bwd.wgsl fallback must — is the documented double fault
// (.agents/rules/kernels.md §C.2): it launches only C threads (64 for an
// IResNet stem, on a 3840-core card), and each of those threads walks a
// contiguous run alone, so a warp's 32 lanes sit H*W floats apart and every
// 32-byte sector fetched serves ONE useful float. Here the 64 threads of a
// workgroup stride the channel's WHOLE N*H*W index space together (thread t
// takes t, t+64, t+128, ...), so every fetched sector is fully used, and the C
// channels run concurrently — the same fix rmsnorm_rows made for RMSNorm and
// gn_stats_wg for GroupNorm.
//
// The walk is over the channel's flat N*H*W space, NOT `for n { stride the
// H*W plane }`. The two are equivalent only while H*W >= 64: below that the
// per-plane form leaves 64-H*W lanes idle in every plane, and at H*W == 1 —
// the flat [N, C] activation this family advertises — it collapses to thread 0
// walking the entire channel with a stride of C floats, which is exactly the
// bn_dgamma fault this header rejects, re-introduced one shape lower down.
// Cost of the flat form: one u32 divide per element to recover `n`. The kernel
// moves 16 bytes per element (x, dy, dx + the a[] scalar), so it is bandwidth
// bound on every card in scope and the divide hides under the traffic.
//
// Fusing dx into the same pass is what makes the channel-owned mapping cheap:
// x and dy are streamed ONCE and serve both outputs. Splitting the backward in
// two would re-read both (5 tensor reads instead of 3). The cost of the fusion
// is that dx — an N*C*H*W elementwise write — is produced by only C*64
// invocations. That is the right trade while C*64 keeps the device busy; if a
// profile ever shows this kernel latency-bound at very small C, the fix is a
// two-pass `_part` split (partials per (c, n-chunk) + a second small fold),
// NOT a cross-workgroup accumulate — there are no atomics in this engine.
//
// Race-freedom: every element of `dx` is written by exactly one invocation, and
// every `da[c]` by exactly one workgroup (its thread 0), because the invocation
// -> element map is a bijection. Only ONE top-level `workgroupBarrier()`, which
// is the CPU JIT's structural limit; the 64 partials are folded by a single
// thread (64 adds, cheaper than a second barrier).
//
// `da` IS ALWAYS [C], one entry per channel, EVEN WHEN nslope == 1. With a
// single shared slope the true gradient is the sum of all C entries, and this
// kernel deliberately does not compute it: two workgroups adding into da[0]
// would race. The caller folds the C values (a C-length sum). Getting this
// wrong is silent: the shapes still match and the numbers are still plausible,
// they are just per-channel instead of summed.
//
// `da` ACCUMULATES into a pre-zeroed buffer (`da[c] = da[c] + s`), like
// conv2d_gd_dw / convtr1d_dw, so the pass composes with a prior grad buffer.
// Callers zero it via `submit`'s clear list: `g.submit(&[&dab], &[step])`.
//
// Reduction order: ascending flat (n, h, w) within each thread's stride-64
// slice, then ascending thread index — deterministic for a fixed shape, but
// NOT the same order as prelu_bwd.wgsl's single ascending run, so the two agree
// to fp32 reassociation and not bit-exactly (`+` reassociates; see
// .agents/rules/kernels.md §E.4).

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

var<workgroup> psum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let c = wg.y * nwg.x + wg.x;
    let t = li.x;
    // Uniform across the workgroup, so returning before the barrier is legal.
    if (c >= p.C) { return; }

    let hw = p.H * p.W;
    let slope = a[select(0u, c, p.nslope > 1u)];

    // Stride 64 over this channel's flat N*H*W space: the 64 lanes hold
    // consecutive words (a single iteration straddles a plane boundary at most
    // once, splitting it into two coalesced runs). See the header for why this
    // is NOT a per-plane loop.
    let cn = p.N * hw;
    var acc = 0.0;
    for (var i = t; i < cn; i = i + 64u) {
        let n = i / hw;  // hw >= 1 wherever cn >= 1, so this never divides by 0
        let idx = (n * p.C + c) * hw + (i - n * hw);
        let v = x[idx];
        let g = dy[idx];
        if (v > 0.0) {
            dx[idx] = g;
        } else {
            dx[idx] = slope * g;
            acc = acc + g * v;
        }
    }

    psum[t] = acc;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var k: u32 = 0u; k < 64u; k = k + 1u) {
            s = s + psum[k];
        }
        da[c] = da[c] + s;
    }
}
