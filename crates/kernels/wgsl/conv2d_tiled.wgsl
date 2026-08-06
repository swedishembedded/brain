// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Weight-staged conv2d: identical math to conv2d.wgsl, but one workgroup loads its output channel's weights into WORKGROUP (on-chip) memory once and reuses them across a block of output spatial positions — so each weight is read from global memory once per block instead of once per output pixel (the H*W weight re-read that makes the naive kernel memory-bound on a GPU)
// @how   64-thread workgroup tile, 2 barriers
// @opt   4
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// Weight-staged conv2d: identical math to conv2d.wgsl, but one workgroup loads
// its output channel's weights into WORKGROUP (on-chip) memory once and reuses
// them across a block of output spatial positions — so each weight is read from
// global memory once per block instead of once per output pixel (the H*W weight
// re-read that makes the naive kernel memory-bound on a GPU).
//
// Work-group layout: one workgroup = one (n, output-channel, 64-position block).
//   * the 64 invocations cooperatively load w[co, :] (Cin*K*K floats) into `wsh`,
//   * workgroupBarrier(),
//   * each invocation computes one output position reading weights from `wsh`.
//
// This is the single source of truth for BOTH backends: wgpu runs it directly,
// and the wgsl-cpu Cranelift JIT compiles it via its work-group execution model
// (workgroup memory = per-workgroup scratch; the barrier splits the body into a
// cooperative-load loop then a per-invocation compute loop).
//
// Dispatch: total invocations = N * Cout * ceil(Ho*Wo / 64) * 64 (the caller
// rounds the spatial extent up to whole 64-wide blocks per channel).

struct Params {
    N: u32,
    Cin: u32,
    H: u32,
    W: u32,
    Cout: u32,
    K: u32,
    stride: u32,
    pad: u32,
    Ho: u32,
    Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// Cin*K*K weights for this workgroup's output channel. 8192 = 32 KiB covers
// every yolov8n layer (max Cin*K*K = 512*9 = 4608).
var<workgroup> wsh: array<f32, 8192>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let kg = p.Cin * p.K * p.K;
    let psz = p.Ho * p.Wo;
    let blocks = (psz + 63u) / 64u;          // 64-position blocks per channel
    let wg = wgid.y * nwg.x + wgid.x;        // flat workgroup id

    // Decode (n, co, spatial-block) from the flat workgroup id.
    let per_n = p.Cout * blocks;
    let n = wg / per_n;
    let cob = wg % per_n;
    let co = cob / blocks;
    let sblock = cob % blocks;

    // Cooperative load: this workgroup's 64 invocations stage w[co, :] into wsh.
    var i: u32 = lid.x;
    loop {
        if (i >= kg) { break; }
        wsh[i] = w[co * kg + i];
        i = i + 64u;
    }
    workgroupBarrier();

    // Each invocation computes one output position from the staged weights.
    let pidx = sblock * 64u + lid.x;
    if (pidx < psz && n < p.N) {
        let wo = pidx % p.Wo;
        let ho = pidx / p.Wo;
        var acc = 0.0;
        for (var ci: u32 = 0u; ci < p.Cin; ci = ci + 1u) {
            for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
                let hi_b = ho * p.stride + kh;
                if (hi_b >= p.pad) {
                    let hi = hi_b - p.pad;
                    if (hi < p.H) {
                        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                            let wi_b = wo * p.stride + kw;
                            if (wi_b >= p.pad) {
                                let wi = wi_b - p.pad;
                                if (wi < p.W) {
                                    let x_idx = ((n * p.Cin + ci) * p.H + hi) * p.W + wi;
                                    let w_off = (ci * p.K + kh) * p.K + kw;
                                    acc = acc + x[x_idx] * wsh[w_off];
                                }
                            }
                        }
                    }
                }
            }
        }
        y[(n * p.Cout + co) * psz + pidx] = acc;
    }
}
