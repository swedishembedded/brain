// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Hyper-connection front of a decoder layer in one workgroup per row: lane maps, the lane mix u, and the block input y = u
// @how   128 threads: 16 threads run the Sinkhorn iteration on 4x4 through workgroup memory while every thread holds its share of the lane mix in registers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// mirage_mhc_maps, mirage_mhc_mix and a row copy as one dispatch, for the released lane count n = 4. The single thread
// per row of mirage_mhc_maps spends its time on 20 dependent iterations of divisions; here one thread per matrix entry
// shares them (row sums and column sums through workgroup memory, double buffered so an iteration costs two barriers),
// and the lane mix, which does not depend on the Sinkhorn result, is loaded into registers before it starts. The
// arithmetic of every output follows the original's order, so results agree with it to rounding.
//
// phi: [rows, 2n + n*n]   b_pre, b_post: [n]   b_res: [n*n]   lanes: [rows, n, d]
// maps: [rows, 2n + n*n] = h_pre, h_post, h_res   u = sum_j h_pre[j] lanes[j]: [rows, d]   y = u (a copy the block updates in place)
// Params: rows, n (must be 4), lane_id, iters, a_pre, a_post, a_res (f32 bits), d. Dispatch: one workgroup per row.

struct Params { rows: u32, n: u32, lane_id: u32, iters: u32, a_pre: f32, a_post: f32, a_res: f32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       lanes: array<f32>;
@group(0) @binding(2) var<storage, read>       phi: array<f32>;
@group(0) @binding(3) var<storage, read>       b_pre: array<f32>;
@group(0) @binding(4) var<storage, read>       b_post: array<f32>;
@group(0) @binding(5) var<storage, read>       b_res: array<f32>;
@group(0) @binding(6) var<storage, read_write> maps: array<f32>;
@group(0) @binding(7) var<storage, read_write> u: array<f32>;
@group(0) @binding(8) var<storage, read_write> y: array<f32>;

var<workgroup> pre: array<f32, 4>;
var<workgroup> sa: array<f32, 16>;
var<workgroup> sb: array<f32, 16>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.rows) { return; }
    let t = li.x;
    let stride = 24u;
    let mb = row * stride;
    // The lane mix's inputs are requested first: they do not wait for the maps.
    var l0 = vec4<f32>(0.0);
    var l1 = vec4<f32>(0.0);
    var l2 = vec4<f32>(0.0);
    var l3 = vec4<f32>(0.0);
    let d4 = p.d / 4u;
    let live = t < d4;
    if (live) {
        let c = 4u * t;
        let b0 = (row * 4u) * p.d + c;
        l0 = vec4<f32>(lanes[b0], lanes[b0 + 1u], lanes[b0 + 2u], lanes[b0 + 3u]);
        l1 = vec4<f32>(lanes[b0 + p.d], lanes[b0 + p.d + 1u], lanes[b0 + p.d + 2u], lanes[b0 + p.d + 3u]);
        l2 = vec4<f32>(lanes[b0 + 2u * p.d], lanes[b0 + 2u * p.d + 1u], lanes[b0 + 2u * p.d + 2u], lanes[b0 + 2u * p.d + 3u]);
        l3 = vec4<f32>(lanes[b0 + 3u * p.d], lanes[b0 + 3u * p.d + 1u], lanes[b0 + 3u * p.d + 2u], lanes[b0 + 3u * p.d + 3u]);
    }
    // Gates: thread j < 4 owns h_pre[j] and h_post[j].
    if (t < 4u) {
        var lane = 0.0;
        if (t == p.lane_id) { lane = 1.0; }
        let hp = 1.0 / (1.0 + exp(-(p.a_pre * phi[mb + t] + b_pre[t] + (8.0 * lane - 4.0))));
        pre[t] = hp;
        maps[mb + t] = hp;
        maps[mb + 4u + t] = 2.0 / (1.0 + exp(-(p.a_post * phi[mb + 4u + t] + b_post[t] - 4.0 * (1.0 - lane))));
    }
    // Sinkhorn: thread e < 16 owns entry (i, j) = (e / 4, e % 4) of the 4x4 matrix.
    let i = t / 4u;
    let j = t % 4u;
    var r = 0.0;
    if (t < 16u) {
        sa[t] = p.a_res * phi[mb + 8u + t] + b_res[t];
    }
    workgroupBarrier();
    if (t < 16u) {
        var mx = -1.0e30;
        for (var e = 0u; e < 16u; e = e + 1u) { mx = max(mx, sa[e]); }
        r = exp(sa[t] - mx);
    }
    workgroupBarrier();
    for (var it = 0u; it < p.iters; it = it + 1u) {
        if (t < 16u) { sb[t] = r; }
        workgroupBarrier();
        if (t < 16u) {
            let rs = ((sb[4u * i] + sb[4u * i + 1u]) + sb[4u * i + 2u]) + sb[4u * i + 3u];
            r = r * (1.0 / rs);
            sa[t] = r;
        }
        workgroupBarrier();
        if (t < 16u) {
            let cs = ((sa[j] + sa[4u + j]) + sa[8u + j]) + sa[12u + j];
            r = r * (1.0 / cs);
        }
    }
    if (t < 16u) { maps[mb + 8u + t] = r; }
    workgroupBarrier();
    if (live) {
        let c = 4u * t;
        let mixed = ((pre[0] * l0 + pre[1] * l1) + pre[2] * l2) + pre[3] * l3;
        let at = row * p.d + c;
        u[at] = mixed.x; u[at + 1u] = mixed.y; u[at + 2u] = mixed.z; u[at + 3u] = mixed.w;
        y[at] = mixed.x; y[at + 1u] = mixed.y; y[at + 2u] = mixed.z; y[at + 3u] = mixed.w;
    }
}
