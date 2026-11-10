// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Hyper-connection lane maps per row: pre and post gates and the Sinkhorn-normalised residual mixing matrix
// @how   one thread per row, serial Sinkhorn on at most 4x4
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// phi: [rows, 2n + n*n] = pre (n), post (n), res (n*n) from ONE fused projection; b_pre, b_post: [n]; b_res: [n*n]; maps: [rows, 2n + n*n] =
// h_pre (n), h_post (n), h_res (n*n). h_pre = sigmoid(a_pre*pre + b_pre + (8*lane - 4)), lane = 1 at lane_id;
// h_post = 2*sigmoid(a_post*post + b_post - 4*(1 - lane)); h_res = Sinkhorn(exp(a_res*res + b_res)) (row and column normalisation).
// n is at most 4. Params: rows, n, lane_id, iters, a_pre, a_post, a_res (f32 bits). Dispatch: rows threads.

struct Params { rows: u32, n: u32, lane_id: u32, iters: u32, a_pre: f32, a_post: f32, a_res: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       phi: array<f32>;
@group(0) @binding(2) var<storage, read>       b_pre: array<f32>;
@group(0) @binding(3) var<storage, read>       b_post: array<f32>;
@group(0) @binding(4) var<storage, read>       b_res: array<f32>;
@group(0) @binding(5) var<storage, read_write> maps: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows) { return; }
    let n = p.n;
    let stride = 2u * n + n * n;
    for (var j = 0u; j < n; j = j + 1u) {
        var lane = 0.0;
        if (j == p.lane_id) { lane = 1.0; }
        maps[idx * stride + j] = 1.0 / (1.0 + exp(-(p.a_pre * phi[idx * stride + j] + b_pre[j] + (8.0 * lane - 4.0))));
        maps[idx * stride + n + j] = 2.0 / (1.0 + exp(-(p.a_post * phi[idx * stride + n + j] + b_post[j] - 4.0 * (1.0 - lane))));
    }
    if (n == 4u) {
        // The released width: four vec4 rows live in registers.
        let rb = idx * stride + 2u * n;
        var r0 = vec4<f32>(phi[rb], phi[rb + 1u], phi[rb + 2u], phi[rb + 3u]) * p.a_res + vec4<f32>(b_res[0], b_res[1], b_res[2], b_res[3]);
        var r1 = vec4<f32>(phi[rb + 4u], phi[rb + 5u], phi[rb + 6u], phi[rb + 7u]) * p.a_res + vec4<f32>(b_res[4], b_res[5], b_res[6], b_res[7]);
        var r2 = vec4<f32>(phi[rb + 8u], phi[rb + 9u], phi[rb + 10u], phi[rb + 11u]) * p.a_res + vec4<f32>(b_res[8], b_res[9], b_res[10], b_res[11]);
        var r3 = vec4<f32>(phi[rb + 12u], phi[rb + 13u], phi[rb + 14u], phi[rb + 15u]) * p.a_res + vec4<f32>(b_res[12], b_res[13], b_res[14], b_res[15]);
        let mx4 = max(max(max(r0, r1), max(r2, r3)), vec4<f32>(-1.0e30));
        let m = max(max(mx4.x, mx4.y), max(mx4.z, mx4.w));
        r0 = exp(r0 - m);
        r1 = exp(r1 - m);
        r2 = exp(r2 - m);
        r3 = exp(r3 - m);
        // One reciprocal per row and one vec4 reciprocal per column sweep, then multiplies: a division per lane
        // (32 an iteration) was the whole cost of this kernel on a device (a single thread per row).
        for (var it = 0u; it < p.iters; it = it + 1u) {
            r0 = r0 * (1.0 / (r0.x + r0.y + r0.z + r0.w));
            r1 = r1 * (1.0 / (r1.x + r1.y + r1.z + r1.w));
            r2 = r2 * (1.0 / (r2.x + r2.y + r2.z + r2.w));
            r3 = r3 * (1.0 / (r3.x + r3.y + r3.z + r3.w));
            let ics = vec4<f32>(1.0) / (r0 + r1 + r2 + r3);
            r0 = r0 * ics;
            r1 = r1 * ics;
            r2 = r2 * ics;
            r3 = r3 * ics;
        }
        let ob = idx * stride + 2u * n;
        maps[ob] = r0.x; maps[ob + 1u] = r0.y; maps[ob + 2u] = r0.z; maps[ob + 3u] = r0.w;
        maps[ob + 4u] = r1.x; maps[ob + 5u] = r1.y; maps[ob + 6u] = r1.z; maps[ob + 7u] = r1.w;
        maps[ob + 8u] = r2.x; maps[ob + 9u] = r2.y; maps[ob + 10u] = r2.z; maps[ob + 11u] = r2.w;
        maps[ob + 12u] = r3.x; maps[ob + 13u] = r3.y; maps[ob + 14u] = r3.z; maps[ob + 15u] = r3.w;
        return;
    }
    // Sinkhorn in the exponent domain: alternately normalise the rows and the columns of exp(logits). The same
    // fixed-point iteration as the log-domain form, without a transcendental per step; logits are shifted by
    // their maximum first, which scaling the matrix does not change after the first normalisation.
    var l: array<f32, 16>;
    var mx = -1.0e30;
    for (var i = 0u; i < n * n; i = i + 1u) {
        l[i] = p.a_res * phi[idx * stride + 2u * n + i] + b_res[i];
        mx = max(mx, l[i]);
    }
    for (var i = 0u; i < n * n; i = i + 1u) { l[i] = exp(l[i] - mx); }
    for (var it = 0u; it < p.iters; it = it + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            var s = 0.0;
            for (var j = 0u; j < n; j = j + 1u) { s = s + l[i * n + j]; }
            for (var j = 0u; j < n; j = j + 1u) { l[i * n + j] = l[i * n + j] / s; }
        }
        for (var j = 0u; j < n; j = j + 1u) {
            var s = 0.0;
            for (var i = 0u; i < n; i = i + 1u) { s = s + l[i * n + j]; }
            for (var i = 0u; i < n; i = i + 1u) { l[i * n + j] = l[i * n + j] / s; }
        }
    }
    for (var i = 0u; i < n * n; i = i + 1u) { maps[idx * stride + 2u * n + i] = l[i]; }
}
