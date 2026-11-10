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
// pre, post: [rows, n]; res: [rows, n*n]; b_pre, b_post: [n]; b_res: [n*n]; maps: [rows, 2n + n*n] =
// h_pre (n), h_post (n), h_res (n*n). h_pre = sigmoid(a_pre*pre + b_pre + (8*lane - 4)), lane = 1 at lane_id;
// h_post = 2*sigmoid(a_post*post + b_post - 4*(1 - lane)); h_res = Sinkhorn(a_res*res + b_res) in log space.
// n is at most 4. Params: rows, n, lane_id, iters, a_pre, a_post, a_res (f32 bits). Dispatch: rows threads.

struct Params { rows: u32, n: u32, lane_id: u32, iters: u32, a_pre: f32, a_post: f32, a_res: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pre: array<f32>;
@group(0) @binding(2) var<storage, read>       post: array<f32>;
@group(0) @binding(3) var<storage, read>       res: array<f32>;
@group(0) @binding(4) var<storage, read>       b_pre: array<f32>;
@group(0) @binding(5) var<storage, read>       b_post: array<f32>;
@group(0) @binding(6) var<storage, read>       b_res: array<f32>;
@group(0) @binding(7) var<storage, read_write> maps: array<f32>;

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
        maps[idx * stride + j] = 1.0 / (1.0 + exp(-(p.a_pre * pre[idx * n + j] + b_pre[j] + (8.0 * lane - 4.0))));
        maps[idx * stride + n + j] = 2.0 / (1.0 + exp(-(p.a_post * post[idx * n + j] + b_post[j] - 4.0 * (1.0 - lane))));
    }
    var l: array<f32, 16>;
    for (var i = 0u; i < n * n; i = i + 1u) { l[i] = p.a_res * res[idx * n * n + i] + b_res[i]; }
    for (var it = 0u; it < p.iters; it = it + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            var mx = -1.0e30;
            for (var j = 0u; j < n; j = j + 1u) { mx = max(mx, l[i * n + j]); }
            var s = 0.0;
            for (var j = 0u; j < n; j = j + 1u) { s = s + exp(l[i * n + j] - mx); }
            let lse = mx + log(s);
            for (var j = 0u; j < n; j = j + 1u) { l[i * n + j] = l[i * n + j] - lse; }
        }
        for (var j = 0u; j < n; j = j + 1u) {
            var mx = -1.0e30;
            for (var i = 0u; i < n; i = i + 1u) { mx = max(mx, l[i * n + j]); }
            var s = 0.0;
            for (var i = 0u; i < n; i = i + 1u) { s = s + exp(l[i * n + j] - mx); }
            let lse = mx + log(s);
            for (var i = 0u; i < n; i = i + 1u) { l[i * n + j] = l[i * n + j] - lse; }
        }
    }
    for (var i = 0u; i < n * n; i = i + 1u) { maps[idx * stride + 2u * n + i] = exp(l[i]); }
}
