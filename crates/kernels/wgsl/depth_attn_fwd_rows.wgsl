// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its
// clients. If your team needs expertise in decoders that attend over their own
// earlier layers then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Depth attention forward, one WORKGROUP per row: depth_attn_fwd's cooperative twin
// @how   128-thread workgroup walks the row coalesced; per state one tree reduction of (sum of squares, 4 query dots)
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Same Params, bindings, outputs and saved statistics as depth_attn_fwd (read its header for the math, and for the
// per-row queries of `row_queries`). The reference gives one THREAD a whole row, so at a prompt of tens of tokens a
// launch runs tens of threads, each walking `count + 1` states of `dim` serially, a warp's loads `dim` floats apart. Here 128 threads share a row: each state's sum of
// squares and query dots are strided partial sums folded by a tree in workgroup memory, thread 0 keeps the state's
// RMS and scores, the softmax runs once, and the mix walks the row coalesced. The sums reassociate, so it is not the
// reference's bits (a numerical gate holds it to the host reference: `model` tests `depth_attn`).
//
// Dispatch: one workgroup per row (`Dispatch::Workgroups(rows)`), 2D-grid safe.

const MAX_STATES: u32 = 33u;
const WG: u32 = 128u;

struct Params {
    rows: u32,
    dim: u32,
    count: u32,
    nq: u32,
    eps: f32,
    scale: f32,
    row_queries: u32,
    pad1: u32,
    slots: array<vec4<u32>, 8>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       store: array<f32>;
@group(0) @binding(2) var<storage, read>       partial: array<f32>;
@group(0) @binding(3) var<storage, read>       query: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;

var<workgroup> red_ss: array<f32, WG>;
var<workgroup> red_dot: array<vec4<f32>, WG>;
// Per state: its RMS factor, then (after the softmax) its weight per query lane.
var<workgroup> rms: array<f32, MAX_STATES>;
var<workgroup> z: array<vec4<f32>, MAX_STATES>;

// Element offset of state i's row `row` (the partial when i == count).
fn state_base(i: u32, row: u32) -> u32 {
    if (i < p.count) {
        let slot = p.slots[i / 4u][i % 4u];
        return (slot * p.rows + row) * p.dim;
    }
    return row * p.dim;
}

fn state_at(i: u32, base: u32) -> f32 {
    if (i < p.count) {
        return store[base];
    }
    return partial[base];
}

// Lane q of a vec4 holds query q (row `row`'s own with row_queries); lanes past nq read zero.
fn queries_at(row: u32, j: u32) -> vec4<f32> {
    var q = vec4<f32>(0.0);
    let rows = select(1u, p.rows, p.row_queries != 0u);
    let r = select(0u, row, p.row_queries != 0u);
    for (var k = 0u; k < p.nq; k = k + 1u) {
        q[k] = query[(k * rows + r) * p.dim + j];
    }
    return q;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Uniform across the workgroup, so the early return keeps every barrier uniform.
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.rows) { return; }
    let t = li.x;
    let n = p.count + 1u;

    for (var i = 0u; i < n; i = i + 1u) {
        let base = state_base(i, row);
        var ss = 0.0;
        var dots = vec4<f32>(0.0);
        for (var j = t; j < p.dim; j = j + WG) {
            let v = state_at(i, base + j);
            ss = ss + v * v;
            dots = dots + v * queries_at(row, j);
        }
        red_ss[t] = ss;
        red_dot[t] = dots;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s = s >> 1u) {
            if (t < s) {
                red_ss[t] = red_ss[t] + red_ss[t + s];
                red_dot[t] = red_dot[t] + red_dot[t + s];
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            let r = inverseSqrt(red_ss[0] / f32(p.dim) + p.eps);
            rms[i] = r;
            z[i] = p.scale * r * red_dot[0];
        }
        // The next state overwrites the reduction slots thread 0 just read.
        workgroupBarrier();
    }

    // Softmax over the stack, every query lane at once, and the saved statistics.
    if (t == 0u) {
        var m = z[0];
        for (var i = 1u; i < n; i = i + 1u) {
            m = max(m, z[i]);
        }
        var sum = vec4<f32>(0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            let e = exp(z[i] - m);
            z[i] = e;
            sum = sum + e;
        }
        let sbase = row * n * (1u + p.nq);
        for (var i = 0u; i < n; i = i + 1u) {
            z[i] = z[i] / sum;
            stats[sbase + i] = rms[i];
            for (var q = 0u; q < p.nq; q = q + 1u) {
                stats[sbase + n + q * n + i] = z[i][q];
            }
        }
    }
    workgroupBarrier();

    for (var j = t; j < p.dim; j = j + WG) {
        var acc = vec4<f32>(0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            acc = acc + z[i] * (rms[i] * state_at(i, state_base(i, row) + j));
        }
        for (var q = 0u; q < p.nq; q = q + 1u) {
            out[(q * p.rows + row) * p.dim + j] = acc[q];
        }
    }
}
