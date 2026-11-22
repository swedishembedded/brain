// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its
// clients. If your team needs expertise in training decoders that attend over
// their own earlier layers then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Depth attention backward to the stacked states, one WORKGROUP per row: depth_attn_bwd's cooperative twin
// @how   128-thread workgroup walks the row coalesced; tree reductions for the score and norm-backward dot products
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Same Params, bindings and outputs as depth_attn_bwd (read its header for the math): d-state ADDED into every
// retained slot's gradient and the partial's, dz written for depth_attn_bwd_query. The reference gives one thread a
// whole row and walks it three times per state; here 128 threads share the row, each dot product a strided partial
// sum folded by a tree in workgroup memory. The sums reassociate, so it is held to finite differences and the
// reference within a tolerance, not to its bits (`model` tests `depth_attn`).
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
    pad0: u32,
    pad1: u32,
    slots: array<vec4<u32>, 8>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       store: array<f32>;
@group(0) @binding(2) var<storage, read>       partial: array<f32>;
@group(0) @binding(3) var<storage, read>       query: array<f32>;
@group(0) @binding(4) var<storage, read>       stats: array<f32>;
@group(0) @binding(5) var<storage, read>       dout: array<f32>;
@group(0) @binding(6) var<storage, read_write> dstore: array<f32>;
@group(0) @binding(7) var<storage, read_write> dpartial: array<f32>;
@group(0) @binding(8) var<storage, read_write> dz: array<f32>;

var<workgroup> red: array<vec4<f32>, WG>;
// Per state: the score gradient (dw, then dz) per query lane.
var<workgroup> gz: array<vec4<f32>, MAX_STATES>;

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

// Lane q of a vec4 holds query q's value; lanes past nq read zero.
fn dout_at(row: u32, j: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    for (var k = 0u; k < p.nq; k = k + 1u) {
        v[k] = dout[(k * p.rows + row) * p.dim + j];
    }
    return v;
}

fn queries_at(j: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    for (var k = 0u; k < p.nq; k = k + 1u) {
        v[k] = query[k * p.dim + j];
    }
    return v;
}

fn weights(sbase: u32, n: u32, i: u32) -> vec4<f32> {
    var wi = vec4<f32>(0.0);
    for (var q = 0u; q < p.nq; q = q + 1u) {
        wi[q] = stats[sbase + n + q * n + i];
    }
    return wi;
}

// Tree-fold `red` into red[0]; every thread must call it (it holds barriers), and red[0] is valid after it returns.
fn fold(t: u32) {
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (t < s) {
            red[t] = red[t] + red[t + s];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.rows) { return; }
    let t = li.x;
    let n = p.count + 1u;
    let sbase = row * n * (1u + p.nq);

    // dw[q,i] = rms_i * <dout[q], s_i>
    for (var i = 0u; i < n; i = i + 1u) {
        let base = state_base(i, row);
        var acc = vec4<f32>(0.0);
        for (var j = t; j < p.dim; j = j + WG) {
            acc = acc + dout_at(row, j) * state_at(i, base + j);
        }
        red[t] = acc;
        fold(t);
        if (t == 0u) {
            gz[i] = red[0] * stats[sbase + i];
        }
        workgroupBarrier();
    }
    // dz[q,i] = w[q,i] * (dw[q,i] - sum_e w[q,e] dw[q,e])
    if (t == 0u) {
        var wdw = vec4<f32>(0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            wdw = wdw + weights(sbase, n, i) * gz[i];
        }
        let zbase = row * p.nq * n;
        for (var i = 0u; i < n; i = i + 1u) {
            let g = weights(sbase, n, i) * (gz[i] - wdw);
            gz[i] = g;
            for (var q = 0u; q < p.nq; q = q + 1u) {
                dz[zbase + q * n + i] = g[q];
            }
        }
    }
    workgroupBarrier();

    let inv_dim = 1.0 / f32(p.dim);
    for (var i = 0u; i < n; i = i + 1u) {
        let wi = weights(sbase, n, i);
        let gi = p.scale * gz[i];
        let r = stats[sbase + i];
        let base = state_base(i, row);
        // <dn_i, n_i>
        var c = 0.0;
        for (var j = t; j < p.dim; j = j + WG) {
            let dn = dot(wi, dout_at(row, j)) + dot(gi, queries_at(j));
            c = c + dn * r * state_at(i, base + j);
        }
        red[t] = vec4<f32>(c, 0.0, 0.0, 0.0);
        fold(t);
        let cc = red[0].x;
        for (var j = t; j < p.dim; j = j + WG) {
            let dn = dot(wi, dout_at(row, j)) + dot(gi, queries_at(j));
            let ds = r * (dn - r * state_at(i, base + j) * cc * inv_dim);
            if (i < p.count) {
                dstore[base + j] = dstore[base + j] + ds;
            } else {
                dpartial[base + j] = dpartial[base + j] + ds;
            }
        }
        // The next state overwrites red[0] every thread just read.
        workgroupBarrier();
    }
}
