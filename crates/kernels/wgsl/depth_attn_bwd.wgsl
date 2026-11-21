// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its
// clients. If your team needs expertise in training decoders that attend over
// their own earlier layers then you can procure our services by sending an email
// to info@swedishembedded.com.

// @what  Depth attention backward to the stacked states: d-state into every retained slot and the partial, plus d-scores
// @how   one thread per row, one pass for the score gradients and two per state for the norm backward
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of depth_attn_fwd (same params and stack). With n_i = rms_i * s_i and dout[q] the output gradient:
//   dw[q,i] = <dout[q], n_i>,   dz[q,i] = w[q,i] * (dw[q,i] - sum_e w[q,e] dw[q,e])
//   dn_i    = sum_q w[q,i] dout[q] + scale * dz[q,i] query[q]
//   ds_i    = rms_i * (dn_i - n_i * <dn_i, n_i> / dim)
// ds_i is ADDED to dstore[slot_i] (or dpartial for the last state): a retained state feeds several consumers and its
// gradient is the sum over all of them, so the caller zeroes these once and every consumer accumulates. dz is
// written for depth_attn_bwd_query: dz[r] = [dz[0,0..n-1], .., dz[nq-1,0..n-1]].
//
//   store, partial, query, stats: as the forward   dout: [nq, rows, dim]
//   dstore: [slots, rows, dim] (+=)   dpartial: [rows, dim] (+=)   dz: [rows, nq*n]
// Dispatch: rows invocations.

const MAX_STATES: u32 = 33u;

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

fn state_index(i: u32, row: u32, j: u32) -> u32 {
    let slot = p.slots[i / 4u][i % 4u];
    return (slot * p.rows + row) * p.dim + j;
}

fn state(i: u32, row: u32, j: u32) -> f32 {
    if (i < p.count) {
        return store[state_index(i, row, j)];
    }
    return partial[row * p.dim + j];
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.rows) { return; }
    let n = p.count + 1u;
    let sbase = row * n * (1u + p.nq);

    var rms: array<f32, MAX_STATES>;
    var w: array<vec4<f32>, MAX_STATES>;
    var g: array<vec4<f32>, MAX_STATES>;
    for (var i = 0u; i < n; i = i + 1u) {
        rms[i] = stats[sbase + i];
        var wi = vec4<f32>(0.0);
        for (var q = 0u; q < p.nq; q = q + 1u) {
            wi[q] = stats[sbase + n + q * n + i];
        }
        w[i] = wi;
        // dw[q,i] = <dout[q], n_i>
        var acc = vec4<f32>(0.0);
        for (var j = 0u; j < p.dim; j = j + 1u) {
            acc = acc + dout_at(row, j) * state(i, row, j);
        }
        g[i] = acc * rms[i];
    }
    var wdw = vec4<f32>(0.0);
    for (var i = 0u; i < n; i = i + 1u) {
        wdw = wdw + w[i] * g[i];
    }
    let zbase = row * p.nq * n;
    for (var i = 0u; i < n; i = i + 1u) {
        g[i] = w[i] * (g[i] - wdw);
        for (var q = 0u; q < p.nq; q = q + 1u) {
            dz[zbase + q * n + i] = g[i][q];
        }
    }
    let inv_dim = 1.0 / f32(p.dim);
    for (var i = 0u; i < n; i = i + 1u) {
        let wi = w[i];
        let gi = p.scale * g[i];
        let r = rms[i];
        // <dn_i, n_i>
        var c = 0.0;
        for (var j = 0u; j < p.dim; j = j + 1u) {
            let dn = dot(wi, dout_at(row, j)) + dot(gi, queries_at(j));
            c = c + dn * r * state(i, row, j);
        }
        for (var j = 0u; j < p.dim; j = j + 1u) {
            let dn = dot(wi, dout_at(row, j)) + dot(gi, queries_at(j));
            let ds = r * (dn - r * state(i, row, j) * c * inv_dim);
            if (i < p.count) {
                let k = state_index(i, row, j);
                dstore[k] = dstore[k] + ds;
            } else {
                let k = row * p.dim + j;
                dpartial[k] = dpartial[k] + ds;
            }
        }
    }
}
