// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its
// clients. If your team needs expertise in decoders that attend over their own
// earlier layers then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Depth attention forward: per row, a softmax over a retained stack of RMS-normed states, mixed by learned queries
// @how   one thread per row, two passes over the stack (norm + scores, then the mix)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// With `row_queries` set, every row has its own query set: query is [nq, rows, dim], query q of row r at
// (q * rows + r) * dim (a served batch whose rows run under different adapters). The backward takes shared queries.
//
// The stack of row r is [store[slot_0][r], ..., store[slot_{count-1}][r], partial[r]] (n = count + 1 states of
// `dim`). Each state is RMS-normed without a gain, n_i = s_i * rsqrt(mean(s_i^2) + eps), and every query q (`nq` of
// them, shared by all rows) mixes the normed states:
//   z[q,i] = scale * <query[q], n_i>,  w[q,:] = softmax_i(z[q,:]),  out[q,r] = sum_i w[q,i] n_i
// The backward (depth_attn_bwd, depth_attn_bwd_query) reads the per-row statistics this kernel saves:
//   stats[r] = [rms_0 .. rms_{n-1}, w[0,0..n-1], .., w[nq-1,0..n-1]]   (stride n * (1 + nq))
// Composing it from rmsnorm + matmul + softmax kernels would write and re-read every normed state per consumer; here
// each state is read twice and only the nq outputs are written.
//
//   store: [slots, rows, dim]  partial: [rows, dim]  query: [nq, dim] (or [nq, rows, dim])  out: [nq, rows, dim]
//   stats: [rows, n*(1+nq)]
//   params: rows, dim, count (<= 32), nq (<= 4), eps (f32), scale (f32), row_queries (0/1), a pad word, then 32 slot
//   indices
// Dispatch: rows invocations.

const MAX_STATES: u32 = 33u;

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

fn state(i: u32, row: u32, j: u32) -> f32 {
    if (i < p.count) {
        let slot = p.slots[i / 4u][i % 4u];
        return store[(slot * p.rows + row) * p.dim + j];
    }
    return partial[row * p.dim + j];
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.rows) { return; }
    let n = p.count + 1u;
    let sbase = row * n * (1u + p.nq);

    var rms: array<f32, MAX_STATES>;
    var z: array<vec4<f32>, MAX_STATES>;
    for (var i = 0u; i < n; i = i + 1u) {
        var ss = 0.0;
        var dots = vec4<f32>(0.0);
        for (var j = 0u; j < p.dim; j = j + 1u) {
            let v = state(i, row, j);
            ss = ss + v * v;
            dots = dots + v * queries_at(row, j);
        }
        let r = inverseSqrt(ss / f32(p.dim) + p.eps);
        rms[i] = r;
        z[i] = p.scale * r * dots;
    }
    // Softmax over the stack, every query lane at once.
    var m = z[0];
    for (var i = 1u; i < n; i = i + 1u) {
        m = max(m, z[i]);
    }
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < n; i = i + 1u) {
        z[i] = exp(z[i] - m);
        sum = sum + z[i];
    }
    for (var i = 0u; i < n; i = i + 1u) {
        z[i] = z[i] / sum;
        stats[sbase + i] = rms[i];
        for (var q = 0u; q < p.nq; q = q + 1u) {
            stats[sbase + n + q * n + i] = z[i][q];
        }
    }
    for (var j = 0u; j < p.dim; j = j + 1u) {
        var acc = vec4<f32>(0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            acc = acc + z[i] * (rms[i] * state(i, row, j));
        }
        for (var q = 0u; q < p.nq; q = q + 1u) {
            out[(q * p.rows + row) * p.dim + j] = acc[q];
        }
    }
}
