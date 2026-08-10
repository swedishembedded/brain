// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet UT-transform backward, half 1: d_attn0 from a finalised row of d_t_mat
// @how   one thread per output element, serial reduction over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Reverse-mode counterpart of `gdn_ut_step.wgsl`'s forward substitution row:
//   t_mat[row,i,j] = attn0[row,i,j] + sum_{k=j+1}^{i-1} attn0[row,i,k]*t_mat[row,k,j]   for j < i
// Differentiating w.r.t. `attn0[row,i,p]` for a fixed row `i` (every term of
// row `i`'s own recurrence that reads `attn0[row,i,p]`, `p` ranging over both
// roles it plays -- the direct `j == p` term of column `p`'s own equation, and
// the `k == p` term inside every OTHER column `j < p`'s sum) gives, together
// with the direct pass-through gradient already resting on `d_t_mat[row,i,p]`:
//   d_attn0[row,i,p] = d_t_mat[row,i,p] + sum_{j=0}^{p-1} d_t_mat[row,i,j]*t_mat[row,p,j]
//
// Requires `d_t_mat[row,i,:]` to be FULLY accumulated already -- true by
// induction when the host processes `i` from `c_len-1` DOWN TO `1` (the
// reverse of the forward's increasing sweep): every row `i' > i` that could
// still add to `d_t_mat[row,i,:]` (via `gdn_ut_bwd_dtmat.wgsl`'s scatter) has
// already run in a strictly earlier host dispatch. `t_mat` (frozen forward
// values) and `d_t_mat` (the evolving gradient, still being written into by
// LOWER rows this same reverse sweep) are kept as separate bindings for the
// same reason `gdn_ut_step.wgsl` keeps `attn0`/`t_mat` separate: this
// dispatch only ever READS `d_t_mat[row,i,:]` and `t_mat[row,p,:]` for
// `p < i`, both already finalised by earlier dispatches, and only ever WRITES
// `d_attn0[row,i,:]` -- a disjoint destination from anything it reads, so no
// race is possible within or across dispatches.
//
// The diagonal (`t_mat[row,i,i] = 1`, a hardcoded constant added by
// `gdn_add_identity.wgsl` AFTER the whole forward loop) carries NO gradient --
// this kernel's `p` only ranges `[0,i)`, matching `gdn_ut_step.wgsl`'s own `j`
// range, so the diagonal is never touched by either the forward or backward
// per-row sweep.
//
// Dispatch: `threads = bhc * i`, `i` from `c_len-1` down to `1` (row `i=0`
// needs no update: `attn0`'s row 0 is all zero, contributing nothing).

struct Params { bhc: u32, c_len: u32, i: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       t_mat:   array<f32>;
@group(0) @binding(2) var<storage, read>       d_t_mat: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_attn0: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bhc * p.i) { return; }
    let row = idx / p.i;
    let pp = idx % p.i;
    let cc = p.c_len * p.c_len;
    let base = row * cc;
    let row_i = base + p.i * p.c_len;
    var acc = d_t_mat[row_i + pp];
    var j: u32 = 0u;
    loop {
        if (j >= pp) { break; }
        acc = acc + d_t_mat[row_i + j] * t_mat[base + pp * p.c_len + j];
        j = j + 1u;
    }
    d_attn0[row_i + pp] = acc;
}
