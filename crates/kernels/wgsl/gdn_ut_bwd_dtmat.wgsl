// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet UT-transform backward, half 2: scatter row i's d_t_mat into earlier rows
// @how   one thread per output element, serial scatter over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// The other half of `gdn_ut_step.wgsl`'s reverse-mode derivative (see
// `gdn_ut_bwd_dattn0.wgsl`'s header for the forward recurrence and the full
// derivation context). Row `i`'s forward equation
//   t_mat[row,i,j] = attn0[row,i,j] + sum_{k=j+1}^{i-1} attn0[row,i,k]*t_mat[row,k,j]
// reads `t_mat[row,k,j]` for every `k` in `(j,i)`, so `d_t_mat[row,i,j]`
// (already fully accumulated, by the same induction `gdn_ut_bwd_dattn0.wgsl`
// relies on) must scatter a further contribution back into EVERY such
// `t_mat[row,k,j]`'s own gradient:
//   d_t_mat[row,k,j] += d_t_mat[row,i,j] * attn0[row,i,k]     for k in (j,i)
//
// One thread per `(row,j)`, `j` in `[0,i)` -- exactly `gdn_ut_step.wgsl`'s own
// dispatch shape, TRANSPOSED: that kernel has one thread per column `j`
// reducing a serial sum over `k` into ONE cell `t_mat[row,i,j]`; this kernel
// has one thread per column `j` SCATTERING a serial loop over `k` into MANY
// cells `d_t_mat[row,k,j]`. No race: within one dispatch (fixed `i`), column
// `j` is unique per thread, so two threads never write the same `(k,j)` cell
// regardless of `k`; across dispatches, each host call has its own `i`,
// strictly decreasing, run in sequence by `docs/kernel-checklist.md`'s
// host-orchestrated multi-dispatch convention (matching
// `gdn_chunk_cumsum_step.wgsl`/`gdn_ut_step.wgsl`) -- no two dispatches
// execute concurrently. `attn0` (frozen forward values) is read-only;
// `d_t_mat` is read-modify-write, but every read (`d_t_mat[row,i,j]`, fixed
// row `i` for this whole dispatch) is disjoint from every write (`row < i`),
// so a thread never races its own read against another thread's write.
//
// Dispatch: `threads = bhc * i`, run for the SAME `i` as, and either before or
// after (order between the two doesn't matter — `gdn_ut_bwd_dattn0.wgsl` only
// reads row `i`, this kernel only writes rows `< i`), `gdn_ut_bwd_dattn0.wgsl`
// in the reverse sweep `i` from `c_len-1` down to `1`.

struct Params { bhc: u32, c_len: u32, i: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       attn0:   array<f32>;
@group(0) @binding(2) var<storage, read_write> d_t_mat: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bhc * p.i) { return; }
    let row = idx / p.i;
    let j = idx % p.i;
    let cc = p.c_len * p.c_len;
    let base = row * cc;
    let dtij = d_t_mat[base + p.i * p.c_len + j];
    var k: u32 = j + 1u;
    loop {
        if (k >= p.i) { break; }
        let cell = base + k * p.c_len + j;
        d_t_mat[cell] = d_t_mat[cell] + dtij * attn0[base + p.i * p.c_len + k];
        k = k + 1u;
    }
}
