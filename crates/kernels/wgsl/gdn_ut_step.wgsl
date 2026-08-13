// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  One sequential row of Gated DeltaNet's intra-chunk UT-transform forward substitution
// @how   one thread per output element, serial reduction over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One row `i` of `torch_chunk_gated_delta_rule`'s forward substitution:
//   attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
//   where row = attn[..., i, :i]   (captured BEFORE this row's own update)
//         sub = attn[..., :i, :i]  (already fully updated by earlier rows)
// `row[k] = attn0[i,k]` (row `i` is untouched before this call, so its
// "current" value IS `attn0`) and `sub[k,j] = t_mat[k,j]` for `k,j < i`
// (already finalised). Expanding the broadcast-multiply-then-sum-over-k and
// dropping the terms that provably vanish (`t_mat[k,j]` is 0 whenever
// `k <= j`: `k == j` is an untouched diagonal entry, still 0 until
// `gdn_add_identity.wgsl` runs after this whole loop; `k < j` is a
// still-strictly-upper entry of row `k`, also always 0) leaves, for output
// column `j` in `0..i`:
//   t_mat[row,i,j] = attn0[row,i,j] + sum_{k=j+1}^{i-1} attn0[row,i,k] * t_mat[row,k,j]
//
// TWO buffers, not one: the PyTorch reference mutates a SINGLE tensor in
// place, which is safe there only because "compute the whole RHS as a fresh
// tensor, then assign" is sequential inside one Python statement — `row`'s
// data is read before the assignment writes anything. WGSL invocations carry
// no such ordering: within ONE dispatch (fixed `i`), different `j` threads
// write different `t_mat[row,i,j]` cells while ALSO reading `attn0[row,i,k]`
// for `k` up to `i-1` — if `attn0` and the evolving matrix were the same
// buffer, thread `j` could race against thread `k=j'>j`'s write to the exact
// cell it is reading. Keeping `attn0` (written once by
// `gdn_mask_strict_lower.wgsl`, read-only forever after) and `t_mat` (the
// evolving output, caller-cleared to 0 before row 1) physically separate
// removes the race entirely: every `t_mat[row,k,j]` this dispatch reads has
// `k < i`, finalised by a STRICTLY EARLIER host dispatch (this kernel runs
// once per `i` in increasing order — the same host-orchestrated
// multi-dispatch pattern as `gdn_chunk_cumsum_step.wgsl`), and no thread in
// this dispatch ever reads another thread's own output cell.
//
// The host calls this once per `i` in `1..c_len` (row 0 needs no update — it
// stays 0, matching `attn0`'s all-zero row 0, since `attn0` is strictly
// lower-triangular). `gdn_add_identity.wgsl` adds the `+1` diagonal AFTER
// this loop finishes every row, never during it — this kernel's `j` only
// ranges `0..i-1`, so it never reaches the diagonal, and the diagonal must
// stay exactly 0 through the whole loop for the "always-0" argument above to
// hold at every `i`.
//
// Dispatch: `threads = bhc * i` (only `j < i` exist for this row). One
// thread per `(row, j)`; the `k` reduction is a plain serial loop, at most
// `c_len` iterations (tens for this model family).

struct Params { bhc: u32, c_len: u32, i: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       attn0: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_mat: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bhc * p.i) { return; }
    let row = idx / p.i;
    let j = idx % p.i;
    let cc = p.c_len * p.c_len;
    let base = row * cc;
    var acc = attn0[base + p.i * p.c_len + j];
    var k = j + 1u;
    loop {
        if (k >= p.i) { break; }
        acc = acc + attn0[base + p.i * p.c_len + k] * t_mat[base + k * p.c_len + j];
        k = k + 1u;
    }
    t_mat[base + p.i * p.c_len + j] = acc;
}
