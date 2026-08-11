// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of Gated DeltaNet's causal decay mask, row-sum or column-sum by mode
// @how   one thread per output element, serial reduction over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of `gdn_decay_mask.wgsl`:
//   decay_mask[row,i,j] = exp(g_cs[row,i] - g_cs[row,j])   if j <= i else 0
// Both `g_cs[row,i]` and `g_cs[row,j]` feed every masked cell in row `i`
// AND column `j` respectively, so `g_cs`'s gradient needs a full row-sum (the
// `+i` role) and a full column-sum (the `-j` role) over the SAME
// `[bhc,c_len,c_len]` tensor:
//   d_g_cs[row,i] += sum_{j<=i} d_decay_mask[row,i,j] * decay_mask[row,i,j]
//   d_g_cs[row,j] -= sum_{i>=j} d_decay_mask[row,i,j] * decay_mask[row,i,j]
//
// One kernel, dispatched TWICE with `mode` selecting which reduction
// direction this call performs (this engine's no-atomics rule: a single
// thread cannot cheaply accumulate both a row's AND a column's worth
// of a shared tensor without either racing or redundant work, so two
// dispatches over two different thread-to-cell mappings is the correct
// decomposition here, not two separate files -- the body is identical enough
// that a `mode` branch is clearer than two near-duplicate kernels).
// `mode == 0`: thread `idx` owns row `i = idx % c_len` of batch `row = idx /
// c_len`, sums `j` in `[0,i]`, ADDS into `d_g_cs[row,i]`.
// `mode == 1`: thread `idx` owns column `j = idx % c_len`, sums `i` in
// `[j,c_len)`, SUBTRACTS from `d_g_cs[row,j]`.
// `d_g_cs` is a genuine multi-source accumulator (also written by
// `gdn_state_decay_bwd_dscale.wgsl`, `gdn_decay_scale_bwd.wgsl`/
// `_last.wgsl`, and `mul.wgsl`'s `exp_g_cs` backward via `splice_add.wgsl`) --
// the caller must zero it before the first of ALL these contributions, not
// just before this kernel's own two dispatches.
//
// Dispatch: `threads = bhc * c_len` for either mode.

struct Params { bhc: u32, c_len: u32, mode: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_decay_mask: array<f32>;
@group(0) @binding(2) var<storage, read>       decay_mask:   array<f32>;
@group(0) @binding(3) var<storage, read_write> d_g_cs:       array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bhc * p.c_len) { return; }
    let row = idx / p.c_len;
    let pos = idx % p.c_len;
    let cc = p.c_len * p.c_len;
    let base = row * cc;
    var acc = 0.0;
    if (p.mode == 0u) {
        var j: u32 = 0u;
        loop {
            if (j > pos) { break; }
            let cell = base + pos * p.c_len + j;
            acc = acc + d_decay_mask[cell] * decay_mask[cell];
            j = j + 1u;
        }
        d_g_cs[idx] = d_g_cs[idx] + acc;
    } else {
        var i: u32 = pos;
        loop {
            if (i >= p.c_len) { break; }
            let cell = base + i * p.c_len + pos;
            acc = acc + d_decay_mask[cell] * decay_mask[cell];
            i = i + 1u;
        }
        d_g_cs[idx] = d_g_cs[idx] - acc;
    }
}
