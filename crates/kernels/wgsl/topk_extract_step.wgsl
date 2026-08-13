// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  One step of iterative top-K extraction
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One step of iterative top-K extraction: given this iteration's per-row
// argmax index (from the existing argmax_part/argmax_final pair, run just
// before this kernel), record that (index, value) pair into column `col` of
// the row's top-K output, then mask the winning position out of `logits` so
// the NEXT argmax_part/argmax_final pass finds the row's next-best value.
//
// Composing this with argmax_part/argmax_final K times extracts a row's true
// top-K (value, index) pairs on-device, so only `[bsz, K]` needs reading back
// -- not the whole `[bsz, vocab]` row -- while reusing the existing,
// already-gated reduction kernels unchanged. `topk_vals`/`topk_idx` are
// therefore sorted descending by construction (each iteration's winner is the
// best of what remains). One invocation per row; no barrier, no reduction.
//
//   idx_this_iter : [bsz]       f32 winner index this iteration (argmax_final's output)
//   logits        : [bsz, vocab]  mutated in place (winner masked to -inf)
//   topk_vals     : [bsz, k_max]  written at column `col`
//   topk_idx      : [bsz, k_max]  written at column `col` (index, as f32)

struct Params {
    bsz: u32,
    vocab: u32,
    k_max: u32,
    col: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       idx_this_iter: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits:        array<f32>;
@group(0) @binding(3) var<storage, read_write> topk_vals:     array<f32>;
@group(0) @binding(4) var<storage, read_write> topk_idx:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.bsz) { return; }
    let i = u32(idx_this_iter[row]);
    let logit_base = row * p.vocab;
    let v = logits[logit_base + i];
    let out_base = row * p.k_max + p.col;
    topk_vals[out_base] = v;
    topk_idx[out_base] = f32(i);
    logits[logit_base + i] = -3.4e38;
}
