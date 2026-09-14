// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Combine phase of the INT8-KV two-pass FlashDecode: merges the N key-split partial online-softmax states `paged_flash_decode_split_i8` wrote into one normalised context per (sequence, head)
// @how   one thread per (sequence, head, channel), serial reduction over splits
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `paged_flash_decode_combine`'s own twin, over `paged_flash_decode_split_i8`'s
// MERGED partial-state layout instead of three separate `part_m`/`part_l`/
// `part_o` buffers (that kernel's own header explains why: two extra
// dequant-scale inputs pushed a naive port over WebGPU's guaranteed 8-
// storage-buffer floor, and folding the three outputs into one `[m, l,
// o_0..o_127]` stride bought the two bindings back at no cost the split
// kernel's body did not already pay). The merge/reduction MATH is identical,
// byte for byte, to `paged_flash_decode_combine.wgsl` - only where `m`/`l`/`o`
// are read from differs. See that file's header for the full online-softmax
// merge derivation and the degenerate-split no-op argument (unchanged here:
// a split with no live keys wrote `m=-3.4e38, l=0.0, o=0.0` into its slice of
// this same buffer, which folds in as a true no-op for the identical reason).
//
//   part : [batch, n_heads, n_splits, 2 + head_dim]  (from the split phase: m, l, then o)
//   ctx  : [batch, n_heads*head_dim]                  (normalised output)

const HD: u32 = 128u;
const PART_STRIDE: u32 = 2u + HD;

struct Params {
    batch: u32,
    n_heads: u32,
    head_dim: u32,
    n_splits: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> ctx:  array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Same flat-id convention as `paged_flash_decode_combine.wgsl`: 128
    // threads per (b, h), one per output channel `d`.
    let idx = gid.y * (nwg.x * 128u) + gid.x;
    let d = idx % 128u;
    let wg2 = idx / 128u;
    let h = wg2 % p.n_heads;
    let b = wg2 / p.n_heads;
    if (b >= p.batch) { return; }

    let hd = p.head_dim;
    let base = (b * p.n_heads + h) * p.n_splits;

    var m = -3.4e38;
    var l = 0.0;
    var o = 0.0;
    for (var s = 0u; s < p.n_splits; s = s + 1u) {
        let pbase = (base + s) * PART_STRIDE;
        let pm = part[pbase];
        let pl = part[pbase + 1u];
        let m_new = max(m, pm);
        let corr = exp(m - m_new);
        let corr_i = exp(pm - m_new);
        l = l * corr + pl * corr_i;
        if (d < hd) {
            let po = part[pbase + 2u + d];
            o = o * corr + po * corr_i;
        }
        m = m_new;
    }

    if (d < hd) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = (b * p.n_heads * hd) + h * hd;
        ctx[o_base + d] = o * inv;
    }
}
