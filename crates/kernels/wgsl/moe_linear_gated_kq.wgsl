// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear, AFFINE K-quant (Q4_K/Q5_K, DP4A): moe_linear_gated_i8.wgsl's row skip, over affine-coded packed weights
// @how   DP4A packed int8, one thread per output element, serial per-group inner reduction with staging-time code unpack + min correction, early exit
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Swedish Embedded AB implements quantized inference kernels for edge and
// embedded GPUs for its clients. If your team needs expertise in shipping
// affine K-quant (GGUF Q4_K/Q5_K-class) inference on commodity GPU hardware
// without an intermediate fp32 detour then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// The AFFINE counterpart of `moe_linear_gated_i8.wgsl`, in the SAME naive
// tier as that kernel (one thread per output element, no workgroup tiling)
// rather than `matmul_kq_dyn`'s 256-thread register-tiled shape, and for the
// identical reason: `matmul_kq_dyn`/`matmul_kq_gemv` stage rows into
// WORKGROUP-SHARED memory across a barrier, and WGSL requires every thread in
// a workgroup to reach a `workgroupBarrier()` uniformly - a per-thread early
// return for a non-routed row would make that undefined behaviour. This
// kernel accepts a tiled kernel's throughput ceiling in exchange for a
// row-level skip that is trivially safe (an ordinary `return`, no barrier in
// this kernel at all) - see `moe_linear_gated_i8.wgsl`'s own header for the
// fuller tradeoff, unchanged here.
//
// Same affine dequant contract as `matmul_kq_dyn`/`matmul_kq_gemv` (read
// those kernels' headers for the full derivation): the weight is an UNSIGNED
// `CODE_BITS`-wide code (4 for Q4_K, 8 for Q5_K) that reconstructs as
// `ds*code - dm` per weight-scale group, and the min correction needs the
// activation-only group-sum prepass `S[m,g] = xgs[m,g]` (`quant_group_sum.
// wgsl`) - NEVER a f32 activation, which would introduce a systematic bias
// proportional to `dm` rather than a rounding difference.
//
//   xq   : [m, k/4]              u32 - 4 int8 activations packed along K per u32
//   wq   : [n, k*CODE_BITS/32]   u32 - K-CONTIGUOUS unsigned codes, `32/CODE_BITS` codes per word, low bits first
//   sx   : [m]                   f32 - per-token activation scale
//   wsz  : [n, 2*k/32]           f32 - interleaved (ds, dm) pairs, one pair per 32-element weight-scale group
//   xgs  : [m, k/32]             f32 - activation group sums (quant_group_sum.wgsl)
//   gate : [m, n_experts]        f32 - dense per-token-per-expert weight (0 = not routed)
//   out  : [m, n]                f32 - out[row,:] = 0 for a non-routed row, else
//                                       sx[row] * Σ_g( ds[n,g]*A[row,n,g] - dm[n,g]*S[row,g] )
//
// `k` (the params field) is the RAW LOGICAL reduction length, NOT a packed-
// word count - the identical reason `matmul_kq_dyn.wgsl`'s own header states
// (`xq` and `wq` have different word densities for the same `k`). `k` must be
// a multiple of 32 (one weight-scale group).
//
// ## No min-correction guard needed here
//
// Unlike `matmul_kq_gemv.wgsl`'s 64-thread k-stride (where 8 threads visit
// one group's 8 quads and only the FIRST must apply the correction, guarded
// by `(g % WPGK) == 0u`), this kernel's inner loop walks K IN ORDER, one
// thread per output element: a whole group's 8 quads are consecutive in the
// SAME thread's own loop, so the group's `ds`/`dm` are read once, the integer
// sum over the group's 8 quads accumulates in `i32` (exact and associative),
// and the correction is applied exactly once per (row, group) as the group
// completes - no cross-thread double-application is possible because no
// other thread ever touches this (row, group) pair.

struct Params {
    m: u32,
    k: u32,   // RAW LOGICAL K
    n: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:   array<u32>;  // [m, k/4]
@group(0) @binding(2) var<storage, read>       wq:   array<u32>;  // [n, k*CODE_BITS/32]
@group(0) @binding(3) var<storage, read>       sx:   array<f32>;  // [m]
@group(0) @binding(4) var<storage, read>       wsz:  array<f32>;  // [n, 2*k/32] interleaved (ds, dm)
@group(0) @binding(5) var<storage, read>       xgs:  array<f32>;  // [m, k/32]
@group(0) @binding(6) var<storage, read>       gate: array<f32>;
@group(0) @binding(7) var<storage, read_write>  out: array<f32>;

const CODE_BITS: u32 = 8u;  // template knob: 4 (Q4_K) or 8 (Q5_K)

// Quads (4-element units, matching xq's own word density) per weight-scale
// group (32 elements / 4).
const WPGK: u32 = 8u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.m * p.n;
    if (idx >= total) { return; }
    let row = idx / p.n;
    let col = idx % p.n;
    if (gate[row * p.n_experts + p.e_idx] <= 0.0) {
        out[idx] = 0.0;
        return;
    }

    let kgx = p.k / 4u;                       // quads per row (xq's own word density)
    let ng = p.k / 32u;                       // weight-scale groups per row
    let wq_row_words = p.k * CODE_BITS / 32u; // wq words per row
    let qpwq = (32u / CODE_BITS) / 4u;        // quads packed per raw wq word
    let cmask = (1u << CODE_BITS) - 1u;       // low-CODE_BITS mask

    let x_base = row * kgx;
    let w_base = col * wq_row_words;
    let sz_base = col * 2u * ng;
    let g_base = row * ng;

    var accf = 0.0;
    for (var gr: u32 = 0u; gr < ng; gr = gr + 1u) {
        var acc = 0i;
        for (var j: u32 = 0u; j < WPGK; j = j + 1u) {
            let g = gr * WPGK + j;
            let word_idx = g / qpwq;
            let qoff = g % qpwq;
            let src = wq[w_base + word_idx];
            let base_bit = qoff * 4u * CODE_BITS;
            let u0 = (src >> (base_bit + 0u * CODE_BITS)) & cmask;
            let u1 = (src >> (base_bit + 1u * CODE_BITS)) & cmask;
            let u2 = (src >> (base_bit + 2u * CODE_BITS)) & cmask;
            let u3 = (src >> (base_bit + 3u * CODE_BITS)) & cmask;
            let wv = u0 | (u1 << 8u) | (u2 << 16u) | (u3 << 24u);
            acc = acc + dot4I8Packed(xq[x_base + g], wv);
        }
        let ds = wsz[sz_base + 2u * gr];
        let dm = wsz[sz_base + 2u * gr + 1u];
        accf = accf + f32(acc) * ds - dm * xgs[g_base + gr];
    }
    out[idx] = accf * sx[row];
}
