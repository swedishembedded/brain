// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched matmul: out[b,m,n] = alpha * sum_k A[b,·]·B[b,·], both operands vary per batch
// @how   one thread per output element, serial inner reduction over k
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Batched matmul — the primitive every Gated DeltaNet (GDN) chunk-recurrence
// step needs (`bmm_acc.wgsl` is its accumulating twin). Unlike `matmul.wgsl`
// (a single shared weight `W` applied to a `[M,K]` batch of rows), here BOTH
// `A` and `B` carry their own batch dimension — e.g. per-(batch,head,chunk)
// `[C,Dk]` query/key blocks:
//
//   out[b,m,n] = alpha * sum_k Â[b,m,k] * B̂[b,k,n]
//
// `Â`/`B̂` are the LOGICAL `[batch,m,k]`/`[batch,k,n]` views; `trans_a`/
// `trans_b` select whether the PHYSICAL layout already matches that (0) or is
// transposed (1 — physically `[batch,k,m]`/`[batch,n,k]`, i.e. this kernel
// reads `A[b,k,m]`/`B[b,n,k]` for the same logical element). `alpha` folds in
// a scale or sign (e.g. GDN's `-(k_beta @ key^T)` needs `alpha = -1`, and an
// attention-score scale such as `1/sqrt(Dk)` can ride here too) without a
// separate negate/scale kernel.
//
// `a_off`/`b_off`/`out_off` are flat ELEMENT offsets added after the normal
// `batch_idx * (per-batch element count)` address — this engine's convention
// for addressing a sub-range of a buffer from a kernel's own `Params`
// (`splice_add.wgsl` is the existing example) instead of binding a
// byte-offset slice, which requires 256-byte alignment; a `Params` offset
// has no such constraint, which matters
// because GDN's tiny test shapes are nowhere near 64-float-aligned. GDN's
// caller picks a flat batch layout with its sequential/recurrent axis (the
// chunk index) OUTERMOST, specifically so "chunk c, every (batch,head)" is
// one CONTIGUOUS batch range `[c * bh_count, (c+1) * bh_count)` addressable
// with a plain offset — no stride override needed. Every whole-tensor caller
// passes 0 for all three; a per-chunk caller passes a multiple of the
// relevant per-batch element count for whichever operand has a chunk axis
// (`crates/model/src/gdn.rs` is the only current caller and documents each
// call site's offsets).
//
// One invocation per `(b,m,n)` output element; the `k`-reduction is a plain
// serial fp32 loop (`@opt 2`, matching `matmul.wgsl`'s own naive tier) — the
// matrices GDN drives this with are small (`C`/`Dk`/`Dv` in the tens to low
// hundreds), so correctness-first beats a register-tiled variant here
// (get it correct, then freeze).

struct Params {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    trans_a: u32,
    trans_b: u32,
    alpha: f32,
    a_off: u32,
    b_off: u32,
    out_off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.batch * p.m * p.n;
    if (idx >= total) { return; }
    let bi = idx / (p.m * p.n);
    let rest = idx % (p.m * p.n);
    let mm = rest / p.n;
    let nn = rest % p.n;

    let a_base = p.a_off + bi * (p.m * p.k);
    let b_base = p.b_off + bi * (p.k * p.n);

    var acc = 0.0;
    for (var kk: u32 = 0u; kk < p.k; kk = kk + 1u) {
        var av: f32;
        if (p.trans_a == 0u) {
            av = a[a_base + mm * p.k + kk];
        } else {
            av = a[a_base + kk * p.m + mm];
        }
        var bv: f32;
        if (p.trans_b == 0u) {
            bv = b[b_base + kk * p.n + nn];
        } else {
            bv = b[b_base + nn * p.k + kk];
        }
        acc = acc + av * bv;
    }
    out[p.out_off + bi * (p.m * p.n) + mm * p.n + nn] = p.alpha * acc;
}
