// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  col2im for a lowered ConvTranspose1d - gathers a `[Cout*K, L]` GEMM output into `[Cout, Lo]` NCL and adds the per-channel bias
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// col2im for a lowered ConvTranspose1d - gathers a `[Cout*K, L]` GEMM output
// into `[Cout, Lo]` NCL and adds the per-channel bias.
//
//   col  : [Cout*K, L]   col[co*K + kw, li] = sum_ci x[ci, li]·w[ci, co, kw]
//   bias : [Cout]
//   y    : [Cout, Lo]    y[co, lo] = bias[co] + sum over the taps landing here
//
// The GEMM that produces `col` is `matmul_dw_reg_splitk` with `s = 1` -
// `out[n,k] = sum_m a[m,n]·b[m,k]`, i.e. the TN form, whose contraction index
// is the leading axis of BOTH operands. That is exactly what a transposed conv
// wants: `a` is the native `[Cin, Cout/G, K]` weight read as `[Cin, Cout*K]`
// and `b` is the native `[Cin, L]` NCL input, so the lowering needs no
// transpose and no weight permute. `_splitk` rather than `matmul_dw_reg`
// because it ASSIGNS (`out[...] = c`) where the latter accumulates - with a
// reused scratch buffer an accumulating GEMM would fold the previous stage's
// values in, and zeroing it instead would cost a full extra pass over the
// largest buffer in the pipeline.
//
// The tap mapping is `convtr1d.wgsl`'s own, from the other side: that kernel
// walks `li = (lo + pad - kw*dilation)/stride` for each output element, and so
// does this one - the only thing the GEMM replaced is the inner sum over Cin.
// So the lowering computes the same taps in the same places; only the Cin
// reduction is reassociated (register accumulators instead of a serial loop).
//
// NOTE the FLOP count does not grow. A stage with `K = 2·stride` does `L` GEMM
// rows rather than `Lo = L·stride`, and `L·Cin·Cout·K == Lo·Cin·Cout·(K/stride)`
// - the naive kernel's `(lo + pad - kw·d) % stride != 0` branch discards
// exactly the taps this form never computes.
//
// Layout choice: `col` is `[Cout*K, L]` (tap-major), not `[L, Cout*K]`. Both
// are reachable from native operands by swapping the GEMM's two inputs, but
// tap-major is the one that reads well here - consecutive `lo` in a warp read
// consecutive `li` for a given tap, so an active tap's lanes touch a contiguous
// run instead of striding by `Cout*K`.
//
// Groups: `G = 1` only (`audio::conv`'s selector keeps grouped transposed
// convs on the direct kernel), so `co_local == co`.
//
// One invocation per OUTPUT element; writes are contiguous in `lo`.

struct Params {
    l: u32,          // input length (col column count)
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    lo: u32,         // output length
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       col:  array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.cout * p.lo;
    if (idx >= total) { return; }

    let lo = idx % p.lo;
    let co = idx / p.lo;

    var acc = bias[co];
    for (var kw: u32 = 0u; kw < p.k; kw = kw + 1u) {
        let num = lo + p.pad;
        let sub = kw * p.dilation;
        if (num >= sub) {
            let num2 = num - sub;
            if ((num2 % p.stride) == 0u) {
                let li = num2 / p.stride;
                if (li < p.l) {
                    acc = acc + col[(co * p.k + kw) * p.l + li];
                }
            }
        }
    }
    y[idx] = acc;
}
