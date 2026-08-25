// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  col2im as a GATHER - the input gradient of a conv, given the gradient of its im2col matrix
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// col2im as a GATHER — the input gradient of a conv, given the gradient of its
// im2col matrix.
//
//   dcol : [Ho*Wo, Cin*K*K]  row-major (the adjoint of `im2col_at`'s output)
//   dx   : [N, Cin, H, W]    read_write, ONE invocation per INPUT element
//
// Why this exists. `conv2d_dx` computes the same thing directly, but each of its
// invocations reduces over `Cout * K * K` — the whole output-channel axis — so a
// 512-channel conv makes every input pixel walk 4608 terms on one lane. Lowering
// the conv backward to a GEMM moves that Cout reduction into `matmul_dx_reg`
// (register-tiled, the card's fastest fp32 path) and leaves this kernel summing only the
// `K*K` taps that touched the pixel — 9 terms for a 3x3, independent of Cout.
//
// GATHER, not scatter. The forward reads input (hi,wi) into output (ho,wo) when
//   ho*stride - pad + kh == hi   =>   ho = (hi + pad - kh)/stride
// is a non-negative multiple of `stride` landing inside [0,Ho). Iterating those
// (kh,kw) per INPUT pixel means every dx[idx] is written by exactly one
// invocation, so there are no atomics — which is `vae::blocks::grad`'s rule 1
// and a hard invariant of this engine, not a preference.
//
// The tap's column in `dcol` is `(ci*K + kh)*K + kw`, matching `im2col_at`'s
// packing exactly; reading it any other way is a silently transposed kernel.

struct Params {
    N: u32,
    Cin: u32,
    H: u32,
    W: u32,
    K: u32,
    stride: u32,
    pad: u32,
    Ho: u32,
    Wo: u32,
    cinkk: u32,     // Cin*K*K — dcol's row stride
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dcol: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cin * p.H * p.W;
    if (idx >= total) { return; }

    // Decode the input coordinate (n, ci, hi, wi).
    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let ci = t2 % p.Cin;
    let n  = t2 / p.Cin;

    var acc = 0.0;
    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        let hi_pad = hi + p.pad;
        if (hi_pad >= kh) {
            let num_h = hi_pad - kh;
            if ((num_h % p.stride) == 0u) {
                let ho = num_h / p.stride;
                if (ho < p.Ho) {
                    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                        let wi_pad = wi + p.pad;
                        if (wi_pad >= kw) {
                            let num_w = wi_pad - kw;
                            if ((num_w % p.stride) == 0u) {
                                let wo = num_w / p.stride;
                                if (wo < p.Wo) {
                                    // dcol row = this output position (the batch
                                    // index is folded in by the caller, which
                                    // binds one image's slice at a time).
                                    let pos = ho * p.Wo + wo;
                                    let tap = (ci * p.K + kh) * p.K + kw;
                                    acc = acc + dcol[pos * p.cinkk + tap];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dx[idx] = acc;
}
