// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Asymmetric REFLECT pad on a collapsed [img, h, w] volume (gather) - spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Asymmetric REFLECT pad (PyTorch `F.pad(..., mode="reflect")`, no edge
// repetition) over the LAST TWO axes of a volume collapsed to `[img, h, w]`
// (`img` folds every axis the pad does not touch - N*C for a plain NCHW
// image, N*C*T for a video's spatial pad, since this kernel does not care
// what "img" means dimensionally, exactly `pad2d.wgsl`'s own "batch and
// channel combined" idiom). Params carry the UNPADDED dims `h`, `w` and pad
// amounts `l`/`r`/`t`/`b` (left/right/top/bottom, u32, each may be 0);
// padded dims are derived: `hp = h+t+b`, `wp = w+l+r`. One thread per OUTPUT
// element `idx < total` (`total = img*hp*wp`):
//   p = idx/(hp*wp); r0 = idx % (hp*wp); ho = r0/wp; wo = r0 % wp
//   ih = mirror(ho - t, h); iw = mirror(wo - l, w)
//   y[idx] = x[p*h*w + ih*w + iw]
// `mirror(i, n)` folds `i` (possibly negative or >= n) into `[0, n)` with NO
// repeated edge sample (`n`'s two boundary elements are visited once, not
// twice) via the `2*(n-1)`-periodic fold `nn.functional.pad(mode="reflect")`
// uses, inlined per-axis in `main` (the CPU JIT's WGSL subset has no user-
// defined function calls). Requires `n >= 2` (every spatial axis this kernel
// is used on is >= 16 in practice, per `crates/minimaxh3/src/video_vae.rs`'s
// causal-conv spatial padding, always `l/r/t/b <= 1`, well inside the valid
// range for any `n >= 2`).

struct Params {
    total: u32,
    h: u32,
    w: u32,
    l: u32,
    r: u32,
    t: u32,
    b: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let hp = p.h + p.t + p.b;
    let wp = p.w + p.l + p.r;
    let img = idx / (hp * wp);
    let r0 = idx % (hp * wp);
    let ho = r0 / wp;
    let wo = r0 % wp;

    // Mirror into [0, n) with NO repeated edge sample (`nn.functional.pad`'s
    // `mode="reflect"`), inlined twice (the CPU JIT's WGSL subset has no
    // user-defined function calls) - `period = 2*(n-1)`, `n >= 2`.
    let hperiod = 2 * (i32(p.h) - 1);
    var hm = (i32(ho) - i32(p.t)) % hperiod;
    if (hm < 0) { hm = hm + hperiod; }
    if (hm >= i32(p.h)) { hm = hperiod - hm; }
    let ih = u32(hm);

    let wperiod = 2 * (i32(p.w) - 1);
    var wm = (i32(wo) - i32(p.l)) % wperiod;
    if (wm < 0) { wm = wm + wperiod; }
    if (wm >= i32(p.w)) { wm = wperiod - wm; }
    let iw = u32(wm);

    y[idx] = x[img * p.h * p.w + ih * p.w + iw];
}
