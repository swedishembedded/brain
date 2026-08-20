// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Snake activation backward - per-channel alpha gradient
// @how   one thread per channel, serial reduction over rows*inner
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-channel alpha gradient of `snake1d.wgsl`'s forward
// (`y = x + (a+eps)^-1 * sin(a*x)^2`, `a = alpha[c]`, `u = a+eps`,
// `s = sin(a*x)`, `co = cos(a*x)`):
//   d(term)/da = (2*x*s*co)/u - s^2/u^2
//   dalpha[c] = sum over rows,inner of dy * d(term)/da
// Same one-thread-per-output-element-with-serial-inner-reduction shape as
// `conv1d_dw.wgsl`'s weight gradient. WRITES (not accumulates) `dalpha` -
// this is the only kernel that touches it per backward pass, unlike a
// conv's `dw` which is shared across dispatches.

struct Params {
    rows: u32,
    c: u32,
    inner: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:     array<f32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read>       alpha:  array<f32>;
@group(0) @binding(4) var<storage, read_write> dalpha: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.c) { return; }
    let a = alpha[c];
    let u = a + p.eps;
    var acc = 0.0;
    for (var row: u32 = 0u; row < p.rows; row = row + 1u) {
        let base = row * (p.c * p.inner) + c * p.inner;
        for (var l: u32 = 0u; l < p.inner; l = l + 1u) {
            let xi = x[base + l];
            let s = sin(a * xi);
            let co = cos(a * xi);
            let dterm_da = (2.0 * xi * s * co) / u - (s * s) / (u * u);
            acc = acc + dy[base + l] * dterm_da;
        }
    }
    dalpha[c] = acc;
}
