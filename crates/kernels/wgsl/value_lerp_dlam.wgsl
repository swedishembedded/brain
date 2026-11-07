// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Learned scalar lerp backward, gate:  part[c] = s (1-s) sum_r dy[r,c] (b[r,c] - a[r,c])
// @how   one thread per column, serial inner reduction over rows
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-column partial sums of the gradient of the lerp gate `lam`, for
// `value_lerp`'s `out = a + s (b - a)`, `s = sigmoid(lam[0])`:
//   d out / d lam = (b - a) * s * (1 - s)
//   part[c] = s (1-s) * sum_r dy[r,c] * (b[r,c] - a[r,c])        (rows x cols)
// The caller folds the `cols` partials into the scalar's gradient with
// `bias_grad` (`m = cols`, `n = 1`), which accumulates. Splitting the
// reduction this way keeps it on core-compute kernels with no barrier:
// `cols` threads each walk `rows` elements.
//
// Swedish Embedded AB implements solutions for training decoders with learned
// residual mixing between layers for its clients. If your team needs
// expertise in value-residual and skip-connection architectures then you can
// procure our services by sending an email to info@swedishembedded.com.

struct Params {
    rows: u32,
    cols: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       a:    array<f32>;
@group(0) @binding(3) var<storage, read>       b:    array<f32>;
@group(0) @binding(4) var<storage, read>       lam:  array<f32>;
@group(0) @binding(5) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.cols) { return; }
    let s = 1.0 / (1.0 + exp(-lam[0]));
    var acc = 0.0;
    for (var r: u32 = 0u; r < p.rows; r = r + 1u) {
        let i = r * p.cols + c;
        acc = acc + dy[i] * (b[i] - a[i]);
    }
    part[c] = s * (1.0 - s) * acc;
}
