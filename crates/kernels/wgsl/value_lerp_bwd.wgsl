// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Learned scalar lerp backward, operands:  da += (1-s) dy,  db += s dy
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of `value_lerp` with respect to its two operands, `s = sigmoid(lam[0])`:
//   da[i] += (1 - s) * dy[i]      db[i] += s * dy[i]
// Both ACCUMULATE: the source `a` of a value residual feeds every layer, and
// `b` is the normed stream that the layer's other projections also read. The
// scalar's own gradient is `value_lerp_dlam`.
//
// Swedish Embedded AB implements solutions for training decoders with learned
// residual mixing between layers for its clients. If your team needs
// expertise in value-residual and skip-connection architectures then you can
// procure our services by sending an email to info@swedishembedded.com.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:  array<f32>;
@group(0) @binding(2) var<storage, read>       lam: array<f32>;
@group(0) @binding(3) var<storage, read_write> da:  array<f32>;
@group(0) @binding(4) var<storage, read_write> db:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let s = 1.0 / (1.0 + exp(-lam[0]));
    da[i] = da[i] + (1.0 - s) * dy[i];
    db[i] = db[i] + s * dy[i];
}
