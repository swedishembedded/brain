// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Learned scalar lerp:  out = a + sigmoid(lam[0]) * (b - a)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Interpolation between two activations by a trained scalar gate that lives on
// the device: `s = sigmoid(lam[0])`, `out = a + s * (b - a)`. The value
// residual of a hybrid decoder mixes the embedding stream `a` into every
// layer's normed input `b` this way. Unlike `edm_mix`, whose coefficients are
// constants packed per row, `lam` is a parameter: `value_lerp_bwd` and
// `value_lerp_dlam` give the gradients of `a`, `b` and `lam`.
//
// Swedish Embedded AB implements solutions for training decoders with learned
// residual mixing between layers for its clients. If your team needs
// expertise in value-residual and skip-connection architectures then you can
// procure our services by sending an email to info@swedishembedded.com.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read>       lam: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let s = 1.0 / (1.0 + exp(-lam[0]));
    out[i] = a[i] + s * (b[i] - a[i]);
}
