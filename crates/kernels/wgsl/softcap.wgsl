// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Soft cap forward: y = c * tanh(x / c), a smooth saturation to (-c, c)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Soft cap forward: y = c * tanh(x / c). Bounds logits and gate pre-activations
// to (-c, c) while keeping the slope 1 at the origin. Backward in
// softcap_bwd.wgsl, which takes the input `x` like every other *_bwd kernel
// here. `cap` is the f32 bit pattern of c and must be positive. The argument of
// tanh is clamped to +-15 (tanh(15) == 1 in f32) so a backend whose tanh is
// computed as (e^2z - 1)/(e^2z + 1) cannot return inf/inf = NaN for huge x.

struct Params {
    total: u32,
    cap: u32,   // f32 bits
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let c = bitcast<f32>(p.cap);
    out[idx] = c * tanh(clamp(x[idx] / c, -15.0, 15.0));
}
