// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training activation backward: dx[i] += dy[i] * f'(x[i]) for SiLU, sigmoid or tanh-GELU
// @how   one thread per element; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// mode 0 SiLU, 1 sigmoid, 2 GELU (tanh approximation). x is the pre-activation, as in the forward kernels.
// Params: total, mode. Dispatch: total threads.

struct Params { total: u32, mode: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let v = x[idx];
    var d = 0.0;
    if (p.mode == 2u) {
        let k = 0.7978845608028654;
        let t = tanh(k * (v + 0.044715 * v * v * v));
        let dinner = k * (1.0 + 3.0 * 0.044715 * v * v);
        d = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * dinner;
    } else {
        let s = 1.0 / (1.0 + exp(-v));
        if (p.mode == 0u) {
            d = s * (1.0 + v * (1.0 - s));
        } else {
            d = s * (1.0 - s);
        }
    }
    dx[idx] = dx[idx] + dy[idx] * d;
}
