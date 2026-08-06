// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  EDM output wrap for the DIAMOND sampler (denoiser.py::wrap_model_output)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// EDM output wrap for the DIAMOND sampler (denoiser.py::wrap_model_output):
//   d[i] = quantize(clamp(coef[0]*x[i] + coef[1]*f[i], -1, 1))
// where coef = [c_skip, c_out] (a per-sigma device buffer, so one recorded
// graph serves every denoise step) and quantize reproduces torch's
// `.clamp(-1,1).add(1).div(2).mul(255).byte()` byte-TRUNCATION exactly:
//   b = floor((c + 1) * 127.5);  d = b/255*2 - 1
// One invocation per element. Inference-only (no backward).

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       f:    array<f32>;
@group(0) @binding(3) var<storage, read>       coef: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    // max(min(..)) instead of clamp, and f32(u32(..)) instead of floor (the
    // value is >= 0, where u32 truncation == floor): the wgsl-cpu JIT's math
    // subset has neither Clamp nor Floor (docs/world-models/PLAYBOOKS.md §2).
    let c = max(min(coef[0u] * x[i] + coef[1u] * f[i], 1.0), -1.0);
    let b = f32(u32((c + 1.0) * 127.5));
    y[i] = b / 255.0 * 2.0 - 1.0;
}
