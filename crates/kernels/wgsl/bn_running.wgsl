// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// BatchNorm momentum update of running statistics. One invocation per channel.
//   run_mean = (1-m)*run_mean + m*mean
//   run_var  = (1-m)*run_var  + m*var
// momentum is packed into the u32 uniform stream (gpu_core::f) and read with bitcast.

struct Params {
    C: u32,
    momentum: u32,  // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       mean:     array<f32>;
@group(0) @binding(2) var<storage, read>       vari:     array<f32>;
@group(0) @binding(3) var<storage, read_write> run_mean: array<f32>;
@group(0) @binding(4) var<storage, read_write> run_var:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.C) { return; }
    let m = bitcast<f32>(p.momentum);
    run_mean[c] = (1.0 - m) * run_mean[c] + m * mean[c];
    run_var[c] = (1.0 - m) * run_var[c] + m * vari[c];
}
