// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  BatchNorm statistic packing: interleave mean/var/gamma/beta into the mv, gb and mvg layouts
// @how   one thread per channel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// BatchNorm statistic packing, on the device. One invocation per channel c:
//   mv[2c]  = mean[c],  mv[2c+1]  = var[c]                  (bn_train, bn_dgamma)
//   gb[2c]  = gamma[c], gb[2c+1]  = beta[c]                 (bn_train)
//   mvg[3c] = mean[c],  mvg[3c+1] = var[c], mvg[3c+2] = gamma[c]   (bn_dstats)
//
// `bn_stats` emits mean and var as separate tensors and the parameters live as
// separate tensors, while the kernels that consume them read interleaved
// arrays (to stay within their binding budget). Interleaving on the HOST meant
// reading every statistic back between `bn_stats` and `bn_train` - a device
// drain per BatchNorm per forward - and splitting the forward into two
// submissions around it. Packed here, the whole train-mode forward is one
// submission. Pure data movement: every value is copied, none is computed.

struct Params {
    C: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       mean:  array<f32>;
@group(0) @binding(2) var<storage, read>       vari:  array<f32>;
@group(0) @binding(3) var<storage, read>       gamma: array<f32>;
@group(0) @binding(4) var<storage, read>       beta:  array<f32>;
@group(0) @binding(5) var<storage, read_write> mv:    array<f32>;
@group(0) @binding(6) var<storage, read_write> gb:    array<f32>;
@group(0) @binding(7) var<storage, read_write> mvg:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.C) { return; }
    let m = mean[c];
    let v = vari[c];
    let g = gamma[c];
    mv[2u * c] = m;
    mv[2u * c + 1u] = v;
    gb[2u * c] = g;
    gb[2u * c + 1u] = beta[c];
    mvg[3u * c] = m;
    mvg[3u * c + 1u] = v;
    mvg[3u * c + 2u] = g;
}
