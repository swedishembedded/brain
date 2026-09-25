// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 2: sum each (tile, gaussian) instance's 256-pixel slot block into one gradient record
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One WORKGROUP per instance of the band. splat_bwd_slots left the instance's
// partials as 11 contiguous runs of 256 floats (one per channel, one float per
// pixel of its tile); each of the 64 threads sums 4 consecutive pixels of
// every channel - lanes read consecutive 16-byte chunks, so every load is
// coalesced - into workgroup memory, and after the one barrier threads 0..10
// finish one channel each. Thread 11 stamps the record with its gaussian id
// (the sort key of the per-gaussian stage).
//
// Record stride 12 in `recs`: {v_xy(2), v_conic(3), v_opacity, v_rgb(3),
// v_depth, |v_xy| summed per pixel, id}.
//
// This is what replaced sorting one record per (pixel, gaussian) by gaussian
// id: that sort was 70% of the backward's device time. The records it now
// feeds are one per (tile, gaussian) instance, a few percent as many.

struct Params {
    n: u32,   // instances in the band
    k0: u32,  // first instance of the band
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       slots: array<f32>; // band_instances*11*256
@group(0) @binding(2) var<storage, read>       vals:  array<u32>; // sorted instance -> gaussian id
@group(0) @binding(3) var<storage, read_write> recs:  array<f32>; // n_isects*12

var<workgroup> partial: array<f32, 704>; // 11 channels x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let inst = wg.y * nwg.x + wg.x;
    if (inst >= p.n) { return; }
    let t = li.x;
    for (var c = 0u; c < 11u; c = c + 1u) {
        let base = (inst * 11u + c) * 256u + t * 4u;
        partial[c * 64u + t] = slots[base] + slots[base + 1u] + slots[base + 2u] + slots[base + 3u];
    }
    workgroupBarrier();
    let r = (p.k0 + inst) * 12u;
    if (t < 11u) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        recs[r + t] = s;
    } else if (t == 11u) {
        recs[r + 11u] = bitcast<f32>(vals[p.k0 + inst]);
    }
}
