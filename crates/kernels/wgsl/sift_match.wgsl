// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  descriptor matching: each descriptor's best and second-best dot product in another image's set
// @how   64-thread workgroup tile, 3 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// For every 128-float descriptor of `a`, the index of its largest dot
// product among `b`'s descriptors and the two largest values - the
// nearest-neighbour search behind Lowe's ratio test on unit-norm RootSIFT
// (`sfm::matching`, whose host loop is the reference). out[i*3] is the
// index (bits), out[i*3 + 1] the best, out[i*3 + 2] the second best; an
// empty `b` leaves index u32::MAX and scores -1.
//
// One workgroup takes 64 rows of `a` into workgroup memory (row stride 129,
// so the 64 threads reading their own rows hit 64 different banks), then
// streams `b` through in tiles of 16 descriptors that every thread reads
// together (a broadcast). Ties keep the first index, as the host loop does.

struct Params {
    na: u32,
    nb: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>; // na*128
@group(0) @binding(2) var<storage, read>       b:   array<f32>; // nb*128
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // na*3

const D: u32 = 128u;
const ROWS: u32 = 64u;
const STRIDE: u32 = 129u;
const TILE: u32 = 16u;

var<workgroup> sa: array<f32, 8256>; // ROWS * STRIDE
var<workgroup> sb: array<f32, 2048>; // TILE * D

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_index) t: u32,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let base = (wg.y * nwg.x + wg.x) * ROWS;
    if (base >= p.na) { return; }
    for (var e = t; e < ROWS * D; e = e + ROWS) {
        let row = e / D;
        var v = 0.0;
        if (base + row < p.na) { v = a[(base + row) * D + e % D]; }
        sa[row * STRIDE + e % D] = v;
    }
    workgroupBarrier();
    var bi = 0xffffffffu;
    var b1 = -1.0;
    var b2 = -1.0;
    let mine = t * STRIDE;
    for (var j0 = 0u; j0 < p.nb; j0 = j0 + TILE) {
        for (var e = t; e < TILE * D; e = e + ROWS) {
            let j = j0 + e / D;
            var v = 0.0;
            if (j < p.nb) { v = b[j * D + e % D]; }
            sb[e] = v;
        }
        workgroupBarrier();
        let n = min(TILE, p.nb - j0);
        for (var jj = 0u; jj < n; jj = jj + 1u) {
            var s = 0.0;
            for (var k = 0u; k < D; k = k + 1u) {
                s = s + sa[mine + k] * sb[jj * D + k];
            }
            if (s > b1) {
                b2 = b1;
                b1 = s;
                bi = j0 + jj;
            } else if (s > b2) {
                b2 = s;
            }
        }
        workgroupBarrier();
    }
    let i = base + t;
    if (i < p.na) {
        out[i * 3u] = bitcast<f32>(bi);
        out[i * 3u + 1u] = b1;
        out[i * 3u + 2u] = b2;
    }
}
