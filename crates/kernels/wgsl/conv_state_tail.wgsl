// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
// clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  The convolution ring state after a prefill chunk: the last K-1 inputs of each channel, continuing the previous state
// @how   one thread per (channel, slot), gathers from the chunk or from the previous state
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The ring state a chunk of `t` token-major rows leaves behind for the decode step or the next chunk: slot i of
// channel c (oldest first, `causal_conv1d_step`'s `hist` layout) holds the input at logical position
// `p = t - (K-1) + i` of the sequence continued from `hist_in`:
//
//   hist_out[c, i] = p >= 0 ? x[p, c] : hist_in[c, K-1+p]
//
// `hist_in` and `hist_out` are separate bindings: in place the slots would be read after another thread wrote them. A
// fresh sequence passes a zeroed `hist_in`. One segment is assumed (rows packed from several documents are a training
// layout and carry no state).
//
//   x : [t, c]   hist_in, hist_out : [c, K-1]   params: t, c, taps

struct Params {
    t: u32,
    c: u32,
    taps: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:        array<f32>;
@group(0) @binding(2) var<storage, read>       hist_in:  array<f32>;
@group(0) @binding(3) var<storage, read_write> hist_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let slots = p.taps - 1u;
    if (idx >= p.c * slots) { return; }
    let ci = idx / slots;
    let i = idx % slots;
    // Logical position of this slot, shifted by K-1 so it stays unsigned: pos_shifted = t + i, pos = pos_shifted - (K-1).
    let shifted = p.t + i;
    if (shifted >= slots) {
        hist_out[idx] = x[(shifted - slots) * p.c + ci];
    } else {
        hist_out[idx] = hist_in[ci * slots + shifted];
    }
}
