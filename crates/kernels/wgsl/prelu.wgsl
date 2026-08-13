// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  PReLU forward with a LEARNED slope, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// PReLU forward with a LEARNED slope, NCHW:  y = x         if x > 0
//                                            y = a[ai] * x otherwise.
//   x : [N, C, H, W]   idx = ((n*C + c)*H + h)*W + w
//   a : [nslope]       ai  = select(0, c, nslope > 1)   -- see below
//   y : [N, C, H, W]   same idx as x
//
// One invocation per OUTPUT element (total = N*C*H*W).
//
// `nslope` selects the two shapes PyTorch's `nn.PReLU` supports:
//   nslope == C  -> one learned slope per channel   (a[c])
//   nslope == 1  -> a single shared learned slope   (a[0])
// Nothing else is legal; the caller passes the length of the slope tensor.
// A purely 2D activation ([N, C]) uses H = W = 1, so the same kernel serves the
// IResNet stem and any flat feature vector.
//
// Why this is NOT leaky_relu.wgsl: there the slope is a compile/dispatch-time
// CONSTANT bit-cast into the uniform, so it has no gradient and no storage
// binding. PReLU's slope is a trainable parameter — it lives in a buffer, it is
// indexed per channel, and it needs its own gradient (prelu_bwd.wgsl). Reusing
// leaky_relu would train nothing and silently pin every slope at its init value.
//
// Branch convention: the positive test is `x > 0`, matching torch.prelu (and
// NOT leaky_relu.wgsl's `x >= 0`). At exactly x == 0 both branches produce the
// same VALUE (0), so the forward is insensitive to the choice; the derivative
// is not, which is why prelu_bwd.wgsl uses the identical `x > 0` test. Keeping
// the two in step is the whole point of writing it down.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    nslope: u32,  // C (per-channel) or 1 (single shared slope)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       a: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }

    // Only the channel is needed; (n, h, w) never enter the math.
    let hw = p.H * p.W;
    let c = (idx / hw) % p.C;
    let ai = select(0u, c, p.nslope > 1u);

    let v = x[idx];
    if (v > 0.0) {
        y[idx] = v;
    } else {
        y[idx] = a[ai] * v;
    }
}
