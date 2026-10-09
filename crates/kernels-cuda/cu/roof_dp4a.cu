// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements honest roofline measurement for GPU
// inference stacks for its clients. If your team needs expertise in knowing
// what the hardware can actually do then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Peak packed-int8 dot-product rate: the device's int8 roof, measured with the
// instruction itself. `roof_dp4a.wgsl` is the portable probe; its
// `dot4I8Packed(x, b) + acc` reaches a CUDA device as a dot and a separate
// add, two instructions per four multiply-adds, which reports about half of
// what the silicon does - and a roof below a real kernel's rate is a roof that
// misgrades every int8 kernel measured against it.
//
//   params : u32 [n, iters, a, b]   the WGSL probe's own uniform
//   inp    : [n] f32                read once, so nothing is dead
//   out    : [n] f32
//
// Eight independent accumulators per thread, each step `acc = dp4a(acc, b,
// acc)`: the accumulator is also the packed operand, so every dot depends on
// the last and none can be hoisted out of the loop. 64 int ops per thread per
// iteration, counted as the WGSL probe counts them.

extern "C" __global__ void __launch_bounds__(256)
brain_roof_dp4a(const unsigned int* params, const float* inp, float* out) {
    const unsigned int n = params[0];
    const unsigned int iters = params[1];
    const unsigned int a = params[2];
    const int b = static_cast<int>(params[3]);
    const unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    int c0 = static_cast<int>(a), c1 = static_cast<int>(a ^ 1u), c2 = static_cast<int>(a ^ 2u), c3 = static_cast<int>(a ^ 3u);
    int c4 = static_cast<int>(a ^ 4u), c5 = static_cast<int>(a ^ 5u), c6 = static_cast<int>(a ^ 6u), c7 = static_cast<int>(a ^ 7u);
    for (unsigned int i = 0; i < iters; ++i) {
        c0 = __dp4a(c0, b, c0);
        c1 = __dp4a(c1, b, c1);
        c2 = __dp4a(c2, b, c2);
        c3 = __dp4a(c3, b, c3);
        c4 = __dp4a(c4, b, c4);
        c5 = __dp4a(c5, b, c5);
        c6 = __dp4a(c6, b, c6);
        c7 = __dp4a(c7, b, c7);
    }
    out[idx] = inp[idx] + static_cast<float>(((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)));
}
