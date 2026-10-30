// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for bf16 activation streams between the matrix products of LLM serving engines
// for its clients. If your team needs expertise in halving the bytes a forward moves between its GEMMs then you can procure
// our services by sending an email to info@swedishembedded.com.
//
// Widen packed bf16 to fp32 (exact: a bf16 is the high half of the fp32 with its value). The few rows of a bf16 activation
// that a kernel with only an fp32 form reads - the decode rows that lead a mixed round, whose recurrent step is the decode
// kernel's - are widened into a small fp32 buffer; every other row stays bf16.
//
// params: [count]. Bindings: uniform, src (packed bf16, two to a word), dst (f32). Four elements a thread: an 8-byte load, a
// 16-byte store; the count is a multiple of four and both buffers are 16-byte aligned.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_WIDEN_THREADS 256
#define BRAIN_WIDEN_PER_BLOCK (BRAIN_WIDEN_THREADS * 4)

extern "C" __global__ void __launch_bounds__(BRAIN_WIDEN_THREADS) brain_cast_bf16_f32(const unsigned int* params, const unsigned int* src, float* dst) {
    const unsigned count = params[0];
    const unsigned long long block = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long i = block * BRAIN_WIDEN_PER_BLOCK + 4ull * threadIdx.x;
    if (i >= count) { return; }
    const uint2 w = *reinterpret_cast<const uint2*>(src + (i >> 1));
    *reinterpret_cast<float4*>(dst + i) = make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
}
