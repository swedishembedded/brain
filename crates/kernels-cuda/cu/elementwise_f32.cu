// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in putting the memory-bound glue of a transformer tower onto the bandwidth roof of the card it
// runs on then you can procure our services by sending an email to info@swedishembedded.com.
//
// The elementwise glue of a transformer block, fp32: the native forms of `bias_add.wgsl` (`out[m, n] +=
// bias[n]`), `add2.wgsl` (`out = a + b`) and `gelu_erf.wgsl` (exact erf GELU), reading the identical buffers and
// the identical uniforms.
//
// The WGSL kernels give one invocation one element and, for the bias, an integer modulo per element. Here a
// block takes a run of 1024 consecutive elements, four a thread with 16-byte accesses; the bias row is found
// once per vector. A vector access needs the buffers 16-byte aligned, the bias width a multiple of four and
// the flat length a whole number of vectors per block run; otherwise the thread walks its run element by
// element with the identical arithmetic.
//
// BIT-IDENTICAL to the WGSL tier: every element evaluates the reference's expression in the reference's order
// with the same `expf` and IEEE division (the backend's `--fmad=false` keeps every product and sum separately
// rounded on both sides). The gate is `gpu-core/tests/elementwise_native.rs`.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_EW_THREADS 256
#define BRAIN_EW_PER_BLOCK (BRAIN_EW_THREADS * 4)

__device__ __forceinline__ unsigned brain_ew_block() { return blockIdx.y * gridDim.x + blockIdx.x; }

__device__ __forceinline__ bool brain_ew_aligned(const void* a) { return (reinterpret_cast<unsigned long long>(a) & 15ull) == 0ull; }

// Flat elementwise over `total` elements, four a thread; `f4` takes the element index of a vector, `f1` of one
// element.
template <class F4, class F1>
__device__ __forceinline__ void brain_ew_flat(unsigned total, bool vec, F4 f4, F1 f1) {
    const unsigned start = brain_ew_block() * BRAIN_EW_PER_BLOCK;
    if (start >= total) { return; }
    const unsigned end = min(total, start + BRAIN_EW_PER_BLOCK);
    const unsigned i = start + 4u * threadIdx.x;
    if (vec && i + 4u <= end) {
        f4(i);
    } else if (!vec) {
        for (unsigned j = start + threadIdx.x; j < end; j += BRAIN_EW_THREADS) { f1(j); }
    } else {
        for (unsigned j = i; j < end && j < i + 4u; ++j) { f1(j); }
    }
}

// out[m, n] += bias[n]. params [m, n].
extern "C" __global__ void __launch_bounds__(BRAIN_EW_THREADS) brain_bias_add(const unsigned int* params, float* out, const float* bias) {
    const unsigned total = params[0] * params[1];
    const unsigned n = params[1];
    // A vector never straddles a row when the row width is a whole number of vectors.
    const bool vec = (n % 4u == 0u) && brain_ew_aligned(out) && brain_ew_aligned(bias);
    brain_ew_flat(
        total, vec,
        [&](unsigned i) {
            const float4 b = *reinterpret_cast<const float4*>(bias + (i % n));
            const float4 v = *reinterpret_cast<const float4*>(out + i);
            *reinterpret_cast<float4*>(out + i) = make_float4(v.x + b.x, v.y + b.y, v.z + b.z, v.w + b.w);
        },
        [&](unsigned i) { out[i] = out[i] + bias[i % n]; });
}

// out = a + b. params [total].
extern "C" __global__ void __launch_bounds__(BRAIN_EW_THREADS) brain_add2(const unsigned int* params, const float* a, const float* b, float* out) {
    const unsigned total = params[0];
    brain_ew_flat(
        total, brain_ew_aligned(a) && brain_ew_aligned(b) && brain_ew_aligned(out),
        [&](unsigned i) {
            const float4 x = *reinterpret_cast<const float4*>(a + i);
            const float4 y = *reinterpret_cast<const float4*>(b + i);
            *reinterpret_cast<float4*>(out + i) = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
        },
        [&](unsigned i) { out[i] = a[i] + b[i]; });
}

// 0.5 v (1 + erf(v / sqrt 2)), erf by Abramowitz & Stegun 7.1.26, in the WGSL kernel's order of operations.
__device__ __forceinline__ float brain_ew_gelu_erf(float v) {
    const float arg = v * 0.7071067811865476f;
    float s = 1.0f;
    if (arg < 0.0f) { s = -1.0f; }
    const float ax = fabsf(arg);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    const float erf = s * (1.0f - poly * expf(-ax * ax));
    return 0.5f * v * (1.0f + erf);
}

// out = gelu(x). params [total].
extern "C" __global__ void __launch_bounds__(BRAIN_EW_THREADS) brain_gelu_erf(const unsigned int* params, const float* x, float* out) {
    const unsigned total = params[0];
    brain_ew_flat(
        total, brain_ew_aligned(x) && brain_ew_aligned(out),
        [&](unsigned i) {
            const float4 v = *reinterpret_cast<const float4*>(x + i);
            *reinterpret_cast<float4*>(out + i) = make_float4(brain_ew_gelu_erf(v.x), brain_ew_gelu_erf(v.y), brain_ew_gelu_erf(v.z), brain_ew_gelu_erf(v.w));
        },
        [&](unsigned i) { out[i] = brain_ew_gelu_erf(x[i]); });
}
