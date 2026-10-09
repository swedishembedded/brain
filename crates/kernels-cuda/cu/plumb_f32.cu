// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements convolutional-network training kernels for
// its clients. If your team needs expertise in getting a network's data
// movement onto the memory roofline of a GPU, you can procure our services by
// sending an email to info@swedishembedded.com.
//
// The memory-bound plumbing of a CSP network's training step, fp32: the
// hand-written forms of `concat_split.wgsl` (take a channel window of an NCHW
// map), `chan_place.wgsl` (write a map into a channel window), `concat2.wgsl`
// (concatenate two maps along channels), `silu.wgsl` and `silu_bwd.wgsl`,
// reading the identical buffers and the identical uniforms.
//
// The WGSL kernels give one invocation one element and recover its (n, c, h,
// w) with four integer divisions; in the generated tier that is most of their
// cost. Each channel window is a run of whole (n, c) planes, so here a block
// copies a run of ONE plane - no division per element - with 16-byte accesses
// where the plane allows it. The activations are flat: a block takes a run of
// the array, four elements a thread.
//
// BIT-IDENTICAL to the WGSL tier: the copies move values untouched, and the
// activations evaluate the reference's expression in the reference's order
// with the same `expf` and IEEE division (the backend's `--fmad=false` keeps
// every product and sum separately rounded on both sides).
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_PL_THREADS 256
#define BRAIN_PL_PER_BLOCK (BRAIN_PL_THREADS * 4)

__device__ __forceinline__ unsigned brain_pl_block() { return blockIdx.y * gridDim.x + blockIdx.x; }

__device__ __forceinline__ bool brain_pl_aligned(const float* a) { return (reinterpret_cast<unsigned long long>(a) & 15ull) == 0ull; }

// Copy `len` floats src[0..len) -> dst[0..len), this block's run starting at
// `start`, four a thread; vectorised when `vec`.
__device__ __forceinline__ void brain_pl_copy_run(const float* src, float* dst, unsigned len, unsigned start, bool vec) {
    const unsigned end = min(len, start + BRAIN_PL_PER_BLOCK);
    if (vec) {
        const unsigned i = start + 4u * threadIdx.x;
        if (i < end) { *reinterpret_cast<float4*>(dst + i) = *reinterpret_cast<const float4*>(src + i); }
    } else {
        for (unsigned i = start + threadIdx.x; i < end; i += BRAIN_PL_THREADS) { dst[i] = src[i]; }
    }
}

// The run of plane `plane` (of `hw` floats) this block covers: blocks are
// numbered plane-major, `runs` per plane.
struct BrainPlRun {
    unsigned plane, start;
    bool ok;
};

__device__ __forceinline__ BrainPlRun brain_pl_run(unsigned planes, unsigned hw) {
    const unsigned runs = (hw + BRAIN_PL_PER_BLOCK - 1u) / BRAIN_PL_PER_BLOCK;
    const unsigned blk = brain_pl_block();
    BrainPlRun r;
    r.ok = runs > 0u && blk < planes * runs;  // block-uniform
    r.plane = runs ? blk / runs : 0u;
    r.start = (blk - r.plane * runs) * BRAIN_PL_PER_BLOCK;
    return r;
}

// da[n, c] = dy[n, c_off + c] over the da planes. params [N, Ctot, Csrc, c_off, H, W].
extern "C" __global__ void __launch_bounds__(BRAIN_PL_THREADS) brain_concat_split(const unsigned int* params, const float* dy, float* da) {
    const unsigned N = params[0], Ctot = params[1], Csrc = params[2], c_off = params[3], hw = params[4] * params[5];
    const BrainPlRun r = brain_pl_run(N * Csrc, hw);
    if (!r.ok) { return; }
    const unsigned n = r.plane / Csrc, c = r.plane - (r.plane / Csrc) * Csrc;
    const float* src = dy + ((unsigned long long)n * Ctot + c_off + c) * hw;
    float* dst = da + (unsigned long long)r.plane * hw;
    brain_pl_copy_run(src, dst, hw, r.start, hw % 4u == 0u && brain_pl_aligned(src) && brain_pl_aligned(dst));
}

// dst[n, c_off + c] = src[n, c] over the src planes. params [N, Ctot, Csrc, c_off, H, W].
extern "C" __global__ void __launch_bounds__(BRAIN_PL_THREADS) brain_chan_place(const unsigned int* params, const float* src, float* dst) {
    const unsigned N = params[0], Ctot = params[1], Csrc = params[2], c_off = params[3], hw = params[4] * params[5];
    const BrainPlRun r = brain_pl_run(N * Csrc, hw);
    if (!r.ok) { return; }
    const unsigned n = r.plane / Csrc, c = r.plane - (r.plane / Csrc) * Csrc;
    const float* s = src + (unsigned long long)r.plane * hw;
    float* d = dst + ((unsigned long long)n * Ctot + c_off + c) * hw;
    brain_pl_copy_run(s, d, hw, r.start, hw % 4u == 0u && brain_pl_aligned(s) && brain_pl_aligned(d));
}

// y = concat(a, b) along channels. params [N, Ca, Cb, H, W].
extern "C" __global__ void __launch_bounds__(BRAIN_PL_THREADS) brain_concat2(const unsigned int* params, const float* a, const float* b, float* y) {
    const unsigned N = params[0], Ca = params[1], Cb = params[2], hw = params[3] * params[4];
    const unsigned Ctot = Ca + Cb;
    const BrainPlRun r = brain_pl_run(N * Ctot, hw);
    if (!r.ok) { return; }
    const unsigned n = r.plane / Ctot, c = r.plane - (r.plane / Ctot) * Ctot;
    const float* src = c < Ca ? a + ((unsigned long long)n * Ca + c) * hw : b + ((unsigned long long)n * Cb + (c - Ca)) * hw;
    float* dst = y + (unsigned long long)r.plane * hw;
    brain_pl_copy_run(src, dst, hw, r.start, hw % 4u == 0u && brain_pl_aligned(src) && brain_pl_aligned(dst));
}

// Flat elementwise over `total` elements, four a thread; `f4` takes the
// element index of a vector, `f1` of one element.
template <class F4, class F1>
__device__ __forceinline__ void brain_pl_flat(unsigned total, bool vec, F4 f4, F1 f1) {
    const unsigned start = brain_pl_block() * BRAIN_PL_PER_BLOCK;
    if (start >= total) { return; }
    const unsigned end = min(total, start + BRAIN_PL_PER_BLOCK);
    const unsigned i = start + 4u * threadIdx.x;
    if (vec && i + 4u <= end) {
        f4(i);
    } else if (!vec) {
        for (unsigned j = start + threadIdx.x; j < end; j += BRAIN_PL_THREADS) { f1(j); }
    } else {
        for (unsigned j = i; j < end && j < i + 4u; ++j) { f1(j); }
    }
}

__device__ __forceinline__ float brain_pl_silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float brain_pl_silu_grad(float v, float d) {
    const float s = 1.0f / (1.0f + expf(-v));
    return d * (s + v * s * (1.0f - s));
}

// out = x / (1 + exp(-x)). params [total].
extern "C" __global__ void __launch_bounds__(BRAIN_PL_THREADS) brain_silu_fwd(const unsigned int* params, const float* x, float* out) {
    const unsigned total = params[0];
    brain_pl_flat(
        total, brain_pl_aligned(x) && brain_pl_aligned(out),
        [&](unsigned i) {
            const float4 v = *reinterpret_cast<const float4*>(x + i);
            *reinterpret_cast<float4*>(out + i) = make_float4(brain_pl_silu(v.x), brain_pl_silu(v.y), brain_pl_silu(v.z), brain_pl_silu(v.w));
        },
        [&](unsigned i) { out[i] = brain_pl_silu(x[i]); });
}

// dx = dy * (s + x * s * (1 - s)), s = sigmoid(x). params [total].
extern "C" __global__ void __launch_bounds__(BRAIN_PL_THREADS) brain_silu_bwd(const unsigned int* params, const float* x, const float* dy, float* dx) {
    const unsigned total = params[0];
    brain_pl_flat(
        total, brain_pl_aligned(x) && brain_pl_aligned(dy) && brain_pl_aligned(dx),
        [&](unsigned i) {
            const float4 v = *reinterpret_cast<const float4*>(x + i);
            const float4 d = *reinterpret_cast<const float4*>(dy + i);
            *reinterpret_cast<float4*>(dx + i) = make_float4(brain_pl_silu_grad(v.x, d.x), brain_pl_silu_grad(v.y, d.y),
                                                             brain_pl_silu_grad(v.z, d.z), brain_pl_silu_grad(v.w, d.w));
        },
        [&](unsigned i) { dx[i] = brain_pl_silu_grad(x[i], dy[i]); });
}
