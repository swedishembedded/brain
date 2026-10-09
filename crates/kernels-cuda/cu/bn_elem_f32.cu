// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements convolutional-network training kernels for
// its clients. If your team needs expertise in getting a CNN's normalisation
// layers onto the memory roofline of a GPU, you can procure our services by
// sending an email to info@swedishembedded.com.
//
// BatchNorm's elementwise passes over an NCHW map, fp32: the hand-written
// forms of `bn_train.wgsl` (normalise with batch statistics) and `bn_dx.wgsl`
// (the input gradient), reading the identical buffers and the identical
// uniform `[N, C, H, W]`.
//
// The WGSL kernels give one invocation one element and recover its channel
// with two integer divisions, then recompute the channel's `1 / sqrt(var +
// eps)` - per element, so the pass is bound by that arithmetic rather than by
// memory. Here a block covers a run of one (n, c) plane: the channel is known
// per block, its constants are computed once per thread, and the elements are
// moved with 16-byte accesses where the plane allows it.
//
// BIT-IDENTICAL to the WGSL tier: per element the same operations in the same
// order (`(x - mean) * inv * gamma + beta`; `(gamma * inv) * ((dy - dsum / M)
// - xhat * dxhat_sum / M)`), the same `1.0f / sqrtf(..)` for `inverseSqrt`,
// and the backend's `--fmad=false` keeps every product and sum separately
// rounded on both sides. There is no reduction, so nothing can reassociate.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_BE_THREADS 256
#define BRAIN_BE_PER_BLOCK (BRAIN_BE_THREADS * 4)  // elements of one plane per block
#define BRAIN_BE_EPS 1e-5f

// The plane (n, c) and the first element of this block's run of it.
struct BrainBePlane {
    unsigned c;
    unsigned long long base;  // flat offset of the plane
    unsigned start;           // first element of the run, within the plane
    unsigned hw;
    bool ok;
};

__device__ __forceinline__ BrainBePlane brain_be_plane(const unsigned* p) {
    const unsigned N = p[0], C = p[1], hw = p[2] * p[3];
    const unsigned runs = (hw + BRAIN_BE_PER_BLOCK - 1u) / BRAIN_BE_PER_BLOCK;
    const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
    BrainBePlane o;
    o.hw = hw;
    o.ok = runs > 0u && blk < N * C * runs;  // block-uniform
    const unsigned plane = runs ? blk / runs : 0u;
    o.c = C ? plane % C : 0u;
    o.base = (unsigned long long)plane * hw;
    o.start = (blk - plane * runs) * BRAIN_BE_PER_BLOCK;
    return o;
}

__device__ __forceinline__ bool brain_be_aligned(const float* a) { return (reinterpret_cast<unsigned long long>(a) & 15ull) == 0ull; }

// Apply f(index_in_plane) -> writes, over this block's run, four elements a
// thread, vectorised when `vec` holds.
template <class F4, class F1>
__device__ __forceinline__ void brain_be_run(const BrainBePlane& pl, bool vec, F4 f4, F1 f1) {
    const unsigned end = min(pl.hw, pl.start + BRAIN_BE_PER_BLOCK);
    if (vec) {
        const unsigned i = pl.start + 4u * threadIdx.x;
        if (i < end) { f4(pl.base + i); }  // hw % 4 == 0, so a whole vector fits
    } else {
        for (unsigned i = pl.start + threadIdx.x; i < end; i += BRAIN_BE_THREADS) { f1(pl.base + i); }
    }
}

// out = (x - mean[c]) * inv * gamma[c] + beta[c], inv = 1 / sqrt(var[c] + eps).
extern "C" __global__ void __launch_bounds__(BRAIN_BE_THREADS) brain_bn_train(const unsigned int* params, const float* x, const float* mv,
                                                                            const float* gb, float* out) {
    const BrainBePlane pl = brain_be_plane(params);
    if (!pl.ok) { return; }
    const float mean = mv[2u * pl.c];
    const float va = mv[2u * pl.c + 1u];
    const float gamma = gb[2u * pl.c];
    const float beta = gb[2u * pl.c + 1u];
    const float inv = 1.0f / sqrtf(va + BRAIN_BE_EPS);
    auto one = [&](float v) { return (v - mean) * inv * gamma + beta; };
    const bool vec = (pl.hw % 4u == 0u) && brain_be_aligned(x) && brain_be_aligned(out);
    brain_be_run(
        pl, vec,
        [&](unsigned long long o) {
            const float4 v = *reinterpret_cast<const float4*>(x + o);
            *reinterpret_cast<float4*>(out + o) = make_float4(one(v.x), one(v.y), one(v.z), one(v.w));
        },
        [&](unsigned long long o) { out[o] = one(x[o]); });
}

// dx = (gamma * inv) * (dy - dsum / M - xhat * dxhat_sum / M),
// xhat = (x - mean) * inv; bp = [mean, var, gamma, dsum, dxhat_sum] per channel.
extern "C" __global__ void __launch_bounds__(BRAIN_BE_THREADS) brain_bn_dx(const unsigned int* params, const float* x, const float* dy,
                                                                         const float* bp, float* dx) {
    const BrainBePlane pl = brain_be_plane(params);
    if (!pl.ok) { return; }
    const float M = (float)(params[0] * params[2] * params[3]);
    const float mean = bp[5u * pl.c + 0u];
    const float va = bp[5u * pl.c + 1u];
    const float gamma = bp[5u * pl.c + 2u];
    const float dsum = bp[5u * pl.c + 3u];
    const float dxhat_sum = bp[5u * pl.c + 4u];
    const float inv = 1.0f / sqrtf(va + BRAIN_BE_EPS);
    const float scale = gamma * inv;
    const float dsum_m = dsum / M;
    auto one = [&](float xv, float d) {
        const float xhat = (xv - mean) * inv;
        return scale * ((d - dsum_m) - (xhat * dxhat_sum) / M);
    };
    const bool vec = (pl.hw % 4u == 0u) && brain_be_aligned(x) && brain_be_aligned(dy) && brain_be_aligned(dx);
    brain_be_run(
        pl, vec,
        [&](unsigned long long o) {
            const float4 a = *reinterpret_cast<const float4*>(x + o);
            const float4 d = *reinterpret_cast<const float4*>(dy + o);
            *reinterpret_cast<float4*>(dx + o) = make_float4(one(a.x, d.x), one(a.y, d.y), one(a.z, d.z), one(a.w, d.w));
        },
        [&](unsigned long long o) { dx[o] = one(x[o], dy[o]); });
}
