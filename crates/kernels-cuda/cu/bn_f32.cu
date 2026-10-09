// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements convolutional-network training kernels for
// its clients. If your team needs expertise in getting a CNN's normalisation
// layers onto the memory roofline of a GPU, you can procure our services by
// sending an email to info@swedishembedded.com.
//
// BatchNorm's per-channel reductions over an NCHW map, fp32: the hand-written
// forms of `bn_stats.wgsl`, `bn_dstats.wgsl`, `bn_dgamma.wgsl` and
// `bn_dbeta.wgsl`, reading the identical buffers and the identical uniform
// `[N, C, H, W]`, writing the identical outputs.
//
// What the WGSL kernels do and why it is slow
// -------------------------------------------
// One invocation per CHANNEL walks all N*H*W elements of it serially. A
// handful of channels is a handful of threads for the whole device, and the
// threads of one warp read addresses a whole channel apart, so every 32-byte
// sector fetched serves one float.
//
// What these kernels do instead
// -----------------------------
// One 512-thread block per channel. The channel is N contiguous runs of H*W
// floats; the block's threads stride across them together (16-byte loads
// where the runs allow it), so every warp reads contiguous memory. Each
// thread keeps a private fp32 partial; the block folds them in a fixed tree
// (warp shuffles, then one warp over the warp partials), so the result does
// not depend on scheduling. The statistics are two passes like the
// reference's: the mean first, then the sum of squared deviations from it.
//
// Accuracy: fp32 accumulation throughout; the sums are re-associated (per
// thread, then a tree) - a shorter error chain than the reference's single
// serial sum, and within the summation bound the gate holds them to. eps and
// the `1 / sqrt` form are the reference's.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_BN_THREADS 512
#define BRAIN_BN_WARPS (BRAIN_BN_THREADS / 32)
#define BRAIN_BN_EPS 1e-5f

// Block-wide sums of two values; every thread gets the totals. Fixed order:
// a shuffle tree inside each warp, then warp 0 folds the warp partials.
__device__ __forceinline__ float2 brain_bn_sum2(float a, float b, float2* part) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, o);
        b += __shfl_down_sync(0xffffffffu, b, o);
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) { part[warp] = make_float2(a, b); }
    __syncthreads();
    if (warp == 0) {
        float2 v = lane < BRAIN_BN_WARPS ? part[lane] : make_float2(0.0f, 0.0f);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v.x += __shfl_down_sync(0xffffffffu, v.x, o);
            v.y += __shfl_down_sync(0xffffffffu, v.y, o);
        }
        if (lane == 0) { part[BRAIN_BN_WARPS] = v; }
    }
    __syncthreads();
    const float2 r = part[BRAIN_BN_WARPS];
    __syncthreads();  // `part` is reused by the caller's next reduction
    return r;
}

// Visit every element of channel `c` once, block-cooperatively:
// f(x_value, dy_value) per element, with `dy` optional (nullptr).
// Elements are taken four at a time where every run is a whole number of
// 16-byte vectors at aligned addresses.
template <class F>
__device__ __forceinline__ void brain_bn_visit(const unsigned* p, unsigned c, const float* x, const float* dy, F f) {
    const unsigned N = p[0], C = p[1], HW = p[2] * p[3];
    const bool vec = (HW % 4u == 0u) && ((reinterpret_cast<unsigned long long>(x) & 15ull) == 0ull) &&
                     (dy == nullptr || (reinterpret_cast<unsigned long long>(dy) & 15ull) == 0ull);
    if (vec) {
        const unsigned hw4 = HW / 4u;
        const unsigned total = N * hw4;
        for (unsigned e = threadIdx.x; e < total; e += BRAIN_BN_THREADS) {
            const unsigned n = e / hw4;
            const unsigned long long off = ((unsigned long long)n * C + c) * HW + 4ull * (e - n * hw4);
            const float4 xv = *reinterpret_cast<const float4*>(x + off);
            if (dy != nullptr) {
                const float4 dv = *reinterpret_cast<const float4*>(dy + off);
                f(xv.x, dv.x); f(xv.y, dv.y); f(xv.z, dv.z); f(xv.w, dv.w);
            } else {
                f(xv.x, 0.0f); f(xv.y, 0.0f); f(xv.z, 0.0f); f(xv.w, 0.0f);
            }
        }
    } else {
        const unsigned total = N * HW;
        for (unsigned e = threadIdx.x; e < total; e += BRAIN_BN_THREADS) {
            const unsigned n = e / HW;
            const unsigned long long off = ((unsigned long long)n * C + c) * HW + (e - n * HW);
            f(x[off], dy != nullptr ? dy[off] : 0.0f);
        }
    }
}

__device__ __forceinline__ unsigned brain_bn_channel() { return blockIdx.y * gridDim.x + blockIdx.x; }

// mean[c] = mean(x), var[c] = mean((x - mean)^2), population variance.
extern "C" __global__ void __launch_bounds__(BRAIN_BN_THREADS) brain_bn_stats(const unsigned int* params, const float* x, float* mean,
                                                                            float* var) {
    __shared__ float2 part[BRAIN_BN_WARPS + 1];
    const unsigned c = brain_bn_channel();
    if (c >= params[1]) { return; }  // block-uniform
    const float m_count = (float)(params[0] * params[2] * params[3]);
    float s = 0.0f;
    brain_bn_visit(params, c, x, nullptr, [&](float v, float) { s += v; });
    const float m = brain_bn_sum2(s, 0.0f, part).x / m_count;
    float q = 0.0f;
    brain_bn_visit(params, c, x, nullptr, [&](float v, float) {
        const float d = v - m;
        q = fmaf(d, d, q);
    });
    const float vq = brain_bn_sum2(q, 0.0f, part).x;
    if (threadIdx.x == 0) {
        mean[c] = m;
        var[c] = vq / m_count;
    }
}

// bp[5c..] = mean, var, gamma (copied from mvg) and sum(dy), sum(dy * xhat).
extern "C" __global__ void __launch_bounds__(BRAIN_BN_THREADS) brain_bn_dstats(const unsigned int* params, const float* x,
                                                                             const float* dy, const float* mvg, float* bp) {
    __shared__ float2 part[BRAIN_BN_WARPS + 1];
    const unsigned c = brain_bn_channel();
    if (c >= params[1]) { return; }
    const float mean = mvg[3u * c], va = mvg[3u * c + 1u], gamma = mvg[3u * c + 2u];
    const float inv = 1.0f / sqrtf(va + BRAIN_BN_EPS);
    float ds = 0.0f, dx = 0.0f;
    brain_bn_visit(params, c, x, dy, [&](float v, float d) {
        ds += d;
        dx = fmaf(d, (v - mean) * inv, dx);
    });
    const float2 r = brain_bn_sum2(ds, dx, part);
    if (threadIdx.x == 0) {
        bp[5u * c + 0u] = mean;
        bp[5u * c + 1u] = va;
        bp[5u * c + 2u] = gamma;
        bp[5u * c + 3u] = r.x;
        bp[5u * c + 4u] = r.y;
    }
}

// dgamma[c] += sum(dy * xhat), mv = mean|var interleaved.
extern "C" __global__ void __launch_bounds__(BRAIN_BN_THREADS) brain_bn_dgamma(const unsigned int* params, const float* x,
                                                                             const float* dy, const float* mv, float* dgamma) {
    __shared__ float2 part[BRAIN_BN_WARPS + 1];
    const unsigned c = brain_bn_channel();
    if (c >= params[1]) { return; }
    const float mean = mv[2u * c];
    const float inv = 1.0f / sqrtf(mv[2u * c + 1u] + BRAIN_BN_EPS);
    float acc = 0.0f;
    brain_bn_visit(params, c, x, dy, [&](float v, float d) { acc = fmaf(d, (v - mean) * inv, acc); });
    const float r = brain_bn_sum2(acc, 0.0f, part).x;
    if (threadIdx.x == 0) { dgamma[c] = dgamma[c] + r; }
}

// dbeta[c] += sum(dy).
extern "C" __global__ void __launch_bounds__(BRAIN_BN_THREADS) brain_bn_dbeta(const unsigned int* params, const float* dy,
                                                                            float* dbeta) {
    __shared__ float2 part[BRAIN_BN_WARPS + 1];
    const unsigned c = brain_bn_channel();
    if (c >= params[1]) { return; }
    float acc = 0.0f;
    // `dy` is the only operand: visit it as the "x" stream.
    brain_bn_visit(params, c, dy, nullptr, [&](float d, float) { acc += d; });
    const float r = brain_bn_sum2(acc, 0.0f, part).x;
    if (threadIdx.x == 0) { dbeta[c] = dbeta[c] + r; }
}
