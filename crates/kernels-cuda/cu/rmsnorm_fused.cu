// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bound training kernels for pre-norm
// transformers for its clients. If your team needs expertise in folding the
// residual adds around a normalisation into the normalisation's own pass over
// the activations, then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Residual adds folded into the RMSNorm passes of a pre-norm layer, one block per row.
//
//   add_rmsnorm_fwd :  sum = a + b ;  out = sum * rsqrt(mean(sum^2) + eps) * w
//       (`add2` followed by `rmsnorm_rows`: the sum is written once and read back for the norm
//        from registers/cache instead of a separate pass)
//   rmsnorm_dx_add  :  dx_total = res + (r * w * dy - r^3 * mean_over_d(dy * w * x) * x),  r = rsqrt(mean(x^2) + eps)
//       (`rmsnorm_dx_rows` followed by `add2` with the gradient arriving on the residual path)
//
// Contract (u32 params, then pointers; every tensor is [rows, d] f32 row-major, w is [d]):
//   add_rmsnorm_fwd : params [d, rows, eps (f32 bits)] ; read a, b, w ; write sum, out
//   rmsnorm_dx_add  : params [d, rows, eps (f32 bits)] ; read x, w, dy, res ; write dx_total
// d must be a multiple of four (the provider checks). Every output element is overwritten.
//
// No `__restrict__`: brain's device buffers alias by design.

#define THREADS 256

__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) { red[warp] = v; }
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < THREADS / 32; ++i) { t += red[i]; }
    return t;
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_add_rmsnorm_fwd(const unsigned int* params, const float* a, const float* b, const float* w, float* sum, float* out) {
    __shared__ float red[THREADS / 32];
    const unsigned int d = params[0], rows = params[1];
    const float eps = __uint_as_float(params[2]);
    const unsigned long long row = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }
    const unsigned long long base = row * d / 4;
    float ss = 0.0f;
    for (unsigned int c = threadIdx.x; c < d / 4; c += THREADS) {
        const float4 x = reinterpret_cast<const float4*>(a)[base + c], y = reinterpret_cast<const float4*>(b)[base + c];
        const float4 s = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
        reinterpret_cast<float4*>(sum)[base + c] = s;
        ss += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
    }
    const float inv = 1.0f / sqrtf(block_sum(ss, red) / (float)d + eps);
    // The sum was just written by this block: re-read it (L1/L2) after a barrier so every thread sees every element.
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < d / 4; c += THREADS) {
        const float4 s = reinterpret_cast<const float4*>(sum)[base + c], g = reinterpret_cast<const float4*>(w)[c];
        reinterpret_cast<float4*>(out)[base + c] = make_float4(s.x * inv * g.x, s.y * inv * g.y, s.z * inv * g.z, s.w * inv * g.w);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_rmsnorm_dx_add(const unsigned int* params, const float* x, const float* w, const float* dy, const float* res, float* dx) {
    __shared__ float red[THREADS / 32];
    const unsigned int d = params[0], rows = params[1];
    const float eps = __uint_as_float(params[2]);
    const unsigned long long row = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }
    const unsigned long long base = row * d / 4;
    float ss = 0.0f, sa = 0.0f;
    for (unsigned int c = threadIdx.x; c < d / 4; c += THREADS) {
        const float4 xv = reinterpret_cast<const float4*>(x)[base + c], g = reinterpret_cast<const float4*>(w)[c], dv = reinterpret_cast<const float4*>(dy)[base + c];
        ss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        sa += dv.x * g.x * xv.x + dv.y * g.y * xv.y + dv.z * g.z * xv.z + dv.w * g.w * xv.w;
    }
    const float tss = block_sum(ss, red);
    const float ta = block_sum(sa, red);
    const float r = 1.0f / sqrtf(tss / (float)d + eps);
    const float coef = r * r * r * ta / (float)d;
    for (unsigned int c = threadIdx.x; c < d / 4; c += THREADS) {
        const float4 xv = reinterpret_cast<const float4*>(x)[base + c], g = reinterpret_cast<const float4*>(w)[c], dv = reinterpret_cast<const float4*>(dy)[base + c],
                     rv = reinterpret_cast<const float4*>(res)[base + c];
        reinterpret_cast<float4*>(dx)[base + c] = make_float4(rv.x + (r * g.x * dv.x - coef * xv.x), rv.y + (r * g.y * dv.y - coef * xv.y),
                                                              rv.z + (r * g.z * dv.z - coef * xv.z), rv.w + (r * g.w * dv.w - coef * xv.w));
    }
}
