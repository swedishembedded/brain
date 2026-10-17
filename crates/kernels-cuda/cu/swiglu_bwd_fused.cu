// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bound training kernels for gated-MLP
// language models for its clients. If your team needs expertise in collapsing
// element-wise gradient passes around a transformer's MLP into single kernels,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The SwiGLU backward as ONE pass: with h = silu(a) * b and the incoming gradient dh,
//   da = dh * b * (sig + a * sig * (1 - sig)),   db = dh * silu(a),   sig = 1 / (1 + exp(-a)).
// The portable path runs `silu_bwd_da` and `silu_bwd_db`, which each read `a` and `dh` and
// recompute the sigmoid: here a, b and dh are read once and the sigmoid is shared.
//
// Contract (u32 params, then pointers; every tensor is [total] f32):
//   params : [total]
//   read   : a (gate pre-activation), b (up), dh
//   write  : da, db  (every element overwritten)
// Elements are handled four at a time when total is a multiple of four.
//
// No `__restrict__`: brain's device buffers alias by design.

#define THREADS 256

__device__ __forceinline__ void swiglu_one(float a, float b, float dh, float& da, float& db) {
    const float sig = 1.0f / (1.0f + expf(-a));
    da = dh * b * (sig + a * sig * (1.0f - sig));
    db = dh * (a * sig);
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_swiglu_bwd_fused(const unsigned int* params, const float* a, const float* b, const float* dh, float* da, float* db) {
    const unsigned int total = params[0];
    const unsigned long long blk = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long i = blk * THREADS + threadIdx.x;
    if ((total & 3u) == 0) {
        if (i * 4 >= total) { return; }
        const float4 av = reinterpret_cast<const float4*>(a)[i], bv = reinterpret_cast<const float4*>(b)[i], dv = reinterpret_cast<const float4*>(dh)[i];
        float4 ga, gb;
        swiglu_one(av.x, bv.x, dv.x, ga.x, gb.x);
        swiglu_one(av.y, bv.y, dv.y, ga.y, gb.y);
        swiglu_one(av.z, bv.z, dv.z, ga.z, gb.z);
        swiglu_one(av.w, bv.w, dv.w, ga.w, gb.w);
        reinterpret_cast<float4*>(da)[i] = ga;
        reinterpret_cast<float4*>(db)[i] = gb;
    } else {
        if (i >= total) { return; }
        swiglu_one(a[i], b[i], dh[i], da[i], db[i]);
    }
}
