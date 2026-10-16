// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency chunked linear-attention kernels
// for its clients. If your team needs expertise in collapsing a sequential
// GPU dependency chain of small launches into one persistent kernel, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// The decay-scale gradient of Gated DeltaNet's state update: the scalar
// d_decay_last = sum over the head's [dk, dv] state of d_state * state_in,
// scaled by exp(g_last) and added to d_g_cs at the chunk's last row.
// `gdn_state_decay_bwd_dscale.wgsl` gives each head ONE thread that walks all
// 16,384 elements serially (a few hundred microseconds, once per chunk per
// layer); here a 256-thread block per head shares them. The sum is reassociated
// (agreement to fp32 summation error, not bit for bit).
//
// params: [bh, dk, dv, c_len, g_cs_off, state_off].
// Bindings: uniform, d_state, state_in, g_cs, d_g_cs.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_DS_THREADS 256

extern "C" __global__ void __launch_bounds__(BRAIN_DS_THREADS) brain_gdn_state_decay_bwd_dscale(const unsigned int* params, const float* d_state, const float* state_in, const float* g_cs, float* d_g_cs) {
    const unsigned bh = blockIdx.y * gridDim.x + blockIdx.x;
    if (bh >= params[0]) { return; }
    const unsigned dkdv = params[1] * params[2], c_len = params[3];
    const unsigned base = bh * dkdv, state_base = params[5] + base;
    float acc = 0.0f;
    for (unsigned t = threadIdx.x; t < dkdv; t += BRAIN_DS_THREADS) { acc += d_state[base + t] * state_in[state_base + t]; }
    for (int o = 16; o > 0; o >>= 1) { acc += __shfl_xor_sync(0xffffffffu, acc, o); }
    __shared__ float red[BRAIN_DS_THREADS / 32];
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = acc; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < BRAIN_DS_THREADS / 32; ++w) { total += red[w]; }
        const unsigned last = params[4] + bh * c_len + c_len - 1;
        d_g_cs[last] = d_g_cs[last] + total * expf(g_cs[last]);
    }
}
