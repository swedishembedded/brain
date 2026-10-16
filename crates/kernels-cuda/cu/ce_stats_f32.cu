// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements large-vocabulary language-model training for
// its clients. If your team needs expertise in cross-entropy over hundreds of
// thousands of classes without materialising the softmax, you can procure our
// services by sending an email to info@swedishembedded.com.
//
// Per-row softmax statistics of the cross-entropy backward, `ce_stats.wgsl`'s
// reduction with a block per row instead of a thread per row. At a 248,320-word
// vocabulary the reference gives each of 1,024 rows to one thread that scans it
// twice serially (77 ms a call); here 256 threads share a row (a fraction of
// a millisecond, the logits' read bandwidth).
//
//   stats[2n]   = max_v logits[n, v]
//   stats[2n+1] = sum_v exp(logits[n, v] - max)       (a row whose target is ignored: 0 and 1)
//
// The maximum is exact; the sum is added in a different order from the serial
// reference, so it agrees to fp32 summation error, not bit for bit.
//
// params: [n_rows, u_bins, ignore]. Bindings: uniform, logits, targets, stats.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_CE_THREADS 256

extern "C" __global__ void __launch_bounds__(BRAIN_CE_THREADS) brain_ce_stats(const unsigned int* params, const float* logits, const unsigned int* targets, float* stats) {
    const unsigned long long row = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned n_rows = params[0], bins = params[1], ignore = params[2];
    if (row >= n_rows) { return; }
    if (targets[row] == ignore) {
        if (threadIdx.x == 0) { stats[2 * row] = 0.0f; stats[2 * row + 1] = 1.0f; }
        return;
    }
    __shared__ float red[BRAIN_CE_THREADS / 32];
    const float* x = logits + row * bins;

    float m = -3.4e38f;
    for (unsigned c = threadIdx.x; c < bins; c += BRAIN_CE_THREADS) { m = fmaxf(m, x[c]); }
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = m; }
    __syncthreads();
    m = red[0];
    for (int w = 1; w < BRAIN_CE_THREADS / 32; ++w) { m = fmaxf(m, red[w]); }
    __syncthreads();

    float s = 0.0f;
    for (unsigned c = threadIdx.x; c < bins; c += BRAIN_CE_THREADS) { s += expf(x[c] - m); }
    for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = s; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < BRAIN_CE_THREADS / 32; ++w) { total += red[w]; }
        stats[2 * row] = m;
        stats[2 * row + 1] = total;
    }
}
