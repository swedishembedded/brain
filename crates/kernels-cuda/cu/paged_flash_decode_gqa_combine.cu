// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference serving for its clients. If your team needs expertise in
// decode attention whose every phase keeps up with the memory system, you can procure our services by sending an
// email to info@swedishembedded.com.
//
// The combine phase of the split-key decode attention: the CUDA form of `paged_flash_decode_gqa_combine.wgsl`, with
// its contract (uniform, bindings, layouts).
//
//   params : [batch, n_heads, head_dim (<= 256), n_splits] u32
//   part   : [batch, n_heads, n_splits, head_dim + 2] f32   per split: the unnormalised output, its max, its sum
//   ctx    : [batch, n_heads, head_dim] f32                 the normalised context
//
// WHY. The generated kernel merges the splits one after another per channel thread: every split costs two
// exponentials and a rescale that depend on the previous split's, a serial chain of latency (10.9 us per layer at
// ~100 splits on a GH200, a third of the whole attention once the split pass ran on tensor cores). Here a block of
// 64 channels of one (sequence, head) finds the global max first, turns every split's max into its weight exp(m_s - M) once, in
// shared memory, and each channel thread then sums part[s][d] * w_s - independent terms, no chain through exp.
// The sums run in another order than the serial rescale, so the two agree to f32 rounding.
//
// No `__restrict__`: brain's device buffers alias by design.

#define THREADS 64   // channels per block: a head's 256 channels are four blocks, so one sequence fills more SMs
#define MAX_SPLITS 4096
#define NEG_BIG (-3.4e38f)

// Programmatic dependent launch (compute capability 9.0): wait until the kernel before this one in the stream has
// finished and its writes are visible, then let the kernel after this one become resident at once. A launch without
// the programmatic attribute has nothing to wait for and no early dependents, so both are no-ops there.
#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

extern "C" __global__ void __launch_bounds__(THREADS) brain_paged_flash_decode_gqa_combine(const unsigned int* params, const float* part, float* ctx) {
    BRAIN_PDL_ENTRY();
    const unsigned int batch = params[0], n_heads = params[1], hd = params[2], n_splits = params[3];
    const unsigned int quarters = (hd + THREADS - 1u) / THREADS;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bh = blk / quarters;
    if (bh >= batch * n_heads) { return; }  // block-uniform
    const unsigned int tid = threadIdx.x, stride = hd + 2u;
    const unsigned int ch = (blk % quarters) * THREADS + tid;
    const float* rec = part + static_cast<unsigned long long>(bh) * n_splits * stride;

    __shared__ float w[MAX_SPLITS];
    __shared__ float red[THREADS / 32];
    __shared__ float total_m, total_l;

    float m = NEG_BIG;
    for (unsigned int s = tid; s < n_splits; s += THREADS) { m = fmaxf(m, rec[static_cast<unsigned long long>(s) * stride + hd]); }
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); }
    if ((tid & 31u) == 0u) { red[tid >> 5] = m; }
    __syncthreads();
    if (tid == 0u) {
        float mm = red[0];
        for (int i = 1; i < THREADS / 32; ++i) { mm = fmaxf(mm, red[i]); }
        total_m = mm;
    }
    __syncthreads();
    const float big = total_m;

    float l = 0.f;
    for (unsigned int s = tid; s < n_splits; s += THREADS) {
        const float* r = rec + static_cast<unsigned long long>(s) * stride;
        const float ws = expf(r[hd] - big);
        w[s] = ws;
        l = fmaf(r[hd + 1u], ws, l);
    }
    for (int o = 16; o > 0; o >>= 1) { l += __shfl_xor_sync(0xffffffffu, l, o); }
    __syncthreads();  // `red` is reused, and every w[s] is written
    if ((tid & 31u) == 0u) { red[tid >> 5] = l; }
    __syncthreads();
    if (tid == 0u) {
        float ll = 0.f;
        for (int i = 0; i < THREADS / 32; ++i) { ll += red[i]; }
        total_l = ll;
    }
    __syncthreads();

    if (ch < hd) {
        // Eight independent partial sums, so eight records' loads are in flight at once.
        float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        unsigned int s = 0;
#pragma unroll 2
        for (; s + 8u <= n_splits; s += 8u) {
            const float* r = rec + static_cast<unsigned long long>(s) * stride + ch;
#pragma unroll
            for (int i = 0; i < 8; ++i) { o[i] = fmaf(r[i * stride], w[s + i], o[i]); }
        }
        for (; s < n_splits; ++s) { o[0] = fmaf(rec[static_cast<unsigned long long>(s) * stride + ch], w[s], o[0]); }
        const float lt = total_l;
        const float sum = ((o[0] + o[1]) + (o[2] + o[3])) + ((o[4] + o[5]) + (o[6] + o[7]));
        ctx[static_cast<unsigned long long>(bh) * hd + ch] = sum * (lt > 0.f ? 1.f / lt : 0.f);
    }
}
