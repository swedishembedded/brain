// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements latency-bound LLM decode on GPUs for its clients. If your team needs expertise in
// removing the serial tails of the small kernels between two weight streams, then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Append a batch of new tokens' K (or V) rows into an int8 paged pool, with a calibrated clip ceiling on the
// per-(token, kv head) scale: `paged_kv_append_i8_clipped_batched.wgsl` with the same uniform, bindings and
// arithmetic, one WARP per (token, kv head) instead of one thread.
//
//   params  : u32 [batch, kv_stride, block_size, head_dim]
//   src     : [batch, kv_stride] f32          blocks, offsets : [batch] u32   clip : [n_kv] f32
//   pool    : int8 packed four to a u32 word  scales : [num_blocks * block_size * n_kv] f32
//
// The reference's thread walks the head twice serially (a 128-step max, then a 128-step quantise); here the lanes
// split both walks. Bit-identical to the reference: the absolute maximum is order-independent, the scale is
// min(max, clip) / 127 (1 for zero), and each value is clamp(rint(src / scale), -127, 127) with a correctly rounded
// division and round-half-to-even, the WGSL `round`.
//
// No `__restrict__`: brain's device buffers alias by design.

#define WARPS 4

extern "C" __global__ void __launch_bounds__(WARPS * 32) brain_kv_append_i8_clipped(const unsigned int* params, const float* src, const unsigned int* blocks, const unsigned int* offsets, const float* clip, unsigned int* pool, float* scales) {
    const unsigned int batch = params[0], kv_stride = params[1], block_size = params[2], hd = params[3];
    const unsigned int n_kv = kv_stride / hd;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int idx = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (idx >= batch * n_kv) { return; }   // warp-uniform
    const unsigned int b = idx / n_kv, head = idx % n_kv;
    const unsigned long long kbase = (unsigned long long)b * kv_stride + (unsigned long long)head * hd;

    float mx = 0.f;
    for (unsigned int i = lane; i < hd; i += 32u) { mx = fmaxf(mx, fabsf(src[kbase + i])); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
    float scale = fminf(mx, clip[head]) / 127.0f;
    if (scale == 0.0f) { scale = 1.0f; }
    const unsigned long long slot = (unsigned long long)blocks[b] * block_size + offsets[b];
    if (lane == 0u) { scales[slot * n_kv + head] = scale; }
    const unsigned long long base_i8 = slot * kv_stride + (unsigned long long)head * hd;
    for (unsigned int i = lane * 4u; i < hd; i += 128u) {
        unsigned int packed = 0u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = max(-127, min(127, static_cast<int>(rintf(src[kbase + i + j] / scale))));
            packed |= (static_cast<unsigned int>(q) & 0xffu) << (8 * j);
        }
        pool[(base_i8 + i) / 4u] = packed;
    }
}
