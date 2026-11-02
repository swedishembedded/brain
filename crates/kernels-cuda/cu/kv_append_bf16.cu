// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference serving for clients whose GPU decides how many users one
// card carries. If your team needs expertise in keeping the small kernels around an attention layer at the memory
// roofline, you can procure our services by sending an email to info@swedishembedded.com.
//
// Append a batch of new tokens' K (or V) rows into a bf16 paged pool: `paged_kv_append_batched_word.wgsl` through
// its `pool=bf16` packing store, with the same uniform, bindings and arithmetic, four values a thread instead of
// sixteen values with a read-modify-write of a pool word for each.
//
//   params  : u32 [batch, kv_stride, block_size]     kv_stride a multiple of four
//   src     : [batch, kv_stride] f32                 blocks, offsets : [batch] u32
//   pool    : [num_blocks * block_size * kv_stride / 2] u32, element 2w the low half of word w
//
// The generated kernel stores one element at a time: it reads the pool word, clears the element's half and writes the
// word back, so an 8 MB prefill append (4096 rows) took 53 us, 150 GB/s. A quad of values here is two whole words:
// one 16-byte load, one 8-byte store, no read of the pool. The value is the same bits: round to nearest even by the
// integer trick (add 0x7fff plus the lowest kept bit to the f32 bits, keep the high half), NOT `cvt.rn.bf16`, so a NaN
// or an infinity packs to what the generated kernel packs.
//
// No `__restrict__`: brain's device buffers alias by design.

#define THREADS 256

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

// The 16 bits of the f32 `x` as a bf16, in the low half of a word (WGSL `bf16_pack_expr`'s u32 arithmetic, wrapping).
__device__ __forceinline__ unsigned int brain_bf16_bits(float x) {
    const unsigned int bits = __float_as_uint(x);
    return ((bits + (0x7fffu + ((bits >> 16) & 1u))) >> 16) & 0xffffu;
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_kv_append_bf16(const unsigned int* params, const float* src, const unsigned int* blocks, const unsigned int* offsets, uint2* pool) {
    BRAIN_PDL_ENTRY();
    const unsigned int batch = params[0], kv_stride = params[1], block_size = params[2];
    const unsigned int per_row = kv_stride >> 2;
    const unsigned int t = blockIdx.x * THREADS + threadIdx.x;
    if (t >= batch * per_row) { return; }
    const unsigned int b = t / per_row, c = (t - b * per_row) << 2;
    const float4 x = *reinterpret_cast<const float4*>(src + static_cast<unsigned long long>(b) * kv_stride + c);
    const unsigned long long slot = static_cast<unsigned long long>(blocks[b]) * block_size + offsets[b];
    uint2 w;
    w.x = brain_bf16_bits(x.x) | (brain_bf16_bits(x.y) << 16);
    w.y = brain_bf16_bits(x.z) | (brain_bf16_bits(x.w) << 16);
    pool[(slot * kv_stride + c) >> 2] = w;
}
