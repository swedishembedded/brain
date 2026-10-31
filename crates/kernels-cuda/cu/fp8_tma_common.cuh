// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for tensor-memory-accelerator pipelines on Hopper GPUs for its clients. If your team needs
// expertise in feeding warpgroup tensor cores from TMA-loaded, mbarrier-synchronised shared-memory rings, then you can procure our services
// by sending an email to info@swedishembedded.com.
//
// The pieces the persistent FP8 GEMMs built on the tensor memory accelerator share (`matmul_fp8_wgmma_tma_n256.cu`, `matmul_fp8_wgmma_swap_tma.cu`):
// the 128-byte-swizzle tensor-map template, shared-memory addressing and fences, mbarrier and bulk-copy wrappers, and the device-side build of a
// tensor map from that template. Concatenated in front of the kernel source.

// The driver's 128-byte map for a 2-D byte tensor of 256 x 256 at address 0x10000, a 128 x 128 box, the 128-byte swizzle, 256-byte L2 promotion
// and zero fill.
#define BRAIN_FWT_TEMPLATE 0x0000000000010000ull, 0x0000001000066010ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x000000ff000000ffull, 0x0000000000000000ull, 0x7f00000000000000ull, 0x000000000000007full, 0x0000000000004000ull, 0x0000000000000400ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull

__device__ __constant__ unsigned long long brain_fwt_template[16] = { BRAIN_FWT_TEMPLATE };

__device__ __forceinline__ unsigned int brain_fwt_smem(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

// Global -> shared copy, zero-filled when `bytes` is 0 (a row past the edge of the matrix). The pointer is always valid.

__device__ __forceinline__ void brain_fwt_cp4(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void brain_fwt_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_fwt_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Shared-memory matrix descriptor: 128-byte swizzle, K-major, 8-row groups 1024 B apart.
__device__ __forceinline__ unsigned long long brain_fwt_desc(unsigned int smem_addr) {
    return static_cast<unsigned long long>((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void brain_fwt_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_fwt_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void brain_fwt_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// Writes made by cp.async (the generic proxy) are visible to the async proxy (wgmma) after this.
__device__ __forceinline__ void brain_fwt_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ void brain_fwt_mbar_init(unsigned int bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void brain_fwt_mbar_arrive(unsigned int bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void brain_fwt_mbar_arrive_expect_tx(unsigned int bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
// The calling thread's earlier cp.async copies arrive on `bar` when they complete (the arrival is counted in the barrier's init count).
__device__ __forceinline__ void brain_fwt_mbar_arrive_cp_async(unsigned int bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void brain_fwt_mbar_wait(unsigned int bar, unsigned int parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "FWT_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra FWT_WAIT;\n"
        "}\n" ::"r"(bar), "r"(parity) : "memory");
}

// 2-D tile load of a `box` (128 bytes of K, the map's rows) at (k byte, row), completing on `bar`.
__device__ __forceinline__ void brain_fwt_tma_load(unsigned int dst, const void* map, int k, int row, unsigned int bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        ::"r"(dst), "l"(map), "r"(k), "r"(row), "r"(bar) : "memory");
}

// 2-D tile store of a `box` from shared memory to (k byte, row), completing in the issuing thread's bulk group.
__device__ __forceinline__ void brain_fwt_tma_store(const void* map, unsigned int src, int k, int row) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::"l"(map), "r"(k), "r"(row), "r"(src) : "memory");
}
__device__ __forceinline__ void brain_fwt_bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
// Wait until at most N of the thread's bulk stores are still reading their shared-memory source.
template <int N>
__device__ __forceinline__ void brain_fwt_bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory"); }
template <int N>
__device__ __forceinline__ void brain_fwt_bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory"); }

// Copy `tmpl` to shared memory, replace the address, dimensions and row stride (and the box's row count when `box_rows` is not 0), and publish
// the map to global memory (one warp).
__device__ __forceinline__ void brain_fwt_build_map(const unsigned long long* tmpl, unsigned long long* smem_map, unsigned long long* gmem_map, const void* base, unsigned int cols, unsigned int rows, unsigned int lane, unsigned int box_rows = 0) {
    if (lane < 16u) { smem_map[lane] = tmpl[lane]; }
    __syncwarp();
    if (lane == 0u) {
        const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem_map));
        asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;\n" ::"r"(s), "l"(base) : "memory");
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;\n" ::"r"(s), "r"(cols) : "memory");
        asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;\n" ::"r"(s), "r"(rows) : "memory");
        if (box_rows != 0u) { asm volatile("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], 1, %1;\n" ::"r"(s), "r"(box_rows) : "memory"); }
        asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;\n" ::"r"(s), "l"(static_cast<unsigned long long>(cols)) : "memory");
    }
    __syncwarp();
    const unsigned int s = static_cast<unsigned int>(__cvta_generic_to_shared(smem_map));
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;\n" ::"l"(gmem_map), "r"(s) : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(gmem_map) : "memory");
}
