// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core inference kernels for linear-attention language models for its clients. If your
// team needs expertise in driving Hopper warpgroup tensor cores from hand-written kernels (shared-memory descriptors,
// register-resident operands, asynchronous operand staging), then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Helpers the Gated DeltaNet warpgroup kernels share, prepended to their sources by the kernel registry (NVRTC takes one
// translation unit): `cp.async` staging, proxy fences, `wgmma` issue and the bf16 wrappers for the shapes they use, and the
// 128-byte-swizzle shared-memory matrix descriptors.

#define ATOM_BYTES 8192         // one 128-byte-swizzle atom column of a 64-row tile

__device__ __forceinline__ void gw_named_barrier(unsigned int id, unsigned int threads) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory"); }
__device__ __forceinline__ void gw_cp16(unsigned int dst, const void* src, unsigned int bytes) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory"); }
__device__ __forceinline__ void gw_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void gw_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void gw_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void gw_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void gw_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void gw_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }

// 128-byte-swizzle shared-memory matrix descriptors. K-major operand: 8-row groups 1024 bytes apart. MN-major operand: the
// 64-value atom columns of the N dimension 8192 bytes apart, the 8-row groups of K 1024 bytes apart.
__device__ __forceinline__ unsigned long long gw_desc_k(unsigned int addr) { return static_cast<unsigned long long>((addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62); }
__device__ __forceinline__ unsigned long long gw_desc_mn(unsigned int addr) { return static_cast<unsigned long long>((addr & 0x3FFFFu) >> 4) | (512ull << 16) | (64ull << 32) | (1ull << 62); }

// D (64 x 64 f32) (=|+=) (+|-) A (64 x 16 bf16, this thread's four registers) . B (64 x 16 bf16, K-major in shared memory)^T.
template <int SCALE_A>
__device__ __forceinline__ void gw_mma_n64(float (&d)[32], const unsigned int (&a)[4], unsigned long long db, bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23,"
        " %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, %38, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
          "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(static_cast<int>(accumulate)), "n"(SCALE_A));
}

// D (64 x 128 f32) += A (64 x 16 bf16 registers) . B (16 x 128 bf16, N-major in shared memory: the transposed-B form).
__device__ __forceinline__ void gw_mma_n128_acc(float (&d)[64], const unsigned int (&a)[4], unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23,"
        " %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
        " %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
          "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]),
          "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
          "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]),
          "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}


