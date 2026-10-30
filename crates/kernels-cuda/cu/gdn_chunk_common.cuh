// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core kernels for linear-attention language models for its clients. If your
// team needs expertise in running the small dense products and triangular solves of a chunked recurrence at the
// card's tensor-core rate, then you can procure our services by sending an email to info@swedishembedded.com.
//
// Device helpers the Gated DeltaNet chunk kernels share, prepended to their sources by the kernel registry (NVRTC
// takes one translation unit, so there is no `#include`): bf16 packing, `ldmatrix`, the bf16 `m16n8k16` MMA and the
// unit-lower-triangular inverse of the chunk's decay-weighted key Gram matrix.

__device__ __forceinline__ unsigned int brain_pack_bf16(float lo, float hi) {
    unsigned int d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));  // the first source takes the upper half
    return d;
}

__device__ __forceinline__ unsigned int brain_smem_addr(const void* p) { return static_cast<unsigned int>(__cvta_generic_to_shared(p)); }

// Four 8x8 b16 matrices from shared memory; lane l supplies the address of row (l & 7) of matrix (l >> 3).
__device__ __forceinline__ void brain_ldmatrix_x4(unsigned int (&r)[4], const void* lane_row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(brain_smem_addr(lane_row)));
}
__device__ __forceinline__ void brain_ldmatrix_x4_trans(unsigned int (&r)[4], const void* lane_row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(brain_smem_addr(lane_row)));
}

__device__ __forceinline__ void brain_mma_bf16(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// T = (I - A)^-1 for a strictly lower triangular A (the first `rows` rows of the 64 x 64 matrix `sA`, row stride `ld`
// floats) into `sT` (same stride, all zero on entry): T = I + A @ T, one panel of eight rows at a time. A panel takes
// everything from the finished panels above in one pass of fused multiply-adds (the k sum is split over four groups
// of 64 threads; each thread owns one column), then substitutes within the panel in registers: two barriers a panel
// instead of one per row. Each element sums its terms in a fixed order, fp32 throughout, never the Neumann series
// (which cancels catastrophically for near-parallel keys). `sRed` is [3][8][64] floats of scratch. Needs 256
// threads; the caller barriers before (A final) and after (T final) the call.
__device__ __forceinline__ void brain_unit_lower_inverse(const float* sA, float* sT, float* sRed, unsigned int ld, unsigned int rows) {
    const unsigned int tid = threadIdx.x, j = tid & 63u, grp = tid >> 6;
    for (unsigned int i0 = 0; i0 < rows; i0 += 8) {
        float r[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) { r[q] = 0.0f; }
        if (j < i0 + 8) {
            for (unsigned int k = 4 * grp; k < i0; k += 16) {
                const float x0 = sT[k * ld + j], x1 = sT[(k + 1) * ld + j], x2 = sT[(k + 2) * ld + j], x3 = sT[(k + 3) * ld + j];
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float4 a4 = *reinterpret_cast<const float4*>(sA + (i0 + q) * ld + k);
                    r[q] = fmaf(a4.x, x0, r[q]);
                    r[q] = fmaf(a4.y, x1, r[q]);
                    r[q] = fmaf(a4.z, x2, r[q]);
                    r[q] = fmaf(a4.w, x3, r[q]);
                }
            }
        }
        if (grp != 0) {
#pragma unroll
            for (int q = 0; q < 8; ++q) { sRed[((grp - 1) * 8 + q) * 64 + j] = r[q]; }
        }
        __syncthreads();
        if (grp == 0 && j < i0 + 8) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                float v = r[q] + sRed[q * 64 + j] + sRed[(8 + q) * 64 + j] + sRed[(16 + q) * 64 + j];
                if (j == i0 + q) { v += 1.0f; }
                r[q] = v;
            }
            // Within the panel: x[q] = r[q] + sum_{p < q} A[i0 + q][i0 + p] * x[p].
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                float v = r[q];
#pragma unroll
                for (int p = 0; p < q; ++p) { v = fmaf(sA[(i0 + q) * ld + i0 + p], r[p], v); }
                r[q] = v;
                if (i0 + q < rows) { sT[(i0 + q) * ld + j] = v; }
            }
        }
        __syncthreads();
    }
}
