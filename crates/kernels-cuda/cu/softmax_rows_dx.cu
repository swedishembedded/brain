// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements attention training kernels for GPUs for its
// clients. If your team needs expertise in the backward pass of large
// attention maps then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The softmax Jacobian over contiguous rows, one warp per row:
//
//   params : u32 [rows, k]
//   y      : [rows, k] f32   the softmax OUTPUT (probabilities)
//   dy     : [rows, k] f32
//   dx     : [rows, k] f32   dx = y * (dy - sum_j dy_j * y_j)
//
// `softmax_k_dx.wgsl` with `M = 1` computes the same thing with one thread per
// row walking the whole row: neighbouring threads read addresses a row apart,
// so every load is a scattered sector, and a 2560-wide row is a 2560-long
// dependent chain. Here a warp reads its row in 16-byte coalesced pieces,
// reduces the dot product with shuffles and writes the row back - the
// bandwidth-bound form. The reduction order differs from the serial one, so
// this kernel is not a drop-in for that one's callers; it is asked for by
// name by a trainer whose contract is agreement with its host reference.
//
// No `__restrict__`: brain's device buffers alias by design.

#define BRAIN_SMDX_WARPS 8

extern "C" __global__ void __launch_bounds__(BRAIN_SMDX_WARPS * 32)
brain_softmax_rows_dx(const unsigned int* params, const float* y, const float* dy, float* dx) {
    const unsigned int rows = params[0];
    const unsigned int k = params[1];
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int row = (blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_SMDX_WARPS + threadIdx.x / 32;
    if (row >= rows) { return; }
    const float* yr = y + (size_t)row * k;
    const float* dyr = dy + (size_t)row * k;
    float* dxr = dx + (size_t)row * k;
    // 16-byte pieces where the row allows them (each row starts on a 16-byte
    // boundary when k and the bases do), single floats otherwise.
    const bool vec = k % 4 == 0 && ((reinterpret_cast<size_t>(y) | reinterpret_cast<size_t>(dy) | reinterpret_cast<size_t>(dx)) & 15u) == 0;
    float dot = 0.0f;
    if (vec) {
        const float4* y4 = reinterpret_cast<const float4*>(yr);
        const float4* d4 = reinterpret_cast<const float4*>(dyr);
        for (unsigned int i = lane; i < k / 4; i += 32) {
            const float4 a = __ldg(y4 + i);
            const float4 b = __ldg(d4 + i);
            dot = fmaf(a.x, b.x, dot);
            dot = fmaf(a.y, b.y, dot);
            dot = fmaf(a.z, b.z, dot);
            dot = fmaf(a.w, b.w, dot);
        }
    } else {
        for (unsigned int i = lane; i < k; i += 32) { dot = fmaf(__ldg(yr + i), __ldg(dyr + i), dot); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { dot += __shfl_xor_sync(0xffffffffu, dot, o); }
    if (vec) {
        const float4* y4 = reinterpret_cast<const float4*>(yr);
        const float4* d4 = reinterpret_cast<const float4*>(dyr);
        float4* x4 = reinterpret_cast<float4*>(dxr);
        for (unsigned int i = lane; i < k / 4; i += 32) {
            const float4 a = __ldg(y4 + i);
            const float4 b = __ldg(d4 + i);
            x4[i] = make_float4(a.x * (b.x - dot), a.y * (b.y - dot), a.z * (b.z - dot), a.w * (b.w - dot));
        }
    } else {
        for (unsigned int i = lane; i < k; i += 32) { dxr[i] = __ldg(yr + i) * (__ldg(dyr + i) - dot); }
    }
}
