// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements attention kernels on the memory roofline for
// its clients. If your team needs expertise in bandwidth-bound passes that
// read their data once then you can procure our services by sending an email
// to info@swedishembedded.com.
//
// Row softmax over contiguous rows, fp32 - the hand-written form of
// `softmax_rows.wgsl`, with its uniform and bindings:
//
//   params : u32 [rows, cols]
//   scores : [rows, cols] f32
//   probs  : [rows, cols] f32   probs = exp(scores - max) / sum
//
// BIT-IDENTICAL to the WGSL kernel. Both give a row 64 lanes; lane t owns
// columns t, t + 64, t + 128, ... in that order. The row maximum is exact in
// any order. Each lane sums its own exponentials in ascending column order,
// and the 64 lane partials are folded in lane order; the exponential is
// `expf` and the normaliser one IEEE division `1 / max(total, 1e-38)`, and the
// backend's `--fmad=false` keeps every operation separately rounded on both
// sides - so every bit agrees.
//
// What differs is the traffic. The WGSL kernel reads the row three times and
// writes it twice (max, exponentials written out, then read back and scaled);
// here a lane holds its columns in registers, so the row is read once and
// written once. The register count is a compile-time bucket (`KMAX` values a
// lane), one entry point per bucket, and `gpu_core::native_upgrade` picks the
// smallest bucket that holds the row: a worst-case bucket would make a short
// row pay for every masked-off slot.
//
// A 256-thread block covers four rows. No `__restrict__`: brain's device
// buffers alias by design (and the row is in registers before anything is
// written, so an in-place call is safe too).

#define BRAIN_SMR_LANES 64
#define BRAIN_SMR_ROWS 4

template <int KMAX>
__device__ __forceinline__ void brain_softmax_rows_body(const unsigned int* params, const float* scores, float* probs) {
    __shared__ float partial[BRAIN_SMR_ROWS][BRAIN_SMR_LANES];
    const unsigned int rows = params[0], cols = params[1];
    const unsigned int sub = threadIdx.x / BRAIN_SMR_LANES;  // which row of the block
    const unsigned int t = threadIdx.x % BRAIN_SMR_LANES;
    const unsigned int row = (blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_SMR_ROWS + sub;
    // A row past the end still takes part in the barriers below.
    const bool live = row < rows;
    const size_t base = (size_t)row * cols;

    float v[KMAX];
    float mx = -3.4e38f;
#pragma unroll
    for (int j = 0; j < KMAX; ++j) {
        const unsigned int c = t + BRAIN_SMR_LANES * j;
        v[j] = (live && c < cols) ? scores[base + c] : -3.4e38f;
        if (live && c < cols) { mx = fmaxf(mx, v[j]); }
    }
    partial[sub][t] = mx;
    __syncthreads();
    float rowmax = -3.4e38f;
    for (int i = 0; i < BRAIN_SMR_LANES; ++i) { rowmax = fmaxf(rowmax, partial[sub][i]); }
    __syncthreads();

    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < KMAX; ++j) {
        const unsigned int c = t + BRAIN_SMR_LANES * j;
        if (live && c < cols) {
            v[j] = expf(v[j] - rowmax);
            sum = sum + v[j];
        }
    }
    partial[sub][t] = sum;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < BRAIN_SMR_LANES; ++i) { total = total + partial[sub][i]; }
    const float inv = 1.0f / fmaxf(total, 1e-38f);
    if (!live) { return; }
#pragma unroll
    for (int j = 0; j < KMAX; ++j) {
        const unsigned int c = t + BRAIN_SMR_LANES * j;
        if (c < cols) { probs[base + c] = v[j] * inv; }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_SMR_LANES * BRAIN_SMR_ROWS)
brain_softmax_rows_k8(const unsigned int* params, const float* scores, float* probs) {
    brain_softmax_rows_body<8>(params, scores, probs);
}

extern "C" __global__ void __launch_bounds__(BRAIN_SMR_LANES * BRAIN_SMR_ROWS)
brain_softmax_rows_k16(const unsigned int* params, const float* scores, float* probs) {
    brain_softmax_rows_body<16>(params, scores, probs);
}

extern "C" __global__ void __launch_bounds__(BRAIN_SMR_LANES * BRAIN_SMR_ROWS)
brain_softmax_rows_k32(const unsigned int* params, const float* scores, float* probs) {
    brain_softmax_rows_body<32>(params, scores, probs);
}

extern "C" __global__ void __launch_bounds__(BRAIN_SMR_LANES * BRAIN_SMR_ROWS)
brain_softmax_rows_k64(const unsigned int* params, const float* scores, float* probs) {
    brain_softmax_rows_body<64>(params, scores, probs);
}
