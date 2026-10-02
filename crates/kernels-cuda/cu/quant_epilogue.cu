// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels between two weight streams into one launch, then you can
// procure our services by sending an email to info@swedishembedded.com.
//
// The activation back end of a decode step's mixer or MLP, in a single launch:
// produce the activation that feeds the next int8 linear AND quantise it.
//
//   y  = f(a, b)                                   f picked by `mode`
//   sx = max(max|y|, 1e-8) / 127                   (per-row activation scale)
//   xq = pack4(clamp(rint(y / sx), -127, 127))     (four int8 per u32)
//
//   mode 0 : y = a                    (a plain activation that only needs quantising)
//   mode 1 : y = silu(a) * b          (the SwiGLU product feeding `down_proj`)
//   mode 2 : y = a * sigmoid(b)       (attention output times its sigmoid gate)
//
// Each replaces a chain of kernels a decode token runs dozens of times
// (`silu_mul` / `sigmoid` + `mul`, then `max_abs_rows`, then `quant_pack`) and
// produces what that chain produces to the last bit.
//
//   params : u32 [k, rows, mode, 0]
//   a, b   : [rows, k] f32   (b is not read in mode 0)
//   y      : [rows, k] f32   the activation itself (an fp32 linear reads it)
//   xq     : [rows, k/4] u32 packed int8 activation
//   sx     : [rows] f32      per-row scale
//
// The arithmetic is the reference kernels' own expressions: silu is
// `x / (1.0 + exp(-x))` and sigmoid `1.0 / (1.0 + exp(-x))` with `expf` (the
// WGSL translator's `exp`), products and sums are explicit round-to-nearest,
// the max is exact in any order, `/` and `rint` are correctly rounded (WGSL
// `round` is half-to-even, i.e. `rintf`). Nothing here is order-sensitive, so
// the result is the chain's, bit for bit, however the row is split over threads.
//
// One 256-thread block per row; a thread keeps its elements in registers from
// the single read to the quantiser, and the four int8 of an output word, which
// belong to four different threads, are exchanged through shared memory so
// every store is a whole coalesced word. A row wider than
// BRAIN_QE_THREADS * BRAIN_QE_MAX_ELEMS is not served; the host keeps the WGSL
// chain for it.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_QE_THREADS 256
// Elements one thread keeps in registers: rows up to 256 * this wide. 72 covers
// 18432, which holds Qwen3.8-27B's widest decode activation, the 17408-wide MLP
// hidden; two such arrays per thread are what the register budget allows.
#define BRAIN_QE_MAX_ELEMS 72

extern "C" __global__ void __launch_bounds__(BRAIN_QE_THREADS)
brain_quant_epilogue(const unsigned int* params, const float* a, const float* b, float* y,
                     unsigned int* xq, float* sx) {
    const unsigned int k = params[0];
    const unsigned int rows = params[1];
    const unsigned int mode = params[2];

    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }  // block-uniform
    const unsigned int t = threadIdx.x;
    const unsigned long long base = (unsigned long long)row * k;

    __shared__ float warp_max[BRAIN_QE_THREADS / 32];
    __shared__ unsigned int qwords[BRAIN_QE_THREADS * BRAIN_QE_MAX_ELEMS / 4];
    signed char* qbytes = reinterpret_cast<signed char*>(qwords);

    // Phase 1: every load, nothing else - a load that has to wait for an
    // earlier store through a possibly aliasing pointer, or whose dependent
    // arithmetic sits right behind it, is a serialised round trip to L2 (the
    // lesson `add_rms_quant.cu` records). Both operands load unconditionally
    // apart from the row tail and mode 0's absent `b`.
    float v[BRAIN_QE_MAX_ELEMS];
    float w[BRAIN_QE_MAX_ELEMS];
    const bool two = mode != 0u;
#pragma unroll
    for (int i = 0; i < BRAIN_QE_MAX_ELEMS; ++i) {
        const unsigned int c = t + BRAIN_QE_THREADS * i;
        const bool live = c < k;
        v[i] = live ? a[base + c] : 0.0f;
        w[i] = (live && two) ? b[base + c] : 0.0f;
    }

    // Phase 2: the producer and the running max, in registers.
    float mx = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_QE_MAX_ELEMS; ++i) {
        float r = v[i];
        if (mode == 1u) {
            // silu(a) * b:  a / (1 + exp(-a)) * b
            r = __fmul_rn(__fdiv_rn(r, __fadd_rn(1.0f, expf(-r))), w[i]);
        } else if (mode == 2u) {
            // a * sigmoid(b):  a * (1 / (1 + exp(-b)))
            r = __fmul_rn(r, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-w[i]))));
        }
        v[i] = r;
        mx = fmaxf(mx, fabsf(r));
    }

    // Row max: exact in any order.
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
    if ((t & 31u) == 0u) { warp_max[t >> 5] = mx; }
    __syncthreads();
    float rowmax = warp_max[0];
#pragma unroll
    for (int i = 1; i < BRAIN_QE_THREADS / 32; ++i) { rowmax = fmaxf(rowmax, warp_max[i]); }
    const float scale = __fdiv_rn(fmaxf(rowmax, 1e-8f), 127.0f);
    if (t == 0) { sx[row] = scale; }
    const float rscale = __fdiv_rn(1.0f, scale);

    // Phase 3: the stores - the product, the quantised bytes into shared
    // memory, then whole packed words.
#pragma unroll
    for (int i = 0; i < BRAIN_QE_MAX_ELEMS; ++i) {
        const unsigned int c = t + BRAIN_QE_THREADS * i;
        if (c < k) {
            if (two) { y[base + c] = v[i]; }
            const float q = fminf(fmaxf(rintf(__fmul_rn(v[i], rscale)), -127.0f), 127.0f);
            qbytes[c] = static_cast<signed char>(static_cast<int>(q));
        }
    }
    __syncthreads();
    const unsigned int words = k >> 2;
    for (unsigned int g = t; g < words; g += BRAIN_QE_THREADS) {
        xq[(unsigned long long)row * words + g] = qwords[g];
    }
}
