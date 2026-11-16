// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements exact on-device sampling for language-model serving for its clients. If your team
// needs expertise in drawing a served model's tokens without reading its vocabulary back, then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// The keyed Gumbel-max draw of every row of a logits block in ONE launch: the native candidate for the WGSL pair
// `draw_part` + `draw_final` (two launches, two more for a min-p floor), with the same answer. A row's token is
//
//   argmax_t (logit_t / temp - ln(-ln(u_t)))   over the row's survivors,
//
// a draw from the softmax over them, with u_t the keyed uniform `model::serve::token_uniform` defines (the same integer
// arithmetic, so the same token), and its log-probability under that softmax.
//
// One block of 1024 threads a row; thread t reads columns t, t + 1024, ... (coalesced), keeping its slice's maximum of
// logit/temp, the slice's mass relative to it and its best perturbed score. The block folds the slices; a row with a
// min-p floor sweeps the row again (out of L2) keeping only the tokens whose weight relative to the row's best is at
// least min_p, and folds again.
//
//   params : u32 [rows, vocab]
//   logits : [rows, vocab] f32
//   spec   : [rows, 4] u32: temperature bits, min-p bits, key, draw flag (0 = row not drawn, left as it was)
//   out    : [rows, 4] f32: the drawn column (as f32, exact below 2^24), its log-probability, the row's best
//            logit/temp and its mass sum_t exp(logit_t / temp - best) (the statistics a candidate list is answered with)
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_DRAW_THREADS 1024
#define BRAIN_DRAW_WARPS (BRAIN_DRAW_THREADS / 32)
#define BRAIN_DRAW_NONE (-3.402823e38f)

struct DrawAcc {
    float mx;    // the slice's maximum of logit / temp
    float mass;  // sum of exp(logit / temp - mx) over the slice's survivors
    float key;   // the best perturbed score
    unsigned int idx;
};

__device__ __forceinline__ unsigned int brain_draw_mix(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// The fold of two slices: the larger maximum with both masses rescaled to it, the better score (the lower column on a
// tie).
__device__ __forceinline__ DrawAcc brain_draw_fold(DrawAcc a, DrawAcc b) {
    DrawAcc r;
    r.mx = fmaxf(a.mx, b.mx);
    r.mass = (a.mass > 0.0f ? a.mass * expf(a.mx - r.mx) : 0.0f) + (b.mass > 0.0f ? b.mass * expf(b.mx - r.mx) : 0.0f);
    const bool take_b = b.key > a.key || (b.key == a.key && b.idx < a.idx);
    r.key = take_b ? b.key : a.key;
    r.idx = take_b ? b.idx : a.idx;
    return r;
}

// One sweep of thread t's columns: every token of positive weight, or with `floor_on` those whose weight
// exp(logit/temp - top) is at least `min_p`.
__device__ __forceinline__ DrawAcc brain_draw_sweep(const float* x, unsigned int vocab, float temp, unsigned int key, bool floor_on, float top, float min_p) {
    DrawAcc a = {BRAIN_DRAW_NONE, 0.0f, BRAIN_DRAW_NONE, 0u};
    for (unsigned int i = threadIdx.x; i < vocab; i += BRAIN_DRAW_THREADS) {
        const float s = x[i] / temp;
        if (!(s > BRAIN_DRAW_NONE)) { continue; }  // NaN and -inf weigh nothing
        if (floor_on) {
            const float w = expf(s - top);
            if (!(w > 0.0f && w >= min_p)) { continue; }
        }
        if (s > a.mx) {
            a.mass = a.mass * expf(a.mx - s) + 1.0f;
            a.mx = s;
        } else {
            a.mass += expf(s - a.mx);
        }
        const unsigned int h = brain_draw_mix(key ^ brain_draw_mix(i + 0x9e3779b9u));
        const float u = (static_cast<float>(h >> 9) + 0.5f) * (1.0f / 8388608.0f);
        const float k = s - logf(-logf(u));
        if (k > a.key) {  // ascending columns: the first of equal scores stays
            a.key = k;
            a.idx = i;
        }
    }
    return a;
}

// The block's fold of every thread's accumulator, returned to every thread.
__device__ __forceinline__ DrawAcc brain_draw_block(DrawAcc a, DrawAcc* shared) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (unsigned int o = 16; o > 0; o >>= 1) {
        DrawAcc b;
        b.mx = __shfl_down_sync(0xffffffffu, a.mx, o);
        b.mass = __shfl_down_sync(0xffffffffu, a.mass, o);
        b.key = __shfl_down_sync(0xffffffffu, a.key, o);
        b.idx = __shfl_down_sync(0xffffffffu, a.idx, o);
        a = brain_draw_fold(a, b);
    }
    if (lane == 0) { shared[warp] = a; }
    __syncthreads();
    if (warp == 0) {
        a = shared[lane];
        for (unsigned int o = 16; o > 0; o >>= 1) {
            DrawAcc b;
            b.mx = __shfl_down_sync(0xffffffffu, a.mx, o);
            b.mass = __shfl_down_sync(0xffffffffu, a.mass, o);
            b.key = __shfl_down_sync(0xffffffffu, a.key, o);
            b.idx = __shfl_down_sync(0xffffffffu, a.idx, o);
            a = brain_draw_fold(a, b);
        }
        if (lane == 0) { shared[BRAIN_DRAW_WARPS] = a; }
    }
    __syncthreads();
    const DrawAcc r = shared[BRAIN_DRAW_WARPS];
    __syncthreads();
    return r;
}

extern "C" __global__ void __launch_bounds__(BRAIN_DRAW_THREADS) brain_draw_rows(const unsigned int* params, const float* logits, const unsigned int* spec, float* out) {
    const unsigned int rows = params[0], vocab = params[1];
    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }  // block-uniform
    const unsigned int* sp = spec + 4ull * row;
    if (sp[3] == 0u) { return; }  // block-uniform: the row is not drawn
    const float temp = fmaxf(__uint_as_float(sp[0]), 1e-6f);
    const float min_p = __uint_as_float(sp[1]);
    const unsigned int key = sp[2];
    const float* x = logits + static_cast<unsigned long long>(row) * vocab;

    __shared__ DrawAcc shared[BRAIN_DRAW_WARPS + 1];
    DrawAcc r = brain_draw_block(brain_draw_sweep(x, vocab, temp, key, false, 0.0f, 0.0f), shared);
    const float top = r.mx, mass = r.mass;
    if (min_p > 0.0f) {
        r = brain_draw_block(brain_draw_sweep(x, vocab, temp, key, true, top, min_p), shared);
    }
    if (threadIdx.x == 0) {
        float* o = out + 4ull * row;
        o[2] = top;
        o[3] = mass;
        if (!(r.mass > 0.0f)) {
            o[0] = 0.0f;
            o[1] = 0.0f;
        } else {
            o[0] = static_cast<float>(r.idx);
            o[1] = (x[r.idx] / temp - top) - logf(r.mass);
        }
    }
}
