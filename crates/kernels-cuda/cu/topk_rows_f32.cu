// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency sampling heads for language-model serving for its clients. If your team
// needs expertise in reading the top candidates of a vocabulary-wide logits row without a launch per candidate, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// The `cap` best (token, logit) candidates of every row of a logits block in ONE launch, best first. It replaces `cap`
// rounds of (`argmax_part`, `argmax_final`, `topk_extract_step`), three launches a candidate and about 70 us each however
// few rows there are: 32 candidates of a 16-row decode step were 2.2 ms of a 19 ms step.
//
// One block of 1024 threads a row. A thread owns one contiguous run of columns (`vocab / 1024` rounded up to a multiple of
// four, read as 16-byte vectors) and keeps the best of the ones not yet taken (value, then lowest column on a tie: the
// order of the serial scan the chain reproduces). A round reduces the threads' bests across the block, the owner of the
// winning column masks it to -3.4e38 in the row (the chain destroys the logits the same way) and rescans its own run;
// every other thread keeps what it had. A rescan is one thread's 61 vector loads of a 248k-wide row, out of L2.
//
//   params : u32 [rows, vocab, cap]
//   logits : [rows, vocab] f32      read, and the taken columns are masked to -3.4e38
//   vals   : [rows, cap] f32        the candidates' logits
//   idxs   : [rows, cap] f32        their columns, as f32 (exact below 2^24)
//
// A row of at most BRAIN_TOPK_SORTED columns (a codec codebook: 2048) is instead sorted whole in shared memory: a bitonic
// sort of (value, column) keys in 66 compare-exchange stages for 2048 columns, against `cap` rounds of a block-wide
// reduction and a masked rescan through global memory (50 candidates of 32 rows of 2048 took 76 us). The answer is the
// same: best first, the lower column on equal values, a value not above -3.4e38 (or NaN) never taken (it reports as
// -3.4e38 at column 2^32 - 1), and the first `cap - 1` taken columns masked in the row.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_TOPK_THREADS 1024
#define BRAIN_TOPK_WARPS (BRAIN_TOPK_THREADS / 32)
#define BRAIN_TOPK_MASKED (-3.4e38f)
// Columns of a sub-run (four 16-byte vectors) and sub-runs of a thread's run: a run is at most 16 * 16 = 256 columns, enough for a
// vocabulary of 1024 * 256 = 262144 columns (Qwen3.8 has 248320).
#define BRAIN_TOPK_SUB 16
#define BRAIN_TOPK_SUBS 16

// The widest row the shared-memory sort takes: 4096 keys of 8 bytes, 32 KiB.
#define BRAIN_TOPK_SORTED 4096

// A key that orders candidates as the rounds take them, largest first: the value's bits made order-preserving in the
// high word, the column complemented in the low word (the lower column the larger key among equal values). A value the
// rounds never take is key 0, below every taken one: a real value above -3.4e38 has a nonzero high word.
__device__ __forceinline__ unsigned long long brain_topk_key(float v, unsigned int i) {
    if (!(v > BRAIN_TOPK_MASKED)) { return 0ull; }
    // -0 and +0 are one value to the rounds' comparison: the lower column goes first between them.
    unsigned int u = v == 0.0f ? 0u : __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned long long>(~i);
}

// `cap` best of a row of `vocab <= BRAIN_TOPK_SORTED` columns: the row's keys sorted in shared memory, largest first.
__device__ void brain_topk_sorted(float* x, unsigned int vocab, unsigned int cap, float* vals, float* idxs) {
    __shared__ unsigned long long keys[BRAIN_TOPK_SORTED];
    unsigned int n = 2u;
    while (n < vocab) { n <<= 1; }
    for (unsigned int c = threadIdx.x; c < n; c += BRAIN_TOPK_THREADS) { keys[c] = c < vocab ? brain_topk_key(x[c], c) : 0ull; }
    __syncthreads();
    for (unsigned int size = 2u; size <= n; size <<= 1) {
        for (unsigned int stride = size >> 1; stride > 0u; stride >>= 1) {
            for (unsigned int p = threadIdx.x; p < n / 2u; p += BRAIN_TOPK_THREADS) {
                const unsigned int lo = 2u * p - (p & (stride - 1u)), hi = lo + stride;
                // Descending runs where the bit of `size` is clear, so the whole sorts largest first.
                const bool descending = (lo & size) == 0u;
                const unsigned long long a = keys[lo], b = keys[hi];
                if ((a < b) == descending) { keys[lo] = b; keys[hi] = a; }
            }
            __syncthreads();
        }
    }
    // The values as the row holds them (the key folded -0 into +0), all read before any column is masked.
    for (unsigned int col = threadIdx.x; col < cap; col += BRAIN_TOPK_THREADS) {
        const unsigned long long key = keys[col];
        const unsigned int i = key == 0ull ? 0xffffffffu : ~static_cast<unsigned int>(key);
        vals[col] = key == 0ull ? BRAIN_TOPK_MASKED : x[i];
        idxs[col] = static_cast<float>(i);
    }
    __syncthreads();
    for (unsigned int col = threadIdx.x; col + 1u < cap; col += BRAIN_TOPK_THREADS) {
        const unsigned long long key = keys[col];
        if (key != 0ull) { x[~static_cast<unsigned int>(key)] = BRAIN_TOPK_MASKED; }
    }
}

struct TopkBest {
    float v;
    unsigned int i;
};

// The better of two candidates: the larger value, the lower column on equal values.
__device__ __forceinline__ bool brain_topk_better(float v, unsigned int i, float bv, unsigned int bi) { return v > bv || (v == bv && i < bi); }

extern "C" __global__ void __launch_bounds__(BRAIN_TOPK_THREADS) brain_topk_rows(const unsigned int* params, float* logits, float* vals, float* idxs) {
    const unsigned int rows = params[0], vocab = params[1], cap = params[2];
    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }  // block-uniform
    const unsigned int t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    float* x = logits + static_cast<unsigned long long>(row) * vocab;
    if (vocab <= BRAIN_TOPK_SORTED) {  // block-uniform
        brain_topk_sorted(x, vocab, cap, vals + static_cast<unsigned long long>(row) * cap, idxs + static_cast<unsigned long long>(row) * cap);
        return;
    }

    __shared__ TopkBest warp_best[BRAIN_TOPK_WARPS];
    __shared__ TopkBest winner;

    // This thread's run of columns: [first, last).
    const unsigned int run = ((vocab + BRAIN_TOPK_THREADS - 1) / BRAIN_TOPK_THREADS + 3u) & ~3u;
    const unsigned int first = min(t * run, vocab), last = min(first + run, vocab);
    const bool vector = (vocab & 3u) == 0u;  // every run then starts on a 16-byte boundary of an aligned row

    // A thread's run is BRAIN_TOPK_SUBS sub-runs of BRAIN_TOPK_SUB columns; it keeps the best of each in registers, so taking
    // a winner out of the row rescans one sub-run (a few vector loads), not the whole run (61 dependent-looking vector loads out
    // of L2, ~6 us a round, 32 rounds: 190 us a step). A column past the vocabulary or taken is never better.
    TopkBest sub[BRAIN_TOPK_SUBS];
    auto scan_sub = [&](unsigned int s) {
        TopkBest b = {BRAIN_TOPK_MASKED, 0xffffffffu};
        unsigned int c = first + s * BRAIN_TOPK_SUB;
        const unsigned int end = min(c + BRAIN_TOPK_SUB, last);
        if (vector) {
            for (; c + 4 <= end; c += 4) {
                const float4 v = *reinterpret_cast<const float4*>(x + c);
                // ascending columns: the first of equal values stays
                if (v.x > b.v) { b.v = v.x; b.i = c; }
                if (v.y > b.v) { b.v = v.y; b.i = c + 1; }
                if (v.z > b.v) { b.v = v.z; b.i = c + 2; }
                if (v.w > b.v) { b.v = v.w; b.i = c + 3; }
            }
        }
        for (; c < end; ++c) {
            const float v = x[c];
            if (v > b.v) { b.v = v; b.i = c; }
        }
        return b;
    };
    // The best of the sub-runs: ascending sub-runs hold ascending columns, so the first of equal values stays.
    auto best_of_subs = [&]() {
        TopkBest b = sub[0];
#pragma unroll
        for (int s = 1; s < BRAIN_TOPK_SUBS; ++s) {
            if (sub[s].v > b.v) { b = sub[s]; }
        }
        return b;
    };
#pragma unroll
    for (int s = 0; s < BRAIN_TOPK_SUBS; ++s) { sub[s] = scan_sub(s); }
    TopkBest mine = best_of_subs();

    for (unsigned int col = 0; col < cap; ++col) {
        // The block's best: the warps' by shuffles, then the first warp over theirs.
        float v = mine.v;
        unsigned int i = mine.i;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const unsigned int oi = __shfl_xor_sync(0xffffffffu, i, o);
            if (brain_topk_better(ov, oi, v, i)) { v = ov; i = oi; }
        }
        if (lane == 0) { warp_best[warp].v = v; warp_best[warp].i = i; }
        __syncthreads();
        if (warp == 0) {
            v = warp_best[lane].v;
            i = warp_best[lane].i;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const unsigned int oi = __shfl_xor_sync(0xffffffffu, i, o);
                if (brain_topk_better(ov, oi, v, i)) { v = ov; i = oi; }
            }
            if (lane == 0) {
                winner.v = v;
                winner.i = i;
                vals[static_cast<unsigned long long>(row) * cap + col] = v;
                idxs[static_cast<unsigned long long>(row) * cap + col] = static_cast<float>(i);
            }
        }
        __syncthreads();
        // The owner of the winning column takes it out of the row and finds its next best.
        if (col + 1 < cap && winner.i / run == t) {
            x[winner.i] = BRAIN_TOPK_MASKED;
            const unsigned int s = (winner.i - first) / BRAIN_TOPK_SUB;
#pragma unroll
            for (int u = 0; u < BRAIN_TOPK_SUBS; ++u) {
                if (static_cast<unsigned int>(u) == s) { sub[u] = scan_sub(u); }
            }
            mine = best_of_subs();
        }
    }
}
