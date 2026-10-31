// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference throughput for its
// clients. If your team needs expertise in putting causal attention on Hopper
// warpgroup tensor cores without giving up a compact paged KV cache, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// Fused causal paged-attention PREFILL at head_dim 256 over an int8 (BRAIN_KV_TIER 2) or FP8 E4M3 (3) cache on Hopper
// warpgroup tensor cores (`wgmma.mma_async`): version 3 of flash_prefill_f16_hd256.cu. It is compiled after that file with
// BRAIN_FLASH_HELPERS_ONLY defined (helpers, `KvT`, tier macros) and keeps its numerics - Q, K, V and the softmax weights
// rounded to fp16, fp32 accumulators - with one deliberate change: the per-row cache scale is applied to the fp32 scores
// (K) and to the fp16 softmax weights (V) instead of to the dequantised fp16 operands, so the producer's widening of a
// byte to fp16 is EXACT and costs no multiply.
//
//   params       : [bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt, splits_max, multiprocessors] u32
//   q            : [bsz, n_heads * 256] f32       pool_k / pool_v : paged 8-bit planes      k_scales / v_scales : [pool rows, n_kv_heads] f32
//   block_tables : [bsz, max_bt] u32, seq_lens : [bsz] u32, ctx : [bsz, n_heads * 256] f32
//   part         : [splits_max * bsz * n_heads * 258] f32 scratch, read only by the combine kernel below
//
// One block is 384 threads: a PRODUCER warpgroup and two CONSUMER warpgroups of 64 query rows each (128 rows a block, one
// head), sharing every K/V tile.
//   * The producer fetches the raw bytes of a 64-key tile (K and V) with cp.async into a double-buffered staging area (so a
//     fetch has a whole tile period to land), widens them to fp16 into the 128-byte-swizzled K / V tiles the tensor cores
//     read, and publishes them with mbarriers (K: two stages, V: one).
//   * A consumer warpgroup runs S = Q K^T as 16 wgmma m64n64k16 (Q and K by descriptor), the online softmax on the
//     accumulator registers, and O += P V as 4 wgmma m64n256k16 with P in registers (the accumulator layout of S IS the
//     register A layout of the next product) and V by descriptor in its transposed (key-major) form.
//   * Register budget: ptxas compiles the whole kernel under the launch bound (168 for 384 threads) and the m64n256 wgmma
//     needs 158 of them; `setmaxnreg` still moves the register file from the producer to the consumers.
//
// SPLIT-KEY. A chunk of 512 rows is 4 blocks of 128 rows a head, 96 blocks on a 132-multiprocessor part: fewer than there
// are multiprocessors. The key range of a block is therefore divided over up to `splits_max` blocks (grid = base blocks x
// splits_max, split slowest); `wg_eff_splits` picks how many do work from the block's key count, so a short context does
// not pay for splits it cannot use and the surplus blocks return at once. A split writes its unnormalised output and its
// (running max, running sum) to `part`; `brain_flash_prefill_combine` merges them. With one effective split the block
// normalises and writes `ctx` itself and the combine kernel does nothing.
//
// Contract differences from version 1/2: a block covers 128 query rows, and every row of it reads the block table of the
// block's FIRST row (one sequence per 128 rows; version 1/2 hold that for 64).
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_FLASH_HELPERS_ONLY
#error "flash_prefill_f16_hd256_wg.cu is compiled after flash_prefill_f16_hd256.cu with BRAIN_FLASH_HELPERS_ONLY defined"
#endif
#if BRAIN_HD != 256 || BRAIN_DENSE || BRAIN_KV_TIER < 2
#error "version 3 serves head_dim 256 over an int8 or FP8 paged cache"
#endif

#ifndef WG_SPLIT_OVERHEAD
#define WG_SPLIT_OVERHEAD 16u
#endif
#define WG_THREADS 384
#define WG_ROWS 128                   // query rows a block: two consumer warpgroups of 64
#define WG_KEYS 64                    // keys a tile
#define WG_SUB_BYTES 8192             // one 64-row x 64-value fp16 swizzle atom column: 64 rows x 128 bytes
#define WG_TILE_BYTES 32768           // a 64-row x 256-value fp16 tile: four atom columns
#define WG_RAW_BYTES (WG_KEYS * 256)      // a staged raw tile: 64 rows of 256 bytes (chunk c of row r at c ^ ((r & 3) << 1))
#define WG_PRODUCER_REGS 40
#define WG_CONSUMER_REGS 232

// Dynamic shared memory layout (offsets from the 1024-aligned base).
#define WG_OFF_Q 0                                     // two warpgroups x 32 KiB
#define WG_OFF_K (WG_OFF_Q + 2 * WG_TILE_BYTES)        // two stages x 32 KiB
#define WG_OFF_V (WG_OFF_K + 2 * WG_TILE_BYTES)        // one stage
#define WG_OFF_RAWK (WG_OFF_V + WG_TILE_BYTES)         // two staging tiles each
#define WG_OFF_RAWV (WG_OFF_RAWK + 2 * WG_RAW_BYTES)
#define WG_OFF_RSK (WG_OFF_RAWV + 2 * WG_RAW_BYTES)    // staged raw K row scales, two x 64 f32
#define WG_OFF_RSV (WG_OFF_RSK + 512)
#define WG_OFF_KS (WG_OFF_RSV + 512)                   // K scale x softmax scale x log2(e), two stages x 64 f32
#define WG_OFF_VS (WG_OFF_KS + 512)                    // V scale, 64 f32
#define WG_OFF_BAR (WG_OFF_VS + 256)                   // six 8-byte mbarriers, padded to 64 bytes
#define WG_SMEM_END (WG_OFF_BAR + 64)

// How many splits of a block's key range do work. `tiles` is the block's key tiles, `base` the blocks of one split, `smax`
// the grid's split count and `nsm` the multiprocessors. Minimises waves x (tiles a split walks + a fixed cost that stands
// for the Q load, the epilogue and the merge). A pure function of values the main and the combine kernel both know.
__device__ __forceinline__ unsigned int wg_eff_splits(unsigned int tiles, unsigned int base, unsigned int smax, unsigned int nsm) {
    unsigned int best = 1u, best_cost = 0xffffffffu;
    for (unsigned int s = 1u; s <= smax; ++s) {
        const unsigned int per = (tiles + s - 1u) / s;
        if (s > 1u && per * (s - 1u) >= tiles) { break; }   // the last split would be empty
        const unsigned int waves = (base * s + nsm - 1u) / nsm;
        const unsigned int cost = waves * (per + WG_SPLIT_OVERHEAD);
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    return best;
}

__device__ __forceinline__ void wg_mbar_init(unsigned int bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void wg_mbar_arrive(unsigned int bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void wg_mbar_wait(unsigned int bar, unsigned int parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WG_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WG_WAIT;\n"
        "}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void wg_named_barrier(unsigned int id) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(id) : "memory");
}
// Two warpgroups take turns on the tensor cores: a warpgroup `sync`s on its slot's barrier (128 threads + the 128 `arrive`s of
// the warpgroup before it in the order S0, S1, PV0, PV1) and, once its wgmma are issued, `arrive`s on the next slot's.
__device__ __forceinline__ void wg_turn_wait(unsigned int id) {
    asm volatile("bar.sync %0, 256;\n" ::"r"(id) : "memory");
}
__device__ __forceinline__ void wg_turn_pass(unsigned int id) {
    asm volatile("bar.arrive %0, 256;\n" ::"r"(id) : "memory");
}
__device__ __forceinline__ void wg_cp16(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wg_cp4(unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wg_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void wg_cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
// Generic-proxy shared-memory writes become visible to the async proxy (wgmma).
__device__ __forceinline__ void wg_fence_proxy() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wg_regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(WG_PRODUCER_REGS)); }
__device__ __forceinline__ void wg_regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(WG_CONSUMER_REGS)); }

// Shared-memory matrix descriptors, 128-byte swizzle. K-major operand (Q, K): 8-row groups 1024 bytes apart. N-major operand
// (V, the transposed B form): the 64-value atom columns are 8192 bytes apart (leading dimension) and the 8-key groups 1024.
__device__ __forceinline__ unsigned long long wg_desc_k(unsigned int addr) {
    return static_cast<unsigned long long>((addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ unsigned long long wg_desc_mn(unsigned int addr) {
    return static_cast<unsigned long long>((addr & 0x3FFFFu) >> 4) | (512ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ unsigned int wg_prmt(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned int d;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
__device__ __forceinline__ unsigned int wg_hsub2(unsigned int a, unsigned int b) {
    unsigned int d;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ float wg_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Four 8-bit codes (one word) -> two f16x2 registers, codes (0,1) and (2,3), EXACT and unscaled. int8: (code ^ 0x80) is
// code + 128 in 0..255, ORed under the f16 exponent of 1024 it is 1024 + code + 128, and subtracting 1152 leaves the signed
// code. E4M3: the hardware conversion.
__device__ __forceinline__ void wg_widen4(unsigned int word, unsigned int& lo, unsigned int& hi) {
#if BRAIN_KV_TIER == 3
    asm("{\n.reg .b16 l, u;\nmov.b32 {l, u}, %2;\ncvt.rn.f16x2.e4m3x2 %0, l;\ncvt.rn.f16x2.e4m3x2 %1, u;\n}\n" : "=r"(lo), "=r"(hi) : "r"(word));
#else
    const unsigned int u = word ^ 0x80808080u;
    lo = wg_hsub2(wg_prmt(u, 0x64646464u, 0x4140u), 0x64806480u);
    hi = wg_hsub2(wg_prmt(u, 0x64646464u, 0x4342u), 0x64806480u);
#endif
}

// S (64 x 64 f32, 32 registers a thread) = or += Q (64 x 16 f16) . K (64 x 16 f16)^T, both operands from shared memory by descriptor.
__device__ __forceinline__ void brain_wg_mma_s(float (&d)[32], unsigned long long da, unsigned long long db, bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23,"
        "%24, %25, %26, %27, %28, %29, %30, %31"
        "}, %32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        :           "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
          "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
          "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(static_cast<int>(accumulate)));
}

// O (64 x 256 f32, 128 registers a thread) += P (64 x 16 f16, the thread's four registers) . V (16 x 256 f16, key-major in shared
// memory: the transposed-B form).
__device__ __forceinline__ void brain_wg_mma_pv(float (&d)[128], const unsigned int (&a)[4], unsigned long long db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23,"
        "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45,"
        "%46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67,"
        "%68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89,"
        "%90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109,"
        "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
        "}, {%128, %129, %130, %131}, %132, p, 1, 1, 1;\n"
        "}\n"
        :           "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
          "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
          "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
          "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
          "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
          "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
          "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]),
          "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]),
          "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
          "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]),
          "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]),
          "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]),
          "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]),
          "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]),
          "+f"(d[126]), "+f"(d[127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

extern "C" __global__ void __launch_bounds__(WG_THREADS, 1) brain_flash_prefill_f16_hd256_wg(
    const unsigned int* params, const float* q, const KvT* pool_k, const KvT* pool_v, const unsigned int* block_tables,
    const unsigned int* seq_lens, float* ctx, const float* k_scales, const float* v_scales, float* part) {
    extern __shared__ unsigned char brain_wg_smem_raw[];
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int n_kv_heads = params[2];
    const unsigned int group = params[4];
    const unsigned int block_size = params[5];
    const unsigned int max_bt = params[6];
    const unsigned int smax = params[7];
    const unsigned int nsm = params[8];
    const unsigned int nrows = bsz;

    const unsigned int ntq = (nrows + WG_ROWS - 1u) / WG_ROWS;
    if (ntq == 0u) { return; }
    const unsigned int base_blocks = ntq * n_heads;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int qt = ntq - 1u - blk % ntq;   // the longest key ranges first
    const unsigned int h = (blk / ntq) % n_heads;
    const unsigned int split = blk / base_blocks;
    if (split >= smax) { return; }   // block-uniform: no barrier is skipped per-thread

    const unsigned int i0 = qt * WG_ROWS;
    const unsigned int max_t = seq_lens[min(i0 + WG_ROWS - 1u, nrows - 1u)];
    const unsigned int tiles = (max_t + WG_KEYS - 1u) / WG_KEYS;
    const unsigned int n_split = wg_eff_splits(tiles, base_blocks, smax, nsm);
    if (split >= n_split) { return; }
    const unsigned int per = (tiles + n_split - 1u) / n_split;
    const unsigned int tile_begin = min(split * per, tiles);
    const unsigned int tile_end = min(tile_begin + per, tiles);

    const unsigned int hkv = h / (group == 0u ? 1u : group);
    const unsigned long long q_row = static_cast<unsigned long long>(n_heads) * HD;
    const unsigned long long kv_row = static_cast<unsigned long long>(n_kv_heads) * HD;

    const unsigned int raw_addr = brain_smem_addr(brain_wg_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_wg_smem_raw + (base - raw_addr);
    const unsigned int bar_full_k = base + WG_OFF_BAR;          // + 8 stage
    const unsigned int bar_empty_k = base + WG_OFF_BAR + 16u;   // + 8 stage
    const unsigned int bar_full_v = base + WG_OFF_BAR + 32u;
    const unsigned int bar_empty_v = base + WG_OFF_BAR + 40u;

    const unsigned int tid = threadIdx.x;
    if (tid == 0u) {
        wg_mbar_init(bar_full_k, 128u);
        wg_mbar_init(bar_full_k + 8u, 128u);
        wg_mbar_init(bar_empty_k, 8u);
        wg_mbar_init(bar_empty_k + 8u, 8u);
        wg_mbar_init(bar_full_v, 128u);
        wg_mbar_init(bar_empty_v, 8u);
    }
    __syncthreads();

    if (tid < 128u) {
        // ================= producer warpgroup =================
        wg_regs_dec();
        if (tile_begin == tile_end) { return; }
        // Thread -> (key row, half of the row's 16-byte chunks): a quarter-warp reads four rows, two chunks each, which land
        // on eight distinct bank slots of the staging rows (chunk c of row r is stored at c ^ ((r & 3) << 1)).
        const unsigned int row = (tid >> 5) * 16u + ((tid & 31u) >> 1);
        const unsigned int half = tid & 1u;
        const unsigned int swz = (row & 3u) << 1;
        const unsigned int rawk_base = base + WG_OFF_RAWK + row * 256u;
        const unsigned int rawv_base = base + WG_OFF_RAWV + row * 256u;
        const unsigned int* const bt = block_tables + static_cast<unsigned long long>(i0) * max_bt;
        const bool pow2_block = (block_size & (block_size - 1u)) == 0u;
        const unsigned int block_shift = __ffs(block_size) - 1u;

        // Fetch key tile `kt` (K and V planes) into staging buffer `kt & 1`.
        auto issue_raw = [&](unsigned int kt) {
            const unsigned int buf = kt & 1u;
            if (kt < tile_end) {
                const unsigned int j = kt * WG_KEYS + row;
                const bool live = j < max_t;
                unsigned long long prow = 0;
                {
                    const unsigned int jb = pow2_block ? (j >> block_shift) : (j / block_size);
                    const unsigned int jo = pow2_block ? (j & (block_size - 1u)) : (j % block_size);
                    prow = static_cast<unsigned long long>(bt[jb]) * block_size + jo;
                }
                const unsigned int bytes = live ? 16u : 0u;
                const KvT* const ksrc = pool_k + prow * kv_row + hkv * HD;
                const KvT* const vsrc = pool_v + prow * kv_row + hkv * HD;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const unsigned int c = 2u * i + half;
                    wg_cp16(rawk_base + buf * WG_RAW_BYTES + ((c ^ swz) << 4), ksrc + c * 16u, bytes);
                    wg_cp16(rawv_base + buf * WG_RAW_BYTES + ((c ^ swz) << 4), vsrc + c * 16u, bytes);
                }
                if (half == 0u) {
                    const unsigned long long srow = prow * n_kv_heads + hkv;
                    wg_cp4(base + WG_OFF_RSK + buf * 256u + row * 4u, k_scales + srow, live ? 4u : 0u);
                    wg_cp4(base + WG_OFF_RSV + buf * 256u + row * 4u, v_scales + srow, live ? 4u : 0u);
                }
            }
            wg_cp_commit();   // an empty group past the end keeps the wait count below uniform
        };
        // Widen this thread's half of staged row (staging base `raw_base`, buffer `buf`) into the swizzled tile at `tile`.
        auto widen = [&](unsigned int raw_base, unsigned int buf, unsigned int tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned int c = 2u * i + half;
                uint4 w;
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w) : "r"(raw_base + buf * WG_RAW_BYTES + ((c ^ swz) << 4)));
                uint4 o0, o1;
                wg_widen4(w.x, o0.x, o0.y);
                wg_widen4(w.y, o0.z, o0.w);
                wg_widen4(w.z, o1.x, o1.y);
                wg_widen4(w.w, o1.z, o1.w);
                // 16 codes -> d-chunks 2c and 2c + 1 of the row (8 values each), in the atom column c / 4.
                const unsigned int dst = tile + (c >> 2) * WG_SUB_BYTES + row * 128u;
                const unsigned int s0 = ((2u * c) & 7u) ^ (row & 7u);
                const unsigned int s1 = ((2u * c + 1u) & 7u) ^ (row & 7u);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst + s0 * 16u), "r"(o0.x), "r"(o0.y), "r"(o0.z), "r"(o0.w));
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst + s1 * 16u), "r"(o1.x), "r"(o1.y), "r"(o1.z), "r"(o1.w));
            }
        };

        float* const ks2 = reinterpret_cast<float*>(gen + WG_OFF_KS);
        float* const vs1 = reinterpret_cast<float*>(gen + WG_OFF_VS);
        const float* const rsk = reinterpret_cast<const float*>(gen + WG_OFF_RSK);
        const float* const rsv = reinterpret_cast<const float*>(gen + WG_OFF_RSV);
        issue_raw(tile_begin);
        issue_raw(tile_begin + 1u);
        for (unsigned int kt = tile_begin; kt < tile_end; ++kt) {
            const unsigned int j = kt - tile_begin;
            const unsigned int buf = kt & 1u;
            const unsigned int ks = j & 1u;
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");   // this tile's copies (the newer group may still fly)
            wg_named_barrier(1u);                                      // and every other thread's
            wg_mbar_wait(bar_empty_k + 8u * ks, ((j >> 1) & 1u) ^ 1u);
            widen(rawk_base, buf, base + WG_OFF_K + ks * WG_TILE_BYTES);
            if (half == 0u) { ks2[ks * 64u + row] = rsk[buf * 64u + row] * SM_SCALE_LOG2; }
            wg_fence_proxy();
            wg_mbar_arrive(bar_full_k + 8u * ks);
            wg_mbar_wait(bar_empty_v, (j & 1u) ^ 1u);
            widen(rawv_base, buf, base + WG_OFF_V);
            if (half == 0u) { vs1[row] = rsv[buf * 64u + row]; }
            wg_fence_proxy();
            wg_mbar_arrive(bar_full_v);
            wg_named_barrier(1u);                                      // nobody reads this staging buffer any more
            issue_raw(kt + 2u);
        }
        return;
    }

    // ================= consumer warpgroups =================
    wg_regs_inc();
    const unsigned int cw = __shfl_sync(0xffffffffu, tid >> 7, 0) - 1u;   // which consumer warpgroup (warp-uniform to the compiler): rows 64 cw .. 64 cw + 63
    const unsigned int ct = tid & 127u;
    const unsigned int warp = ct >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t4 = lane & 3u;

    const unsigned int wg_row0 = i0 + cw * 64u;
    const unsigned int row_a = wg_row0 + warp * 16u + g;
    const unsigned int row_b = row_a + 8u;
    const bool live_a = row_a < nrows, live_b = row_b < nrows;
    const unsigned int len_a = live_a ? seq_lens[row_a] : 0u;
    const unsigned int len_b = live_b ? seq_lens[row_b] : 0u;
    const unsigned int wg_min_len = wg_row0 < nrows ? seq_lens[wg_row0] : 0u;

    // ---- Q (this warpgroup's 64 rows) to shared memory as fp16, swizzled: 64 rows x 32 chunks of 8 values.
    const unsigned int q_tile = base + WG_OFF_Q + cw * WG_TILE_BYTES;
#pragma unroll 4
    for (int c = 0; c < 16; ++c) {
        const unsigned int e = ct + 128u * c;
        const unsigned int r = e >> 5, cc = e & 31u;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (wg_row0 + r < nrows) {
            const float* src = q + static_cast<unsigned long long>(wg_row0 + r) * q_row + h * HD + cc * 8u;
            const float4 lo = *reinterpret_cast<const float4*>(src);
            const float4 hi = *reinterpret_cast<const float4*>(src + 4);
            v.x = brain_pack_f16x2(lo.x, lo.y);
            v.y = brain_pack_f16x2(lo.z, lo.w);
            v.z = brain_pack_f16x2(hi.x, hi.y);
            v.w = brain_pack_f16x2(hi.z, hi.w);
        }
        const unsigned int dst = q_tile + (cc >> 3) * WG_SUB_BYTES + r * 128u + (((cc & 7u) ^ (r & 7u)) << 4);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    }
    wg_fence_proxy();
    wg_named_barrier(2u + cw);

    float o[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) { o[i] = 0.f; }
    float m_a = NEG_BIG, m_b = NEG_BIG, l_a = 0.f, l_b = 0.f;
    const float* const ks2 = reinterpret_cast<const float*>(gen + WG_OFF_KS);
    const float* const vs1 = reinterpret_cast<const float*>(gen + WG_OFF_VS);

    // Every warp runs every tile, a tile past a warpgroup's last row included (the mask weighs all of it 0): a branch around the
    // wgmma that the compiler cannot prove warp-uniform makes it wait for each one before issuing the next.
    for (unsigned int kt = tile_begin; kt < tile_end; ++kt) {
        const unsigned int j = kt - tile_begin;
        const unsigned int ks = j & 1u;
        wg_mbar_wait(bar_full_k + 8u * ks, (j >> 1) & 1u);

        float s[32];
        {
            const unsigned int k_tile = base + WG_OFF_K + ks * WG_TILE_BYTES;
            wg_fence();
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                brain_wg_mma_s(s, wg_desc_k(q_tile + (k >> 2) * WG_SUB_BYTES + (k & 3) * 32u), wg_desc_k(k_tile + (k >> 2) * WG_SUB_BYTES + (k & 3) * 32u), k > 0);
            }
            wg_commit();
        }
        {
            wg_wait0();

            // ---- row scale of K (already times 1/sqrt(d) log2 e) and, on the few tiles that cross a row's own position, the causal mask.
            if ((kt + 1u) * WG_KEYS <= wg_min_len) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const float2 kx = *reinterpret_cast<const float2*>(ks2 + ks * 64u + n * 8u + t4 * 2u);
                    s[4 * n + 0] *= kx.x;
                    s[4 * n + 1] *= kx.y;
                    s[4 * n + 2] *= kx.x;
                    s[4 * n + 3] *= kx.y;
                }
            } else {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const float2 kx = *reinterpret_cast<const float2*>(ks2 + ks * 64u + n * 8u + t4 * 2u);
                    const unsigned int j0 = kt * WG_KEYS + n * 8u + t4 * 2u;
                    s[4 * n + 0] = j0 < len_a ? s[4 * n + 0] * kx.x : NEG_BIG;
                    s[4 * n + 1] = j0 + 1u < len_a ? s[4 * n + 1] * kx.y : NEG_BIG;
                    s[4 * n + 2] = j0 < len_b ? s[4 * n + 2] * kx.x : NEG_BIG;
                    s[4 * n + 3] = j0 + 1u < len_b ? s[4 * n + 3] * kx.y : NEG_BIG;
                }
            }
        }
        __syncwarp();
        if (lane == 0u) { wg_mbar_arrive(bar_empty_k + 8u * ks); }

        unsigned int pf[4][4];
        {
            float mx_a = NEG_BIG, mx_b = NEG_BIG;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                mx_a = fmaxf(mx_a, fmaxf(s[4 * n + 0], s[4 * n + 1]));
                mx_b = fmaxf(mx_b, fmaxf(s[4 * n + 2], s[4 * n + 3]));
            }
            mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
            mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
            mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
            mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
            const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
            const float al_a = wg_ex2(m_a - mn_a), al_b = wg_ex2(m_b - mn_b);
            m_a = mn_a;
            m_b = mn_b;
            // A row with no live key yet has mn = NEG_BIG; its masked scores must weigh 0, not exp2(0).
            const float ref_a = mn_a > NEG_BIG ? mn_a : 0.f, ref_b = mn_b > NEG_BIG ? mn_b : 0.f;
            float sum_a = 0.f, sum_b = 0.f;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                s[4 * n + 0] = wg_ex2(s[4 * n + 0] - ref_a);
                s[4 * n + 1] = wg_ex2(s[4 * n + 1] - ref_a);
                s[4 * n + 2] = wg_ex2(s[4 * n + 2] - ref_b);
                s[4 * n + 3] = wg_ex2(s[4 * n + 3] - ref_b);
                sum_a += s[4 * n + 0] + s[4 * n + 1];
                sum_b += s[4 * n + 2] + s[4 * n + 3];
            }
            l_a = l_a * al_a + sum_a;
            l_b = l_b * al_b + sum_b;
            // O only needs rescaling when some row's running maximum grew: nearly never past the first tiles. The first tile's
            // O is still zero, and the vote keeps the warp together.
            if (j > 0u && !__all_sync(0xffffffffu, al_a == 1.0f && al_b == 1.0f)) {
#pragma unroll
                for (int n = 0; n < 32; ++n) {
                    o[4 * n + 0] *= al_a;
                    o[4 * n + 1] *= al_a;
                    o[4 * n + 2] *= al_b;
                    o[4 * n + 3] *= al_b;
                }
            }
        }

        wg_mbar_wait(bar_full_v, j & 1u);   // the V row scales and the V tile of this key tile
        {
            // P as fp16 A fragments, straight from the score accumulators, times the V row scale.
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const float2 v0 = *reinterpret_cast<const float2*>(vs1 + (2 * kk) * 8u + t4 * 2u);
                const float2 v1 = *reinterpret_cast<const float2*>(vs1 + (2 * kk + 1) * 8u + t4 * 2u);
                pf[kk][0] = brain_pack_f16x2(s[8 * kk + 0] * v0.x, s[8 * kk + 1] * v0.y);
                pf[kk][1] = brain_pack_f16x2(s[8 * kk + 2] * v0.x, s[8 * kk + 3] * v0.y);
                pf[kk][2] = brain_pack_f16x2(s[8 * kk + 4] * v1.x, s[8 * kk + 5] * v1.y);
                pf[kk][3] = brain_pack_f16x2(s[8 * kk + 6] * v1.x, s[8 * kk + 7] * v1.y);
            }
        }
        {
            wg_fence();
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) { brain_wg_mma_pv(o, pf[kk], wg_desc_mn(base + WG_OFF_V + kk * 2048u)); }
            wg_commit();
        }
        wg_wait0();
        __syncwarp();
        if (lane == 0u) { wg_mbar_arrive(bar_empty_v); }
    }

    // ---- epilogue. l is a per-thread partial of its quad's row sum.
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    if (n_split == 1u) {
        const float inv_a = l_a > 0.f ? 1.0f / l_a : 0.f;
        const float inv_b = l_b > 0.f ? 1.0f / l_b : 0.f;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const unsigned int d = n * 8u + t4 * 2u;
            if (live_a) { *reinterpret_cast<float2*>(ctx + static_cast<unsigned long long>(row_a) * q_row + h * HD + d) = make_float2(o[4 * n] * inv_a, o[4 * n + 1] * inv_a); }
            if (live_b) { *reinterpret_cast<float2*>(ctx + static_cast<unsigned long long>(row_b) * q_row + h * HD + d) = make_float2(o[4 * n + 2] * inv_b, o[4 * n + 3] * inv_b); }
        }
    } else {
        const unsigned long long split_rows = static_cast<unsigned long long>(split) * bsz;
        float* const part_ml = part + static_cast<unsigned long long>(smax) * bsz * q_row;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const unsigned int d = n * 8u + t4 * 2u;
            if (live_a) { *reinterpret_cast<float2*>(part + ((split_rows + row_a) * n_heads + h) * HD + d) = make_float2(o[4 * n], o[4 * n + 1]); }
            if (live_b) { *reinterpret_cast<float2*>(part + ((split_rows + row_b) * n_heads + h) * HD + d) = make_float2(o[4 * n + 2], o[4 * n + 3]); }
        }
        if (t4 == 0u) {
            if (live_a) { *reinterpret_cast<float2*>(part_ml + ((split_rows + row_a) * n_heads + h) * 2u) = make_float2(m_a, l_a); }
            if (live_b) { *reinterpret_cast<float2*>(part_ml + ((split_rows + row_b) * n_heads + h) * 2u) = make_float2(m_b, l_b); }
        }
    }
}

// Merges the splits of `brain_flash_prefill_f16_hd256_wg`: one block of 256 threads (one per output value; the head is 256
// wide) for each (row, head), grid = bsz * n_heads. Does nothing for a row whose block ran a single split.
extern "C" __global__ void __launch_bounds__(256) brain_flash_prefill_combine(const unsigned int* params, const unsigned int* seq_lens, const float* part, float* ctx) {
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int smax = params[7];
    const unsigned int nsm = params[8];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int row = blk / n_heads;
    const unsigned int h = blk % n_heads;
    if (row >= bsz) { return; }
    const unsigned int ntq = (bsz + WG_ROWS - 1u) / WG_ROWS;
    const unsigned int i0 = (row / WG_ROWS) * WG_ROWS;
    const unsigned int tiles = (seq_lens[min(i0 + WG_ROWS - 1u, bsz - 1u)] + WG_KEYS - 1u) / WG_KEYS;
    const unsigned int n_split = wg_eff_splits(tiles, ntq * n_heads, smax, nsm);
    if (n_split <= 1u) { return; }
    const unsigned long long q_row = static_cast<unsigned long long>(n_heads) * HD;
    const float* const part_ml = part + static_cast<unsigned long long>(smax) * bsz * q_row;
    float m = NEG_BIG;
    for (unsigned int s = 0; s < n_split; ++s) { m = fmaxf(m, part_ml[((static_cast<unsigned long long>(s) * bsz + row) * n_heads + h) * 2u]); }
    const float ref = m > NEG_BIG ? m : 0.f;
    float l = 0.f, acc = 0.f;
    const unsigned int d = threadIdx.x;
    for (unsigned int s = 0; s < n_split; ++s) {
        const unsigned long long at = (static_cast<unsigned long long>(s) * bsz + row) * n_heads + h;
        const float w = wg_ex2(part_ml[at * 2u] - ref);
        l += w * part_ml[at * 2u + 1u];
        acc += w * part[at * HD + d];
    }
    ctx[static_cast<unsigned long long>(row) * q_row + h * HD + d] = l > 0.f ? acc / l : 0.f;
}
