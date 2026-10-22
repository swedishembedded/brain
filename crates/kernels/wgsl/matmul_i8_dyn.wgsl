// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled int8 (DP4A) GEMM with GROUP-WISE (32-element) activation and weight scales, both read from buffers - the prefill/DiT int8 GEMM
// @how   DP4A packed int8, vec4 shared tiles, TMxTN register block per thread, staged group scales, per-k-chunk group dequant, 256-thread workgroup tile, 3 barriers
// @opt   5
// @cpu   no
// @gpu   yes-wg256
// @npu   yes
// @quant int8
// @dtype f32
//
// `matmul_i8`'s dynamic-scale sibling: both scales come from buffers rather
// than from the uniform, so one build serves every shape and the activation
// scale can be recomputed per forward.
//
//   x_q : [M, K/4] u32   - 4 int8 activations packed along K per u32 (row-major)
//   w_q : [N, K/4] u32   - 4 int8 weights    packed along K per u32 (row-major)
//   sx  : [M, K/32] f32  - GROUP-WISE activation scale (`max_abs_row`'s)
//   sw  : [N, K/32] f32  - GROUP-WISE weight scale (`model::int8::GROUP` = 32)
//   out : [M, N]   f32   - out[m,n] = Σ_g acc_i32[m,n,g] * (sw[n,g] * sx[m,g])
//
// K must be a multiple of 32 (the group), which subsumes the multiple of 4 the
// packing needs.
//
// DP4A (`dot4I8Packed`) does four int8 multiply-accumulates in one
// instruction. The int8 weights also move 1/4 the bytes of fp32.
//
// ## Tile shape: TM x TN outputs per thread
//
// 256 threads on a 16x16 lane grid; each thread owns a TM x TN block of int32
// accumulators, so the workgroup tile is BM = 16*TM rows by BN = 16*TN
// columns. The default TM = TN = 8 is the 128x128 tile every caller
// dispatches (`m.div_ceil(128) * n.div_ceil(128)` workgroups). `TN = 4`
// (128x64) keeps the per-thread accumulators under 128 registers, which lets
// two workgroups share an SM and hides shared-memory and barrier latency that
// a single 128-register-heavy workgroup cannot; on a Tesla P40 it is faster
// than 8x8 at Qwen3-8B prefill shapes (`qwen_bench gemm8` reproduces the
// comparison). A caller that picks another
// tile specialises TM/TN through `kernels::template` and sizes its grid with
// the matching BM x BN.
//
// The WGSL `dot4I8Packed` lowers to a plain DP4A followed by a separate
// integer add (the accumulate form is not reachable from WGSL), so the inner
// loop issues two integer instructions per DP4A; that, not the shared-memory
// traffic, is the arithmetic ceiling of this kernel.
//
// Staging is software-pipelined through registers: the next chunk's global
// loads are in flight while the current one is consumed. The staging arrays
// hold `[row][k-group]` with the k-groups contiguous, four to a `vec4<u32>`,
// so one load instruction feeds four DP4A, and the row stride is padded to
// `SP4` vec4s so a tx-group's reads hit distinct banks. A thread's rows and
// columns are interleaved at stride 16 (rows {ty, ty+16, ...}) so the
// epilogue stores 16 consecutive elements per instruction.
//
// ## Where the group scale is applied
//
// `BKG` is 8 packed groups = 32 int8 along K, which is EXACTLY `QPG` (a
// template knob, default 2) quads' worth of weight-scale groups. At the
// default `QPG=2` one k-chunk is one quantization group (group=32 - Q8_0,
// Q5_0, Q4_0): the TM*TN integer accumulators sum a chunk exactly, then fold
// into f32 running totals scaled by that chunk's `sw[n, sg] * sx[m, ch]` (a
// chunk is always one 32-element activation group). `QPG=1` folds twice per
// chunk (group=16 - Q6_K), both with the chunk's activation scale. The integer sum inside a group is exact
// and re-orderable; the cross-group folds are floating point, in ascending
// group order.
//
// Two details keep the fold cheap. The weight scales of a chunk are fetched
// one chunk ahead by `BN` threads into shared memory (`Ss`), so the fold reads
// them from shared memory instead of eight dependent global gathers. And the
// integer accumulators start at the bit pattern of 1.5*2^23 (`MAGIC`): a group
// sum is far below 2^22 in magnitude, so the accumulator's bits reinterpreted
// as f32 are exactly 1.5*2^23 + sum, and subtracting `MAGIC_F` recovers
// f32(sum) exactly with one FADD instead of an int-to-float convert. The
// result is bit-identical to `f32(sum)`.
//
// @workgroup_size(256). Not CPU-JIT'able (multi-barrier work-group); the CPU
// int8 reference lives in the validation test, so parity is still gated.

struct Params { m: u32, kg: u32, n: u32 };  // group-wise sx and sw, kg = K/4
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<u32>;
@group(0) @binding(2) var<storage, read>       w:   array<u32>;
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

const TM: u32 = 8u;    // output rows per thread
const TN: u32 = 8u;    // output columns per thread
const BKG: u32 = 8u;    // packed K-groups per chunk (= 32 int8 along K)
const BKQ: u32 = 2u;    // BKG / 4 - vec4 quads of k-groups per row per chunk
const QPG: u32 = 2u;    // quads per weight-scale group: 2 => group=32, 1 => group=16 (Q6_K)
const GPC: u32 = BKQ / QPG;  // weight-scale groups per chunk
const SP4: u32 = 3u;    // padded shared stride in vec4s (BKQ + 1), bank-spread
const LN: u32 = 16u;    // lane grid: 16 x 16 threads, stride-16 interleave
const BM: u32 = TM * LN;  // workgroup tile rows
const BN: u32 = TN * LN;  // workgroup tile columns
const RS: u32 = LN * SP4; // vec4 step between a thread's own rows/columns
const MAGIC: i32 = 0x4B400000;  // bits of 1.5 * 2^23; see the header
const MAGIC_F: f32 = 12582912.0;

var<workgroup> As: array<vec4<u32>, BM * SP4>;
var<workgroup> Bs: array<vec4<u32>, BN * SP4>;
var<workgroup> Ss: array<f32, GPC * BN>;  // [group][col] weight scales of the live chunk
var<workgroup> Sx: array<f32, BM>;        // [row] activation scales of the live chunk (one group)

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let tid = lid.x; let ty = tid / LN; let tx = tid % LN;
  let wg = wgid.y * nwg.x + wgid.x;
  let tiles_n = (p.n + BN - 1u) / BN;
  let row0 = (wg / tiles_n) * BM; let col0 = (wg % tiles_n) * BN;
  let sr = tid / BKQ; let sq = tid % BKQ;
  let arow = row0 + sr; let brow = col0 + sr;
  let a_ok = tid < BM * BKQ && arow < p.m; let b_ok = tid < BN * BKQ && brow < p.n;
  let a_base = arow * p.kg; let b_base = brow * p.kg; let sh_idx = sr * SP4 + sq;
  // c: exact int32 sums of the live group; d: f32 running totals across groups.
  var c: array<i32, TM * TN>;
  var d: array<f32, TM * TN>;
  for (var i = 0u; i < TM * TN; i++) { c[i] = MAGIC; d[i] = 0.0; }
  var rA: vec4<u32>; var rB: vec4<u32>;
  let nchunks = (p.kg + BKG - 1u) / BKG;
  let ng = p.kg / (4u * QPG);
  // Weight scales are staged like the operands: thread t < BN owns column
  // col0 + t and fetches a chunk's scales one chunk ahead.
  let sc_ok = tid < BN && col0 + tid < p.n; let sc_base = (col0 + tid) * ng;
  var rS = vec2<f32>(0.0, 0.0);
  // Activation scales likewise: thread t < BM owns row row0 + t; a chunk is
  // exactly one activation group, so it is scale `ch` of that row.
  let ng_x = p.kg / 8u;
  let sx_ok = tid < BM && row0 + tid < p.m; let sx_base = (row0 + tid) * ng_x;
  var rX = 0.0;
  {
    let g0 = sq * 4u;
    var av = vec4<u32>(0u); var bv = vec4<u32>(0u);
    if (a_ok && g0 + 3u < p.kg) { av = vec4<u32>(x[a_base + g0], x[a_base + g0 + 1u], x[a_base + g0 + 2u], x[a_base + g0 + 3u]); }
    else if (a_ok) { if (g0 < p.kg) { av.x = x[a_base + g0]; } if (g0 + 1u < p.kg) { av.y = x[a_base + g0 + 1u]; } if (g0 + 2u < p.kg) { av.z = x[a_base + g0 + 2u]; } }
    if (b_ok && g0 + 3u < p.kg) { bv = vec4<u32>(w[b_base + g0], w[b_base + g0 + 1u], w[b_base + g0 + 2u], w[b_base + g0 + 3u]); }
    else if (b_ok) { if (g0 < p.kg) { bv.x = w[b_base + g0]; } if (g0 + 1u < p.kg) { bv.y = w[b_base + g0 + 1u]; } if (g0 + 2u < p.kg) { bv.z = w[b_base + g0 + 2u]; } }
    if (tid < BM * BKQ) { As[sh_idx] = av; }
    if (tid < BN * BKQ) { Bs[sh_idx] = bv; }
    if (tid < BN) {
      Ss[tid] = select(0.0, sw[sc_base], sc_ok && 0u < ng);
      if (GPC > 1u) { Ss[BN + tid] = select(0.0, sw[sc_base + 1u], sc_ok && 1u < ng); }
    }
    if (tid < BM) { Sx[tid] = select(0.0, sx[sx_base], sx_ok && 0u < ng_x); }
  }
  workgroupBarrier();
  for (var ch = 0u; ch < nchunks; ch = ch + 1u) {
    let has_next = ch + 1u < nchunks;
    if (has_next) {
      let g1 = (ch + 1u) * BKG + sq * 4u;
      let sg1 = (ch + 1u) * GPC;
      rS = vec2<f32>(select(0.0, sw[sc_base + sg1], sc_ok && sg1 < ng), select(0.0, sw[sc_base + sg1 + 1u], sc_ok && GPC > 1u && sg1 + 1u < ng));
      rX = select(0.0, sx[sx_base + ch + 1u], sx_ok && ch + 1u < ng_x);
      rA = vec4<u32>(0u); rB = vec4<u32>(0u);
      if (a_ok && g1 + 3u < p.kg) { rA = vec4<u32>(x[a_base + g1], x[a_base + g1 + 1u], x[a_base + g1 + 2u], x[a_base + g1 + 3u]); }
      else if (a_ok) { if (g1 < p.kg) { rA.x = x[a_base + g1]; } if (g1 + 1u < p.kg) { rA.y = x[a_base + g1 + 1u]; } if (g1 + 2u < p.kg) { rA.z = x[a_base + g1 + 2u]; } }
      if (b_ok && g1 + 3u < p.kg) { rB = vec4<u32>(w[b_base + g1], w[b_base + g1 + 1u], w[b_base + g1 + 2u], w[b_base + g1 + 3u]); }
      else if (b_ok) { if (g1 < p.kg) { rB.x = w[b_base + g1]; } if (g1 + 1u < p.kg) { rB.y = w[b_base + g1 + 1u]; } if (g1 + 2u < p.kg) { rB.z = w[b_base + g1 + 2u]; } }
    }
    for (var q = 0u; q < BKQ; q = q + 1u) {
      let ao = ty * SP4 + q; let bo = tx * SP4 + q;
      var a: array<vec4<u32>, TM>;
      for (var i = 0u; i < TM; i++) { a[i] = As[ao + i * RS]; }
      for (var j = 0u; j < TN; j++) {
        let b = Bs[bo + j * RS];
        for (var i = 0u; i < TM; i++) {
          c[i * TN + j] += dot4I8Packed(a[i].x, b.x) + dot4I8Packed(a[i].y, b.y) + dot4I8Packed(a[i].z, b.z) + dot4I8Packed(a[i].w, b.w);
        }
      }
      if ((q + 1u) % QPG == 0u) {
        let sb = (q / QPG) * BN + tx;
        var xs: array<f32, TM>;
        for (var i = 0u; i < TM; i++) { xs[i] = Sx[ty + i * LN]; }
        for (var j = 0u; j < TN; j++) {
          let e = Ss[sb + j * LN];
          for (var i = 0u; i < TM; i++) {
            d[i * TN + j] += (bitcast<f32>(c[i * TN + j]) - MAGIC_F) * (e * xs[i]);
            c[i * TN + j] = MAGIC;
          }
        }
      }
    }
    workgroupBarrier();
    if (has_next) {
      if (tid < BM * BKQ) { As[sh_idx] = rA; }
      if (tid < BN * BKQ) { Bs[sh_idx] = rB; }
      if (tid < BN) { Ss[tid] = rS.x; if (GPC > 1u) { Ss[BN + tid] = rS.y; } }
      if (tid < BM) { Sx[tid] = rX; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    let mi = row0 + ty + i * LN;
    if (mi < p.m) {
      let r = mi * p.n + col0 + tx;
      for (var j = 0u; j < TN; j++) {
        if (col0 + tx + j * LN < p.n) { out[r + j * LN] = d[i * TN + j]; }
      }
    }
  }
}
