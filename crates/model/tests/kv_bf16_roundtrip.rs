// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Paged-KV-cache `bf16` dual-backend roundtrip (B9) - the core deliverable
//! of the KV-cache storage-tier phase.
//!
//! Builds a paged KV cache spanning MULTIPLE pages via `model::paged::
//! {BlockAllocator, BlockTable}` (the real host-side bookkeeping
//! `qwen3::serve::Engine` uses), appends a long sequence of random K/V pairs
//! through BOTH `Ops::kv_append_batched`'s `F32` tier (the existing default)
//! and its new `BF16` tier (B9), then runs the SAME decode-time attention
//! (`Ops::decode_scores_batched` + `decode_softmax_batched` +
//! `Ops::decode_apply_batched`) against both pools and checks the bf16 tier
//! against the fp32 one within an EXPLICITLY DERIVED tolerance - the same
//! "per-output-element, not a flat epsilon" discipline
//! `bf16_roundtrip.rs`/`f16_roundtrip.rs` established for the matmul family.
//!
//! **Tolerance - the ULP-level math, explicit, isolated per stage.**
//! Attention has a nonlinear step (softmax) between K's use (scores) and V's
//! use (apply), so this file proves the two narrowed operands correct
//! SEPARATELY, each with a tight, rigorously derived bound, rather than
//! fighting softmax's error-amplification analysis for one combined number:
//!
//! * **Scores** (`scores[h,j] = scale * sum_d q[h,d] * k[j,d]`): only the
//!   CACHE (`k`) narrows to bf16 (7 explicit mantissa bits, round-to-nearest -
//!   see `kernels::template::bf16_pack_expr`'s own doc comment) - `q` is a
//!   fresh f32 activation, never touched. Each term's absolute error is
//!   therefore bounded by `|q[h,d] * k[j,d]| * 2^-8`, so the DOT PRODUCT's
//!   absolute error is bounded by `scale * 2^-8 * sum_d |q[h,d] * k[j,d]|`
//!   ([`scores_tol`]), computed per `(h,j)` - exactly [`bf16_roundtrip.
//!   rs`]'s own derivation, applied to a cache read instead of a weight read.
//! * **Apply, isolated from softmax** (`ctx[h,d] = sum_j probs[h,j] *
//!   v[j,d]`): dispatched TWICE against the SAME reference `probs` (computed
//!   ONCE from the fp32-K scores, shared by both the fp32-V and bf16-V apply
//!   calls) - this isolates V's own narrowing error from any score-level
//!   perturbation the nonlinear softmax step would otherwise mix in. Only
//!   `v` narrows, so the bound is `2^-8 * sum_j |probs[h,j] * v[j,d]|`
//!   ([`apply_tol`]), computed per `(h,d)`.
//! * **Full pipeline, as a sanity check, not the rigor gate** (bf16 K AND V,
//!   the bf16 pipeline's OWN softmax over its own slightly-perturbed
//!   scores): a GENEROUS bound, `apply_tol + 2 * (worst per-head score
//!   tolerance) * sum_j |v[j,d]|` - the second term crudely bounds how far
//!   softmax's own output can move for a score perturbation no larger than
//!   the scores-stage tolerance already proved (softmax's Jacobian row-sum
//!   is bounded by `2 * probs[i]` by a standard first-order argument; folding
//!   in `sum_j |v[j,d]|` as the worst case is intentionally loose, not tight,
//!   which is exactly why the two isolated checks above - not this one - are
//!   the rigor gate). Reported the same "worst observed err/tol ratio" way as
//!   `bf16_roundtrip.rs`.
//!
//! **The read-modify-write stress test** ([`kv_bf16_append_rmw_shared_word_
//! preserves_both_adjacent_slots_on_cpu`]/`_on_gpu`) deliberately uses an ODD
//! `kv_stride` (every REAL head_dim in this tree is even, so two adjacent
//! cache slots normally never share a packed `u32` word - see `kernels::
//! template::rewrite_packed_stores`'s own doc comment for exactly when they
//! do) to force the case a botched pack would corrupt: appends TWO tokens to
//! ADJACENT slots via two SEPARATE, sequential dispatches (never concurrent -
//! the normal way a decode loop appends one token at a time), then reads
//! BOTH back via a one-hot `probs` vector through `Ops::decode_apply_batched`
//! and checks each survived within bf16 tolerance.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{KvPage, Ops, PagedDecodeShape};
use model::paged::{BlockAllocator, BlockTable};

/// Two of this file's tests each build a real `Gpu::new_wgpu` device
/// directly (same reason as `conv_dtype_roundtrip.rs` - forcing a specific
/// backend rather than sharing the ambient one via `gpu_core::testgpu::dev`),
/// so under `cargo test`'s default multi-threaded run they can race their
/// own independent device builds against each other. See
/// `crates/gpu-core/tests/device_sharing.rs`'s `DEVICE_SERIAL`.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The full façade kernel set `Ops::new` requires, plus `decode_softmax_batched`
/// (dispatched directly, not through `Ops` - softmax has no weight/cache
/// operand to narrow, so it was never a candidate for the `Ops` façade).
/// Mirrors `model::ops::tests::kernel_list` / `bf16_roundtrip.rs`'s own copy.
fn kernel_list() -> Vec<(&'static str, &'static str)> {
    let dv = kernels::template::dtype_variant;
    let bf16_matmul = dv("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
    let bf16_gemv = dv("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
    let bf16_reg3 = dv("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
    let f16_matmul = dv("matmul", kernels::MATMUL, "w", Dtype::F16).unwrap();
    let f16_gemv = dv("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::F16).unwrap();
    let f16_reg3 = dv("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::F16).unwrap();
    let bf16_embed = dv("embed", kernels::EMBED, "emb", Dtype::BF16).unwrap();
    let f16_embed = dv("embed", kernels::EMBED, "emb", Dtype::F16).unwrap();
    let bf16_moe = dv("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::BF16).unwrap();
    let f16_moe = dv("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::F16).unwrap();
    let bf16_kv_append = kernels::template::dtype_variant_store(
        "paged_kv_append_batched_word",
        kernels::PAGED_KV_APPEND_BATCHED_WORD,
        "pool",
        Dtype::BF16,
    )
    .unwrap();
    let bf16_decode_scores =
        dv("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED, "pool_k", Dtype::BF16).unwrap();
    let bf16_decode_apply =
        dv("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED, "pool_v", Dtype::BF16).unwrap();
    // B10: matmul_dx's bf16-weight-read backward variant. matmul_dw has no
    // bf16 variant at all (it never reads the weight).
    let bf16_matmul_dx = dv("matmul_dx", kernels::MATMUL_DX, "w", Dtype::BF16).unwrap();
    vec![
        ("matmul", kernels::MATMUL),
        ("matmul_gemv", kernels::MATMUL_GEMV),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
        ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
        ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
        ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
        ("max_abs_row", kernels::MAX_ABS_ROW),
        ("quant_pack", kernels::QUANT_PACK),
        bf16_matmul,
        bf16_gemv,
        bf16_reg3,
        f16_matmul,
        f16_gemv,
        f16_reg3,
        ("embed", kernels::EMBED),
        bf16_embed,
        f16_embed,
        ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
        bf16_moe,
        f16_moe,
        ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
        bf16_kv_append,
        ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
        bf16_decode_scores,
        ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
        bf16_decode_apply,
        ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
        ("matmul_dx", kernels::MATMUL_DX),
        ("matmul_dw", kernels::MATMUL_DW),
        bf16_matmul_dx,
    ]
}

fn qk_scale(head_dim: u32) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// Per-`(h,j)` scores tolerance - see this file's module doc for the
/// derivation. `f64` accumulation so the tolerance's own arithmetic is not
/// itself the source of round-off being measured against.
fn scores_tol(q_head: &[f32], k_tok: &[f32], scale: f32) -> f32 {
    let mut abs_sum = 0f64;
    for (qi, ki) in q_head.iter().zip(k_tok) {
        abs_sum += (*qi as f64 * *ki as f64).abs();
    }
    (abs_sum * scale as f64 * 2f64.powi(-8)) as f32 + 1e-5
}

/// Per-`(h,d)` apply tolerance (V-only narrowing, at fixed/shared `probs`).
fn apply_tol(probs_row: &[f32], v_col: &[f32]) -> f32 {
    let mut abs_sum = 0f64;
    for (p, v) in probs_row.iter().zip(v_col) {
        abs_sum += (*p as f64 * *v as f64).abs();
    }
    (abs_sum * 2f64.powi(-8)) as f32 + 1e-5
}

/// Long-context parity: a real multi-block sequence, appended through both
/// tiers, decode-time attention run against both, checked at every stage
/// described in this file's module doc.
fn run_long_context_parity(gpu: Gpu, label: &str) {
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let (n_heads, n_kv, head_dim) = (4u32, 2u32, 8u32);
    let group = n_heads / n_kv;
    let kv_stride = n_kv * head_dim;
    let (block_size, num_blocks) = (4u32, 16u32);
    // Long enough to span MULTIPLE blocks (37 tokens / block_size 4 = 10
    // blocks, the last partially filled) - the paging-granularity case the
    // task asked for, not a single-block toy shape.
    let t = 37u32;

    let mut alloc = BlockAllocator::new(num_blocks, block_size);
    let mut bt = BlockTable::new();
    let mut blocks_h = Vec::with_capacity(t as usize);
    let mut offsets_h = Vec::with_capacity(t as usize);
    for _ in 0..t {
        let (b, o) = bt.append(&mut alloc).expect("pool has room for t tokens");
        blocks_h.push(b);
        offsets_h.push(o);
    }
    assert!(bt.blocks().len() > 1, "{label}: sequence must span more than one physical block");

    let mut rng = Lcg::new(0xB9_1000 ^ label.len() as u64);
    let k_h: Vec<f32> = rng.vec_scaled((t * kv_stride) as usize, 1.0);
    let v_h: Vec<f32> = rng.vec_scaled((t * kv_stride) as usize, 1.0);
    let q_h: Vec<f32> = rng.vec_scaled((n_heads * head_dim) as usize, 1.0);

    let src_k = g.storage_init("src_k", &k_h);
    let src_v = g.storage_init("src_v", &v_h);
    let blocks_buf = g.storage(t as u64);
    g.write(&blocks_buf, &blocks_h);
    let offsets_buf = g.storage(t as u64);
    g.write(&offsets_buf, &offsets_h);

    // `kv_stride` is even here (the realistic case), so every token's packed
    // word is entirely private to it - no cross-token RMW hazard, so all `t`
    // tokens can be appended in ONE batched dispatch (the batch axis just
    // means "t independent (src, block, offset) triples", it does not care
    // whether they came from t different sequences or t steps of ONE
    // sequence). The read-modify-write stress test below is what exercises
    // the ODD-kv_stride, shared-word, sequential-dispatch case.
    let pool_k_f32 = KvPage::zeros(&ops, num_blocks, block_size, kv_stride, Dtype::F32);
    let pool_k_bf16 = KvPage::zeros(&ops, num_blocks, block_size, kv_stride, Dtype::BF16);
    let pool_v_f32 = KvPage::zeros(&ops, num_blocks, block_size, kv_stride, Dtype::F32);
    let pool_v_bf16 = KvPage::zeros(&ops, num_blocks, block_size, kv_stride, Dtype::BF16);

    let mut append_steps = Vec::new();
    ops.kv_append_batched(&mut append_steps, &pool_k_f32, &src_k, &blocks_buf, &offsets_buf, t, kv_stride, block_size);
    ops.kv_append_batched(&mut append_steps, &pool_k_bf16, &src_k, &blocks_buf, &offsets_buf, t, kv_stride, block_size);
    ops.kv_append_batched(&mut append_steps, &pool_v_f32, &src_v, &blocks_buf, &offsets_buf, t, kv_stride, block_size);
    ops.kv_append_batched(&mut append_steps, &pool_v_bf16, &src_v, &blocks_buf, &offsets_buf, t, kv_stride, block_size);
    g.submit(&[], &append_steps);

    // VRAM claim, exercised for real (not just the pure arithmetic check in
    // `model::ops::tests`): the bf16 pool's own word count is exactly half.
    let f32_words = KvPage::word_count(num_blocks, block_size, kv_stride, Dtype::F32);
    let bf16_words = KvPage::word_count(num_blocks, block_size, kv_stride, Dtype::BF16);
    assert_eq!(bf16_words * 2, f32_words, "{label}: bf16 KV pool must be exactly half the f32 pool's word count");

    // Decode-time attention: one query, seq_lens=[t].
    let q_buf = g.storage_init("q", &q_h);
    let block_table_buf = g.storage(bt.blocks().len() as u64);
    g.write(&block_table_buf, bt.blocks());
    let seq_lens_buf = g.storage(1);
    g.write(&seq_lens_buf, &[t]);
    let max_bt = bt.blocks().len() as u32;
    let cap = t;
    let scale = qk_scale(head_dim);
    let shape =
        PagedDecodeShape { batch: 1, n_heads, group, head_dim, block_size, kv_stride, cap, max_bt, scale };

    let scores_f32 = g.storage((n_heads * cap) as u64);
    let scores_bf16 = g.storage((n_heads * cap) as u64);
    let mut score_steps = Vec::new();
    ops.decode_scores_batched(&mut score_steps, &q_buf, &pool_k_f32, &block_table_buf, &seq_lens_buf, &scores_f32, shape);
    ops.decode_scores_batched(&mut score_steps, &q_buf, &pool_k_bf16, &block_table_buf, &seq_lens_buf, &scores_bf16, shape);
    g.submit(&[], &score_steps);
    let got_scores_f32 = g.read(&scores_f32, (n_heads * cap) as usize);
    let got_scores_bf16 = g.read(&scores_bf16, (n_heads * cap) as usize);

    let mut worst_scores = 0f32;
    let mut max_score_tol_per_head = vec![0f32; n_heads as usize];
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q_h[(h * head_dim) as usize..((h + 1) * head_dim) as usize];
        for j in 0..t {
            let kbase = (j * kv_stride + kvh * head_dim) as usize;
            let k_tok = &k_h[kbase..kbase + head_dim as usize];
            let tol = scores_tol(qh, k_tok, scale);
            max_score_tol_per_head[h as usize] = max_score_tol_per_head[h as usize].max(tol);
            let idx = (h * cap + j) as usize;
            let err = (got_scores_f32[idx] - got_scores_bf16[idx]).abs();
            worst_scores = worst_scores.max(err / tol.max(1e-12));
            assert!(
                err <= tol,
                "{label} scores h={h} j={j}: f32={} bf16={} (err {err}, tol {tol})",
                got_scores_f32[idx],
                got_scores_bf16[idx]
            );
        }
    }
    eprintln!("{label} scores (K-only bf16): worst err/tol ratio {worst_scores:.4}");

    // Softmax over the EXACT fp32-K scores -> the shared reference probs
    // used to isolate V's own narrowing error from softmax's nonlinearity.
    let softmax_idx = g.kernel_index("decode_softmax_batched").expect("decode_softmax_batched registered");
    let probs_ref = g.storage((n_heads * cap) as u64);
    let sm = vec![g.step(softmax_idx, &[&scores_f32, &seq_lens_buf, &probs_ref], &[1, n_heads, cap], n_heads)];
    g.submit(&[], &sm);
    let probs_ref_h = g.read(&probs_ref, (n_heads * cap) as usize);

    let ctx_f32 = g.storage((n_heads * head_dim) as u64);
    let ctx_bf16_v = g.storage((n_heads * head_dim) as u64);
    let mut apply_steps = Vec::new();
    ops.decode_apply_batched(&mut apply_steps, &probs_ref, &pool_v_f32, &block_table_buf, &seq_lens_buf, &ctx_f32, shape);
    ops.decode_apply_batched(&mut apply_steps, &probs_ref, &pool_v_bf16, &block_table_buf, &seq_lens_buf, &ctx_bf16_v, shape);
    g.submit(&[], &apply_steps);
    let got_ctx_f32 = g.read(&ctx_f32, (n_heads * head_dim) as usize);
    let got_ctx_bf16_v = g.read(&ctx_bf16_v, (n_heads * head_dim) as usize);

    let mut worst_apply = 0f32;
    for h in 0..n_heads {
        let kvh = h / group;
        let probs_row = &probs_ref_h[(h * cap) as usize..(h * cap + t) as usize];
        for d in 0..head_dim {
            let v_col: Vec<f32> = (0..t).map(|j| v_h[(j * kv_stride + kvh * head_dim + d) as usize]).collect();
            let tol = apply_tol(probs_row, &v_col);
            let idx = (h * head_dim + d) as usize;
            let err = (got_ctx_f32[idx] - got_ctx_bf16_v[idx]).abs();
            worst_apply = worst_apply.max(err / tol.max(1e-12));
            assert!(
                err <= tol,
                "{label} apply(shared probs) h={h} d={d}: f32={} bf16_v={} (err {err}, tol {tol})",
                got_ctx_f32[idx],
                got_ctx_bf16_v[idx]
            );
        }
    }
    eprintln!("{label} apply (V-only bf16, shared probs): worst err/tol ratio {worst_apply:.4}");

    // Full end-to-end bf16 pipeline (K+V both bf16, its OWN softmax) -
    // sanity check with a generous bound, see module doc.
    let probs_bf16 = g.storage((n_heads * cap) as u64);
    let sm2 = vec![g.step(softmax_idx, &[&scores_bf16, &seq_lens_buf, &probs_bf16], &[1, n_heads, cap], n_heads)];
    g.submit(&[], &sm2);
    let ctx_full_bf16 = g.storage((n_heads * head_dim) as u64);
    let mut full_steps = Vec::new();
    ops.decode_apply_batched(&mut full_steps, &probs_bf16, &pool_v_bf16, &block_table_buf, &seq_lens_buf, &ctx_full_bf16, shape);
    g.submit(&[], &full_steps);
    let got_full = g.read(&ctx_full_bf16, (n_heads * head_dim) as usize);

    let mut worst_full = 0f32;
    for h in 0..n_heads {
        let kvh = h / group;
        let probs_row = &probs_ref_h[(h * cap) as usize..(h * cap + t) as usize];
        for d in 0..head_dim {
            let v_col: Vec<f32> = (0..t).map(|j| v_h[(j * kv_stride + kvh * head_dim + d) as usize]).collect();
            let v_abs_sum: f32 = v_col.iter().map(|v| v.abs()).sum();
            let tol = apply_tol(probs_row, &v_col) + 2.0 * max_score_tol_per_head[h as usize] * v_abs_sum;
            let idx = (h * head_dim + d) as usize;
            let err = (got_ctx_f32[idx] - got_full[idx]).abs();
            worst_full = worst_full.max(err / tol.max(1e-12));
            assert!(
                err <= tol,
                "{label} full pipeline h={h} d={d}: f32={} full_bf16={} (err {err}, tol {tol})",
                got_ctx_f32[idx],
                got_full[idx]
            );
        }
    }
    eprintln!("{label} full pipeline (own bf16 softmax, generous bound): worst err/tol ratio {worst_full:.4}");
}

/// The read-modify-write stress test: an ODD `kv_stride` so slot 0's LAST
/// element and slot 1's FIRST element share one packed `u32` word - see this
/// file's module doc for why. Two tokens, two SEPARATE sequential dispatches,
/// read back via a one-hot `probs` vector through `Ops::decode_apply_batched`.
fn run_rmw_stress(gpu: Gpu, label: &str) {
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let (n_heads, group, head_dim) = (1u32, 1u32, 3u32); // n_kv=1, kv_stride=3 (ODD)
    let kv_stride = head_dim;
    let block_size = 2u32; // slot0=offset0, slot1=offset1, one physical block
    let num_blocks = 1u32;

    let pool_bf16 = KvPage::zeros(&ops, num_blocks, block_size, kv_stride, Dtype::BF16);

    let mut rng = Lcg::new(0xB9_5EED ^ label.len() as u64);
    let tok_a: Vec<f32> = rng.vec_scaled(kv_stride as usize, 1.0);
    let tok_b: Vec<f32> = rng.vec_scaled(kv_stride as usize, 1.0);

    // Append slot 0 (token A), fully complete before slot 1 starts - the
    // sequencing that makes this a valid read-modify-write, not a race (see
    // `kernels::template::rewrite_packed_stores`'s own doc comment).
    let src_a = g.storage_init("tok_a", &tok_a);
    let blocks_a = g.storage(1);
    g.write(&blocks_a, &[0u32]);
    let offsets_a = g.storage(1);
    g.write(&offsets_a, &[0u32]);
    let mut sa = Vec::new();
    ops.kv_append_batched(&mut sa, &pool_bf16, &src_a, &blocks_a, &offsets_a, 1, kv_stride, block_size);
    g.submit(&[], &sa);

    // Append slot 1 (token B) - its FIRST element shares a packed word with
    // slot 0's LAST element (see this file's module doc for the exact
    // parity argument). A botched pack that fails to preserve the sibling
    // half would corrupt token A's last element right here.
    let src_b = g.storage_init("tok_b", &tok_b);
    let blocks_b = g.storage(1);
    g.write(&blocks_b, &[0u32]);
    let offsets_b = g.storage(1);
    g.write(&offsets_b, &[1u32]);
    let mut sb = Vec::new();
    ops.kv_append_batched(&mut sb, &pool_bf16, &src_b, &blocks_b, &offsets_b, 1, kv_stride, block_size);
    g.submit(&[], &sb);

    let block_table_buf = g.storage(1);
    g.write(&block_table_buf, &[0u32]);
    let seq_lens_buf = g.storage(1);
    g.write(&seq_lens_buf, &[2u32]);
    let shape =
        PagedDecodeShape { batch: 1, n_heads, group, head_dim, block_size, kv_stride, cap: 2, max_bt: 1, scale: 1.0 };

    for (slot, tok) in [(0u32, &tok_a), (1u32, &tok_b)] {
        let probs_h: [f32; 2] = if slot == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
        let probs_buf = g.storage_init("probs", &probs_h);
        let ctx = g.storage(head_dim as u64);
        let mut steps = Vec::new();
        ops.decode_apply_batched(&mut steps, &probs_buf, &pool_bf16, &block_table_buf, &seq_lens_buf, &ctx, shape);
        g.submit(&[], &steps);
        let got = g.read(&ctx, head_dim as usize);
        for d in 0..head_dim as usize {
            let want = tok[d];
            let tol = want.abs() * 2f32.powi(-8) + 1e-4;
            let err = (got[d] - want).abs();
            assert!(
                err <= tol,
                "{label} RMW stress slot={slot} d={d}: got {} want {want} (err {err}, tol {tol}) -- a botched \
                 pack that clobbers the sibling half would fail exactly here",
                got[d]
            );
        }
    }
    eprintln!("{label} RMW stress: both adjacent-slot tokens (sharing one packed word) survived intact");
}

#[test]
fn kv_bf16_long_context_parity_on_cpu() {
    run_long_context_parity(Gpu::new_cpu(&kernel_list()), "cpu");
}

#[test]
fn kv_bf16_long_context_parity_on_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("kv_bf16_long_context_parity_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("kv_bf16_long_context_parity_on_gpu: running on a real wgpu device");
    run_long_context_parity(Gpu::new_wgpu(&kernel_list()), "gpu");
}

#[test]
fn kv_bf16_append_rmw_shared_word_preserves_both_adjacent_slots_on_cpu() {
    run_rmw_stress(Gpu::new_cpu(&kernel_list()), "cpu");
}

#[test]
fn kv_bf16_append_rmw_shared_word_preserves_both_adjacent_slots_on_gpu() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("kv_bf16_append_rmw_shared_word_preserves_both_adjacent_slots_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("kv_bf16_append_rmw_shared_word_preserves_both_adjacent_slots_on_gpu: running on a real wgpu device");
    run_rmw_stress(Gpu::new_wgpu(&kernel_list()), "gpu");
}
