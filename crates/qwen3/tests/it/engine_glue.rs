// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping the memory passes between a served language model's products
// to the ones the arithmetic needs for its clients. If your team needs expertise in GPU inference servers then you can
// procure our services by sending an email to info@swedishembedded.com.

//! What runs between the products of a forward of the paged serving engine on bf16 tensor cores. Each fused weight
//! (`wq;wk;wv`, `gate;up`) was one product into a wide scratch that `concat_split` then copied apart, and the SwiGLU
//! was `silu_mul` followed by the down product's own bf16 cast re-reading the whole activation: at a long prefill those
//! copies moved more bytes than the products that surround them. On the library path each part of a fused weight is
//! its own product into the buffer its reader takes, sharing one activation cast, and the SwiGLU writes the down
//! product's bf16 operand in the same pass, as does each residual norm whose only readers are bf16 products. The QK norm and rotation run one block a token, not a workgroup a head.

use qwen3::serve::Engine;
use qwen3::{Dtype, QwenConfig};

#[test]
fn a_bf16_tensor_core_forward_copies_no_fused_product_apart() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    // Library-eligible widths (16-element rows), as the prefill census takes them.
    let cfg = QwenConfig { d_model: 64, d_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 64, block_size: 256, max_position_embeddings: 256, tie_embeddings: false, ..QwenConfig::tiny() };
    let w = qwen3::init_weights(&cfg, 5); // seed: any weights serve, the test names kernels
    let (bs, mbt) = (16u32, 16u32);
    let e = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, bs, 64, 4, mbt, 32, true, Dtype::BF16);
    // Portable kernels by catalogue name, native and library steps by the name they report.
    let names = |steps: &[gpu_core::Step]| -> Vec<String> {
        steps.iter().filter_map(|s| s.lowering().map(|l| l.physical.to_string()).or_else(|| s.meta().and_then(|m| e.gpu().kernel_name(m.kernel).map(str::to_string)))).collect()
    };
    let count = |names: &[String], kernel: &str| names.iter().filter(|n| *n == kernel).count();

    // A prefill chunk of one sequence, and a decode step of four.
    let rows = 32u32;
    let tokens: Vec<u32> = (0..rows).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let positions: Vec<u32> = (0..rows).collect();
    let seqlens: Vec<u32> = (1..=rows).collect();
    let blocks: Vec<u32> = (0..rows).map(|i| i / bs).collect();
    let offsets: Vec<u32> = (0..rows).map(|i| i % bs).collect();
    let bt: Vec<u32> = (0..rows).flat_map(|_| 0..mbt).collect();
    let prefill = names(&e.steps_for_profile(rows, &tokens, &positions, &seqlens, &blocks, &offsets, &bt, true));
    let decode = names(&e.steps_for_profile(4, &tokens[..4], &[0, 1, 2, 3], &[1; 4], &[4, 5, 6, 7], &[0; 4], &(4..8).flat_map(|b| std::iter::once(b).chain(std::iter::repeat_n(0, mbt as usize - 1))).collect::<Vec<u32>>(), false));
    for (what, names) in [("prefill", &prefill), ("decode", &decode)] {
        assert!(!names.is_empty(), "{what}: the tape names its kernels");
        assert_eq!(count(names, "concat_split"), 0, "{what}: a fused product was copied apart: {names:?}");
        assert_eq!(count(names, "silu_mul"), 0, "{what}: the SwiGLU's bf16 operand took a pass of its own: {names:?}");
        // The query's and key's norms and rotations run as one native launch (angles once for four heads of a token).
        assert_eq!(names.iter().filter(|n| n.contains("qknorm_rope")).count(), cfg.n_layers as usize, "{what}: the query and key norms and rotations are one native launch a layer: {names:?}");
        // The int8 K and V appends of a layer go as one launch.
        assert_eq!(count(names, "paged_kv_append_i8_clipped_batched"), 0, "{what}: the K and V appends were two launches: {names:?}");
        // A residual norm read only by bf16 products writes their operand itself: what is left to cast is the first
        // layer's input norm and each layer's attention output.
        let casts = names.iter().filter(|n| n.contains("cast_f32_bf16")).count();
        assert!(casts <= cfg.n_layers as usize + 1, "{what}: {casts} activation casts for {} layers: {names:?}", cfg.n_layers);
    }
}

/// On the bf16 library path the fused products' wide outputs are never written, so the engine never allocates them;
/// the paths that write them (here the reference GEMM tier) still get them.
#[test]
fn a_bf16_library_engine_holds_no_fused_product_scratch() {
    let Ok(gpu) = gpu_core::Gpu::try_new_cuda(qwen3::model::pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let cfg = QwenConfig { d_model: 64, d_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 64, block_size: 256, max_position_embeddings: 256, tie_embeddings: false, ..QwenConfig::tiny() };
    let w = qwen3::init_weights(&cfg, 7); // seed: any weights serve
    let prompt: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let run = |e: &mut Engine| {
        let mut t = model::paged::BlockTable::new();
        e.prefill_for_perf(&mut t, &prompt);
        let mut refs = [&mut t];
        model::serve::PagedDecoder::forward_batched_greedy(e, &mut refs, &[1]);
    };
    let mut e = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 64, 4, 16, 32, true, Dtype::BF16);
    run(&mut e);
    assert_eq!(e.fused_scratch_bytes(), 0, "the library path allocated the fused outputs it never writes");
    let mut reference = Engine::from_map_tier_on(&gpu, cfg.clone(), &w, 16, 64, 4, 16, 32, true, Dtype::BF16);
    reference.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Fp32Reference);
    run(&mut reference);
    assert!(reference.fused_scratch_bytes() > 0, "the fused path ran without its scratch");
}
