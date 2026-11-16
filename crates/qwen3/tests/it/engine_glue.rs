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
//! product's bf16 operand in the same pass. The QK norm and rotation run one block a token, not a workgroup a head.

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
    let names = |steps: &[gpu_core::Step]| -> Vec<String> { steps.iter().filter_map(|s| s.meta()).filter_map(|m| e.gpu().kernel_name(m.kernel).map(str::to_string)).collect() };
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
        // The native rows compute the rotation angles once for four heads of a token.
        assert_eq!(count(names, "qknorm_rope_fused"), 0, "{what}: the QK norm and rotation ran a workgroup per head: {names:?}");
    }
}
