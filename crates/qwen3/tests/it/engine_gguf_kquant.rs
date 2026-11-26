// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of quantized checkpoints in the
// precision they were released in for its clients. If your team needs
// expertise in running GGUF K-quant models on the hardware it already owns,
// you can procure our services by sending an email to info@swedishembedded.com.

//! A GGUF whose linears are block-quantized the way llama.cpp's Q4_K_M mixes
//! formats is served as stored, on every device: each linear is held at the
//! tier that keeps its format (Q4_K as Q4K, Q5_K as Q8K, the rest int8), a
//! fused group of mixed formats takes int8, the planner charges exactly the
//! bytes the engine uploads, and the engine computes what an fp32 engine does
//! on the stored values (within the int8 re-quantization of the mixed group
//! and the head, and on a GPU the int8 activations).

use std::collections::HashMap;

use checkpoint::gguf::{GgmlType, MmapGguf};
use checkpoint::gguf_write::{write, TensorOut};
use model::paged::BlockTable;
use qwen3::serve::Engine;
use qwen3::{Dtype, QwenConfig};

/// The storage format of each tensor: layer 0 mixes formats inside the Q/K/V group as a Q4_K_M does (Q6_K `wv`),
/// layer 1 is Q4_K throughout, and the tied embedding (the head) is Q6_K.
fn format_of(name: &str) -> GgmlType {
    match name {
        "blocks.0.attn.wv.weight" | "tok.weight" => GgmlType::Q6K,
        "blocks.0.attn.wo.weight" => GgmlType::Q5K,
        "blocks.0.mlp.down.weight" => GgmlType::Q8_0,
        n if qwen3::q8::Q8::is_i8_linear(n) => GgmlType::Q4K,
        _ => GgmlType::F32,
    }
}

#[test]
fn a_mixed_k_quant_gguf_is_held_as_stored_priced_as_held_and_computes_the_stored_model() {
    let _device = gpu_core::testgpu::exclusive();
    // Every contraction width a whole number of 256-element K-quant super-blocks.
    let cfg = QwenConfig { vocab: 512, d_model: 256, n_heads: 4, n_kv_heads: 2, head_dim: 64, d_ff: 512, block_size: 64, max_position_embeddings: 64, ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 11);
    let path = std::env::temp_dir().join(format!("brain-qwen3-kquant-{}.gguf", std::process::id())).to_string_lossy().into_owned();
    let tensors: Vec<TensorOut> = cfg
        .param_list()
        .into_iter()
        .map(|(name, count)| {
            let ty = format_of(&name);
            let shape = if count == cfg.d_model as usize || name.contains("norm") { vec![count] } else { let k = if name.ends_with("wo.weight") { cfg.q_dim() } else if name.ends_with("down.weight") { cfg.d_ff } else { cfg.d_model } as usize; vec![k, count / k] };
            let data = if ty == GgmlType::F32 { init[&name].iter().flat_map(|x| x.to_le_bytes()).collect() } else { checkpoint::quant::quantize_par(ty.id(), &init[&name]).unwrap() };
            TensorOut { name, shape, ty: ty.id(), data }
        })
        .collect();
    write(&path, &[], &tensors, 32).unwrap();
    let mg = MmapGguf::open(&path).unwrap();

    let policy = qwen3::stored_weight_policy(&cfg, &mg).expect("the linears are block-quantized");
    let expected = |name: &str| match name {
        "blocks.0.attn.wq.weight" | "blocks.0.attn.wk.weight" | "blocks.0.attn.wv.weight" | "blocks.0.mlp.down.weight" | "tok.weight" => Dtype::I8,
        "blocks.0.attn.wo.weight" => Dtype::Q8K,
        _ => Dtype::Q4K,
    };
    let linears: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| qwen3::q8::Q8::is_i8_linear(n)).chain([cfg.head_weight().to_string()]).collect();
    for name in &linears {
        assert_eq!(policy.want(name), expected(name), "{name}");
    }
    Engine::check_tiers(&cfg, &policy).expect("the stored policy is a servable one");

    let mut engine = Engine::from_source_tiers(cfg.clone(), &mg, 4, 32, 1, 16, 64, false, &policy);
    let held = engine.linear_tiers();
    // The engine holds Q/K/V and gate/up as one fused weight each.
    let fused = |name: &str| name.replace(".attn.wq.", ".attn.wqkv.").replace(".attn.wk.", ".attn.wqkv.").replace(".attn.wv.", ".attn.wqkv.").replace(".mlp.gate.", ".mlp.gateup.").replace(".mlp.up.", ".mlp.gateup.");
    for name in &linears {
        assert_eq!(held[&fused(name)], expected(name), "{name} is held as stored on this device, never widened");
    }
    let tables: u64 = cfg.param_list().into_iter().filter(|(n, _)| !qwen3::q8::Q8::is_i8_linear(n)).map(|(_, c)| c as u64 * 4).sum();
    assert_eq!(qwen3::footprint::held_weight_bytes_by(&cfg, |n| policy.want(n)), engine.linear_weight_bytes() + tables, "the planner charges what the engine holds");

    // The reference: fp32 arithmetic on the values the file stores.
    let stored: HashMap<String, Vec<f32>> = cfg.param_list().into_iter().map(|(name, _)| (name.clone(), mg.tensor(&name).unwrap().unwrap())).collect();
    let mut reference = Engine::from_map(cfg.clone(), &stored, 4, 32, 1, 16, 64, false, false);
    for eng in [&mut engine, &mut reference] {
        eng.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Fp32Reference);
    }
    let prompt: Vec<u32> = (0..40).map(|i| (i * 37 + 5) % cfg.vocab).collect();
    let run = |eng: &mut Engine| {
        let mut table = BlockTable::new();
        eng.prefill_for_perf(&mut table, &prompt)
    };
    let (got, want) = (run(&mut engine), run(&mut reference));
    let rel = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt() / want.iter().map(|b| b * b).sum::<f32>().sqrt();
    assert!(rel < 2e-2, "prefill hidden state differs from the stored model's by {rel:e} (relative L2)");
    std::fs::remove_file(&path).ok();
}
