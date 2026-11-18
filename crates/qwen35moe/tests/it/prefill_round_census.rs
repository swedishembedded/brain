// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping an accelerator busy
// through the prefill of a sparse mixture-of-experts model for its clients. If
// your team needs expertise in GPU inference servers then you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the host does per chunked-prefill round of the int8 engine on CUDA, from
//! the call totals of the engine's own handle. A 16K prompt of the 35B-A3B is
//! dozens of rounds of 40 layers each, and a `cuMemAlloc`/`cuMemFree` (each one
//! waits for the whole device) or a host wait per layer left the card idle for
//! a fifth of the prefill. A steady-state round does neither: its temporaries come
//! from blocks the handle keeps, and on a backend whose submissions run in order
//! nothing in a round needs the host to wait for the device.

use std::collections::HashMap;

use gpu_core::{select::Dtype, Gpu};
use model::ops::TierPolicy;
use model::paged::BlockTable;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::pipelines;
use checkpoint::gguf::MmapGguf;
use qwen35moe::gguf_load;
use qwen35moe::serve::{Engine, EngineOptions};

/// `engine_tiers`'s int8-friendly tiny model: every quantised width a multiple of 32.
fn cfg() -> Qwen35Config {
    Qwen35Config {
        vocab: 29,
        block_size: 256,
        n_layers: 8,
        d_model: 32,
        rms_eps: 1e-6,
        max_position_embeddings: 256,
        tie_embeddings: false,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        attn_bias: false,
        rope_theta: 1.0e6,
        partial_rotary_factor: 0.5,
        mrope_section: [1, 1, 1],
        full_attention_interval: 4,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 3,
        n_experts: 6,
        top_k: 2,
        moe_intermediate_size: 32,
        shared_expert_intermediate_size: 32,
        lora: None,
    }
}

#[test]
fn a_steady_state_prefill_round_neither_allocates_nor_waits_for_the_device() {
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let c = cfg();
    let w: HashMap<String, Vec<f32>> = qwen35moe::init::init_weights(&c, 5); // seed: any weights serve, the census counts calls
    const ROUND: u32 = 32;
    let mut e = Engine::from_source_on_parent(&gpu, c.clone(), &w, EngineOptions::new(256, 2).with_tier(TierPolicy::uniform(Dtype::I8)).with_prefill_chunk(ROUND));
    let prompt: Vec<u32> = (0..(4 * ROUND)).map(|i| (i * 7 + 3) % c.vocab).collect();
    let calls = |e: &Engine| e.gpu().call_totals().expect("the CUDA backend counts its host calls");

    // Warm-up: kernels compile, the round's blocks and the sequence state are created once.
    let mut t = BlockTable::new();
    let first = e.prefill(&mut t, &prompt);
    e.release_table(&mut t);

    for pass in 1..3 {
        let before = calls(&e);
        let mut t = BlockTable::new();
        let hidden = e.prefill(&mut t, &prompt);
        let after = calls(&e);
        e.release_table(&mut t);
        assert_eq!(hidden, first, "pass {pass}: the same prompt gave another hidden state");
        assert_eq!(after.device_alloc_calls, before.device_alloc_calls, "pass {pass}: a steady-state prefill called cuMemAlloc, which waits for the device");
        // The one wait is the read of the last row's hidden state, which the caller asked for.
        assert_eq!(after.host_waits - before.host_waits, 1, "pass {pass}: a prefill of {} rounds made the host wait for the device inside it", prompt.len() as u32 / ROUND);
    }
}

/// The memory a placement keeps free for activations ([`qwen35moe::serve::activation_bytes`]) holds what the widest
/// round of the released checkpoint really allocates. Under-charged, a long prompt runs out of device memory after the
/// model loaded, or the handles hand blocks back to the driver every round (knowledge 431).
#[test]
fn the_activations_of_a_prefill_round_of_the_real_checkpoint_fit_their_charge() {
    // The release as `brain pull unsloth/Qwen3.6-35B-A3B-GGUF` stores it.
    let path = format!("{}/Q8_0.gguf", brain_testutil::model_dir("unsloth/Qwen3.6-35B-A3B-GGUF").unwrap_or_default());
    if !std::path::Path::new(&path).is_file() {
        return brain_testutil::skip_unavailable("unsloth/Qwen3.6-35B-A3B-GGUF (Q8_0) is not in the model store");
    }
    // The charge is the int8 engine's; a device without a packed int8 dot holds the request at fp32, a different build.
    if !gpu_core::testgpu::dev(pipelines()).caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("the int8 engine's prefill charge needs the int8_dot capability, absent on this device");
    }
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    const ROWS: u32 = 4096;
    let cfg = gguf_load::resident_config(&mg, 2 * ROWS).expect("the release's config");
    let src = gguf_load::source(&mg, &cfg).expect("the release's tensors");
    let opts = EngineOptions::new(2 * ROWS, 1).with_tier(TierPolicy::uniform(Dtype::I8)).with_kv_tier(model::kv_tier::KvTier::BF16).with_prefill_chunk(ROWS);
    let charged = qwen35moe::serve::activation_bytes(&cfg, &opts);
    let mut e = Engine::from_source(cfg.clone(), &src, opts);
    let Some(resident) = e.gpu().memory_usage().map(|m| m.live) else {
        return brain_testutil::skip_unavailable("this backend reports no device memory use");
    };
    e.gpu().reset_memory_peak();
    let prompt: Vec<u32> = (0..2 * ROWS).map(|i| (i * 7919 + 13) % cfg.vocab).collect();
    let mut t = BlockTable::new();
    e.prefill(&mut t, &prompt);
    e.release_table(&mut t);
    let peak = e.gpu().memory_usage().expect("reported above").peak_live - resident;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!("two {ROWS}-row rounds: activation peak {:.2} GiB, charged {:.2} GiB", gib(peak), gib(charged));
    assert!(peak <= charged, "a {ROWS}-row round allocated {:.2} GiB of activations, the placement charges {:.2} GiB", gib(peak), gib(charged));
}
