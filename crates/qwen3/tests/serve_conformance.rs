// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching LLM serving for clients
// whose agents resend long conversation histories. If your team needs
// expertise in LLM serving throughput then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The qwen3 serving engine through the generic [`model::serve::conformance`]
//! suite, at both KV tiers, with an adapter bank. Everything the suite checks
//! (continuous batching, no head-of-line blocking, prefix reuse, adapter KV
//! isolation) is the scheduler's behaviour over the engine contract; this file
//! only builds a tiny engine and an adapter loader.

use model::serve::conformance::{self, Conformance, MIN_SEQ_TOKENS};
use model::serve::AdapterId;
use qwen3::lora::{AdapterSite, DeviceAdapter};
use qwen3::serve::{AdapterBankConfig, Engine, LoraSite, VerifiedAdapter};
use qwen3::QwenConfig;
use std::sync::atomic::{AtomicU64, Ordering};

const BLOCK: u32 = 4;
const R_MAX: u32 = 4;

fn engine(kv_int8: bool) -> Engine {
    let cfg = QwenConfig::tiny();
    let w = qwen3::init_weights(&cfg, 7);
    let blocks_per_seq = (MIN_SEQ_TOKENS as u32).div_ceil(BLOCK);
    let mut e = Engine::from_map_on(&gpu_core::testgpu::dev(&[]), cfg, &w, BLOCK, 96, 4, blocks_per_seq, 16, kv_int8, false);
    // Outputs are compared bit for bit between a cold prefill and one that reuses cached blocks: pin exact fp32.
    e.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Fp32Reference);
    e.enable_adapters(AdapterBankConfig::new(2, R_MAX, &LoraSite::ALL)).unwrap();
    e
}

/// A synthetic adapter over every linear of every layer, big enough to move
/// the logits visibly.
fn adapter(cfg: &QwenConfig, rank: u32, seed: u64) -> DeviceAdapter {
    let (d, ff, hq, hkv) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.q_dim() as usize, cfg.kv_dim() as usize);
    let leaves = [("attn.wq", hq, d), ("attn.wk", hkv, d), ("attn.wv", hkv, d), ("attn.wo", d, hq), ("mlp.gate", ff, d), ("mlp.up", ff, d), ("mlp.down", d, ff)];
    let mut rng = seed | 1;
    let mut next = move |amp: f32| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
    };
    let mut sites = Vec::new();
    for l in 0..cfg.n_layers {
        for (leaf, out, inn) in leaves {
            let a = (0..rank as usize * inn).map(|_| next(1.0 / (inn as f32).sqrt())).collect();
            let b = (0..out * rank as usize).map(|_| next(0.8)).collect();
            sites.push(AdapterSite { base: format!("blocks.{l}.{leaf}.weight"), a, b });
        }
    }
    DeviceAdapter { rank, alpha: 2.0 * rank as f32, sites }
}

fn verified(cfg: &QwenConfig, rank: u32, seed: u64) -> VerifiedAdapter {
    VerifiedAdapter { label: format!("adapter-{seed}"), digest: format!("sha256:{seed:064x}"), adapter: adapter(cfg, rank, seed) }
}


fn load_adapter(e: &mut Engine) -> AdapterId {
    static NEXT: AtomicU64 = AtomicU64::new(100);
    let seed = NEXT.fetch_add(1, Ordering::SeqCst);
    let cfg = QwenConfig::tiny();
    e.load_adapter(verified(&cfg, 2, seed)).unwrap()
}

#[test]
fn qwen3_with_fp32_kv_conforms() {
    conformance::run_all(&Conformance { make: || engine(false), prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: Some(load_adapter) });
}

#[test]
fn qwen3_with_int8_kv_conforms() {
    conformance::run_all(&Conformance { make: || engine(true), prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: Some(load_adapter) });
}
