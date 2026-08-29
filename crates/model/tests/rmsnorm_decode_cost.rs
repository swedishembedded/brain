// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What one generated token's worth of RMSNorm costs, per-element kernel vs
//! coalesced, at the decode shapes real models really dispatch.
//!
//! `#[ignore]`d: it is a MEASUREMENT, not a gate. The gate is
//! `rmsnorm_variant_agreement.rs` next to it.
//!
//! Why it exists here rather than per model: the models whose decode tapes
//! this prices are 35B-155B parameters and do not fit on development
//! hardware, so a whole-pass A/B is impossible for them - but the RMSNorm
//! dispatch list per token is a property of the config alone, and that list
//! is exactly what the kernel swap changes. Pricing the list is the honest
//! measurement available.
//!
//! Read it as an upper bound on the win, not as a whole-pass figure: it says
//! what the norms alone cost, and what fraction of a served token that is
//! depends on the rest of the pass, which is NOT measured here. The one
//! whole-pass number this change can point at is qwen35's, on real weights on
//! real hardware: the same swap took its decode from 3.94 to 7.44 tok/s, with
//! RMSNorm falling from 48% of device time to 4.6%.
//!
//! The shapes are transcribed from each model's decode step at its named
//! config. Transcribed tables go stale: re-read the tape rather than trusting
//! a row here, and re-run rather than trusting a number printed in a comment.
//!
//! Run:
//!   cargo test --release -p brain-model --test rmsnorm_decode_cost -- --ignored --nocapture
//!
//! Swedish Embedded AB implements GPU kernel selection and profiling for its
//! clients. If your team needs expertise in decode-regime inference
//! optimization then you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::block::{self, rmsnorm_fwd, KernelIds};

const PIPELINES: &[(&str, &str)] = &[("rmsnorm", kernels::RMSNORM), ("rmsnorm_rows", kernels::RMSNORM_ROWS)];

/// `(rows, dim, dispatches per generated token)`.
type Profile = &'static [(u32, u32, u32)];

/// One decode step's RMSNorm dispatch list, per model.
const MODELS: &[(&str, Profile)] = &[
    // qwen35moe, Qwen3.5-35B-A3B: 40 layers, `full_attention_interval = 4`
    // -> 10 GQA + 30 GDN. ln1 + ln2 per layer plus the final norm are the
    // one-row residual norms; the GQA layers add QK-norm, the GDN layers the
    // gated output norm.
    ("qwen35moe 35B-A3B", &[(1, 2048, 81), (16, 256, 10), (2, 256, 10), (32, 128, 30)]),
    // qwen3omnimoe Thinker: 48 layers, hidden 2048, 32/4 heads of 128.
    ("qwen3omnimoe thinker", &[(1, 2048, 97), (32, 128, 48), (4, 128, 48)]),
    // qwen3omnimoe Talker: 20 layers, hidden 1024, 16/2 heads of 128.
    ("qwen3omnimoe talker", &[(1, 1024, 41), (16, 128, 20), (2, 128, 20)]),
    // glmdsa GLM-5.2: 78 layers, d_model 6144, q_lora_rank 2048,
    // kv_lora_rank 512. Four one-row norms per layer (input_ln, q_a_norm,
    // kv_a_norm, post_ln) plus the final norm - MLA norms the two low-rank
    // latents, so three different widths.
    ("glmdsa GLM-5.2", &[(1, 6144, 157), (1, 2048, 78), (1, 512, 78)]),
    // qwen3tts Talker: 28 layers, d_model 1024, 16/8 heads of 128.
    ("qwen3tts talker", &[(1, 1024, 57), (16, 128, 28), (8, 128, 28)]),
    // qwen3tts MTP, per AUDIO FRAME: the 5-layer tape is replayed 15 times,
    // and its rows are the 16 code groups, never 1 - the one adopting tape
    // here that is not decode-shaped.
    ("qwen3tts mtp (per frame)", &[(16, 1024, 165), (256, 128, 75), (128, 128, 75)]),
    // deepseek2 DeepSeek-OCR: 12 layers, d_model 1280, plain MHA (no QK-norm).
    ("deepseek2 DeepSeek-OCR", &[(1, 1280, 25)]),
];

fn ids(rmsnorm_rows: usize) -> KernelIds {
    KernelIds {
        rmsnorm: 0,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        rope: block::UNREGISTERED,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: block::UNREGISTERED,
        gqa_apply: block::UNREGISTERED,
        attn_softmax: block::UNREGISTERED,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: block::UNREGISTERED,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows,
    }
}

#[test]
#[ignore]
fn price_one_decode_token_of_rmsnorm_per_model() {
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let reps = 20;

    println!("{:<26} {:>12} {:>12} {:>9}", "model (one decode token)", "rmsnorm", "rmsnorm_rows", "speedup");
    for (name, profile) in MODELS {
        let mut ms = [0.0f64; 2];
        for (arm, slot) in [(0usize, block::UNREGISTERED), (1usize, 1usize)] {
            let ids = ids(slot);
            // Build every buffer once; the measurement is dispatch cost, not
            // allocation cost.
            let bufs: Vec<_> = profile
                .iter()
                .map(|&(rows, dim, _)| {
                    let n = (rows * dim) as usize;
                    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 + 0.1).sin()).collect();
                    let w: Vec<f32> = (0..dim as usize).map(|i| (i as f32 * 0.31).cos()).collect();
                    (gpu.storage_init("x", &x), gpu.storage_init("w", &w), gpu.storage(n as u64))
                })
                .collect();
            let mut steps = Vec::new();
            for (&(rows, dim, count), (xb, wb, ob)) in profile.iter().zip(&bufs) {
                for _ in 0..count {
                    steps.push(rmsnorm_fwd(&gpu, &ids, xb, wb, ob, dim, rows));
                }
            }
            // `submit` only QUEUES: without a readback to drain it the loop
            // below times enqueue cost, which is the same for both kernels and
            // reports a dead heat (or worse, noise) for a real 20x. Read one
            // float back each rep - the same "the flush is the measurement"
            // discipline the qwen35 decode profiler needed.
            gpu.submit(&[], &steps); // warm
            let _ = gpu.read(&bufs[0].2, 1);
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                gpu.submit(&[], &steps);
                let _ = gpu.read(&bufs[0].2, 1);
            }
            ms[arm] = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        }
        println!("{name:<26} {:>9.3} ms {:>9.3} ms {:>8.1}x", ms[0], ms[1], ms[0] / ms[1]);
    }
}
