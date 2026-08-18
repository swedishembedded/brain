// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real end-to-end validation of `qwen3omnimoe::generate::generate_greedy`: the
//! FULL 48-layer/128-expert Thinker, real weights, real KV-cache decode
//! loop, compared token-for-token against `Qwen3OmniMoeThinkerForConditionalGeneration
//! .generate(...)` (HF's own greedy generation, `use_cache=True`) — the
//! validation the "download the rest, validate for real" decision was for.
//!
//! Every stage this loop composes has already been proven exact in
//! isolation (layer0 cosine 1.000000 on M6a/M6b, the KV-cache primitive's
//! own algebraic-equivalence test in `model::block`) — this test's own risk
//! surface is purely the LOOP: prefill/decode sequencing, cache indexing,
//! tokenizer round-trip, and greedy sampling, chained across 48 real layers
//! for 5 real generated tokens. A single wrong index anywhere in that chain
//! (an off-by-one in `cache_row`, a stale M-RoPE table, an EOS check against
//! the wrong id) would show up here as a token mismatch even though every
//! per-layer forward is individually exact — that's precisely why this rung
//! of the parity ladder exists on top of the per-layer ones.
//!
//! No sampling randomness: both sides are greedy (argmax), so an EXACT
//! token-id match is the bar, not a cosine floor. Measured on the real
//! checkpoint (5 new tokens, `BRAIN_DEVICE=cpu` — the GPU run hit an
//! unrelated pre-existing VRAM shortfall):
//! the prefill and the first 3 KV-cache decode steps matched EXACTLY,
//! diverging only on the 4th. `BRAIN_QWEN3OMNIMOE_DEBUG_LOGITS=1` at that step
//! showed the top-2 candidates 0.17 logits apart and the golden's actual
//! pick in 3rd place, 1.3 logits back — a closely-contested position, not a
//! confidently-wrong one, consistent with accumulated bf16 (HF's reference
//! compute, `torch_dtype=bfloat16`) vs. fp32 (this engine's fp32-arithmetic-
//! only convention, never claimed
//! bit-identical to a bf16 reference) rounding rather than a loop-control
//! bug: the SAME `decode_step` code path ran correctly 3 times immediately
//! before diverging, and every primitive it composes is independently
//! cosine-1.000000-exact against these same real weights (M6a/M6b/M8b's
//! `model::block` test). The assertion below stays a strict exact match
//! (weakening it would hide a REAL future regression) — this is a known,
//! documented, environment-specific mismatch on this one 5-token prompt, not
//! a passing test; a longer run or a different prompt may or may not hit the
//! same near-tie.
//!
//! Real-weight-adjacent, real-golden-adjacent: skips cleanly when
//! `BRAIN_QWEN3OMNIMOE_HF_DIR` is unset or the golden (`tools/goldens/
//! qwen3omnimoe_dump_generate.py`'s output) is missing.
//!
//! Expected to be SLOW — `crate::generate`'s own module doc: every layer's
//! weights are streamed fresh from the checkpoint on every prefill call and
//! every decode step (no resident weights yet), so this is minutes per
//! token, not milliseconds. Marked `#[ignore]`, matching every other
//! real-weight test in this crate.
//!
//! usage: `BRAIN_QWEN3OMNIMOE_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test generate_e2e -- --ignored --nocapture`

use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use checkpoint::weightio::WeightReader;
use gpu_core::Gpu;
use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::generate::{generate_greedy, EmbedTable, ThinkerStack};
use qwen3omnimoe::thinker::thinker_pipelines;

#[test]
#[ignore]
fn matches_the_real_hf_greedy_generation() {
    let Some(hf_dir) = std::env::var("BRAIN_QWEN3OMNIMOE_HF_DIR").ok().filter(|p| !p.is_empty()) else {
        brain_testutil::skip("BRAIN_QWEN3OMNIMOE_HF_DIR unset");
        return;
    };
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_generate.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip(&format!("{golden_path:?} missing (run tools/goldens/qwen3omnimoe_dump_generate.py against the real checkpoint)"));
        return;
    }

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let want_prompt: Vec<u32> = golden.tensor_f32("prompt_ids").expect("golden prompt_ids").iter().map(|&v| v as u32).collect();
    let want_ids: Vec<u32> = golden.tensor_f32("generated_ids").expect("golden generated_ids").iter().map(|&v| v as u32).collect();
    assert!(want_ids.len() > want_prompt.len(), "golden generated_ids must be longer than prompt_ids");
    let max_new = (want_ids.len() - want_prompt.len()) as u32;

    let dir = PathBuf::from(&hf_dir);
    let reader = WeightReader::open_hf_dir(&dir).expect("open real checkpoint");
    let config_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let root: serde_json::Value = serde_json::from_str(&config_json).expect("parse config.json");
    let cfg = MoeTextConfig::thinker_from_json(&root);

    // The golden was dumped from the bare `Thinker` class directly
    // (`Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained`, not the
    // full `Qwen3OmniMoeForConditionalGeneration` wrapper), which carries no
    // bundled `generation_config.eos_token_id` -- `qwen3omnimoe_dump_generate.py`'s
    // own `model.generate(...)` call passes none either, so HF ran all
    // `MAX_NEW` steps unconditionally (confirmed: the golden's own generated
    // ids contain 151645, `<|im_end|>`, mid-sequence, not just at the tail).
    // Passing real `eos_ids` here would make `generate_greedy` stop early at
    // that token and diverge from the golden for a harmless reason (a
    // stopping-policy difference, not a numerics one) -- empty matches what
    // was actually dumped.
    let eos_ids: Vec<u32> = Vec::new();

    let embed = EmbedTable::open(&reader).expect("open embed table");
    // Every discovered card, at its real usable capacity: whatever fits is
    // resident and the rest streams, decided by `qwen3omnimoe::thinker_plan`.
    let devices = qwen3omnimoe::thinker_plan::discovered_devices(2 << 30);
    let (gpus, caps): (Vec<Gpu>, Vec<u64>) = if devices.is_empty() {
        (vec![Gpu::new(thinker_pipelines())], vec![u64::MAX])
    } else {
        (
            devices.iter().map(|&(i, _)| Gpu::new_on_index(i, thinker_pipelines()).expect("open gpu")).collect(),
            devices.iter().map(|&(_, c)| c).collect(),
        )
    };
    let stack = ThinkerStack::build(&reader, &cfg, &gpus, &caps).expect("place the thinker");

    println!("running real 48-layer/128-expert generation for {max_new} tokens -- layers that do not fit stream fresh per step, expect minutes...");
    let got_ids = generate_greedy(&stack, &gpus, &reader, &cfg, &embed, &want_prompt, max_new, &eos_ids);

    let common_prefix = got_ids.iter().zip(&want_ids).take_while(|(a, b)| a == b).count();
    println!("want: {want_ids:?}");
    println!("got:  {got_ids:?}");
    println!("common prefix: {common_prefix} of {} tokens (re-run with BRAIN_QWEN3OMNIMOE_DEBUG_LOGITS=1 to see the diverging step's top candidates)", want_ids.len());
    assert_eq!(got_ids, want_ids, "generate_greedy's token ids must exactly match HF's real greedy generate()");
}
