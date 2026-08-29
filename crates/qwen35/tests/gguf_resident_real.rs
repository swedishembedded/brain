// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint gates for `qwen35::int8_gguf_resident` - the multi-GPU,
//! INT8, load-straight-from-Q8_0-GGUF serving path.
//!
//! Everything here needs the REAL `Qwen3.8-27B*.gguf` (~29 GB), named by
//! `BRAIN_QWEN35_GGUF` - the same variable `crate::gguf_import`'s own
//! real-checkpoint gate already uses. Self-skips loudly when it is unset
//! (`brain_testutil::skip`, a fixture skip) or when the box has too few
//! GPUs (`brain_testutil::skip_unavailable`, a hardware skip no flag may
//! turn fatal).
//!
//! Run them like:
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/.local/share/brain/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --test gguf_resident_real -- --nocapture --test-threads=1
//! ```
//!
//! The end-to-end test is a COLD load of ~29 GB of Q8_0 (dequantized tensor
//! by tensor and re-quantized to brain's group-wise INT8 on the way to the
//! cards) followed by real decode steps. It takes minutes, on purpose - it is
//! the only thing that proves the whole path, and it prints the numbers it
//! measured rather than asserting a performance figure it cannot control.

use std::time::Instant;

use checkpoint::TensorSource;
use checkpoint::gguf::MmapGguf;
use model::shard::Shard;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::int8_gguf_resident::{endpoint_names, layer_cost, resident_config, shard_fetch_plan, shard_source, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// `prompt + max_new` capacity used by the tests here. Small on purpose: it
/// is charged into the placement (a GQA layer's KV cache scales with it), and
/// the smoke test only generates a handful of tokens.
const CAP: u32 = 512;

/// Bytes kept free per card - `brain serve`'s own default `--reserve-gb 2`,
/// so what these tests plan against is what a real deployment plans against.
const RESERVE: u64 = 2 << 30;

fn gguf_path() -> Option<String> {
    match std::env::var("BRAIN_QWEN35_GGUF") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => {
            brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
            None
        }
    }
}

/// The real cards, as `(Device, usable bytes)` - queried, never assumed, the
/// same shape `crates/cli`'s `build_executor` hands its multi-device
/// residents.
fn real_devices() -> Vec<(Device, u64)> {
    gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect()
}

/// The `RemapSource` fetch plan for one stage must name EXACTLY the tensors
/// that stage needs, resolve every one of them to a real tensor in the real
/// file, and hand back correctly-shaped data.
///
/// This is the gate on the whole loading story: `Qwen35::new_i8_shard` asks a
/// `TensorSource` for brain-canonical names, the file offers llama.cpp ones,
/// and the only thing between them is this plan. A plan that silently omits a
/// leaf would surface tens of gigabytes into a load as a bare
/// "missing init weight"; a plan that maps the wrong leaf (a swapped `k`/`v`,
/// which is shape-compatible on every GQA layer) would not surface at all.
#[test]
fn the_fetch_plan_for_one_stage_names_exactly_that_stages_real_tensors() {
    let Some(path) = gguf_path() else { return };
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let cfg = resident_config(&mg, CAP).expect("resident_config on the real checkpoint");
    assert_eq!(cfg.n_layers, 64);
    assert!(!cfg.mtp, "a multi-card resident must not enable MTP (new_impl_on forces MTP to a whole shard)");
    assert!(!cfg.tie_embeddings, "this release ships an untied output.weight");

    // Layers 0 (GDN) and 1 (GDN) plus the embedding: a two-layer embed stage.
    let shard = Shard { start: 0, end: 2, embed: true, head: false, gpu_index: Shard::ANY_GPU };
    let plan = shard_fetch_plan(&mg, &cfg, &shard).expect("plan for layers 0..2");

    let mut want: Vec<String> = vec!["tok.weight".to_string()];
    for l in 0..2 {
        assert_eq!(cfg.layer_types()[l], LayerType::Linear, "layers 0..2 are GDN at interval 4");
        for leaf in [
            "ln1.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "ln2.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
            "mlp.down.weight",
        ] {
            want.push(format!("blocks.{l}.{leaf}"));
        }
    }
    want.sort();
    let mut got: Vec<String> = plan.keys().cloned().collect();
    got.sort();
    assert_eq!(got, want, "the stage's plan must be exactly its own tensors - no head, no other layer, nothing missing");

    // A non-embed, non-head middle stage must NOT claim the endpoints.
    let mid = Shard { start: 8, end: 12, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let mid_plan = shard_fetch_plan(&mg, &cfg, &mid).expect("plan for layers 8..12");
    assert!(!mid_plan.contains_key("tok.weight"));
    assert!(!mid_plan.contains_key("lm_head.weight"));
    assert!(!mid_plan.contains_key("norm.weight"));
    // The MTP block is `blk.64.*`; nothing in any stage's plan may come from it.
    for fetch in mid_plan.values() {
        let checkpoint::remap::Fetch::Whole(src) = fetch else { panic!("every qwen35 GGUF leaf is a 1:1 rename") };
        assert!(!src.starts_with("blk.64."), "the MTP block must never enter a decoder-stage plan, got {src}");
    }

    // A shard that DOES declare the endpoints gets them mapped to the right
    // llama.cpp tensors. The resident itself never builds such a shard (its
    // stages are `embed: false, head: false` - the endpoints are too large to
    // be fp32 device buffers, see the module doc), but the mapping is what
    // `endpoint_names` below resolves against, so it is worth pinning here.
    let head = Shard { start: 60, end: 64, embed: false, head: true, gpu_index: Shard::ANY_GPU };
    let head_plan = shard_fetch_plan(&mg, &cfg, &head).expect("plan for layers 60..64");
    assert_eq!(head_plan.get("norm.weight"), Some(&checkpoint::remap::Fetch::Whole("output_norm.weight".to_string())));
    assert_eq!(head_plan.get("lm_head.weight"), Some(&checkpoint::remap::Fetch::Whole("output.weight".to_string())));
    assert!(!head_plan.contains_key("tok.weight"));

    // The three tensors the RESIDENT holds itself, resolved through the same
    // classifier - no llama.cpp spelling is written down twice.
    let (embed_src, norm_src, head_src) = endpoint_names(&mg, &cfg).expect("endpoint_names on the real checkpoint");
    assert_eq!((embed_src.as_str(), norm_src.as_str(), head_src.as_str()), ("token_embd.weight", "output_norm.weight", "output.weight"));
    // And the embedding really reads a ROW at a time out of the Q8_0 mapping
    // - the accessor that makes a 5.09 GB table cost 20 KiB.
    let d = cfg.d_model as usize;
    for id in [0usize, 1, 1000, cfg.vocab as usize - 1] {
        let row = mg.tensor_range(&embed_src, id * d, d).unwrap_or_else(|| panic!("embedding row {id} must be readable")).expect("dequantize");
        assert_eq!(row.len(), d);
        assert!(row.iter().all(|v| v.is_finite()), "embedding row {id} has a non-finite entry");
        assert!(row.iter().any(|v| *v != 0.0), "embedding row {id} is all zero");
    }

    // ... and the plan really reads: shapes validated from the header, then a
    // few tensors pulled through the RemapSource for real, dequantized data.
    let src = shard_source(&mg, &cfg, &shard).expect("validate + build the RemapSource for layers 0..2");
    for (name, numel) in [
        ("tok.weight", cfg.vocab as usize * cfg.d_model as usize),
        ("blocks.0.linear_attn.in_proj_qkv.weight", cfg.linear_conv_dim() as usize * cfg.d_model as usize),
        ("blocks.1.mlp.down.weight", cfg.d_model as usize * cfg.intermediate_size as usize),
        ("blocks.0.linear_attn.A_log", cfg.linear_num_value_heads as usize),
    ] {
        assert_eq!(src.numel(name), Some(numel), "{name}: wrong element count through the RemapSource");
    }
    // Real values, not just real shapes: a norm gain the reference clusters
    // around 1.0 (llama.cpp's conversion already applied Qwen3.5's `1+w`
    // fold - see `qwen35::gguf_import`'s module doc), read through the plan.
    let mut ln1 = Vec::new();
    assert!(src.with_tensor("blocks.0.ln1.weight", &mut |d| ln1 = d.to_vec()), "blocks.0.ln1.weight must resolve");
    assert_eq!(ln1.len(), cfg.d_model as usize);
    assert!(ln1.iter().all(|v| v.is_finite()), "a dequantized norm gain must be finite");
    let mean = ln1.iter().sum::<f32>() / ln1.len() as f32;
    assert!((0.5..2.0).contains(&mean), "blocks.0.ln1.weight mean {mean} is nowhere near the reference's ~1.0");
}

/// Placement of the REAL file across the REAL cards, printed. Cheap (header
/// only, no tensor data, no GPU allocation) and the number the end-to-end
/// test below then actually loads.
#[test]
fn the_real_checkpoint_plans_across_the_real_cards() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM - this resident is GPU-only");
        return;
    }
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let cfg = resident_config(&mg, CAP).expect("resident_config");
    let cost = layer_cost(&cfg, CAP);
    println!("qwen35 gguf resident: cap={CAP}");
    println!("  total device bytes : {} ({:.2} GiB)", cost.total(), cost.total() as f64 / (1u64 << 30) as f64);
    println!("  embed              : {:.2} GiB", cost.embed as f64 / (1u64 << 30) as f64);
    println!("  head               : {:.2} GiB", cost.head as f64 / (1u64 << 30) as f64);
    for (d, usable) in &devices {
        println!("  candidate {d:?}: {:.2} GiB usable (total minus {} GiB reserve)", *usable as f64 / (1u64 << 30) as f64, RESERVE >> 30);
    }

    let r = Qwen35GgufResident::new(path, devices.clone(), CAP);
    let placement = r.placement();
    assert!(!placement.is_empty(), "the real checkpoint must be placeable across {} card(s)", devices.len());
    for (d, start, end, bytes) in &placement {
        println!("  stage {d:?}: layers {start}..{end} ({} layers), {bytes} bytes ({:.2} GiB)", end - start, *bytes as f64 / (1u64 << 30) as f64);
    }
    // Every stage must independently fit its own card - the property that
    // makes this a plan and not a hope.
    for ((d, _, _, bytes), (dd, usable)) in placement.iter().zip(devices.iter()) {
        assert_eq!(d, dd, "stage i must land on device i");
        assert!(bytes <= usable, "stage on {d:?} needs {bytes} but only {usable} is usable");
    }
    assert_eq!(placement[0].1, 0);
    assert_eq!(placement[placement.len() - 1].2, cfg.n_layers as usize);

    // `estimate_multi` must agree with the placement exactly - the scheduler
    // reserves what the loader will place, or the budget is a lie.
    let cost_multi = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new()));
    assert_eq!(cost_multi.devices().count(), placement.len());
    for (d, _, _, bytes) in &placement {
        assert_eq!(cost_multi.on(*d), *bytes);
    }
}

/// Build the whole thing on the real hardware and drive it: plan the real
/// file across the real cards, build every stage straight from the Q8_0
/// GGUF, tokenize with the GGUF's own embedded tokenizer, and decode real
/// tokens.
///
/// This is the PLUMBING gate. It asserts what this change is responsible
/// for - the model is placeable, loads, produces tokens, and decodes
/// deterministically under greedy sampling (which is what catches
/// per-sequence state a second request inherits instead of resetting) - and
/// PRINTS the cold-load wall clock, the split between prefill and decode,
/// and the text, rather than gating on a performance number.
///
/// Whether the tokens are the RIGHT ones is a separate question with a
/// separate test: [`the_two_card_stack_continues_a_factual_prompt_correctly`]
/// below. Keeping them apart matters, because they fail for different
/// reasons and one of them currently fails - see that test's own doc.
#[test]
fn a_real_two_card_load_runs_end_to_end() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.len() < 2 {
        brain_testutil::skip_unavailable(&format!(
            "this gate wants a genuinely MULTI-card load; this box has {} usable GPU(s)",
            devices.len()
        ));
        return;
    }
    let total: u64 = devices.iter().map(|&(_, c)| c).sum();
    println!("qwen35 gguf resident e2e: {} card(s), {:.1} GiB usable total", devices.len(), total as f64 / (1u64 << 30) as f64);

    let r = Qwen35GgufResident::new(path, devices, CAP);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let cost = r.estimate_multi(&key);
    let placed: Vec<Device> = cost.devices().collect();
    assert!(placed.len() >= 2, "the real 27B model must need more than one 24 GiB card at INT8");

    let t0 = Instant::now();
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint across the real cards");
    let load = t0.elapsed();
    println!("  cold load (Q8_0 -> INT8, {} stage(s)): {:.1} s", placed.len(), load.as_secs_f64());

    let inv = capability::Invocation::new()
        .set("prompt", serde_json::json!("Give one short sentence explaining what a Kalman filter does."))
        .set("max_new", serde_json::json!(32))
        .set("temp", serde_json::json!(0.0))
        .set("enable_thinking", serde_json::json!(false));

    let t1 = Instant::now();
    let out = inst.run("generate", &inv, &mut |_| {}).expect("generate on the real checkpoint");
    let gen = t1.elapsed();

    let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
    let completion = out.outputs["completion_tokens"].as_i64().unwrap_or(0);
    let prompt_tokens = out.outputs["prompt_tokens"].as_i64().unwrap_or(0);
    println!("  whole call: prompt {prompt_tokens} tok, generated {completion} tok in {:.1} s", gen.as_secs_f64());
    // The instance's OWN split of that wall clock - prefill and decode are the
    // same one-token-per-pass primitive here, so only the model can say which
    // half the time went to.
    for (k, v) in inst.metrics() {
        println!("  metric {k:<20} {v}");
    }
    println!("  finish_reason: {}", out.outputs["finish_reason"]);
    println!("  generated text: {text:?}");

    assert!(completion > 0, "the model must produce at least one token");
    assert!(!text.trim().is_empty(), "the generated text must not be empty");
    // There is deliberately NO assertion on the SHAPE of this text - not
    // character variety, not length, not content.
    //
    // This gate is the plumbing gate, and the text this path produces is
    // known-wrong: [`the_two_card_stack_continues_a_factual_prompt_correctly`]
    // below is red precisely because of it. A text-quality proxy asserted here
    // is therefore a coin flip on garbage, and it behaved like one - the same
    // check passed on `"Give one"` while the red gate's own output
    // (`"..\n\n\n\n..."`) would have failed it, and a legitimate ~1e-6
    // reduction-order change in the RMSNorm kernel (gated for correctness by
    // `tests/rmsnorm_variant_agreement.rs`) was enough to move which garbage
    // token comes out and flip it.
    //
    // What a broken shard boundary or a mis-mapped tensor really looks like is
    // caught precisely, and on real weights, by the gates that compare against
    // a reference instead of eyeballing a string: `gguf_reference_parity_real`,
    // `gguf_i8_vs_fp32_real`, and `model`'s own
    // `two_shard_int8_decode_matches_the_whole_shard_model`.
    //
    // Restore a text-shape assertion here when the red gate below goes green -
    // at that point it is checking something real rather than which way the
    // garbage fell.

    // Greedy decoding is deterministic: the same prompt through the same
    // resident instance must produce the same text. This is what catches
    // per-sequence state (the GDN recurrent state, the GQA cache) that a
    // second run inherits instead of resetting.
    let again = inst.run("generate", &inv, &mut |_| {}).expect("second greedy generate");
    assert_eq!(again.outputs["text"].as_str().unwrap_or_default(), text, "greedy decoding must be deterministic across runs on one instance");

}

/// **This gate is RED, on purpose, and it is the honest state of this path.**
///
/// A greedy RAW continuation (`chat: false`, no template at all) of a prompt
/// whose answer the model cannot plausibly get wrong. It depends on no
/// template convention, no sampling parameter and no chat markup, so a
/// failure here is arithmetic: the weights this path loads do not compute
/// what Qwen3.8-27B computes.
///
/// It is separate from [`a_real_two_card_load_runs_end_to_end`] because the
/// two answer different questions and only one of them is settled. The
/// plumbing works: the model plans across two P40s, loads 29 GB of Q8_0
/// straight into INT8 device weights with no fp32 intermediate, decodes at a
/// measured ~3.9 tok/s, and is bit-stable across requests. The OUTPUT is
/// still wrong - measured `"The capital city of France is"` ->
/// `"..\n\n\n\n..."`, with the debug dump
/// (`BRAIN_QWEN35_GGUF_DEBUG=1`) showing a model that predicts sensibly at
/// short range and degrades with context length.
///
/// What has been ruled out, and how: it is not the
/// sharding (bit-exact at tiny scale), not the int8 decode tape (bit-exact
/// against int8 prefill), not `ssm_a` (found and fixed - lesson #70), not
/// the norm folds, and not an `ssm_alpha`/`ssm_beta` swap (tried; strictly
/// worse). It IS a difference between what this GGUF route loads and what
/// the safetensors route loads, because the SAME engine at the SAME INT8
/// tier over 64 real layers already produced `" Paris."` from the FP8
/// checkpoint (roadmap M16).
///
/// Deleting or weakening this assertion would turn a known-broken path into
/// a green one. It stays as it is until the difference is found.
#[test]
fn the_two_card_stack_continues_a_factual_prompt_correctly() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.len() < 2 {
        brain_testutil::skip_unavailable(&format!("this gate wants a MULTI-card load; this box has {} usable GPU(s)", devices.len()));
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, CAP);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint across the real cards");

    let raw = capability::Invocation::new()
        .set("prompt", serde_json::json!("The capital city of France is"))
        .set("chat", serde_json::json!(false))
        .set("max_new", serde_json::json!(8))
        .set("temp", serde_json::json!(0.0));
    let cont = inst.run("generate", &raw, &mut |_| {}).expect("raw greedy continuation");
    let text = cont.outputs["text"].as_str().unwrap_or_default().to_string();
    println!("  raw greedy continuation: {text:?}");
    for (k, v) in inst.metrics() {
        println!("  metric {k:<20} {v}");
    }
    assert!(
        text.contains("Paris"),
        "the two-card int8 stack must continue \"The capital city of France is\" with Paris, got {text:?} \
         - see this test's own doc comment for what has already been ruled out"
    );
}

/// The cost model at the tested capacity, printed alongside the config the
/// real file declares - cheap, header-only, and the arithmetic the placement
/// above is built on. Runs without a GPU.
#[test]
fn the_real_checkpoints_cost_model_is_reported() {
    let Some(path) = gguf_path() else { return };
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let cfg = resident_config(&mg, CAP).expect("resident_config");
    let cost = layer_cost(&cfg, CAP);
    let gdn = cfg.layer_types().iter().position(|t| *t == LayerType::Linear).unwrap();
    let gqa = cfg.layer_types().iter().position(|t| *t == LayerType::Full).unwrap();
    println!("  per-layer GDN: {} bytes, GQA: {} bytes", cost.per_layer[gdn], cost.per_layer[gqa]);
    // The published dims are fixed, so these are facts about the release, not
    // about this box: they must match `Qwen35Config::qwen38_27b`'s own numbers.
    let reference = Qwen35Config::qwen38_27b();
    assert_eq!(cfg.n_layers, reference.n_layers);
    assert_eq!(cfg.d_model, reference.d_model);
    assert_eq!(cfg.vocab, reference.vocab);
    assert_eq!(cfg.intermediate_size, reference.intermediate_size);
    assert_eq!(cost.per_layer[gdn], reference.layer_i8_bytes(LayerType::Linear) + 4 * (48 * 128 * 128 + 10240 * 3));
    assert_eq!(cost.per_layer[gqa], reference.layer_i8_bytes(LayerType::Full) + 2 * CAP as u64 * reference.kv_dim() as u64 * 4);
}
