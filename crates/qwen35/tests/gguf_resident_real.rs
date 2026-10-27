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
use data::tokenizer::Tokenizer;
use checkpoint::gguf::MmapGguf;
use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use model::ops::TierPolicy;
use model::shard::Shard;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::int8_gguf_resident::{decode_scratch_bytes, endpoint_names, layer_cost, prefill_round_rows_for, prefill_workspace_bytes, resident_config, shard_fetch_plan, shard_source, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::model::Instance;
use residency::{Device, ResidentModel};

/// `prompt + max_new` capacity used by the tests here. Small on purpose: it
/// is charged into the placement (a GQA layer's KV cache scales with it), and
/// the smoke test only generates a handful of tokens.
const CAP: u32 = 512;

/// `prompt + max_new` capacity for the long-context prefill measurement
/// ([`prefill_throughput_at_a_real_long_context`]), which needs room for a
/// ~1550-token prompt. Charged into the placement exactly like [`CAP`], so it
/// is kept as small as that measurement allows.
const LONG_CAP: u32 = 2048;

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
    qwen35::int8_gguf_resident::budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE)
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
    let cost = layer_cost(&cfg, CAP, &TierPolicy::uniform(Dtype::I8), KvTier::F32, 1);
    println!("qwen35 gguf resident: cap={CAP}");
    println!("  total device bytes : {} ({:.2} GiB)", cost.total(), cost.total() as f64 / (1u64 << 30) as f64);
    println!("  embed              : {:.2} GiB", cost.embed as f64 / (1u64 << 30) as f64);
    println!("  head               : {:.2} GiB", cost.head as f64 / (1u64 << 30) as f64);
    for (d, usable) in &devices {
        println!("  candidate {d:?}: {:.2} GiB usable (total minus {} GiB reserve)", *usable as f64 / (1u64 << 30) as f64, RESERVE >> 30);
    }

    let r = Qwen35GgufResident::new(path, devices.clone(), CAP, TierPolicy::uniform(Dtype::I8));
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

    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8));
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

/// **M23: GREEN.** Was RED for several milestones before this fix landed.
/// The root cause: llama.cpp's GGUF conversion stores EVERY GDN leaf indexed by
/// VALUE HEAD (`ssm_a`/`ssm_dt.bias`/`ssm_conv1d.weight`'s v-channels/
/// `attn_qkv.weight`'s v-rows/`ssm_alpha.weight`/`ssm_beta.weight`/
/// `attn_gate.weight`'s rows/`ssm_out.weight`'s columns) in GROUP-MAJOR
/// order, not brain's (and the reference HF model's) SUB-MAJOR
/// (repeat_interleave) order - so every head's own decay/gate/projection
/// was individually plausible but applied to the WRONG head's key/value
/// state. Fixed by `int8_gguf_resident::GdnHeadOrder`
/// (`crate::int8_gguf_resident::SsmALogFix`), found by a direct per-tensor
/// diff against the FP8 checkpoint's own weights (M23.1): three of the eight
/// affected leaves are unquantized and matched the FP8 side at cosine
/// EXACTLY 1.0000000 once degrouped, the strongest possible signal that
/// this was a pure reordering, not a value or precision difference.
///
/// It is separate from [`a_real_two_card_load_runs_end_to_end`] because the
/// two answer different questions. The plumbing has always worked: the
/// model plans across two P40s, loads 29 GB of Q8_0 straight into INT8
/// device weights with no fp32 intermediate, decodes at a measured
// perf-number: measured by this test file (command in its module header)
/// 7.44-7.57 tok/s (M22), and is bit-stable across requests. What was wrong
/// was the OUTPUT: `"The capital city of France is"` used to continue
/// `"..\n\n\n\n.."` (an earlier, already-fixed `ssm_a` bug - lesson #70),
/// then `" is is is is is is is is"` (this bug, before the full fix landed),
/// now correctly `" Paris. Paris is the largest city in"`.
///
/// What was ruled out on the way here, and how: not the sharding (bit-exact
/// at tiny scale), not the int8 decode tape (bit-exact against int8
/// prefill), not the norm folds, not an `ssm_alpha`/`ssm_beta` SWAP (tried;
/// strictly worse - the real bug was a HEAD reordering, not a two-tensor
/// swap).
#[test]
fn the_two_card_stack_continues_a_factual_prompt_correctly() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.len() < 2 {
        brain_testutil::skip_unavailable(&format!("this gate wants a MULTI-card load; this box has {} usable GPU(s)", devices.len()));
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8));
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

/// **The batched-decode gate, on the real checkpoint.** Several DIFFERENT
/// factual prompts, submitted as one `Instance::run_batch` call, must each get
/// its own right answer - and the call must report a real aggregate decode
/// rate above what the same prompts get one at a time.
///
/// Two independent things are being checked and they fail in different ways:
///
/// * **Correctness.** Every row must answer ITS OWN question. A batched decode
///   step carries four sequences' rows through one dispatch set, sharing one
///   KV pool (each row addressed by its own physical block) and one batched
///   Gated-DeltaNet state slab (each row gathered in and scattered back). A
///   row reading another row's slice of either would still produce fluent
///   text - it would just be answering the wrong question, or a blend - which
///   is exactly what four distinct, unambiguous factual questions catch and a
///   single repeated prompt never could.
/// * **Throughput.** The whole point of batching a decode step is that its
///   cost is dominated by reading every weight in the model, and those reads
///   are shared by the batch. The numbers are printed, and the aggregate rate
///   is asserted only to BEAT the serial one - not to hit a target, which is a
///   property of the card, not of this code.
///
/// The serial baseline is the same four prompts through the same instance, one
/// `run` call each, so the comparison is on one load of one checkpoint on one
/// set of cards.
#[test]
fn batched_decode_answers_each_prompt_and_beats_serial_throughput() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.len() < 2 {
        brain_testutil::skip_unavailable(&format!("this gate wants a MULTI-card load; this box has {} usable GPU(s)", devices.len()));
        return;
    }
    // Eight unambiguous questions with eight DIFFERENT answers, at
    // deliberately different prompt lengths - equal lengths would make a
    // stale-`seq_lens` or off-by-one causal bound invisible, since every row
    // would then want the same bound anyway.
    let asks: [(&str, &str); 8] = [
        ("The capital city of France is", "Paris"),
        ("The largest planet in our solar system is called", "Jupiter"),
        ("Water freezes at a temperature of zero degrees", "Celsius"),
        ("The chemical symbol for gold is", "Au"),
        ("The tallest mountain above sea level on Earth is Mount", "Everest"),
        ("The ocean that lies between Africa and Australia is the", "Indian"),
        ("In the year 1969 the first human being to walk on the", "Moon"),
        ("The programming language created by Guido van Rossum is called", "Python"),
    ];
    const NEW: usize = 8;

    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8)).with_max_batch(asks.len() as u32);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    println!("  batched decode: max_batch={}, placed on {} card(s)", r.max_batch(), placed.len());
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint across the real cards");

    let inv_for = |prompt: &str| {
        capability::Invocation::new()
            .set("prompt", serde_json::json!(prompt))
            .set("chat", serde_json::json!(false))
            .set("max_new", serde_json::json!(NEW))
            .set("temp", serde_json::json!(0.0))
    };
    // The answer WORD is what is being checked, not its casing: a
    // mid-sentence continuation legitimately writes "moon" where a sentence
    // start would write "Moon", and that is the model's business, not this
    // gate's.
    let answers = |text: &str, want: &str| text.to_lowercase().contains(&want.to_lowercase());
    let decode_rate = |inst: &dyn Instance| -> (f64, f64) {
        let m: std::collections::HashMap<String, serde_json::Value> = inst.metrics().into_iter().collect();
        (m["last_batch_generated_tokens"].as_f64().unwrap_or(0.0), m["last_batch_seconds"].as_f64().unwrap_or(0.0))
    };

    // --- serial baseline: one `run` per prompt, same instance -------------
    let mut serial_tokens = 0.0;
    let mut serial_seconds = 0.0;
    for (prompt, want) in asks {
        let out = inst.run("generate", &inv_for(prompt), &mut |_| {}).expect("serial generate");
        let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
        println!("  serial  {prompt:?} -> {text:?}");
        assert!(answers(&text, want), "serial: {prompt:?} must continue with {want:?}, got {text:?}");
        let (t, s) = decode_rate(inst.as_ref());
        serial_tokens += t;
        serial_seconds += s;
    }
    let serial_rate = serial_tokens / serial_seconds;
    println!("  serial  aggregate: {serial_tokens} decode tok in {serial_seconds:.2} s = {serial_rate:.2} tok/s");

    // --- batched: all of them in ONE run_batch call -----------------------
    let invs: Vec<capability::Invocation> = asks.iter().map(|&(p, _)| inv_for(p)).collect();
    let outs = inst.run_batch("generate", &invs, &mut |_, _| {});
    assert_eq!(outs.len(), asks.len(), "run_batch must return one result per request");
    let (batched_tokens, batched_seconds) = decode_rate(inst.as_ref());
    let batched_rate = batched_tokens / batched_seconds;
    for (i, (prompt, want)) in asks.iter().enumerate() {
        let out = outs[i].as_ref().unwrap_or_else(|e| panic!("batched request {i} ({prompt:?}) failed: {e}"));
        let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
        println!("  batched {prompt:?} -> {text:?}");
        assert!(
            answers(&text, want),
            "batched request {i}: {prompt:?} must continue with {want:?}, got {text:?} - a row answering the WRONG \
             question means the batch mixed per-sequence state (KV pool window or Gated-DeltaNet slab)"
        );
    }
    println!("  batched aggregate: {batched_tokens} decode tok in {batched_seconds:.2} s = {batched_rate:.2} tok/s");
    println!("  speedup at batch {}: {:.2}x", asks.len(), batched_rate / serial_rate);
    assert!(
        batched_rate > serial_rate,
        "a batch of {} must decode at a HIGHER aggregate rate than the same prompts served one at a time \
         ({batched_rate:.2} vs {serial_rate:.2} tok/s) - if it does not, the batch is not sharing the weight reads",
        asks.len()
    );
}

/// M24: the SAME factual-continuation check as
/// [`the_two_card_stack_continues_a_factual_prompt_correctly`], at the Q4
/// (W4A8) tier instead of I8 - and, unlike that gate, on ONE card (Q4's
/// ~15.7-15.8 GiB footprint fits a single 24 GiB P40, where I8's ~27 GiB
/// needs two - `int8_gguf_resident::tests::
/// a_q4_mlp_tier_with_gdn_gates_held_at_f32_fits_one_24gb_card` is the
/// standing arithmetic gate this real-weight run confirms in practice).
/// Reports BOTH candidate policies since neither has an a-priori quality
/// guarantee (Q4 halves the weight levels I8 already has): uniform Q4, and
/// policy C (Q4 on the MLP, F32 on the two GDN state-sensitive gates -
/// `in_proj_a`/`in_proj_b`, ~94 MB total, essentially free).
#[test]
fn the_q4_tier_continues_the_same_factual_prompt_on_one_card() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    for (label, tier) in [
        ("uniform Q4", TierPolicy::uniform(Dtype::Q4)),
        ("policy C (Q4 MLP, F32 GDN gates)", TierPolicy::uniform(Dtype::Q4).with(&["in_proj_a.weight", "in_proj_b.weight"], Dtype::F32)),
    ] {
        let r = Qwen35GgufResident::new(path.clone(), devices.clone(), CAP, tier);
        let key = r.instance_key("generate", &capability::Invocation::new());
        let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
        println!("  {label}: placed on {} device(s): {placed:?}", placed.len());
        let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint at the q4 tier");

        let raw = capability::Invocation::new()
            .set("prompt", serde_json::json!("The capital city of France is"))
            .set("chat", serde_json::json!(false))
            .set("max_new", serde_json::json!(8))
            .set("temp", serde_json::json!(0.0));
        let cont = inst.run("generate", &raw, &mut |_| {}).expect("raw greedy continuation at the q4 tier");
        let text = cont.outputs["text"].as_str().unwrap_or_default().to_string();
        println!("  {label}: raw greedy continuation: {text:?}");
        for (k, v) in inst.metrics() {
            println!("    metric {k:<20} {v}");
        }
        assert!(
            text.contains("Paris"),
            "the q4 tier ({label}) must continue \"The capital city of France is\" with Paris, got {text:?} - \
             if this fails, Q4 is not usable for this model even though the wiring and placement are correct; \
             report which policy failed and its actual output"
        );
    }
}

/// Every OTHER real-checkpoint gate in this file uses a 6-token prompt and an
/// 8-token decode window (`CAP = 512` is plumbing headroom, not what gets
/// used) - enough to prove correctness, nowhere near enough to show how
/// decode throughput actually behaves as context grows. This test measures
/// that directly: a genuinely long real prompt (~1000+ tokens, not a toy),
/// uniform Q4 on one card (the fastest already-correctness-gated tier), real
/// prefill + decode tok/s reported from the instance's own metrics - not
/// predicted. GQA layers (16 of 64) read `O(context)` bytes of KV per decode
/// step, so decode tok/s is expected to degrade somewhat as context grows
/// past this file's other tests' ~14-token depth; GDN layers (the other 48)
/// carry an O(1) recurrent state and should not. This is the real measurement
/// that answers that question instead of assuming it.
#[test]
fn decode_throughput_at_a_real_long_context() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    // ~1050 real English tokens (not repeated filler - repetition can let a
    // decoder degenerate into a trivial attention pattern) sized so a full
    // serving-sized context budget isn't needed: prompt + a modest decode
    // window must fit under this test's own CAP.
    // perf-number: a calendar decade in prose fixture text, not a duration
    let paragraph = "The Antikythera mechanism is an ancient Greek hand-powered orrery, described as the oldest known example of an analogue computer, used to predict astronomical positions and eclipses decades in advance. It could also be used to track the four-year cycle of athletic games which was similar to an Olympiad, the cycle of the ancient Olympic Games. This artefact was retrieved from the sea in 1901, and identified in 1902 as containing a gear, which was found in the same location as a shipwreck off Point Antikythera in Greece. Study of the object was subsequently neglected until Derek J. de Solla Price took an interest in it and, with Charalampos Karakalos, published a seminal paper in 1974. After much modern scholarship since the 1970s, current understanding is that the artefact was made from a single sheet of bronze that was formed and then engraved with astronomical scales and text, and further sheets were later added to hold the mechanism together, making the object more resemble a book than a machine. ";
    let mut prompt = String::new();
    while prompt.split_whitespace().count() < 1050 {
        prompt.push_str(paragraph);
    }

    let cap: u32 = 2048;
    let r = Qwen35GgufResident::new(path, devices, cap, TierPolicy::uniform(Dtype::Q4));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    println!("  placed on {} device(s): {placed:?}", placed.len());
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint at the q4 tier");

    let inv = capability::Invocation::new()
        .set("prompt", serde_json::json!(prompt))
        .set("chat", serde_json::json!(false))
        .set("max_new", serde_json::json!(32))
        .set("temp", serde_json::json!(0.0));
    let out = inst.run("generate", &inv, &mut |_| {}).expect("generate at a real long context");
    let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
    let prompt_tokens = out.outputs["prompt_tokens"].as_i64().unwrap_or(0);
    let completion = out.outputs["completion_tokens"].as_i64().unwrap_or(0);
    println!("  prompt_tokens: {prompt_tokens}, completion_tokens: {completion}");
    println!("  continuation: {text:?}");
    for (k, v) in inst.metrics() {
        println!("  metric {k:<20} {v}");
    }
    assert!(completion > 0, "the model must produce at least one token even at long context");
    assert!(!text.trim().is_empty(), "the generated text must not be empty at long context");
}

/// A sequence that is generating keeps generating while a long prompt is read
/// in: with background prefill (each card's normal stream reads the prompt, its
/// urgent stream runs the decode steps) the longest gap between the running
/// sequence's tokens while the prompt goes in is a fraction of the gap the
/// sliced baseline leaves (a whole slice of prefill between two steps), and the
/// prompt's own continuation is the same either way.
#[test]
fn a_generating_sequence_keeps_generating_while_a_long_prompt_is_read_in() {
    use model::serve::{PagedDecoder, Request, Scheduler};

    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, 4096, TierPolicy::uniform(Dtype::I8)).with_max_batch(2);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let (mut engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");
    engine.set_background_prefill(true);
    if !engine.background_prefill() {
        brain_testutil::skip_unavailable("background prefill needs devices that can prioritise a stream");
        return;
    }
    let short = tok.encode("The capital city of France is");
    let long = tok.encode(&long_factual_prompt());

    // (the gaps between the running sequence's tokens while the long prompt went
    // in, sorted; seconds the prompt took to go in; the long prompt's continuation)
    let run = |engine: model::serve::Scheduler<_>| -> (Vec<f64>, f64, Vec<u32>, _) {
        let mut sched = engine;
        let runner = sched.submit(Request { prompt: short.clone(), max_new: 2000, stop: Vec::new() });
        while sched.running_len() == 0 {
            sched.step_report();
        }
        let t0 = std::time::Instant::now();
        let reader = sched.submit(Request { prompt: long.clone(), max_new: 6, stop: Vec::new() });
        let (mut last, mut gaps) = (std::time::Instant::now(), Vec::new());
        let mut read_in = None;
        let mut out = None;
        while out.is_none() {
            let report = sched.step_report();
            if report.produced.iter().any(|&(id, n)| id == runner && n > 0) && read_in.is_none() {
                gaps.push(last.elapsed().as_secs_f64());
                last = std::time::Instant::now();
            }
            if report.admitted.contains(&reader) {
                read_in = Some(t0.elapsed().as_secs_f64());
            }
            out = report.completed.into_iter().find(|(id, _)| *id == reader).map(|(_, t)| t);
        }
        sched.cancel(runner);
        sched.run();
        gaps.sort_by(f64::total_cmp);
        (gaps, read_in.expect("the long prompt was admitted"), out.expect("finished"), sched.into_decoder())
    };
    let summary = |g: &[f64]| format!("median {:.2} s, p90 {:.2} s, longest {:.2} s over {} tokens", g[g.len() / 2], g[g.len() * 9 / 10], g[g.len() - 1], g.len());
    let (bg_gap, bg_read, bg_out, mut engine) = run(Scheduler::new(engine, 2));
    engine.set_background_prefill(false);
    let (sl_gap, sl_read, sl_out, _) = run(Scheduler::new(engine, 2));
    println!("  {}-token prompt read in: background {bg_read:.1} s (decode gaps {}), sliced {sl_read:.1} s (decode gaps {})", long.len(), summary(&bg_gap), summary(&sl_gap));
    assert_eq!(bg_out, sl_out, "the long prompt's continuation must not depend on how it was read in");
    let (bg_worst, sl_worst) = (bg_gap[bg_gap.len() - 1], sl_gap[sl_gap.len() - 1]);
    assert!(bg_worst < 0.5 * sl_worst, "the running sequence still waited for the prompt: longest gap {bg_worst:.2} s against {sl_worst:.2} s sliced");
}

/// With nothing decoding, a prompt read in the background runs its rounds as a
/// pipeline across the cards (the next card takes a round as soon as the
/// previous one hands it over), so on a multi-card stack it goes in clearly
/// faster than in slices, which run each round through the cards in turn, and
/// on one card no slower.
#[test]
fn a_lone_prompt_reads_in_at_least_as_fast_in_the_background() {
    use model::serve::{PagedDecoder, Request, Scheduler};

    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, 4096, TierPolicy::uniform(Dtype::I8)).with_max_batch(2);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let (mut engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");
    engine.set_background_prefill(true);
    if !engine.background_prefill() {
        brain_testutil::skip_unavailable("background prefill needs devices that can prioritise a stream");
        return;
    }
    let stages = engine.instance().stages();
    let long = tok.encode(&long_factual_prompt());
    // Seconds from submission to the first token, best of two (the first also warms the kernels).
    let first_token = |engine| -> (f64, _) {
        let mut sched = Scheduler::new(engine, 2);
        let mut best = f64::INFINITY;
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let id = sched.submit(Request { prompt: long.clone(), max_new: 1, stop: Vec::new() });
            while !sched.step_report().admitted.contains(&id) {}
            best = best.min(t.elapsed().as_secs_f64());
            sched.run();
        }
        (best, sched.into_decoder())
    };
    let (bg, mut engine) = first_token(engine);
    engine.set_background_prefill(false);
    let (sliced, _) = first_token(engine);
    println!("  {}-token prompt alone over {stages} card(s): background {bg:.2} s, sliced {sliced:.2} s ({:.0} against {:.0} tok/s)", long.len(), long.len() as f64 / bg, long.len() as f64 / sliced);
    let bound = if stages > 1 { 0.85 } else { 1.05 };
    assert!(bg <= bound * sliced, "a lone prompt over {stages} card(s) went in at {bg:.2} s in the background against {sliced:.2} s in slices");
}

/// A ~1550-token raw prompt that ENDS on the same factual cue
/// [`the_two_card_stack_continues_a_factual_prompt_correctly`] uses, so the
/// same "did it answer Paris?" question can be asked at a length that takes
/// SEVERAL prefill rounds instead of one.
///
/// The filler is deliberately on-topic (a repeated paragraph about France)
/// rather than lorem ipsum: this is a real checkpoint, and a gate that
/// demands a specific continuation after 1500 tokens of unrelated noise would
/// be testing the model's temperament, not this crate's prefill.
fn long_factual_prompt() -> String {
    let para = "France is a country in Western Europe. Its territory stretches from the Rhine to the Atlantic \
                Ocean, and its people speak French. The country is known for its cuisine, its literature and \
                its long history. Many travellers visit each year to see its museums, its cathedrals and its \
                countryside. The Seine flows through the north of the country and past its largest urban area. ";
    let mut s = String::new();
    // ~66 tokens per paragraph on this tokenizer; 23 repeats lands near 1550.
    for _ in 0..23 {
        s.push_str(para);
    }
    s.push_str("The capital city of France is");
    s
}

/// **The prefill measurement the chunked-prefill work is judged by**, and a
/// real-weight correctness gate for it in the same run.
///
/// A REAL ~1550-token prompt (see [`long_factual_prompt`]) through the REAL
/// pipeline-parallel INT8 stack, with `max_new = 4` so nearly the whole wall
/// clock is prompt replay, and the instance's own `prefill_seconds`/
/// `prefill_tok_per_s` metrics reported. It PRINTS rather than asserts a
/// rate: what a P40 pair does per second is not this repo's to control, and a
/// hard floor here would go red on a busier box for reasons that have nothing
/// to do with the code.
///
/// What it DOES assert is what the code is responsible for. The prompt is
/// long enough to need several rounds of `MAX_PREFILL_TOKENS`, and it still
/// has to be answered correctly: a chunked replay that drops the ragged last
/// round, mis-masks a round's interior, writes KV rows at the wrong offset or
/// loses the GDN recurrent state across a round boundary all produce a
/// continuation that is not "Paris". At tiny scale the same claim is gated
/// bit-for-bit by `qwen35::model`'s own
/// `two_shard_chunked_prefill_matches_token_by_token_replay`; this is that
/// claim on 27B of real weights.
///
// perf-number: hardware count (2x P40), not a rate
/// Measured on this box (2x Tesla P40, INT8 tier, Q4_K_M source file, 1731
// perf-number: measured by this test (command in the module header) and `qwen35_gguf_prefill_profile`
/// prompt tokens): **262.8 s / 6.6 tok/s** with the old one-token-per-pass
// perf-number: measured by this test (command in the module header) and `qwen35_gguf_prefill_profile`
/// replay, **26.5 s / 65.4 tok/s** once the prompt is consumed in rounds -
// perf-number: measured by this test (command in the module header) and `qwen35_gguf_prefill_profile`
/// 9.9x. See `int8_gguf_resident::MAX_PREFILL_TOKENS` for the round-size
/// sweep behind that second number.
#[test]
fn prefill_throughput_at_a_real_long_context() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.len() < 2 {
        brain_testutil::skip_unavailable(&format!(
            "this gate measures the PIPELINE-PARALLEL prefill path; this box has {} usable GPU(s)",
            devices.len()
        ));
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, LONG_CAP, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    assert!(placed.len() >= 2, "the INT8 27B stack must span more than one card - this gate exists to measure the seam");

    let t0 = Instant::now();
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint across the real cards");
    println!("  cold load ({} stage(s)): {:.1} s", placed.len(), t0.elapsed().as_secs_f64());

    let inv = capability::Invocation::new()
        .set("prompt", serde_json::json!(long_factual_prompt()))
        .set("chat", serde_json::json!(false))
        .set("max_new", serde_json::json!(4))
        .set("temp", serde_json::json!(0.0));

    let t1 = Instant::now();
    let out = inst.run("generate", &inv, &mut |_| {}).expect("generate on the real long prompt");
    println!("  whole call: {:.1} s", t1.elapsed().as_secs_f64());
    let prompt_tokens = out.outputs["prompt_tokens"].as_i64().unwrap_or(0);
    for (k, v) in inst.metrics() {
        println!("  metric {k:<20} {v}");
    }
    let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
    println!("  long-context greedy continuation: {text:?}");

    // Several rounds of `int8_gguf_resident::MAX_PREFILL_TOKENS` (256), with
    // a ragged last one - the case a single-round prompt cannot reach.
    assert!(prompt_tokens > 3 * 256, "this gate needs a prompt spanning several prefill rounds, got {prompt_tokens} tokens");
    assert!(
        text.contains("Paris"),
        "after {prompt_tokens} prompt tokens the stack must still continue \"The capital city of France is\" with Paris, got {text:?}"
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
    let cost = layer_cost(&cfg, CAP, &TierPolicy::uniform(Dtype::I8), KvTier::F32, 1);
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
    assert_eq!(cost.per_layer[gdn], reference.layer_weight_bytes(LayerType::Linear, &TierPolicy::uniform(Dtype::I8)) + 4 * (48 * 128 * 128 + 10240 * 3));
    assert_eq!(cost.per_layer[gqa], reference.layer_weight_bytes(LayerType::Full, &TierPolicy::uniform(Dtype::I8)) + 2 * CAP as u64 * reference.kv_dim() as u64 * 4);
}

/// Diagnostic (not gated - prints, does not assert): sweeps the SAME
/// "France" filler prompt at increasing repeat counts on ONE loaded
/// instance, to find whether the real-checkpoint garbage seen at ~1500-1700
/// tokens (`decode_throughput_at_a_real_long_context`,
/// `prefill_throughput_at_a_real_long_context`) onsets at a length tied to
/// `int8_gguf_resident::MAX_PREFILL_TOKENS` round boundaries (256, 512, 768,
/// ...) or is simply a function of total context length regardless of
/// round count - every prompt here below `MAX_PREFILL_TOKENS` still goes
/// through `replay_prompt`'s SINGLE-round chunked path, so a break at a
/// short, single-round length would rule out round-to-round state carry
/// entirely and point at long-context GDN/GQA state drift instead.
#[test]
fn where_does_the_real_checkpoint_start_producing_garbage() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let cap: u32 = 2048;
    let r = Qwen35GgufResident::new(path, devices, cap, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    println!("placed on {} device(s): {placed:?}", placed.len());
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint");

    let para = "France is a country in Western Europe. Its territory stretches from the Rhine to the Atlantic \
                Ocean, and its people speak French. The country is known for its cuisine, its literature and \
                its long history. Many travellers visit each year to see its museums, its cathedrals and its \
                countryside. The Seine flows through the north of the country and past its largest urban area. ";
    for repeats in [1usize, 2, 4, 6, 9, 12, 16, 20, 23, 27] {
        let mut prompt = String::new();
        for _ in 0..repeats {
            prompt.push_str(para);
        }
        prompt.push_str("The capital city of France is");
        let inv = capability::Invocation::new()
            .set("prompt", serde_json::json!(prompt))
            .set("chat", serde_json::json!(false))
            .set("max_new", serde_json::json!(8))
            .set("temp", serde_json::json!(0.0));
        let out = inst.run("generate", &inv, &mut |_| {}).expect("generate on the real checkpoint");
        let prompt_tokens = out.outputs["prompt_tokens"].as_i64().unwrap_or(0);
        let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
        let ok = text.contains("Paris");
        println!("repeats={repeats:2} prompt_tokens={prompt_tokens:5} ok={ok:5} text={text:?}");
    }
}

/// Diagnostic (not gated - prints, does not assert): a clean, single-
/// instance sweep of DECODE tok/s as a function of context depth, now that
/// [`where_does_the_real_checkpoint_start_producing_garbage`]'s bug is
/// fixed and the numbers it prints are trustworthy. One loaded instance,
/// `max_new = 32` at every length (enough decode steps to average out
/// per-call noise), real prompt tokens from ~100 to ~3800 - the real curve
/// this crate's own `layer_decode_state_bytes` doc already predicts the
/// SHAPE of (GDN layers are `O(1)` in context, GQA layers are `O(context)`
/// KV-read), measured rather than assumed.
#[test]
fn decode_tok_per_s_across_context_depth() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let cap: u32 = 4096;
    let r = Qwen35GgufResident::new(path, devices, cap, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    println!("placed on {} device(s): {placed:?}", placed.len());
    let mut inst = r.activate_multi(&key, &placed).expect("activate the real checkpoint");

    let para = "France is a country in Western Europe. Its territory stretches from the Rhine to the Atlantic \
                Ocean, and its people speak French. The country is known for its cuisine, its literature and \
                its long history. Many travellers visit each year to see its museums, its cathedrals and its \
                countryside. The Seine flows through the north of the country and past its largest urban area. ";
    for repeats in [1usize, 3, 6, 9, 12, 16, 20, 25, 30, 35, 40, 48, 54] {
        let mut prompt = String::new();
        for _ in 0..repeats {
            prompt.push_str(para);
        }
        prompt.push_str("The capital city of France is");
        let inv = capability::Invocation::new()
            .set("prompt", serde_json::json!(prompt))
            .set("chat", serde_json::json!(false))
            .set("max_new", serde_json::json!(32))
            .set("temp", serde_json::json!(0.0));
        let out = inst.run("generate", &inv, &mut |_| {}).expect("generate on the real checkpoint");
        let prompt_tokens = out.outputs["prompt_tokens"].as_i64().unwrap_or(0);
        let text = out.outputs["text"].as_str().unwrap_or_default().to_string();
        let metrics: std::collections::HashMap<String, serde_json::Value> = inst.metrics().into_iter().collect();
        // The scheduler path reports one wall time per call (prefill and
        // decode together), so the rate is end-to-end generated tokens/s.
        let tok_s = metrics.get("last_batch_tok_per_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ok = text.contains("Paris");
        println!("prompt_tokens={prompt_tokens:5} generated_tok_s_incl_prefill={tok_s:6.3} ok={ok:5}");
    }
}

/// The released checkpoint under the generic continuous-batching scheduler:
/// every prompt's greedy continuation, decoded with its neighbours in ONE
/// batched step per iteration (slots joining and leaving as requests finish:
/// four requests through two running slots), is token-for-token what the
/// single-sequence reference loop (`Qwen35GgufInstance::generate`) produces
/// for it alone, and a request sampled under a seed reproduces that seed's
/// tokens when its neighbours change.
#[test]
fn scheduled_batches_reproduce_the_single_sequence_reference_on_the_real_checkpoint() {
    scheduled_batches_reproduce_the_reference(0);
}

/// The same gate with every sequence's KV in 16-row blocks of the shared pool (the layout that is not yet the default).
#[test]
fn scheduled_batches_reproduce_the_single_sequence_reference_with_paged_kv() {
    scheduled_batches_reproduce_the_reference(16);
}

fn scheduled_batches_reproduce_the_reference(kv_block_rows: u32) {
    use model::serve::{Request, SampleParams, Scheduler};

    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8)).with_max_batch(4).with_kv_block_rows(kv_block_rows);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let (engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");

    let prompts = ["The capital city of France is", "The chemical symbol for gold is", "Water freezes at a temperature of zero degrees", "The tallest mountain above sea level on Earth is Mount"];
    let new = [10usize, 6, 10, 8];
    let ids: Vec<Vec<u32>> = prompts.iter().map(|p| tok.encode(p)).collect();
    // Reference first: `generate` resets every slot, so it cannot run once
    // the scheduler owns live sequences.
    let want: Vec<Vec<u32>> = ids.iter().zip(new).map(|(p, n)| engine.instance().generate(p, n as u32, 0.0, 0, 1.0, 0, &mut |_| false).expect("reference generate")).collect();

    let mut sched = Scheduler::new(engine, 2);
    let submitted: Vec<u64> = ids.iter().zip(new).map(|(p, n)| sched.submit(Request { prompt: p.clone(), max_new: n, stop: Vec::new() })).collect();
    let done = sched.run();
    for (i, id) in submitted.iter().enumerate() {
        let got = &done[id];
        println!("  {:?} -> {:?}", prompts[i], tok.decode(got));
        assert_eq!(got, &want[i], "request {i} ({:?}): the batched continuation must be the single-sequence one", prompts[i]);
    }

    // Seeded sampling is per sequence: the same seed gives the same tokens
    // whether the sequence runs alone or beside others.
    let sample = SampleParams { temp: 0.8, top_k: 40, top_p: 0.95, ..SampleParams::greedy() };
    let solo = {
        let id = sched.submit_sampled(Request { prompt: ids[0].clone(), max_new: 8, stop: Vec::new() }, sample, 7);
        sched.run().remove(&id).unwrap()
    };
    let beside = {
        let id = sched.submit_sampled(Request { prompt: ids[0].clone(), max_new: 8, stop: Vec::new() }, sample, 7);
        for p in &ids[1..] {
            sched.submit_sampled(Request { prompt: p.clone(), max_new: 8, stop: Vec::new() }, sample, 99);
        }
        sched.run().remove(&id).unwrap()
    };
    assert_eq!(solo, beside, "a seeded sampled sequence must not depend on its batch neighbours");
}

/// Prefix reuse on the real checkpoint: a prompt that starts with an already
/// served one (an agent resending its history plus a new turn) resumes from the
/// cached KV rows and the recurrent-state snapshot taken at that prompt's
/// checkpoint (`checkpoint_len`: short of its generation prompt, on a whole
/// recurrent chunk), prefills only the tokens after it, and continues token for
/// token like a cold prefill of the whole prompt. An identical prompt resumes
/// from its checkpoint the same way.
#[test]
fn a_prompt_extending_a_cached_one_prefills_only_its_new_tokens_and_continues_identically() {
    a_prompt_extending_a_cached_one_resumes_from_it(0);
}

/// The same gate with the KV in 16-row blocks: the extension shares the cached prompt's blocks.
#[test]
fn a_prompt_extending_a_cached_one_resumes_identically_with_paged_kv() {
    a_prompt_extending_a_cached_one_resumes_from_it(16);
}

fn a_prompt_extending_a_cached_one_resumes_from_it(kv_block_rows: u32) {
    use model::serve::{Request, Scheduler};

    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let r = Qwen35GgufResident::new(path, devices, 2048, TierPolicy::uniform(Dtype::I8)).with_max_batch(2).with_prefix_slots(2).with_kv_block_rows(kv_block_rows);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let (engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");

    let para = "France is a country in Western Europe. Its territory stretches from the Rhine to the Atlantic \
                Ocean, and its people speak French. The country is known for its cuisine, its literature and \
                its long history. Many travellers visit each year to see its museums, its cathedrals and its \
                countryside. The Seine flows through the north of the country and past its largest urban area. ";
    let history = para.repeat(5);
    let first = tok.encode(&format!("{history}The capital city of France is"));
    let second = tok.encode(&format!("{history}The capital city of France is Paris. The largest river in France is the"));
    assert!(second.starts_with(&first) || second.len() > first.len(), "the second prompt extends the first");
    let shared = first.iter().zip(&second).take_while(|(a, b)| a == b).count();
    assert!(shared >= 64, "the prompts must share a real prefix, shared {shared}");

    // Cold references first (`generate` resets every slot and bypasses the cache).
    let want_first = engine.instance().generate(&first, 8, 0.0, 0, 1.0, 0, &mut |_| false).expect("cold first");
    let want_second = engine.instance().generate(&second, 8, 0.0, 0, 1.0, 0, &mut |_| false).expect("cold second");

    let mut sched = Scheduler::new(engine, 2);
    let run = |sched: &mut Scheduler<_>, prompt: &[u32]| {
        let t = std::time::Instant::now();
        let id = sched.submit(Request { prompt: prompt.to_vec(), max_new: 8, stop: Vec::new() });
        let out = sched.run().remove(&id).unwrap();
        (out, t.elapsed().as_secs_f64())
    };
    let (got_first, cold_s) = run(&mut sched, &first);
    assert_eq!(got_first, want_first);
    let (hit0, _, cached0) = sched.prefix_stats();
    assert_eq!((hit0, cached0), (0, 1), "the first prompt is cached, nothing was reused");

    let (got_second, warm_s) = run(&mut sched, &second);
    let (hit1, looked1, cached1) = sched.prefix_stats();
    println!("  prefix reuse: first {} tokens cold {cold_s:.2} s, second {} tokens resumed {warm_s:.2} s; hit {hit1}/{looked1} tokens, {cached1} cached", first.len(), second.len());
    // What is cached is the first prompt's checkpoint (short of its generation prompt, on a whole recurrent chunk), and
    // the extension shares it.
    let checkpoint = qwen35::int8_gguf_resident::checkpoint_len(first.len());
    assert!(checkpoint >= 64 && checkpoint <= shared, "the first prompt has a checkpoint the second shares ({checkpoint} of {shared})");
    assert_eq!(hit1, checkpoint as u64, "the second prompt resumed from the first prompt's checkpoint");
    assert_eq!(got_second, want_second, "resuming from a cached prefix must continue like a cold prefill");
    assert_eq!(cached1, 1, "one entry: the second prompt's own checkpoint is no longer than the one it resumed from");

    let (again, _) = run(&mut sched, &second);
    assert_eq!(again, want_second, "an identical prompt resumed from the cached checkpoint continues like a cold prefill");
    assert_eq!(sched.prefix_stats().0, hit1 + qwen35::int8_gguf_resident::checkpoint_len(second.len()) as u64);
}

/// With speculation from the checkpoint's own MTP head over a paged KV cache (16-row blocks), the drafts stay inside
/// the rows a sequence's blocks hold (the head writes its KV for proposal `j` at `len + j - 2` before the verify step
/// appends any row; a sequence ending on a block boundary used to write past its table and panic the decode), and a
/// prompt resumed from the previous turn's cached checkpoint still drafts from the head: the entry's head rows were
/// filled by the feed of the whole earlier prompt (before, a resumed agent turn drafted nothing and decoded at the plain
/// rate; knowledge 347).
#[test]
fn mtp_drafts_stay_inside_paged_blocks_and_a_resumed_checkpoint_still_drafts() {
    use model::serve::{Request, Scheduler};
    use qwen35::int8_gguf_resident::{SpecServing, SpeculativeMode};

    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("this gate wants at least one usable GPU");
        return;
    }
    let spec = SpecServing { mode: SpeculativeMode::Mtp, ..SpecServing::default() };
    let r = Qwen35GgufResident::new(path, devices, 2048, TierPolicy::uniform(Dtype::I8)).with_max_batch(2).with_prefix_slots(2).with_kv_block_rows(16).with_speculation(spec);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let (engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");
    assert!(engine.instance().has_mtp(), "the released checkpoint carries its MTP head");

    let lines: String = (1..=40).map(|i| format!("let value_{i} = compute(value_{}, {i});\n", i - 1)).collect();
    let first = tok.encode(&format!("Copy this code exactly:\n{lines}\nCopy:\n"));
    // The next turn agrees with the first prompt up to its last few tokens, as a re-rendered generation prompt does.
    let mut second = first[..first.len() - 4].to_vec();
    second.extend(tok.encode(" again, from the top:\n"));

    let mut sched = Scheduler::new(engine, 2);
    let id = sched.submit(Request { prompt: first.clone(), max_new: 96, stop: Vec::new() });
    sched.run().remove(&id).expect("the first turn completes");
    let (hit0, before) = (sched.prefix_stats().0, sched.spec_stats());
    assert!(before.accepted > 0, "the head's proposals are accepted on copied code");
    let id = sched.submit(Request { prompt: second.clone(), max_new: 96, stop: Vec::new() });
    sched.run().remove(&id).expect("the second turn completes");
    let (hit, after) = (sched.prefix_stats().0 - hit0, sched.spec_stats());
    println!("  resumed from {hit} of {} cached tokens; second turn proposed {} accepted {} in {} rounds", first.len(), after.proposed - before.proposed, after.accepted - before.accepted, after.rounds - before.rounds);
    assert!(hit > 0 && (hit as usize) < first.len(), "the second turn resumed from the first prompt's checkpoint, not the whole prompt (hit {hit})");
    assert!(after.proposed > before.proposed, "a prompt resumed from a checkpoint drafts from the MTP head");
    assert!(after.accepted > before.accepted, "and the target accepts some of the head's proposals");
}

/// Host time of gathering a 512-token prompt round's embedding rows from the real checkpoint: one `tensor_range` per
/// row (what a round did) against one `tensor_rows` gather (what it does now), best of several rounds. A timing
/// report, not a spec; CPU only.
#[test]
#[ignore = "a timing report for host work, not a spec"]
fn embedding_gather_timing_at_a_prompt_round() {
    let Some(path) = gguf_path() else { return };
    let mg = MmapGguf::open(&path).expect("open the real checkpoint");
    let (name, d) = ("token_embd.weight", 5120usize);
    let rows: Vec<usize> = (0..512usize).map(|i| (i * 7919 + 13) % 248_000).collect();
    let best = |f: &dyn Fn() -> Vec<f32>| (0..5).map(|_| { let t = Instant::now(); std::hint::black_box(f()); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min);
    let per_row = best(&|| rows.iter().flat_map(|&r| mg.tensor_range(name, r * d, d).expect("in range").expect("decodes")).collect());
    let gather = best(&|| mg.tensor_rows(name, &rows, d).expect("in range").expect("decodes"));
    assert_eq!(rows.iter().flat_map(|&r| mg.tensor_range(name, r * d, d).unwrap().unwrap()).collect::<Vec<f32>>(), mg.tensor_rows(name, &rows, d).unwrap().unwrap());
    println!("  512 embedding rows: one range per row {:.2} ms, one gather {:.2} ms", per_row * 1e3, gather * 1e3);
}

/// The released checkpoint is held as it is stored: a 4-bit release (its MLP
/// gate in Q4_K) costs well under two thirds of the int8 bytes, and every
/// other release exactly the int8 bytes. Reads headers only.
#[test]
fn the_checkpoint_is_held_as_it_is_stored() {
    let Some(path) = gguf_path() else { return };
    let mg = MmapGguf::open(&path).expect("open the GGUF");
    let cfg = resident_config(&mg, CAP).expect("a Qwen3.8 GGUF");
    let held = qwen35::int8_gguf_resident::held_tiers(&mg, &cfg).expect("the held tiers");
    let bytes = |tier: &TierPolicy| -> u64 { cfg.layer_types().into_iter().enumerate().map(|(l, ty)| cfg.layer_weight_bytes_at(&format!("blocks.{l}."), ty, tier)).sum() };
    let (held_bytes, i8_bytes) = (bytes(&held), bytes(&TierPolicy::uniform(Dtype::I8)));
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!("held as stored: {} with {} per-tensor exceptions, {:.2} GiB of layers (int8: {:.2} GiB)", held.default_dtype().name(), held.rule_count(), gib(held_bytes), gib(i8_bytes));
    if mg.dtype("blk.0.ffn_gate.weight") == Some("Q4_K") {
        assert_eq!(held.default_dtype(), Dtype::Q4K);
        assert!(held_bytes * 3 < i8_bytes * 2, "a 4-bit release is held near 4 bits: {:.2} of {:.2} GiB", gib(held_bytes), gib(i8_bytes));
    } else {
        assert_eq!(held_bytes, i8_bytes, "every other release is held at int8");
    }
}

/// Contexts the gate loads the checkpoint at: the decode step's scratch grows with the context.
const CAPS: [u32; 2] = [8192, 100_000];

/// The planner's prefill and decode workspace against what a real prefill round and decode step allocate on a
/// card: the placement keeps this free, so a figure under the measurement lets a served model run out of device
/// memory on a long prompt after it has loaded. Needs one CUDA card (`--device gpuN --backend cuda` through
/// `BRAIN_DEVICE`/`BRAIN_BACKEND`) beside the checkpoint.
#[test]
fn the_planned_workspace_covers_what_a_real_step_allocates() {
    let Some(path) = gguf_path() else { return };
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM");
        return;
    }
    let probe = gpu_core::Gpu::new(&[]);
    if probe.memory_usage().is_none() {
        brain_testutil::skip_unavailable("this backend does not account for its allocations");
        return;
    }
    let mg = checkpoint::gguf::MmapGguf::open(&path).expect("open the checkpoint");
    let cfg = resident_config(&mg, 1).expect("a servable checkpoint");
    for cap in CAPS {
        covers_a_round_and_a_step_at(&path, &devices, &cfg, &probe, cap);
    }
}

/// Load the checkpoint at `cap` and compare the planner's figures with the peak of a long prefill and of decode steps.
fn covers_a_round_and_a_step_at(path: &str, devices: &[(Device, u64)], cfg: &Qwen35Config, probe: &gpu_core::Gpu, cap: u32) {
    let r = Qwen35GgufResident::new(path.to_string(), devices.to_vec(), cap, TierPolicy::uniform(Dtype::I8)).with_kv_tier(KvTier::INT8).with_max_batch(8);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");

    // Several full rounds, so the largest round runs after earlier ones have filled the pool.
    let prompt: Vec<u32> = (0..5000u32).map(|i| 1000 + i % 5000).collect();
    inst.prefill_timed(&prompt[..64]).expect("warm");
    let live = probe.memory_usage().unwrap().live;
    probe.reset_memory_peak();
    inst.prefill_timed(&prompt).expect("prefill");
    let transient = probe.memory_usage().unwrap().peak_live - live;
    let rows = prefill_round_rows_for(true);
    let planned = prefill_workspace_bytes(cfg, rows);
    println!("cap {cap}: prefill rows {rows}: measured {} MiB, planned {} MiB", transient >> 20, planned >> 20);
    assert!(planned >= transient, "the planner charges {planned} bytes of prefill workspace but a real round allocated {transient}");

    // Widest first: the first step also fills the replay arena for its row count.
    for rows in [8u32, 4, 2, 1] {
        probe.reset_memory_peak();
        let toks = vec![5u32; rows as usize];
        let pos: Vec<u32> = (0..rows).map(|i| cap - 100 - i * 10).collect();
        let live = probe.memory_usage().unwrap().live;
        inst.decode_batch_at(&toks, &pos).expect("decode");
        let measured = probe.memory_usage().unwrap().peak_live - live;
        let planned = decode_scratch_bytes(cfg, rows, cap);
        println!("cap {cap}: decode {rows} rows: measured {} MiB, planned {} MiB", measured >> 20, planned >> 20);
        assert!(planned >= measured, "the planner charges {planned} bytes of decode scratch at {rows} rows but a step allocated {measured}");
    }
}
