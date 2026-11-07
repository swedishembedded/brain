// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight tiers, KV tiers and the stage split of the serving engine, on the tiny model.
//!
//! Swedish Embedded AB implements solutions for serving large language models in the precision a deployment
//! can afford, with the loss of every format measured against the full-precision reference, for its clients. If
//! your team needs expertise in quantized weights and KV caches for long-context serving then you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! Three questions, kept apart:
//!
//! 1. **Is the tier executed right?** The engine at a tier must equal the f32 engine run on the weights that
//!    tier's codec decodes to (`Weight::round_trip`, or the independent fixture encoder for NVFP4): the only
//!    difference left is the order of f32 accumulation.
//! 2. **What does the tier cost in quality?** Measured against the original f32 weights as the mean KL divergence
//!    of the next-token distribution and the share of rows whose top-1 token is unchanged.
//! 3. **Does the split change anything?** A two-stage engine must produce the single-stage engine's logits.
//!
//! The measured numbers are printed (`--nocapture`); the bounds below are those measurements with margin.

use std::path::Path;

use brain_testutil::hfquant::{self, Raw};
use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use model::ops::TierPolicy;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use model::shard::{Shard, StreamPlan};
use ricursive::{manifest, open_safetensors, RicursiveEngine, ServeConfig};

use crate::serve::{prefill_in_slices, serve_config, shared_weights, stream, tiny_config, tiny_dir, weights};
use crate::taps::scaled_error;

/// `(label, logits row)` of a scripted run: a prefill row then one row per decode step.
type Trace = Vec<Vec<f32>>;

const PROMPT: usize = 10;
const DECODES: usize = 14;

/// Prefill the first `PROMPT` tokens of a stream in slices of 2 (window 4 wraps) and decode `DECODES` more with the
/// stream's own tokens as inputs: the logits of every step.
fn run(e: &mut RicursiveEngine, ids: &[u32]) -> Trace {
    let mut table = BlockTable::new();
    let hidden = prefill_in_slices(e, &mut table, &ids[..PROMPT], 2);
    let mut rows = vec![e.logits(&hidden)];
    for &token in &ids[PROMPT..PROMPT + DECODES] {
        rows.push(e.decode_logits(&mut [&mut table], &[token]).remove(0));
    }
    e.release_table(&mut table);
    rows
}

fn ids(vocab: usize) -> Vec<u32> {
    stream(7, PROMPT + DECODES, vocab as u32)
}

fn softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = z.iter().map(|&v| f64::from(v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn argmax(z: &[f32]) -> usize {
    z.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
}

/// `(mean KL(reference || got), share of rows with the same top-1 token)` over the trace.
fn quality(reference: &Trace, got: &Trace) -> (f64, f64) {
    assert_eq!(reference.len(), got.len());
    let mut kl = 0.0;
    let mut same = 0usize;
    for (r, g) in reference.iter().zip(got) {
        let (p, q) = (softmax(r), softmax(g));
        kl += p.iter().zip(&q).map(|(p, q)| if *p > 0.0 { p * (p / q.max(1e-300)).ln() } else { 0.0 }).sum::<f64>();
        same += usize::from(argmax(r) == argmax(g));
    }
    (kl / reference.len() as f64, same as f64 / reference.len() as f64)
}

/// The worst scaled error over the rows of two traces.
fn worst_scaled(label: &str, got: &Trace, want: &Trace) -> f32 {
    got.iter().zip(want).enumerate().map(|(i, (g, w))| scaled_error(&format!("{label} row {i}"), g, w)).fold(0.0, f32::max)
}

fn open_engine(dir: &str, serve: ServeConfig) -> Result<RicursiveEngine, String> {
    let cfg = tiny_config(dir);
    RicursiveEngine::from_source(cfg.clone(), shared_weights(dir, &cfg), serve, None)
}

/// The tiny weights with every linear replaced by what the device decodes its `tier` upload to (the engine's own
/// load, `upload_from_source`, read back with `Ops::read_host`), written as an f32 checkpoint: the f32 engine over
/// it is the reference for an engine at `tier`.
fn dequantized_checkpoint(dir: &str, tier: Dtype, into: &Path) -> String {
    let cfg = tiny_config(dir);
    let src = weights(dir, &cfg);
    let ops = model::ops::Ops::new(gpu_core::Gpu::open(None, &ricursive::pipelines(&cfg, &[]))).expect("ops");
    let staging = checkpoint::packed::BoundedWorkspace::new(checkpoint::packed::MemoryBudget::new(64 << 20, 0));
    let linear: Vec<String> = ricursive::linear_names(&cfg);
    let tensors: Vec<Raw> = manifest(&cfg, &(0..cfg.num_hidden_layers).collect::<Vec<_>>())
        .into_iter()
        .map(|(name, shape)| {
            let mut data = Vec::new();
            checkpoint::TensorSource::with_tensor(&src, &name, &mut |d| data = d.to_vec());
            if linear.contains(&name) {
                let w = model::ops::upload_from_source(&ops, &src, &name, shape[0], shape[1], tier, &staging).expect("tensor");
                assert_eq!(w.dtype(), tier, "{name} is held at the tier");
                data = ops.read_host(&w);
            }
            hfquant::f32_raw(&name, &shape, &data)
        })
        .collect();
    let path = into.join("dequantized.safetensors");
    hfquant::write_safetensors(&path, &tensors);
    path.to_string_lossy().into_owned()
}

fn engine_over(dir: &str, file: &str, serve: ServeConfig) -> RicursiveEngine {
    let cfg = tiny_config(dir);
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    RicursiveEngine::from_source(cfg.clone(), ricursive::open_shared_checkpoint(file, &cfg, &layers).expect("open"), serve, None).expect("engine")
}

fn uniform(tier: Dtype) -> Option<TierPolicy> {
    Some(TierPolicy::uniform(tier))
}

// ---- weight tiers ------------------------------------------------------------------------------------------------

/// Spec: a weight tier is executed as its codec defines it. `Weight::round_trip` is the codec's host decode (held
/// equal to the device's by `model`'s own tests); the f32 engine over those values must give the tiered engine's
/// logits to accumulation noise.
#[test]
fn each_weight_tier_executes_as_its_codec_decodes() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let ids = ids(cfg.vocab_size);
    let tmp = tempfile::tempdir().unwrap();
    for (tier, tolerance) in [(Dtype::BF16, 2e-5f32), (Dtype::F16, 2e-5), (Dtype::F8E4M3, 2e-5)] {
        let file = dequantized_checkpoint(&dir, tier, tmp.path());
        let want = run(&mut engine_over(&dir, &file, serve_config(4, 0)), &ids);
        let mut tiered = open_engine(&dir, ServeConfig { weights: uniform(tier), ..serve_config(4, 0) }).unwrap_or_else(|e| panic!("{tier:?}: {e}"));
        assert_eq!(tiered.weight_report().count(tier), 3 * 8 + 1, "{tier:?}: every linear (8 a layer, 3 layers) and the head is held at the tier");
        let got = run(&mut tiered, &ids);
        let worst = worst_scaled(&format!("{tier:?}"), &got, &want);
        eprintln!("{tier:?}: engine vs f32 engine on the codec's decode: worst scaled error {worst:.3e} (tolerance {tolerance:.0e})");
        assert!(worst <= tolerance, "{tier:?}: {worst:.3e} > {tolerance:.0e}");
    }
}

/// Spec: what a tier costs. The original f32 weights are the reference; bf16 and f16 keep the next-token
/// distribution essentially intact, FP8 (a 3-bit mantissa) a little less. The bounds are the measured values
/// with margin, recorded in the roadmap.
#[test]
fn weight_tiers_keep_the_next_token_distribution_of_the_f32_model() {
    let Some(dir) = tiny_dir() else { return };
    let ids = ids(tiny_config(&dir).vocab_size);
    let reference = run(&mut open_engine(&dir, serve_config(4, 0)).expect("f32"), &ids);
    for (tier, max_kl, min_top1) in [(Dtype::BF16, 1e-3, 0.9), (Dtype::F16, 1e-5, 1.0), (Dtype::F8E4M3, 5e-2, 0.6)] {
        let got = run(&mut open_engine(&dir, ServeConfig { weights: uniform(tier), ..serve_config(4, 0) }).unwrap(), &ids);
        let (kl, top1) = quality(&reference, &got);
        eprintln!("{tier:?} vs f32: mean KL {kl:.3e}, top-1 agreement {:.1}% ({} rows)", top1 * 100.0, reference.len());
        assert!(kl <= max_kl, "{tier:?}: KL {kl:.3e} > {max_kl:.0e}");
        assert!(top1 >= min_top1, "{tier:?}: top-1 {top1}");
    }
}

/// Spec: a tier the device cannot hold is refused by name, with the tier it would have used, and never widened
/// silently. The CPU JIT has no packed-int8 dot, so int8 is refused here; a device that has it runs it.
#[test]
fn int8_runs_where_the_device_has_the_packed_dot_and_is_refused_by_name_where_it_has_not() {
    let Some(dir) = tiny_dir() else { return };
    let ids = ids(tiny_config(&dir).vocab_size);
    match open_engine(&dir, ServeConfig { weights: uniform(Dtype::I8), ..serve_config(4, 0) }) {
        Err(e) => {
            assert!(e.contains("int8") && e.contains("f32") && e.contains("model.layers.0"), "{e}");
            eprintln!("int8 on this device is refused: {e}");
        }
        Ok(mut e) => {
            assert_eq!(e.weight_report().count(Dtype::I8), 3 * 8 + 1);
            let tmp = tempfile::tempdir().unwrap();
            let file = dequantized_checkpoint(&dir, Dtype::I8, tmp.path());
            let reference = run(&mut engine_over(&dir, &file, serve_config(4, 0)), &ids);
            let original = run(&mut open_engine(&dir, serve_config(4, 0)).unwrap(), &ids);
            let got = run(&mut e, &ids);
            // The activations are quantized as well (W8A8), which the weight-only decode above does not model.
            let (kl, top1) = quality(&original, &got);
            eprintln!("int8: vs the codec's decode scaled error {:.3e}; vs f32 KL {kl:.3e}, top-1 {:.1}%", worst_scaled("int8", &got, &reference), top1 * 100.0);
            assert!(kl < 5e-3 && top1 >= 0.8);
        }
    }
}

/// Spec: the tier policy comes from the model-agnostic option types: `--weights` and `--kv-cache` are parsed by
/// `model::policy`, the model declares what it supports, and an unsupported choice is refused naming the model.
#[test]
fn the_engine_takes_its_tiers_from_the_serving_policy_and_refuses_what_it_does_not_declare() {
    let mut policy = model::policy::ServePolicy::default();
    policy.set("--weight-format", "bf16").unwrap();
    policy.set("--kv-cache", "int8").unwrap();
    let mut serve = ServeConfig::new(4, 8, 32, 1, 0, 4);
    serve.apply_policy(&policy).expect("declared");
    assert_eq!(serve.kv_cache, KvTier::INT8);
    assert_eq!(serve.weights.as_ref().and_then(TierPolicy::uniform_dtype), Some(Dtype::BF16));

    let mut nf4 = model::policy::ServePolicy::default();
    nf4.set("--weight-format", "nf4").unwrap();
    let err = ServeConfig::new(4, 8, 32, 1, 0, 4).apply_policy(&nf4).unwrap_err();
    assert!(err.contains("ricursive") && err.contains("nf4") && err.contains("supported"), "{err}");
    let mut gdn = model::policy::ServePolicy::default();
    gdn.set("--gdn-state", "bf16").unwrap();
    assert!(ServeConfig::new(4, 8, 32, 1, 0, 4).apply_policy(&gdn).unwrap_err().contains("ricursive"));
}

// ---- NVFP4 ---------------------------------------------------------------------------------------------------------

/// The tiny model's MLP linears stored as a ModelOpt NVFP4 release stores them (U8 codes, E4M3 block scales, an
/// f32 `weight_scale_2` and `input_scale`), everything else as f32, written by an encoder that shares no code
/// with brain's. Returns the checkpoint file and the f32 file holding the values the NVFP4 tensors decode to.
fn nvfp4_release(dir: &str, into: &Path, input_scale: f32) -> (String, String) {
    let cfg = tiny_config(dir);
    let src = weights(dir, &cfg);
    let (mut stored, mut decoded): (Vec<Raw>, Vec<Raw>) = (Vec::new(), Vec::new());
    for (name, shape) in manifest(&cfg, &(0..cfg.num_hidden_layers).collect::<Vec<_>>()) {
        let mut data = Vec::new();
        checkpoint::TensorSource::with_tensor(&src, &name, &mut |d| data = d.to_vec());
        if name.contains(".mlp.") && name.ends_with("_proj.weight") {
            let module = name.trim_end_matches(".weight");
            let values = hfquant::nvfp4(&mut stored, module, &shape, &data);
            // The fixture writes a fixed input scale; the test sets the one it wants.
            let last = stored.len() - 1;
            stored[last] = hfquant::f32_raw(&format!("{module}.input_scale"), &[], &[input_scale]);
            decoded.push(hfquant::f32_raw(&name, &shape, &values));
        } else {
            stored.push(hfquant::f32_raw(&name, &shape, &data));
            decoded.push(hfquant::f32_raw(&name, &shape, &data));
        }
    }
    let (a, b) = (into.join("nvfp4.safetensors"), into.join("nvfp4_decoded.safetensors"));
    hfquant::write_safetensors(&a, &stored);
    hfquant::write_safetensors(&b, &decoded);
    (a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned())
}

/// Spec: a ModelOpt NVFP4 release loads as it is stored, with no conversion: the MLP is resident as NVFP4, the
/// rest stays as stored, and the engine's logits are the f32 engine's over the values the NVFP4 tensors decode to
/// (E2M1 code x E4M3 block scale x f32 `weight_scale_2`). The activations are held in float (W4A16), so the
/// reference needs no activation quantization.
#[test]
fn a_modelopt_nvfp4_release_loads_as_stored_and_executes_its_dequantization() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let ids = ids(cfg.vocab_size);
    let tmp = tempfile::tempdir().unwrap();
    let (release, decoded) = nvfp4_release(&dir, tmp.path(), 0.0123);

    let mut e = engine_over(&dir, &release, serve_config(4, 0));
    assert_eq!(e.weight_report().count(Dtype::F4E2M1), 3 * cfg.num_hidden_layers, "every MLP linear is NVFP4");
    assert_eq!(e.weight_report().count(Dtype::F32), 5 * cfg.num_hidden_layers + 1, "attention and lm_head stay as stored (5 attention linears a layer)");
    let got = run(&mut e, &ids);
    let want = run(&mut engine_over(&dir, &decoded, serve_config(4, 0)), &ids);
    let worst = worst_scaled("nvfp4", &got, &want);
    eprintln!("NVFP4 engine vs f32 engine on the decoded values: worst scaled error {worst:.3e}");
    assert!(worst <= 2e-5, "{worst:.3e}");

    let original = run(&mut open_engine(&dir, serve_config(4, 0)).unwrap(), &ids);
    let (kl, top1) = quality(&original, &got);
    eprintln!("NVFP4 (tiny random MLP) vs the unquantized f32 model: mean KL {kl:.3e}, top-1 {:.1}%", top1 * 100.0);
    // Random weights have no structure for 4 bits to preserve; the real tensors are measured in `real_nvfp4.rs`.
    assert!(kl < 0.3 && top1 >= 0.4);

    let scales = ricursive::import::activation_scales(&release).expect("valid scales");
    assert_eq!((scales.count, scales.quantized_without), (3 * cfg.num_hidden_layers, 0));
    assert_eq!(scales.range, (0.0123, 0.0123));
}

/// Spec: the static activation scales are checked, not ignored: a release whose `input_scale` is not a positive
/// finite scalar, or that has them for some quantized weights only, is refused by name.
#[test]
fn a_release_with_a_bad_static_activation_scale_is_refused() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let tmp = tempfile::tempdir().unwrap();
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let (release, _) = nvfp4_release(&dir, tmp.path(), bad);
        let err = open_safetensors(&release, &cfg, &layers).err().unwrap_or_else(|| panic!("{bad}: accepted"));
        assert!(err.contains("input_scale") && err.contains("positive"), "{bad}: {err}");
    }
}

/// Spec: NVFP4 is held as the release stores it. Asking for it from a checkpoint that does not store it, or
/// quantizing a stored NVFP4 weight into some other 4-bit format, is an error, not a conversion.
#[test]
fn nvfp4_cannot_be_asked_of_a_checkpoint_that_does_not_store_it() {
    let Some(dir) = tiny_dir() else { return };
    let err = open_engine(&dir, ServeConfig { weights: uniform(Dtype::F4E2M1), ..serve_config(4, 0) }).err().expect("refused");
    assert!(err.contains("NVFP4") && err.contains("does not store"), "{err}");
}

// ---- KV tiers ------------------------------------------------------------------------------------------------------

/// Spec: a compact KV cache changes the attention inputs by its rounding and nothing else: decode through
/// several window wraps and block boundaries stays within a measured distance of the f32 cache, in the rings and
/// in the global pool. (bf16 truncates the mantissa to 8 bits; int8 and fp8 keep one scale per (token, kv head).)
#[test]
fn compact_kv_tiers_decode_within_a_measured_distance_of_the_f32_cache() {
    let Some(dir) = tiny_dir() else { return };
    let ids = ids(tiny_config(&dir).vocab_size);
    let reference = run(&mut open_engine(&dir, serve_config(4, 0)).unwrap(), &ids);
    for (tier, scaled, max_kl, min_top1) in [(KvTier::BF16, 5e-3f32, 1e-4, 1.0), (KvTier::INT8, 5e-2, 1e-2, 0.9), (KvTier::FP8, 1e-1, 3e-2, 0.85)] {
        let mut e = open_engine(&dir, ServeConfig { kv_cache: tier, ..serve_config(4, 0) }).unwrap();
        let got = run(&mut e, &ids);
        let worst = worst_scaled(&format!("kv {tier}"), &got, &reference);
        let (kl, top1) = quality(&reference, &got);
        eprintln!("KV {tier}: worst scaled logits error {worst:.3e}, mean KL {kl:.3e}, top-1 {:.1}%", top1 * 100.0);
        assert!(worst <= scaled && kl <= max_kl && top1 >= min_top1, "KV {tier}: scaled {worst:.3e} (max {scaled:.0e}), KL {kl:.3e} (max {max_kl:.0e}), top-1 {top1}");
        let (global, rings) = e.kv_bytes();
        assert_eq!(e.kv_pool_bytes(), global + rings);
        assert!(global < ricursive::global_kv_bytes_per_token(&tiny_config(&dir), KvTier::F32) * 160 * 4, "{tier}: the pool is smaller than f32's");
    }
}

/// Spec: prefix reuse is exact within a tier: the ring snapshot and the shared blocks are copied as stored (words and
/// row scales), so a resumed prompt decodes as the same prompt prefilled cold at that tier.
#[test]
fn prefix_reuse_over_a_compact_cache_resumes_to_the_cold_logits() {
    let Some(dir) = tiny_dir() else { return };
    let vocab = tiny_config(&dir).vocab_size as u32;
    for tier in [KvTier::BF16, KvTier::INT8, KvTier::FP8] {
        let mut worst = 0.0f32;
        // 20 tokens end on a block boundary; 18 end inside a block (the shared tail is copied before it is written).
        for filed_len in [20usize, 18] {
            let ids = stream(11, filed_len + 6 + 2, vocab);
            let extended = &ids[..filed_len + 6];
            let cold = {
                let mut e = open_engine(&dir, ServeConfig { kv_cache: tier, ..serve_config(4, 0) }).unwrap();
                let mut t = BlockTable::new();
                let h = prefill_in_slices(&mut e, &mut t, extended, usize::MAX);
                let mut rows = vec![e.logits(&h)];
                rows.push(e.decode_logits(&mut [&mut t], &[ids[filed_len + 6]]).remove(0));
                rows
            };
            let mut e = open_engine(&dir, ServeConfig { kv_cache: tier, ..serve_config(4, 2) }).unwrap();
            let mut first = BlockTable::new();
            prefill_in_slices(&mut e, &mut first, &ids[..filed_len], usize::MAX);
            let mut second = BlockTable::new();
            let h = prefill_in_slices(&mut e, &mut second, extended, usize::MAX);
            assert_eq!(second.cached_rows() as usize, filed_len, "{tier}: the second request resumes from the filed prompt");
            let warm = vec![e.logits(&h), e.decode_logits(&mut [&mut second], &[ids[filed_len + 6]]).remove(0)];
            worst = worst.max(worst_scaled(&format!("{tier} filed {filed_len}"), &warm, &cold));
        }
        eprintln!("KV {tier}: warm vs cold worst scaled error {worst:.3e}");
        assert!(worst <= 1e-5, "{tier}: {worst:.3e}");
    }
}

// ---- stages --------------------------------------------------------------------------------------------------------

/// Two stages on the ambient device (a CPU box has no card indices to pin): each opens its own device handle.
fn two_stages() -> StreamPlan {
    StreamPlan::balanced(&model::shard::ShardCost { n_layers: 3, per_layer: 1.0, embed: 0.0, head: 0.0, boundary_words: 0 }, &[Shard::ANY_GPU, Shard::ANY_GPU])
}

/// Spec: a stage split moves the layers, not the numbers: the residual crosses the boundary unchanged (an f32 copy
/// between two device handles) and block tables, rings and the prefix cache are shared, so a two-stage engine
/// reproduces the single-stage engine's logits exactly, through prefill, decode and prefix reuse.
#[test]
fn a_two_stage_engine_reproduces_the_single_stage_logits() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let ids = ids(cfg.vocab_size);
    let single = run(&mut open_engine(&dir, serve_config(4, 2)).unwrap(), &ids);
    let plan = two_stages();
    assert_eq!(plan.n_stages(), 2);
    let staged_engine = |serve: ServeConfig| RicursiveEngine::from_source_staged(cfg.clone(), shared_weights(&dir, &cfg), serve, None, &plan).expect("staged engine");
    let mut staged = staged_engine(serve_config(4, 2));
    assert_eq!(staged.stage_plan().n_stages(), 2);
    let got = run(&mut staged, &ids);
    let worst = worst_scaled("two stages", &got, &single);
    eprintln!("two stages vs one: worst scaled error {worst:.3e}");
    assert!(worst <= 1e-6, "{worst:.3e}");

    // Prefix reuse across the boundary: both stages' rings and blocks are copied.
    let vocab = cfg.vocab_size as u32;
    let long = stream(11, 26, vocab);
    let (mut e, mut one) = (staged_engine(serve_config(4, 2)), open_engine(&dir, serve_config(4, 2)).unwrap());
    for engine in [&mut e, &mut one] {
        let mut t = BlockTable::new();
        prefill_in_slices(engine, &mut t, &long[..20], usize::MAX);
    }
    let mut rows = Vec::new();
    for engine in [&mut e, &mut one] {
        let mut t = BlockTable::new();
        let h = prefill_in_slices(engine, &mut t, &long[..26], usize::MAX);
        assert_eq!(t.cached_rows(), 20);
        rows.push(vec![engine.logits(&h)]);
    }
    assert!(worst_scaled("staged prefix reuse", &rows[0], &rows[1]) <= 1e-6);
}

/// Spec: the D-LSA capture and the scoring pass work across stages (each global layer's queries and keys are read
/// from the stage that owns it), and a compact cache is refused by name because the capture reads unquantized keys.
#[test]
fn attention_capture_and_scoring_work_across_stages_and_refuse_a_compact_cache() {
    use model::lsa::capture::AttentionCapture;
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let ids = stream(3, 20, cfg.vocab_size as u32);
    let positions: Vec<u32> = vec![3, 9, 15, 19];
    let mut one = open_engine(&dir, serve_config(4, 0)).unwrap();
    let plan = two_stages();
    let mut two = RicursiveEngine::from_source_staged(cfg.clone(), shared_weights(&dir, &cfg), serve_config(4, 0), None, &plan).expect("staged engine");
    let (a, b) = (one.capture_attention(&ids, &positions).unwrap(), two.capture_attention(&ids, &positions).unwrap());
    let (a, b) = (a[2].as_ref().unwrap(), b[2].as_ref().unwrap());
    assert!(a.q == b.q && a.k == b.k, "the staged capture is the single-stage capture");
    assert_eq!(one.position_logprobs(&ids, 4), two.position_logprobs(&ids, 4));

    let mut int8 = open_engine(&dir, ServeConfig { kv_cache: KvTier::INT8, ..serve_config(4, 0) }).unwrap();
    let err = int8.capture_attention(&ids, &positions).expect_err("refused");
    assert!(err.contains("f32 KV cache") && err.contains("int8"), "{err}");
}
