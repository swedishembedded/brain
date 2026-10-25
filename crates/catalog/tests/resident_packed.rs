// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded serving of quantized models for its clients. If your team needs
// expertise in serving bf16 and int8 checkpoints without expanding them to full precision then you can procure
// our services by sending an email to info@swedishembedded.com.

//! INVARIANT M for the Qwen3.8 serving residents: a brain-native checkpoint stored in f32, bf16 or packed int8 is
//! served through the catalog's engine constructor ([`catalog::resident_qwen35::serving_engine`],
//! [`catalog::resident_qwen35moe::serving_engine`]) as the same function the whole-map engine computes from the
//! decoded weights, on the host tier a residency fallback builds on, and the host never holds the weights as a
//! whole-model f32 map: the heap high-water of the build stays far under the model's f32 size, where the
//! whole-container load the residents used to do reaches it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use gpu_core::select::Dtype;
use model::paged::BlockTable;

mod common;
use common::packed_fixture::{decoded, write, Stored};

// ---- counting allocator: peak live heap bytes of the measuring thread ----

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}
fn measuring() -> bool {
    MEASURING.try_with(|m| m.get()).unwrap_or(false)
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() && measuring() {
            let live = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if measuring() {
            let _ = LIVE.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(l.size())));
        }
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

/// `f`'s value, the peak live heap bytes allocated on this thread while it ran, and the bytes still live when it
/// returned. Callers are serialized: the counters are shared.
fn peak_live<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    let r = f();
    MEASURING.with(|m| m.set(false));
    (r, PEAK.load(Ordering::Relaxed), LIVE.load(Ordering::Relaxed))
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-resident-packed-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---- dense Qwen3.8 ----

mod dense {
    use super::*;
    use qwen35::config::Qwen35Config;
    use qwen35::serve::Engine;

    /// An int8-legal shape wide enough that the model's f32 size dwarfs a staging tile.
    pub fn cfg() -> Qwen35Config {
        Qwen35Config { vocab: 512, d_model: 256, intermediate_size: 1024, n_layers: 8, ..Qwen35Config::tiny_i8() }
    }

    pub fn linears(cfg: &Qwen35Config) -> HashMap<String, (usize, usize)> {
        let mut shapes = HashMap::new();
        for (l, ty) in cfg.layer_types().iter().enumerate() {
            for (suffix, n, k) in cfg.layer_leaves(*ty) {
                shapes.insert(format!("blocks.{l}.{suffix}"), (n as usize, k as usize));
            }
        }
        shapes
    }

    /// The old route: the whole container as an f32 map, `weights` given to an f32 engine.
    pub fn reference(cfg: &Qwen35Config, weights: &HashMap<String, Vec<f32>>, ctx: u32, concurrent: u32) -> Engine {
        Engine::from_source_tier(cfg.clone(), weights, ctx, concurrent, model::kv_tier::KvTier::F32, &model::ops::TierPolicy::uniform(Dtype::F32))
    }

    /// Hidden state after a prompt and the logits of the next three decode steps.
    pub fn run(engine: &mut Engine, prompt: &[u32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut table = BlockTable::new();
        let hidden = engine.prefill(&mut table, prompt);
        let steps = [3u32, 9, 5].iter().map(|&t| engine.forward_batched_logits(&mut [&mut table], &[t])).collect();
        (hidden, steps)
    }
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Spec: the engine the catalog builds from an f32, bf16 or packed-int8 checkpoint computes the function of the
/// weights that checkpoint decodes to - the whole-map engine's logits - on the host tier.
#[test]
fn a_dense_checkpoint_serves_the_function_it_decodes_to_in_every_stored_encoding() {
    let cfg = dense::cfg();
    let weights = qwen35::init::init_weights(&cfg, 41);
    let linears = dense::linears(&cfg);
    let prompt: Vec<u32> = (0..9).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let dir = scratch("dense-logits");
    for stored in [Stored::F32, Stored::Bf16, Stored::Int8] {
        let path = dir.join(format!("{stored:?}.safetensors"));
        write(&path, &cfg.to_json(), &weights, &linears, stored);
        let want_weights = decoded(&weights, &linears, stored);
        let (want, got) = gpu_core::devices::with_host_tier(|| {
            let mut reference = dense::reference(&cfg, &want_weights, 24, 2);
            let mut served = catalog::resident_qwen35::serving_engine(path.to_str().unwrap(), 24, 2).expect("the checkpoint is served");
            (dense::run(&mut reference, &prompt), dense::run(&mut served, &prompt))
        });
        let err = max_err(&got.0, &want.0);
        assert!(err < 1e-4, "{stored:?}: prefill differs from the decoded-weights engine by {err:e}");
        for (g, w) in got.1.iter().zip(&want.1) {
            let err = max_err(g, w);
            assert!(err < 1e-4, "{stored:?}: decode logits differ from the decoded-weights engine by {err:e}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Spec: a bf16 checkpoint stays bf16 in the engine (a linear is not widened to f32), and building it never holds
/// the weights as a whole-model f32 map: on the host tier, where the engine's own buffers are heap, what the build
/// holds beyond the engine it returns is a staging tile and one tensor, a small fraction of the model's f32 size,
/// while the whole-container load the resident used to do holds all of it.
#[test]
fn a_bf16_checkpoint_is_served_without_a_whole_model_f32_copy() {
    let cfg = dense::cfg();
    let weights = qwen35::init::init_weights(&cfg, 43);
    let linears = dense::linears(&cfg);
    let dir = scratch("dense-peak");
    let path = dir.join("bf16.safetensors");
    write(&path, &cfg.to_json(), &weights, &linears, Stored::Bf16);
    let f32_bytes: usize = weights.values().map(|w| w.len() * 4).sum();
    let path = path.to_str().unwrap();

    gpu_core::devices::with_host_tier(|| {
        let (engine, peak, resident) = peak_live(|| catalog::resident_qwen35::serving_engine(path, 24, 2).expect("the checkpoint is served"));
        assert_eq!(engine.weight_tier("blocks.0.mlp.gate.weight"), Some(Dtype::BF16), "a bf16 linear must stay bf16");
        let transient = peak - resident;
        println!("bf16 serving build: model {f32_bytes} B as f32, engine kept {resident} B, host heap beyond the engine {transient} B (peak {peak} B)");
        assert!(transient < f32_bytes / 4, "building from the bf16 checkpoint held {transient} B beyond the engine it kept; the whole model is {f32_bytes} B as f32");
        assert!(resident < f32_bytes * 3 / 4, "the engine kept {resident} B for a model of {f32_bytes} B as f32: the bf16 linears were widened");
        // The measurement can tell: the whole-container load this replaces holds every tensor as f32 at once.
        let (_, old_peak, _) = peak_live(|| checkpoint::load(path).into_by_role(""));
        println!("whole-container load of the same file: peak {old_peak} B");
        assert!(old_peak >= f32_bytes, "the whole-container load peaked at {old_peak} B, under the model's {f32_bytes} B");
    });
    std::fs::remove_dir_all(&dir).ok();
}

/// Spec: on a device with a packed-int8 path the int8 checkpoint stays int8 in the engine. (The host tier widens an
/// int8 linear to f32 because it has no int8 dot; the equality above covers that case.)
#[test]
fn an_int8_checkpoint_stays_int8_where_the_device_has_the_path() {
    let cfg = dense::cfg();
    let weights = qwen35::init::init_weights(&cfg, 47);
    let dir = scratch("dense-int8");
    let path = dir.join("int8.safetensors");
    write(&path, &cfg.to_json(), &weights, &dense::linears(&cfg), Stored::Int8);
    let engine = catalog::resident_qwen35::serving_engine(path.to_str().unwrap(), 24, 2).expect("the checkpoint is served");
    let tier = engine.weight_tier("blocks.0.mlp.gate.weight");
    if gpu_core::Gpu::new(qwen35::model::pipelines()).caps().numeric.int8_dot {
        assert_eq!(tier, Some(Dtype::I8), "an int8 linear must stay int8 on a device with the int8 path");
    } else {
        brain_testutil::skip_unavailable("this device has no packed-int8 path");
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ---- Qwen3.5 mixture of experts ----

/// Spec: the MoE engine the catalog builds from a bf16 checkpoint computes the function of the bf16 weights - the
/// whole-map engine's hidden state and greedy tokens - and reads the routed experts one tensor at a time.
#[test]
fn a_moe_checkpoint_serves_the_function_it_decodes_to() {
    use qwen35moe::config::Qwen35Config;
    use model::serve::PagedDecoder;
    use qwen35moe::serve::Engine;

    let cfg = Qwen35Config::tiny();
    let weights = qwen35moe::init::init_weights(&cfg, 29);
    let (d, ff, sff) = (cfg.d_model as usize, cfg.moe_intermediate_size as usize, cfg.shared_expert_intermediate_size as usize);
    let mut linears = HashMap::new();
    for name in weights.keys() {
        let shape = if name.contains(".mlp.experts.") {
            Some(if name.ends_with(".down.weight") { (d, ff) } else { (ff, d) })
        } else if name.contains(".mlp.shared_expert.") {
            Some(if name.ends_with(".down.weight") { (d, sff) } else { (sff, d) })
        } else {
            None
        };
        if let Some(shape) = shape {
            assert_eq!(weights[name].len(), shape.0 * shape.1, "{name}");
            linears.insert(name.clone(), shape);
        }
    }
    assert!(!linears.is_empty());
    let dir = scratch("moe");
    let path = dir.join("bf16.safetensors");
    write(&path, &cfg.to_json(), &weights, &linears, Stored::Bf16);
    let want_weights = decoded(&weights, &linears, Stored::Bf16);
    let prompt = [2u32, 7, 4, 11];

    let run = |engine: &mut Engine| {
        let mut table = BlockTable::new();
        let hidden = engine.prefill(&mut table, &prompt);
        let mut tok = engine.admit_greedy(&hidden);
        let mut toks = vec![tok];
        for _ in 0..3 {
            tok = engine.forward_batched_greedy(&mut [&mut table], &[tok]).pop().expect("one row");
            toks.push(tok);
        }
        (hidden, toks)
    };
    let (want, got) = gpu_core::devices::with_host_tier(|| {
        let mut reference = Engine::from_map(cfg.clone(), &want_weights, 16, 2);
        let mut served = catalog::resident_qwen35moe::serving_engine(path.to_str().unwrap(), 16, 2).expect("the checkpoint is served");
        (run(&mut reference), run(&mut served))
    });
    let err = max_err(&got.0, &want.0);
    assert!(err < 1e-4, "prefill differs from the decoded-weights engine by {err:e}");
    assert_eq!(got.1, want.1, "greedy tokens differ from the decoded-weights engine");
    std::fs::remove_dir_all(&dir).ok();
}

// ---- residency demotion ----

/// A byte-level tokenizer file over ids `0..256`, so any ASCII prompt encodes inside the model's vocabulary.
fn write_byte_tokenizer(dir: &std::path::Path) -> std::path::PathBuf {
    let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), serde_json::json!(i))).collect();
    let tokenizer = serde_json::json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}});
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    path
}

fn bytes_live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Spec: a resident that is demoted keeps the checkpoint's packed bytes and nothing else on the host, and a
/// promotion brings back the model it was - same text, same footprint. The warm and cold estimates charge no
/// device and no heap, only the mapping of the file.
#[test]
fn a_demoted_resident_holds_no_engine_and_is_promoted_to_the_same_model() {
    use catalog::resident_qwen35::{Qwen35Resident, Qwen35Sizing};
    use capability::Invocation;
    use residency::{Device, InstanceKey, ResidentModel, Tier};

    let cfg = dense::cfg();
    let weights = qwen35::init::init_weights(&cfg, 53);
    let dir = scratch("demote");
    let path = dir.join("model.safetensors");
    write(&path, &cfg.to_json(), &weights, &dense::linears(&cfg), Stored::Bf16);
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    let tokenizer = write_byte_tokenizer(&dir);
    let card = checkpoint::st::ModelCard::new("brain/qwen35", "qwen35");
    let resident = Qwen35Resident::from_card_sized(path.to_str().unwrap(), &card, Some(tokenizer.to_str().unwrap()), Qwen35Sizing { ctx: 48, max_batch: 2, exact_decode: false, refused: false }).expect("a resident for the checkpoint");
    let key = InstanceKey::new("brain/qwen35", "default");

    for tier in [Tier::Warm, Tier::Cold] {
        let cost = resident.estimate_at(&key, tier);
        assert_eq!((cost.vram, cost.ram, cost.mapped), (0, 0, file_bytes), "{tier:?}: only the mapping of the packed file");
    }
    assert!(resident.estimate_at(&key, Tier::Hot).vram >= file_bytes, "the hot estimate still covers the weights");

    let ask = |instance: &mut dyn residency::Instance| -> Result<String, String> {
        let inv = Invocation::new().set("prompt", serde_json::json!("hello")).set("chat", serde_json::json!(false)).set("max_new", serde_json::json!(6)).set("temp", serde_json::json!(0.0)).set("seed", serde_json::json!(1));
        instance.run_batch("generate", &[inv], &mut |_, _| {}).pop().expect("one result").map(|o| o.outputs.get("text").and_then(|t| t.as_str().map(str::to_string)).expect("text output"))
    };
    gpu_core::devices::with_host_tier(|| {
        let ((hot_text, hot_bytes, warm_bytes, parked_err, promoted_text, promoted_bytes), _, _) = peak_live(|| {
            let mut instance = resident.activate(&key, Device::Cpu).expect("the resident activates");
            let hot_text = ask(&mut *instance).expect("a hot instance generates");
            let hot_bytes = bytes_live();
            instance.demote(Tier::Warm).expect("the resident opts into demotion");
            let warm_bytes = bytes_live();
            let parked_err = ask(&mut *instance).expect_err("a demoted instance does not run");
            instance.promote(Device::Cpu).expect("a demoted resident promotes");
            let promoted_text = ask(&mut *instance).expect("a promoted instance generates");
            let promoted_bytes = bytes_live();
            // Cold also gives the file pages back; the mapping still promotes to the same model.
            instance.demote(Tier::Cold).expect("the resident demotes to cold");
            instance.promote(Device::Cpu).expect("a cold resident promotes");
            assert_eq!(ask(&mut *instance).expect("a promoted cold instance generates"), hot_text, "a cold round trip must generate what the hot instance did");
            (hot_text, hot_bytes, warm_bytes, parked_err, promoted_text, promoted_bytes)
        });
        assert!(warm_bytes < hot_bytes / 4, "a warm instance holds {warm_bytes} B of heap against {hot_bytes} B hot: the engine was not released");
        assert!(parked_err.contains("demoted"), "{parked_err}");
        assert_eq!(promoted_text, hot_text, "the promoted model must generate what the hot one did");
        assert!(promoted_bytes > hot_bytes / 2 && promoted_bytes < hot_bytes * 2, "promotion rebuilt {promoted_bytes} B against {hot_bytes} B before");
    });
    std::fs::remove_dir_all(&dir).ok();
}
