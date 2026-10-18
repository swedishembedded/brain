// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35::fp8_source::Qwen35Source`: the lazy, memory-bounded view of a
//! HuggingFace Qwen3.5/3.8 checkpoint directory that the LoRA/QLoRA trainer
//! loads its frozen base from.
//!
//! Spec, checked on a synthetic checkpoint that has the real layout (FP8
//! e4m3 block-scaled linears with bf16 `weight_scale_inv`, bf16 norms and
//! embeddings, f32 `A_log`, `layers-N.safetensors` + `outside.safetensors` +
//! `mtp.safetensors` + an index):
//!   * every tensor equals what the eager importers produce, bit for bit;
//!   * the chunked path yields the same values, in order, in bounded chunks;
//!   * host allocation while streaming the whole model stays near ONE tensor;
//!   * a checkpoint that does not match the config is refused by name;
//!   * a model built from it equals one built from the eager map.
//!
//! The `#[ignore]`d test at the end streams the real 27B checkpoint and
//! measures host RSS and load time:
//! `BRAIN_QWEN35_DIR=[path/to/Qwen3.8-27B-FP8] cargo test -p brain-qwen35 --test fp8_source -- --ignored --nocapture`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use checkpoint::mmap::MmapSafetensors;
use checkpoint::TensorSource;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::base_source::{lora_base, open_base, BaseDtype};
use qwen35::fp8_source::{hf_config, hf_fp8_block_size, Qwen35Source};
use qwen35::import::{import_dir, import_layer};

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
        let p = System.alloc(l);
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
        System.dealloc(p, l)
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

/// Peak live heap bytes allocated on this thread while `f` runs. Callers are
/// serialized: the counters are shared between test threads.
fn peak_live<R>(f: impl FnOnce() -> R) -> (R, usize) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    let r = f();
    MEASURING.with(|m| m.set(false));
    (r, PEAK.load(Ordering::Relaxed))
}

// ---- synthetic checkpoint in the real on-disk layout ----

const BLOCK: usize = 32;

fn fixture_cfg() -> Qwen35Config {
    // Big enough that one tensor is much smaller than the whole model, and
    // distinct non-square dims (rows not a multiple of BLOCK: ragged last band).
    Qwen35Config { d_model: 160, intermediate_size: 584, vocab: 1000, n_layers: 8, ..Qwen35Config::tiny() }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

struct Raw {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn write_st(path: &Path, tensors: &[Raw]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for t in tensors {
        header.insert(t.name.clone(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.bytes.len()]}));
        off += t.bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for t in tensors {
        out.extend(&t.bytes);
    }
    std::fs::write(path, out).unwrap();
}

/// brain-canonical suffix -> HF leaf (inverse of `import::classify`'s renames).
fn hf_leaf(suffix: &str) -> String {
    match suffix {
        "ln1.weight" => "input_layernorm.weight".into(),
        "ln2.weight" => "post_attention_layernorm.weight".into(),
        s => s.replace("mlp.gate.", "mlp.gate_proj.").replace("mlp.up.", "mlp.up_proj.").replace("mlp.down.", "mlp.down_proj."),
    }
}

/// A fresh scratch dir holding the checkpoint; `skip` drops one canonical
/// tensor (to test the refusal path).
fn write_checkpoint(cfg: &Qwen35Config, tag: &str, skip: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwen35_fp8_source_{tag}_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    // seed: any fixed bytes do; equality is against an independent reader of the same files
    let mut rng = Lcg(0x5eed);
    let d = cfg.d_model as usize;

    let mut linear_shapes: HashMap<String, (usize, usize)> = HashMap::new();
    for (l, ty) in cfg.layer_types().iter().enumerate() {
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            linear_shapes.insert(format!("blocks.{l}.{suffix}"), (n as usize, k as usize));
        }
    }
    let mut by_file: HashMap<String, Vec<Raw>> = HashMap::new();
    let mut weight_map = serde_json::Map::new();
    for (canon, numel) in cfg.param_list() {
        if skip == Some(canon.as_str()) {
            continue;
        }
        let (file, hf, shape) = if let Some(rest) = canon.strip_prefix("blocks.") {
            let (l, suffix) = rest.split_once('.').unwrap();
            let hf = format!("model.language_model.layers.{l}.{}", hf_leaf(suffix));
            let shape = match linear_shapes.get(&canon) {
                Some(&(n, k)) => vec![n, k],
                None if suffix == "linear_attn.conv1d.weight" => vec![cfg.linear_conv_dim() as usize, 1, cfg.linear_conv_kernel_dim as usize],
                None => vec![numel],
            };
            (format!("layers-{l}.safetensors"), hf, shape)
        } else {
            let (hf, shape) = match canon.as_str() {
                "tok.weight" => ("model.language_model.embed_tokens.weight", vec![cfg.vocab as usize, d]),
                "lm_head.weight" => ("lm_head.weight", vec![cfg.vocab as usize, d]),
                "norm.weight" => ("model.language_model.norm.weight", vec![d]),
                other => panic!("fixture does not know {other}"),
            };
            ("outside.safetensors".to_string(), hf.to_string(), shape)
        };
        // FP8-quantized in the real release: the big linears, except the GDN
        // in_proj_a/b gates (kept bf16, which also covers a mixed layer).
        let fp8 = linear_shapes.contains_key(&canon) && !canon.contains("in_proj_a") && !canon.contains("in_proj_b");
        let list = by_file.entry(file.clone()).or_default();
        weight_map.insert(hf.clone(), file.clone().into());
        if fp8 {
            let bytes: Vec<u8> = (0..numel).map(|_| match (rng.next() & 0xff) as u8 {
                b if b & 0x7f == 0x7f => 0x38, // e4m3fn NaN
                b => b,
            }).collect();
            let (rows, cols) = (shape[0], shape[1]);
            let (rb, cb) = model::fp8::scale_shape(rows, cols, BLOCK);
            let scales: Vec<u8> = (0..rb * cb).flat_map(|_| bf16(0.001 + (rng.next() % 1000) as f32 / 100_000.0)).collect();
            list.push(Raw { name: hf.clone(), dtype: "F8_E4M3", shape, bytes });
            let sname = format!("{hf}_scale_inv");
            weight_map.insert(sname.clone(), file.clone().into());
            list.push(Raw { name: sname, dtype: "BF16", shape: vec![rb, cb], bytes: scales });
        } else if canon.ends_with("A_log") {
            let bytes = (0..numel).flat_map(|_| rng.unit().to_le_bytes()).collect();
            list.push(Raw { name: hf, dtype: "F32", shape, bytes });
        } else {
            // bf16, including the norms (which the (1+w) fold must touch)
            let bytes = (0..numel).flat_map(|_| bf16(rng.unit() * 0.5)).collect();
            list.push(Raw { name: hf, dtype: "BF16", shape, bytes });
        }
    }
    // Real, present, out-of-scope tensors in their own shard.
    by_file.entry("mtp.safetensors".into()).or_default().push(Raw { name: "mtp.fc.weight".into(), dtype: "BF16", shape: vec![d, 2 * d], bytes: vec![0; d * 2 * d * 2] });
    weight_map.insert("mtp.fc.weight".into(), "mtp.safetensors".into());
    by_file.get_mut("outside.safetensors").unwrap().push(Raw { name: "model.visual.patch_embed.weight".into(), dtype: "BF16", shape: vec![4], bytes: vec![0; 8] });
    weight_map.insert("model.visual.patch_embed.weight".into(), "outside.safetensors".into());

    for (file, tensors) in &by_file {
        write_st(&dir.join(file), tensors);
    }
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&hf_config_json(cfg, BLOCK)).unwrap()).unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap()).unwrap();
    dir
}

/// The `config.json` of the multimodal release for `cfg`: the text model
/// nested under `text_config`, with HF's key names, plus the FP8 block size.
fn hf_config_json(cfg: &Qwen35Config, block: usize) -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [block, block]},
        "text_config": {
            "vocab_size": cfg.vocab, "num_hidden_layers": cfg.n_layers, "hidden_size": cfg.d_model,
            "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim,
            "rms_norm_eps": cfg.rms_eps, "max_position_embeddings": cfg.max_position_embeddings,
            "attention_bias": cfg.attn_bias, "full_attention_interval": cfg.full_attention_interval,
            "intermediate_size": cfg.intermediate_size, "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads, "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim, "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "tie_word_embeddings": cfg.tie_embeddings,
            "rope_parameters": {"mrope_section": cfg.mrope_section, "partial_rotary_factor": cfg.partial_rotary_factor, "rope_theta": cfg.rope_theta, "rope_type": "default"}
        }
    })
}

fn read_all(src: &Qwen35Source, name: &str) -> Vec<f32> {
    let mut out = None;
    assert!(src.with_tensor(name, &mut |d| out = Some(d.to_vec())), "{name} not served");
    out.unwrap()
}

#[test]
fn every_tensor_equals_the_eager_importers_bit_for_bit() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "equal", None);
    let src = Qwen35Source::open_dir(&dir, &cfg, BLOCK).unwrap();
    let eager = import_dir(&dir, &cfg, BLOCK).unwrap();

    let mut names: Vec<&String> = eager.keys().collect();
    names.sort();
    assert_eq!(src.names(), names.iter().map(|s| s.as_str()).collect::<Vec<_>>(), "served names differ from the eager import");
    for (name, want) in &eager {
        assert_eq!(&read_all(&src, name), want, "{name} differs from import_dir");
        assert_eq!(src.numel(name), Some(want.len()), "{name} numel");
    }
    // ... and the per-layer importer the existing trainer uses.
    for l in 0..cfg.n_layers as usize {
        let reader = MmapSafetensors::open(dir.join(format!("layers-{l}.safetensors"))).unwrap();
        for (name, want) in import_layer(&reader, &cfg, l, BLOCK).unwrap() {
            assert_eq!(read_all(&src, &name), want, "{name} differs from import_layer");
        }
    }
    assert!(!src.with_tensor("blocks.0.nope.weight", &mut |_| panic!("served an unknown tensor")));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chunked_reads_concatenate_to_the_same_values_in_bounded_chunks() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "chunks", None);
    let src = Qwen35Source::open_dir(&dir, &cfg, BLOCK).unwrap();
    let eager = import_dir(&dir, &cfg, BLOCK).unwrap();
    // The widest matrix in the fixture: its block band is the chunk floor.
    let widest = cfg.intermediate_size as usize;
    // fp8 (mlp), bf16 head and embedding, a folded norm.
    for name in ["blocks.1.mlp.gate.weight", "blocks.2.mlp.down.weight", "lm_head.weight", "tok.weight", "blocks.0.ln1.weight"] {
        for max_elems in [0usize, 1, 5000, 4 * BLOCK * widest + 3, 1 << 20] {
            let mut got = vec![f32::NAN; eager[name].len()];
            let (mut next, mut max_chunk) = (0u64, 0usize);
            assert!(src.with_tensor_chunks(name, max_elems, &mut |off, chunk| {
                assert_eq!(off, next, "{name}: chunks must be contiguous and in order");
                got[off as usize..off as usize + chunk.len()].copy_from_slice(chunk);
                next += chunk.len() as u64;
                max_chunk = max_chunk.max(chunk.len());
            }));
            assert_eq!(got, eager[name], "{name} max_elems={max_elems}");
            // One block-row band is the unit an FP8 matrix can be dequantized
            // in, so it is the floor of the bound.
            assert!(max_elems == 0 || max_chunk <= max_elems.max(BLOCK * widest), "{name} max_elems={max_elems}: chunk of {max_chunk}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn streaming_the_whole_model_keeps_host_allocation_near_one_tensor() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "peak", None);
    let src = Qwen35Source::open_dir(&dir, &cfg, BLOCK).unwrap();
    let total: usize = cfg.param_list().iter().map(|(_, n)| n * 4).sum();
    let largest: usize = cfg.param_list().iter().map(|(_, n)| n * 4).max().unwrap();

    let (sum, peak) = peak_live(|| {
        let mut sum = 0f64;
        for name in src.names() {
            let found = src.with_tensor(name, &mut |d| sum += d[0] as f64);
            assert!(found);
        }
        sum
    });
    assert!(sum.is_finite());
    assert!(total > 12 * largest, "fixture is too small to tell one tensor from the model ({total} vs {largest})");
    assert!(peak <= 2 * largest, "whole-tensor streaming peaked at {peak} B; one tensor is {largest} B, the model {total} B");

    let band = 8 * 1024;
    let (_, peak) = peak_live(|| {
        for name in ["lm_head.weight", "tok.weight", "blocks.1.mlp.gate.weight"] {
            src.with_tensor_chunks(name, band, &mut |_, c| assert!(c[0].is_finite()));
        }
    });
    assert!(peak <= 4 * (band + BLOCK * cfg.d_model as usize) * 4, "chunked streaming peaked at {peak} B for {band}-element chunks");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_checkpoint_that_disagrees_with_the_config_is_refused_by_name() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "missing", Some("blocks.3.mlp.up.weight"));
    let err = Qwen35Source::open_dir(&dir, &cfg, BLOCK).err().expect("a missing tensor must be refused");
    assert!(err.contains("missing blocks.3.mlp.up.weight"), "{err}");
    std::fs::remove_dir_all(&dir).ok();

    let dir = write_checkpoint(&cfg, "dims", None);
    let wrong = Qwen35Config { intermediate_size: cfg.intermediate_size + 1, ..cfg.clone() };
    let err = Qwen35Source::open_dir(&dir, &wrong, BLOCK).err().expect("a dimension mismatch must be refused");
    assert!(err.contains("blocks.0.mlp.gate.weight") && err.contains("elements"), "{err}");
    let err = Qwen35Source::open_dir(&dir, &cfg, 64).err().expect("a wrong FP8 block size must be refused");
    assert!(err.contains("_scale_inv") && err.contains("shape"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A multimodal wrapper whose own components (a vision tower, a projector)
/// sit beside the decoder in one checkpoint: their tensors are refused unless
/// the caller names their prefixes as owned elsewhere, and naming them never
/// loosens the decoder's own coverage.
#[test]
fn tensors_owned_by_another_component_are_refused_unless_declared_foreign() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "foreign", None);
    let index = dir.join("model.safetensors.index.json");
    let mut idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
    let extra = [("model.vision_tower.post_layernorm.weight", 4usize), ("model.merger.mlp.0.linear_1.bias", 6)];
    let raws: Vec<Raw> = extra.iter().map(|(n, k)| Raw { name: (*n).into(), dtype: "BF16", shape: vec![*k], bytes: vec![0; k * 2] }).collect();
    write_st(&dir.join("vision.safetensors"), &raws);
    for (n, _) in extra {
        idx["weight_map"][n] = serde_json::json!("vision.safetensors");
    }
    std::fs::write(&index, serde_json::to_vec(&idx).unwrap()).unwrap();

    let err = Qwen35Source::open_dir(&dir, &cfg, BLOCK).err().expect("an undeclared foreign tensor must be refused");
    assert!(err.contains("model.vision_tower.post_layernorm.weight") || err.contains("model.merger."), "{err}");
    let foreign = ["model.vision_tower.", "model.merger."];
    let src = Qwen35Source::open_dir_with(&dir, &cfg, BLOCK, &foreign).expect("declared foreign prefixes are skipped");
    assert!(src.names().iter().all(|n| !n.starts_with("model.")), "a foreign tensor leaked into the decoder's names");
    let narrow = Qwen35Config { intermediate_size: cfg.intermediate_size + 1, ..cfg.clone() };
    assert!(Qwen35Source::open_dir_with(&dir, &narrow, BLOCK, &foreign).is_err(), "declaring foreign prefixes must not relax decoder coverage");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_model_built_from_the_lazy_source_equals_one_built_from_the_eager_map() {
    use qwen35::model::Qwen35;
    use model::Shard;

    // The standard tiny config at fp8-compatible dims, small enough for the CPU backend.
    let cfg = Qwen35Config { vocab: 64, n_layers: 4, d_model: 96, intermediate_size: 112, ..Qwen35Config::tiny() };
    let dir = write_checkpoint(&cfg, "model", None);
    let src = Qwen35Source::open_dir(&dir, &cfg, BLOCK).unwrap();
    let eager = import_dir(&dir, &cfg, BLOCK).unwrap();
    let t = cfg.block_size;
    let shard = Shard { start: 0, end: cfg.n_layers as usize, embed: true, head: true, gpu_index: Shard::ANY_GPU };

    let lazy = Qwen35::new_fp32_shard_src(cfg.clone(), 1, t, &src, shard.clone());
    let want = Qwen35::new_fp32_shard_src(cfg.clone(), 1, t, &eager, shard);
    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let (a, b) = (lazy.logits_all(&tokens), want.logits_all(&tokens));
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "lazy-source model diverged from the eager-map model");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn hf_config_json_maps_onto_the_model_config() {
    let want = Qwen35Config::qwen38_27b();
    let dir = std::env::temp_dir().join(format!("qwen35_fp8_source_cfg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&hf_config_json(&want, 128)).unwrap()).unwrap();
    let got = hf_config(&dir).unwrap();
    let same = |c: &Qwen35Config| Qwen35Config { block_size: 0, ..c.clone() }.to_json();
    assert_eq!(same(&got), same(&want));
    assert_eq!(got.block_size, want.max_position_embeddings, "block_size is the model's full window until a trainer narrows it");
    assert_eq!(hf_fp8_block_size(&dir).unwrap(), 128);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn open_base_serves_an_hf_directory_and_a_brain_native_file_alike() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "base", None);
    let eager = import_dir(&dir, &cfg, BLOCK).unwrap();

    let hf = open_base(&dir).unwrap();
    assert_eq!(hf.cfg.to_json(), Qwen35Config { block_size: hf.cfg.block_size, ..cfg.clone() }.to_json());
    for name in ["blocks.1.mlp.up.weight", "blocks.0.ln1.weight", "tok.weight"] {
        assert!(hf.src.with_tensor(name, &mut |d| assert_eq!(d, &eager[name][..], "{name}")));
    }

    let native = dir.join("native.safetensors");
    let plan: Vec<(String, Vec<u64>)> = cfg.param_list().into_iter().map(|(n, k)| (n, vec![k as u64])).collect();
    let mut w = checkpoint::weightio::StWriter::create(native.to_str().unwrap(), &plan, &cfg.to_json(), None).unwrap();
    for (name, _) in &plan {
        w.write(name, &eager[name]).unwrap();
    }
    w.finish().unwrap();
    let nat = open_base(&native).unwrap();
    assert_eq!(nat.cfg.to_json(), cfg.to_json());
    assert!(nat.src.with_tensor("blocks.1.mlp.up.weight", &mut |d| assert_eq!(d, &eager["blocks.1.mlp.up.weight"][..])));

    let err = open_base(&dir.join("model.gguf")).err().expect("a GGUF must be refused with the way forward");
    assert!(err.contains("import-gguf"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lora_base_layers_zero_delta_adapters_over_the_streamed_base() {
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "lora", None);
    let eager = import_dir(&dir, &cfg, BLOCK).unwrap();
    // seed: any draw; the assertions are about structure (shapes, B == 0, base untouched)
    let (lcfg, src) = lora_base(&dir, 4, 8.0, 3, None).unwrap();
    let lora = lcfg.lora.as_ref().expect("lora_base must enable LoRA");
    assert_eq!((lora.rank, lora.alpha), (4, 8.0));
    for (name, numel) in lcfg.param_list() {
        assert_eq!(src.numel(&name), Some(numel), "{name}");
        if name.ends_with(".lora_b") {
            assert!(src.with_tensor(&name, &mut |d| assert!(d.iter().all(|&v| v == 0.0), "{name} must start at zero")));
        } else if name.ends_with(".lora_a") {
            assert!(src.with_tensor(&name, &mut |d| assert!(d.iter().any(|&v| v != 0.0), "{name} must be drawn")));
        } else {
            assert!(src.with_tensor(&name, &mut |d| assert_eq!(d, &eager[&name][..], "{name}")));
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn resuming_an_adapter_continues_its_factors_and_refuses_a_mismatched_one() {
    use checkpoint::st::{Adapter, ModelCard};
    let cfg = fixture_cfg();
    let dir = write_checkpoint(&cfg, "resume", None);
    let (rank, alpha) = (4u32, 8.0f32);
    // An adapter on the `gate` projections only, with recognisable factors.
    let mut card = ModelCard::new("resume-adapter", "qwen35");
    card.adapter = Some(Adapter { kind: "lora".into(), rank: Some(rank), base: None, alpha: Some(alpha), targets: Some(vec!["gate".into()]), dataset_id: None, per_target: None });
    let (d, ff, r) = (cfg.d_model as usize, cfg.intermediate_size as usize, rank as usize);
    let mut tensors = Vec::new();
    for l in 0..cfg.n_layers {
        let a: Vec<f32> = (0..r * d).map(|i| 0.001 * (i + l as usize) as f32).collect();
        let b: Vec<f32> = (0..ff * r).map(|i| -0.002 * (i + 1) as f32).collect();
        tensors.push((format!("blocks.{l}.mlp.gate.weight.lora_a"), vec![a.len() as u64], a));
        tensors.push((format!("blocks.{l}.mlp.gate.weight.lora_b"), vec![b.len() as u64], b));
    }
    let file = dir.join("adapter.safetensors");
    checkpoint::st::save_safetensors(file.to_str().unwrap(), &tensors, &serde_json::json!({}), Some(&card)).unwrap();

    // seed: any draw; resumed factors must come from the file, not the draw
    let (lcfg, src) = lora_base(&dir, rank, alpha, 11, Some(&file)).unwrap();
    assert_eq!(lcfg.lora.as_ref().unwrap().targets, vec!["gate".to_string()]);
    for (name, _, want) in &tensors {
        assert!(src.with_tensor(name, &mut |got| assert_eq!(got, &want[..], "{name} must continue the saved factors")));
    }
    assert!(src.with_tensor("blocks.1.mlp.up.weight", &mut |_| {}), "the base still streams through");

    let err = lora_base(&dir, rank + 1, alpha, 11, Some(&file)).err().expect("a rank mismatch must be refused");
    assert!(err.contains("rank"), "{err}");
    let not_adapter = dir.join("config.json");
    assert!(lora_base(&dir, rank, alpha, 11, Some(&not_adapter)).is_err(), "a non-adapter file must be refused");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn base_dtype_names_round_trip_and_refuse_unknowns() {
    for (s, d) in [("f32", BaseDtype::F32), ("bf16", BaseDtype::Bf16), ("nf4", BaseDtype::Nf4)] {
        assert_eq!(s.parse::<BaseDtype>(), Ok(d));
        assert_eq!(d.name(), s);
    }
    let err = "fp8".parse::<BaseDtype>().unwrap_err();
    assert!(err.contains("f32, bf16 or nf4"), "{err}");
}

/// Real-checkpoint gate: stream every tensor of the shipped 27B FP8 release
/// through the source the way the model builder consumes it (head and
/// embedding in row bands, everything else whole) and check the host RSS
/// high-water mark. Prints tensor count, volume and wall time.
#[test]
#[ignore = "needs the 27B FP8 checkpoint (BRAIN_QWEN35_DIR); reads ~30 GB"]
fn real_27b_checkpoint_streams_under_8_gib_host_rss() {
    let dir = PathBuf::from(std::env::var_os("BRAIN_QWEN35_DIR").expect("BRAIN_QWEN35_DIR must name the Qwen3.8-27B-FP8 checkpoint directory"));
    let cfg = hf_config(&dir).unwrap();
    let want = Qwen35Config { block_size: cfg.block_size, ..Qwen35Config::qwen38_27b() };
    assert_eq!(cfg.to_json(), want.to_json(), "hf_config disagrees with the built-in 27B shape");

    let t0 = std::time::Instant::now();
    let src = Qwen35Source::open_dir(&dir, &cfg, 128).unwrap();
    let mut bytes = 0u64;
    let mut peak_rss_kib = 0u64;
    let names: Vec<String> = src.names().iter().map(|s| s.to_string()).collect();
    for name in &names {
        // The head and embedding go in row bands, everything else whole - what the model builder does.
        if name == "tok.weight" || name == "lm_head.weight" {
            src.with_tensor_chunks(name, 8 << 20, &mut |_, c| bytes += c.len() as u64 * 4);
        } else {
            assert!(src.with_tensor(name, &mut |d| bytes += d.len() as u64 * 4));
        }
        src.advise_drop(name);
        peak_rss_kib = peak_rss_kib.max(rss_hwm_kib());
    }
    println!("real 27B: {} tensors, {:.1} GB f32 streamed in {:.1} s, VmHWM {:.2} GiB", names.len(), bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), peak_rss_kib as f64 / (1024.0 * 1024.0));
    assert!(peak_rss_kib < 8 * 1024 * 1024, "host RSS high-water mark {peak_rss_kib} KiB");
    // Spot-check against the per-layer importer on real bytes: a GDN layer, the
    // first and the last GQA layer.
    for l in [0usize, 3, 63] {
        let reader = MmapSafetensors::open(dir.join(format!("layers-{l}.safetensors"))).unwrap();
        for (name, want) in import_layer(&reader, &cfg, l, 128).unwrap() {
            assert!(src.with_tensor(&name, &mut |d| assert!(d == &want[..], "{name} differs from import_layer")));
        }
    }
    assert!(cfg.layer_types().contains(&LayerType::Full));
}

fn rss_hwm_kib() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    s.lines().find_map(|l| l.strip_prefix("VmHWM:")).and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok()).expect("VmHWM")
}
