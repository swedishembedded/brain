// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving very large language
// models on the accelerators a customer owns, sized to the byte before
// anything is loaded, for its clients. If your team needs expertise in
// placing long-context decoders over one or several cards then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A Ripple directory (tiny dimensions, the real tensor layout) served through the residency adapter: a batch answers
//! as the same requests do one by one, a repeated prompt is answered from the prefix cache, the memory the scheduler
//! places is the one the engine allocates, the deployment is cut over the cards the selection offers, and a choice
//! Ripple cannot honour is refused by name.

use capability::Invocation;
use catalog::resident_ricursive::{RicursiveResident, RicursiveServeConfig};
use data::rng::Lcg;
use model::kv_tier::KvTier;
use model::policy::ServePolicy;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, Instance, InstanceKey, ResidentModel};
use ricursive::{manifest, RicursiveConfig};

const GIB: u64 = 1 << 30;
const ID: &str = "ricursiveAI/Ripple-tiny";

fn config() -> RicursiveConfig {
    RicursiveConfig::from_json(
        r#"{"model_type":"ricursive","vocab_size":264,"hidden_size":64,"intermediate_size":96,"num_hidden_layers":5,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":16,"sliding_window":8,"use_valres":true,"use_gated_attention":true,"ffn_clamp_threshold":20.0,"embed_scale":8.0,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]}"#,
    )
    .unwrap()
}

/// A pulled directory: config, random weights, a byte-level tokenizer that covers any ASCII prompt.
fn pulled_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-ricursive-served-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = config();
    let mut rng = Lcg::new(31);
    let order: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let tensors: Vec<brain_testutil::hfquant::Raw> = manifest(&cfg, &order)
        .into_iter()
        .map(|(name, dims)| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = if name.contains("norm") || name.contains("layernorm") {
                vec![1.0; n]
            } else if name.ends_with("value_res_lambda") {
                vec![0.0; n]
            } else {
                let fan_in = dims.last().copied().unwrap_or(1) as f32;
                (0..n).map(|_| rng.signed() * 1.7 / fan_in.sqrt()).collect()
            };
            brain_testutil::hfquant::f32_raw(&name, &dims, &data)
        })
        .collect();
    brain_testutil::hfquant::write_safetensors(&dir.join("model.safetensors"), &tensors);
    let config = serde_json::json!({
        "architectures": ["RicursiveForCausalLM"], "model_type": "ricursive", "vocab_size": 264, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 5,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "sliding_window": 8, "use_valres": true, "use_gated_attention": true, "ffn_clamp_threshold": 20.0,
        "embed_scale": 8.0, "max_position_embeddings": 512, "eos_token_id": [257],
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), serde_json::json!(i))).collect();
    let tokenizer = serde_json::json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}});
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    dir
}

fn ask(prompt: &str) -> Invocation {
    Invocation::new().set("prompt", serde_json::json!(prompt)).set("chat", serde_json::json!(false)).set("max_new", serde_json::json!(6)).set("temp", serde_json::json!(0.0)).set("seed", serde_json::json!(1))
}

fn text(result: capability::ActionResult) -> String {
    result.expect("the request is answered").outputs.get("text").and_then(|t| t.as_str().map(str::to_string)).expect("text output")
}

fn metric(instance: &dyn Instance, name: &str) -> u64 {
    instance.metrics().into_iter().find(|(k, _)| k == name).and_then(|(_, v)| v.as_u64()).unwrap_or_else(|| panic!("metric {name}"))
}

fn resident(dir: &std::path::Path, knobs: &RicursiveServeConfig, gpus: &[(u32, u64)]) -> Result<RicursiveResident, String> {
    RicursiveResident::from_hf_dir(dir.to_str().unwrap(), ID, knobs, gpus, 0)
}

#[test]
fn a_pulled_directory_is_served_batched_with_prefix_reuse() {
    let dir = pulled_dir("served");
    let knobs = RicursiveServeConfig { max_batch: Some(3), prefix_slots: Some(4), ..Default::default() };
    let resident = resident(&dir, &knobs, &[]).expect("a Ripple directory");
    let key = InstanceKey::new(ID, "default");
    assert_eq!(resident.manifest().max_context_tokens, Some(512), "the advertised context is the sized one: the model's limit on the host tier");
    assert_eq!(resident.batch_limit(), Some(3));
    let prompts = ["the vault code is 4817, remember it, and say it back", "a second and different conversation about sliding windows", "and a third one, shorter than both"];
    gpu_core::devices::with_host_tier(|| {
        let mut instance = resident.activate(&key, Device::Cpu).expect("the resident activates");
        let alone: Vec<String> = prompts.iter().map(|p| text(instance.run("generate", &ask(p), &mut |_| {}))).collect();
        assert!(alone.iter().all(|t| !t.is_empty()));
        let invs: Vec<Invocation> = prompts.iter().map(|p| ask(p)).collect();
        let together: Vec<String> = instance.run_batch("generate", &invs, &mut |_, _| {}).into_iter().map(text).collect();
        assert_eq!(together, alone, "a batch answers each request as it is answered alone");
        assert!(metric(&*instance, "kv_prefix_hit_tokens") >= prompts[0].len() as u64, "the repeated prompts were answered from the prefix cache");
    });
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_estimate_is_the_footprint_the_engine_allocates_and_the_context_follows_the_cards() {
    let dir = pulled_dir("sizing");
    let cfg = config();
    // One card that holds everything: the longest context the model allows, at the default bf16 cache.
    let roomy = resident(&dir, &RicursiveServeConfig::default(), &[(0, 80 * GIB)]).expect("places");
    assert_eq!(roomy.max_seq_len(), 512);
    let key = InstanceKey::new(ID, "default");
    let cost = roomy.estimate_multi(&key);
    assert_eq!(cost.devices().collect::<Vec<_>>(), [Device::Gpu(0)]);
    // A card too small for the asked context refuses it with the arithmetic rather than shortening it.
    let err = resident(&dir, &RicursiveServeConfig { ctx: Some(512), ..Default::default() }, &[(0, GIB / 4)]).err().expect("does not place on a quarter GiB");
    assert!(err.contains("512 tokens") && err.contains("GiB") && err.contains("--ctx"), "{err}");
    // The int8 cache costs less than bf16, which costs less than f32, at the same context.
    let bytes = |kv: KvTier| resident(&dir, &RicursiveServeConfig { kv_cache: Some(kv), ..Default::default() }, &[(0, 80 * GIB)]).unwrap().estimate(&key).vram;
    assert!(bytes(KvTier::INT8) < bytes(KvTier::BF16) && bytes(KvTier::BF16) < bytes(KvTier::F32));
    // Cards that each hold most of the deployment but not all of it share its layers (each stage pays its fixed bytes).
    let whole = resident(&dir, &RicursiveServeConfig::default(), &[(0, 80 * GIB)]).unwrap().estimate(&key).vram;
    let total_for = |usable: u64| (usable..).step_by(1 << 20).find(|&t| ricursive::placement::card_budget(t) >= usable).unwrap();
    let total = total_for(whole * 8 / 10);
    let split = resident(&dir, &RicursiveServeConfig::default(), &[(0, total), (1, total)]).expect("places over two cards");
    let devices: Vec<Device> = split.estimate_multi(&key).devices().collect();
    assert_eq!(devices, [Device::Gpu(0), Device::Gpu(1)], "one card cannot hold it, two can");
    let one = resident(&dir, &RicursiveServeConfig::default(), &[(0, total)]).err().expect("one of them does not hold it");
    assert!(one.contains("1 card(s)"), "{one}");
    assert_eq!(cfg.num_hidden_layers, 5);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_serving_policy_reaches_ripple_and_a_choice_it_cannot_honour_is_refused_by_name() {
    let mut p = ServePolicy::default();
    p.set("--weight-format", "int8").unwrap();
    p.set("--kv-cache", "fp8").unwrap();
    p.set("--ctx", "4096").unwrap();
    p.set("--max-batch", "6").unwrap();
    p.set("--sparse-attention", "ix.safetensors").unwrap();
    let mut cfg = RicursiveServeConfig::default();
    let notes = cfg.apply_policy(&p).expect("Ripple honours it");
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!((cfg.kv_cache.unwrap().as_str(), cfg.ctx, cfg.max_batch, cfg.sparse_attention.as_deref()), ("fp8", Some(4096), Some(6), Some("ix.safetensors")));

    let mut nf4 = ServePolicy::default();
    nf4.set("--weight-format", "nf4").unwrap();
    let mut cfg = RicursiveServeConfig::default();
    let err = cfg.apply_policy(&nf4).unwrap_err();
    assert!(err.contains("ricursive") && err.contains("nf4") && err.contains("supported"), "{err}");
    assert_eq!(cfg.refused.as_deref(), Some(err.as_str()));
    let dir = pulled_dir("refused");
    assert_eq!(resident(&dir, &cfg, &[]).err().as_deref(), Some(err.as_str()), "a refused policy builds no resident");

    let mut sliced = ServePolicy::default();
    sliced.set("--itl-target-ms", "40").unwrap();
    let notes = RicursiveServeConfig::default().apply_policy(&sliced).unwrap();
    assert!(notes.iter().any(|n| n.contains("itl-target-ms") && n.contains("ignored")), "{notes:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_directory_that_is_not_ripple_or_has_no_tokenizer_is_refused_by_name() {
    let dir = pulled_dir("broken");
    std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
    let err = resident(&dir, &RicursiveServeConfig::default(), &[]).err().expect("no tokenizer");
    assert!(err.contains("tokenizer.json"), "{err}");
    let dir = pulled_dir("foreign-indexer");
    let err = resident(&dir, &RicursiveServeConfig { sparse_attention: Some("/nonexistent/ix.safetensors".into()), ..Default::default() }, &[]).err().expect("no indexer file");
    assert!(err.contains("--sparse-attention"), "{err}");
}

/// An indexer file for the tiny model's global layer: `index_dim` 4, cluster blocks of 16 keys (one KV block), a window that covers one.
fn write_indexer(path: &std::path::Path, heads: usize) {
    write_indexer_blocks(path, heads, 16);
}

fn write_indexer_blocks(path: &std::path::Path, heads: usize, cluster_block: usize) {
    use model::lsa::LsaConfig;
    use model::lsa::indexer::{IndexGeometry, Indexer};
    let cfg = config();
    let geometry = IndexGeometry { n_layers: cfg.num_hidden_layers, n_heads: heads, n_kv_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, rope_base: 0.0 };
    let lsa = LsaConfig { index_dim: 4, pool_size: 1, bucket_size: 2, bucket_cap: 4, cluster_block, top_p: 3, top_k: 6, sink: 1, window: 16, early: None };
    let mut indexer = Indexer::random(lsa, geometry, 5).expect("random indexer");
    for (l, layer) in indexer.layers.iter_mut().enumerate() {
        if cfg.layer_type(l) != ricursive::LayerType::Full {
            *layer = None;
        }
    }
    indexer.save(&path.to_string_lossy()).expect("save indexer");
}

#[test]
fn an_indexer_for_another_model_is_refused_at_startup_and_a_fitting_one_is_read_through() {
    let dir = pulled_dir("sparse");
    // Indexer files stay out of the model directory: every `*.safetensors` file in it is a shard of the model.
    let ix_dir = dir.with_extension("ix");
    std::fs::create_dir_all(&ix_dir).unwrap();
    let foreign = ix_dir.join("foreign.safetensors");
    write_indexer(&foreign, config().num_attention_heads * 2);
    let err = resident(&dir, &RicursiveServeConfig { sparse_attention: Some(foreign.to_string_lossy().into_owned()), ..Default::default() }, &[]).err().expect("heads differ");
    assert!(err.contains("--sparse-attention"), "{err}");

    let fitting = ix_dir.join("fitting.safetensors");
    // A cluster block that is not a whole number of KV blocks cannot be read, and is said so at startup.
    let ragged = ix_dir.join("ragged.safetensors");
    write_indexer_blocks(&ragged, config().num_attention_heads, 8);
    let err = resident(&dir, &RicursiveServeConfig { sparse_attention: Some(ragged.to_string_lossy().into_owned()), ..Default::default() }, &[]).err().expect("cluster block 8 over KV blocks of 16");
    assert!(err.contains("cluster_block 8"), "{err}");
    write_indexer(&fitting, config().num_attention_heads);
    let knobs = RicursiveServeConfig { sparse_attention: Some(fitting.to_string_lossy().into_owned()), ..Default::default() };
    let sparse = resident(&dir, &knobs, &[]).expect("a fitting indexer is accepted");
    assert!(sparse.chosen_defaults().iter().any(|l| l == "kv-cache int8"), "sparse attention reads an int8 cache unless another is named: {:?}", sparse.chosen_defaults());
    let key = InstanceKey::new(ID, "default");
    let dense = resident(&dir, &RicursiveServeConfig::default(), &[]).unwrap();
    assert!(sparse.estimate(&key).vram > dense.estimate(&key).vram - dense.estimate(&key).vram / 2, "the index is charged");

    gpu_core::devices::with_host_tier(|| {
        let mut instance = match sparse.activate(&key, Device::Cpu) {
            Ok(instance) => instance,
            Err(e) if e.contains("workgroup barriers") => return brain_testutil::skip_unavailable("sparse attention: this backend does not run workgroup-barrier kernels (use BRAIN_DEVICE=software)"),
            Err(e) => panic!("the sparse resident activates: {e}"),
        };
        let prompt = "the vault code is 4817, remember it and say it back; ".repeat(8);
        let answer = |instance: &mut Box<dyn Instance>| text(instance.run("generate", &ask(&prompt).set("max_new", serde_json::json!(4)), &mut |_| {}));
        assert!(!answer(&mut instance).is_empty());
        assert!(metric(&*instance, "lsa_sparse_row_reads") > 0, "a prompt past the window is read through the index");
    });
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&ix_dir).ok();
}
