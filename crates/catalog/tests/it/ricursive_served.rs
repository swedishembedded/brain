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
    pulled_dir_with(tag, None)
}

/// [`pulled_dir`] whose greedy decode walks the cycle `a b c d a b c ...` whatever the prompt, when `cycle` is set: each
/// of the four tokens has a random unit code for its embedding, and the output row of its successor is that code scaled
/// up, so the successor wins the argmax by a wide margin while the layers (with small gains on their branches) still run.
fn pulled_dir_with(tag: &str, cycle: Option<[u8; 4]>) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-ricursive-served-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = config();
    let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
    let mut rng = Lcg::new(31);
    let codes: Vec<Vec<f32>> = (0..vocab)
        .map(|_| {
            let v: Vec<f32> = (0..hidden).map(|_| rng.signed()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect();
    let order: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let tensors: Vec<brain_testutil::hfquant::Raw> = manifest(&cfg, &order)
        .into_iter()
        .map(|(name, dims)| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = if cycle.is_some() && name == "model.embed_tokens.weight" {
                codes.iter().flatten().copied().collect()
            } else if let (Some(ring), "lm_head.weight") = (cycle, name.as_str()) {
                let mut head = vec![0.0f32; n];
                for (i, &cur) in ring.iter().enumerate() {
                    let next = ring[(i + 1) % ring.len()] as usize;
                    head[next * hidden..(next + 1) * hidden].iter_mut().zip(&codes[cur as usize]).for_each(|(h, c)| *h = c * 40.0);
                }
                head
            } else if name.contains("post_attention_layernorm") || name.contains("post_feedforward_layernorm") {
                vec![if cycle.is_some() { 0.05 } else { 1.0 }; n]
            } else if name.contains("norm") {
                vec![1.0; n]
            } else if name.ends_with("value_res_lambda") {
                vec![0.0; n]
            } else {
                let fan_in = dims.last().copied().unwrap_or(1) as f32;
                (0..n).map(|_| rng.signed() * if cycle.is_some() { 0.5 } else { 1.7 } / fan_in.sqrt()).collect()
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

#[test]
fn prompt_lookup_speculation_returns_the_plain_greedy_tokens_and_reports_what_it_proposed() {
    use qwen35::int8_gguf_resident::SpeculativeMode;
    let dir = pulled_dir_with("speculative", Some(*b"abcd"));
    let plain = resident(&dir, &RicursiveServeConfig::default(), &[]).unwrap();
    let speculating = resident(&dir, &RicursiveServeConfig { speculative: Some(SpeculativeMode::Ngram), ..Default::default() }, &[]).unwrap();
    let key = InstanceKey::new(ID, "default");
    // The model continues the cycle its prompts are written in, so the lookup drafter's proposals are right.
    let prompts = ["abcdabcdabcdabcdabcd", "xabcdabcdabcdabcdabcdabcd", "bcdabcdabcdabcdabcdabcdabcdabcdabcd"];
    let run = |resident: &RicursiveResident| {
        gpu_core::devices::with_host_tier(|| {
            let mut instance = resident.activate(&key, Device::Cpu).expect("activates");
            let invs: Vec<Invocation> = prompts.iter().map(|p| ask(p).set("max_new", serde_json::json!(24))).collect();
            let outputs: Vec<String> = instance.run_batch("generate", &invs, &mut |_, _| {}).into_iter().map(text).collect();
            let (proposed, accepted, steps) = (metric(&*instance, "spec_proposed"), metric(&*instance, "spec_accepted"), metric(&*instance, "spec_verify_steps"));
            eprintln!("speculation: {proposed} proposed, {accepted} accepted over {steps} verify steps");
            assert!(proposed == 0 || accepted > 0, "right proposals are accepted");
            (outputs, proposed, steps)
        })
    };
    let (want, none, _) = run(&plain);
    let (got, proposed, steps) = run(&speculating);
    assert_eq!(none, 0, "plain decoding proposes nothing");
    assert_eq!(got, want, "speculation never changes greedy output");
    assert!(proposed > 0 && steps > 0, "the drafter proposed and the engine verified: {proposed} proposals over {steps} steps");
    assert!(want.iter().all(|t| t.starts_with(['a', 'b', 'c', 'd'])) && want.iter().all(|t| t.chars().count() > 8), "the model walks the cycle: {want:?}");

    // What the drafters of the release would need is refused by name, and the lookup drafter beside sparse attention too.
    for (mode, needle) in [(SpeculativeMode::Mtp, "multi-token-prediction"), (SpeculativeMode::Dflash, "hidden states")] {
        let err = resident(&dir, &RicursiveServeConfig { speculative: Some(mode), ..Default::default() }, &[]).err().expect("refused");
        assert!(err.contains(needle), "{err}");
    }
    let err = resident(&dir, &RicursiveServeConfig { speculative: Some(SpeculativeMode::Ngram), sparse_attention: Some("ix.safetensors".into()), ..Default::default() }, &[]).err().expect("refused");
    assert!(err.contains("--sparse-attention"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A trained adapter file over the pulled directory `dir`, written by the trainer itself: random factors on every
/// linear (a zero B would leave the base's answers alone), and the model that trained it.
fn trained_adapter(dir: &std::path::Path, path: &std::path::Path, card: &str, seed: u64) -> ricursive::Ripple {
    use model::adapter::device::LoraCfg;
    let cfg = config();
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let base = ricursive::open_safetensors(dir.join("model.safetensors").to_str().unwrap(), &cfg, &layers).unwrap();
    let train = ricursive::TrainConfig::new(cfg, 64).with_lora(LoraCfg::new(4, 8.0, ricursive::train::default_targets()));
    let init = ricursive::train::AdapterInit::fresh(&train, &base, 5); // seed: the factors are overwritten just below
    let model = ricursive::Ripple::build(train, 1, 64, &init, &ricursive::TrainOpts::default(), true).unwrap();
    model::conformance_grad::randomize_adapters(&model, seed);
    model.save_adapter(path.to_str().unwrap(), card, ID, None, None).unwrap();
    model
}

/// The greedy continuation of `prompt` by the training model's own forward, as the served text.
fn trained_text(model: &ricursive::Ripple, dir: &std::path::Path, prompt: &str, new: usize) -> String {
    use data::Tokenizer;
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let mut ids = tok.encode(prompt);
    let (start, vocab) = (ids.len(), model.model_config().vocab_size);
    use model::Model;
    for _ in 0..new {
        let logits = model.logits_all(&ids).expect("the training model answers");
        let last = &logits[(ids.len() - 1) * vocab..ids.len() * vocab];
        let best = last.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
        ids.push(best as u32);
    }
    tok.decode(&ids[start..])
}

const A: &str = "ricursiveAI/Ripple-tiny:lab:alpha:v1";
const B: &str = "ricursiveAI/Ripple-tiny:lab:beta:v1";

fn banked(dir: &std::path::Path) -> std::sync::Arc<RicursiveResident> {
    let knobs = RicursiveServeConfig { adapter_bank: Some(catalog::adapter_registry::AdapterBankSettings { slots: 2, rank_cap: 4, targets: catalog::adapter_registry::AdapterTargets::ALL }), max_batch: Some(4), ..Default::default() };
    std::sync::Arc::new(resident(dir, &knobs, &[]).expect("a Ripple directory"))
}

fn on(inv: Invocation, id: &str) -> Invocation {
    inv.set(catalog::adapter_registry::ADAPTER_PARAM, serde_json::json!(id))
}

/// Spec: two trained adapters and the base are served from ONE resident instance: a batch holding a request for each
/// answers every one as it is answered alone, and the adapters' answers are the training model's greedy forward, token for
/// token, and differ from the base's.
#[test]
fn two_adapters_and_the_base_are_served_from_one_instance_as_the_training_forward_does() {
    use catalog::adapter_registry::AdapterHost;
    let dir = pulled_dir("adapters");
    let (fa, fb) = (dir.with_extension("a.safetensors"), dir.with_extension("b.safetensors"));
    let (train_a, train_b) = (trained_adapter(&dir, &fa, A, 9), trained_adapter(&dir, &fb, B, 10));
    let base = banked(&dir);
    let (va, vb) = (base.adapter_view(catalog::adapter_release::verify_adapter(&fa, &base.served_base()).unwrap()).unwrap(), base.adapter_view(catalog::adapter_release::verify_adapter(&fb, &base.served_base()).unwrap()).unwrap());
    assert_eq!((va.id(), vb.id()), (A, B));
    let prompt = "the vault code is 4817";
    gpu_core::devices::with_host_tier(|| {
        let mut one = base.activate(&InstanceKey::new(ID, "default"), Device::Cpu).expect("the base activates once");
        let alone: Vec<String> = [None, Some(A), Some(B)].iter().map(|id| text(one.run("generate", &id.map_or(ask(prompt), |id| on(ask(prompt), id)), &mut |_| {}))).collect();
        let together: Vec<String> = one.run_batch("generate", &[ask(prompt), on(ask(prompt), A), on(ask(prompt), B)], &mut |_, _| {}).into_iter().map(text).collect();
        assert_eq!(together, alone, "a batch answers each request as it is answered alone");
        assert_eq!(alone[1], trained_text(&train_a, &dir, prompt, 6), "adapter a answers what the training forward does");
        assert_eq!(alone[2], trained_text(&train_b, &dir, prompt, 6), "adapter b answers what the training forward does");
        assert!(alone[0] != alone[1] && alone[1] != alone[2], "the adapters changed the answers: {alone:?}");
        // An adapter that was never offered is an unknown model, not the base's answer.
        assert!(one.run("generate", &on(ask(prompt), "ricursiveAI/Ripple-tiny:lab:gamma:v1"), &mut |_| {}).is_err());
    });
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_file(&fa).ok();
    std::fs::remove_file(&fb).ok();
}

/// Spec: an adapter that does not fit the base is refused by name before anything is served: another architecture's
/// card, a layer the model does not have, a linear it does not have, rank 0, a base it was not trained on, a rank above the
/// bank's.
#[test]
fn an_adapter_that_does_not_fit_the_base_is_refused_by_name() {
    use catalog::adapter_registry::AdapterHost;
    let dir = pulled_dir("adapter-refused");
    let file = dir.with_extension("adapter.safetensors");
    trained_adapter(&dir, &file, A, 9);
    let base = banked(&dir);

    let other = catalog::adapter_release::ServedBase::of_family("ricursive", dir.clone(), Some("someone/else".into()));
    let err = catalog::adapter_release::verify_adapter(&file, &other).err().expect("a different base");
    assert!(err.contains("refusing to put it on a different base"), "{err}");
    let qwen = catalog::adapter_release::ServedBase::of_family("qwen", dir.clone(), Some(ID.into()));
    let err = catalog::adapter_release::verify_adapter(&file, &qwen).err().expect("a ricursive adapter on a Qwen base");
    assert!(err.contains("ricursive") && err.contains("qwen"), "{err}");

    let fresh = || catalog::adapter_release::verify_adapter(&file, &base.served_base()).unwrap();
    let mut deeper = fresh();
    deeper.adapter.sites[0].base = "model.layers.9.self_attn.q_proj.weight".into();
    assert!(base.adapter_view(deeper).err().expect("a layer past the model's depth").contains("model.layers.9"));
    let mut bogus = fresh();
    bogus.adapter.sites[0].base = "model.layers.0.self_attn.zz_proj.weight".into();
    assert!(base.adapter_view(bogus).err().expect("a linear the model does not have").contains("zz_proj"));
    let mut rankless = fresh();
    rankless.adapter.rank = 0;
    assert!(base.adapter_view(rankless).err().expect("rank 0").contains("rank"));
    let narrow = std::sync::Arc::new(resident(&dir, &RicursiveServeConfig { adapter_bank: Some(catalog::adapter_registry::AdapterBankSettings { slots: 1, rank_cap: 2, targets: catalog::adapter_registry::AdapterTargets::ALL }), ..Default::default() }, &[]).unwrap());
    let err = narrow.adapter_view(catalog::adapter_release::verify_adapter(&file, &narrow.served_base()).unwrap()).err().expect("above the bank's rank cap");
    assert!(err.contains("--adapter-rank-cap"), "{err}");
    // A server started without an adapter bank offers no adapter at all.
    let bare = std::sync::Arc::new(resident(&dir, &RicursiveServeConfig::default(), &[]).unwrap());
    let loaded = catalog::adapter_release::verify_adapter(&file, &bare.served_base()).unwrap();
    assert!(bare.adapter_view(loaded).err().expect("no bank").contains("--adapter-slots"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_file(&file).ok();
}

/// Spec: the bank is part of the placement: the footprint grows by the bank's bytes, which are the bytes the engine
/// allocates.
#[test]
fn the_adapter_bank_is_charged_in_the_placement() {
    let dir = pulled_dir("adapter-bytes");
    let key = InstanceKey::new(ID, "default");
    let plain = resident(&dir, &RicursiveServeConfig::default(), &[]).unwrap().estimate(&key).vram;
    let with = banked(&dir).estimate(&key).vram;
    let bank = model::adapter::bank::AdapterBankConfig::new(2, 4, &["q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "gate_proj", "up_proj", "down_proj"]);
    let wanted = bank.device_bytes(&ricursive::TrainConfig::new(config(), 1), 256);
    assert_eq!(with - plain, wanted, "the placement charges the bank's bytes");
    std::fs::remove_dir_all(&dir).ok();
}

fn executor(models: Vec<std::sync::Arc<dyn ResidentModel>>) -> residency::Executor {
    let mut budgets = residency::budget::Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    residency::Executor::start(models, budgets, residency::Policy::default())
}

fn say(exec: &residency::Executor, model: &str) -> Result<String, String> {
    exec.run_blocking(model, "generate", ask("the vault code is 4817"), |_| {}).map(|o| o.outputs["text"].as_str().expect("text output").to_string())
}

/// Spec: adapters come and go while the server runs (the data plane under `--adapter-manifest` and `--watch-adapters`):
/// one offered after the instance was built answers at once beside the base, one withdrawn is an unknown model and
/// leaves the engine's bank, and a new release under a card id replaces the one served under it. Ripple's resident
/// is driven by the same `LiveAdapters` the Qwen residents are.
#[test]
fn adapters_are_offered_replaced_and_withdrawn_on_a_running_server() {
    use catalog::adapter_registry::{AdapterHost, LiveAdapters};
    let dir = pulled_dir("adapters-live");
    let files: Vec<std::path::PathBuf> = (0..3).map(|i| dir.with_extension(format!("{i}.safetensors"))).collect();
    trained_adapter(&dir, &files[0], A, 9);
    trained_adapter(&dir, &files[1], B, 10);
    trained_adapter(&dir, &files[2], A, 11); // a second release under the card id of the first
    let base = banked(&dir);
    let exec = executor(vec![base.clone()]);
    let live = LiveAdapters::new(base.clone(), exec.clone());
    let offer = |file: &std::path::Path| live.serve(catalog::adapter_release::verify_adapter(file, &base.served_base()).unwrap()).expect("the adapter is offered");

    let plain = say(&exec, ID).expect("the base answers");
    assert!(say(&exec, A).unwrap_err().starts_with("no model"), "an adapter not yet offered is an unknown model");
    offer(&files[0]);
    let first = say(&exec, A).expect("an adapter offered to a running server answers");
    assert_ne!(first, plain);
    assert_eq!(say(&exec, ID).unwrap(), plain, "and the base is unchanged beside it");
    offer(&files[1]);
    let second = say(&exec, B).unwrap();
    assert_ne!(second, first);
    // The same card id under a new digest is the new release.
    offer(&files[2]);
    let replaced = say(&exec, A).unwrap();
    assert_ne!(replaced, first, "the new release under the id answers differently");
    live.retire(B);
    assert!(say(&exec, B).unwrap_err().starts_with("no model"), "a withdrawn adapter is an unknown model");
    assert_eq!(say(&exec, A).unwrap(), replaced, "the others are untouched");
    exec.shutdown();
    std::fs::remove_dir_all(&dir).ok();
    files.iter().for_each(|f| drop(std::fs::remove_file(f)));
}
