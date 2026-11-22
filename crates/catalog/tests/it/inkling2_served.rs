// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of looped mixture-of-experts language models for its clients. If your team
// needs expertise in putting a new architecture behind a served chat contract with bounded, cancellable requests then
// you can procure our services by sending an email to info@swedishembedded.com.

//! A pulled Inkling 2 directory served by its resident: the tiny reference checkpoint with a byte-level vocabulary and a
//! typed-message chat template of the checkpoint's own shape. The resident serves a chat request on the CPU within its
//! context through the checkpoint's template (with `reasoning_effort` as a template variable, an out-of-range value
//! refused by name, an image part refused by a checkpoint without a vision tower), the request observes cancellation before admission and between decode tokens, and the footprint it
//! reports holds the weights and the whole visit-state pool.

use std::path::PathBuf;

use capability::Invocation;
use catalog::resident_inkling2::Inkling2Resident;
use catalog::resident_qwen35::Qwen35Sizing;
use capability_mock::cancellation::{certify, Bound};
use data::rng::Lcg;
use model::kv_tier::KvTier;
use residency::{Device, InstanceKey, ResidentModel};
use serde_json::{json, Value};

const SPECIALS: [&str; 9] = [
    "<|message_system|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_image|>",
    "<|unused_200054|>",
    "<|content_model_end_sampling|>",
];
/// The image placeholder's id in the byte-level vocabulary.
const IMAGE_TOKEN: u32 = 256 + 7;
/// Stored unembedding rows and the valid ones (bytes and the specials, then padding).
const VOCAB: usize = 272;
const VALID: usize = 256 + SPECIALS.len();

/// The checkpoint's template shape: a system message with the effort, typed user and model messages (a user message's
/// parts each its own message, an image part its placeholder), the reply ended by the sampling eos.
const TEMPLATE: &str = "<|message_system|><|content_text|>effort {{ reasoning_effort }}<|end_message|>{% for m in messages %}{% if m.role == 'user' %}{% if m.content is string %}<|message_user|><|content_text|>{{ m.content }}<|end_message|>{% else %}{% for p in m.content %}{% if p.type == 'text' %}<|message_user|><|content_text|>{{ p.text }}<|end_message|>{% elif p.type in ('image', 'input_image', 'image_url') %}<|message_user|><|content_image|><|unused_200054|><|end_message|>{% else %}{{ raise_exception('Unsupported content part type: ' ~ p.type) }}{% endif %}{% endfor %}{% endif %}{% else %}{% if m.reasoning_content %}<|message_model|><|content_thinking|>{{ m.reasoning_content }}<|end_message|>{% endif %}<|message_model|><|content_text|>{{ m.content }}<|end_message|><|content_model_end_sampling|>{% endif %}{% endfor %}";

/// The tiny reference checkpoint widened to a byte-level vocabulary, with its tokenizer, template and generation
/// config, as a pull leaves a release directory.
fn pulled_dir(tag: &str) -> Option<PathBuf> {
    let tiny = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if !tiny.join("model.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("inkling2-served-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let mut config: Value = serde_json::from_str(&std::fs::read_to_string(tiny.join("config.json")).unwrap()).unwrap();
    config["text_config"]["vocab_size"] = json!(VOCAB);
    config["text_config"]["unpadded_vocab_size"] = json!(VALID);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    // seed: any rows serve; the test reads renderings, bounds and cancellation, never the model's quality.
    let mut rng = Lcg::new(11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = checkpoint::safetensors::read(&tiny.join("model.safetensors").to_string_lossy())
        .unwrap()
        .into_iter()
        .map(|t| {
            if t.name == inkling2::names::embed() || t.name == inkling2::names::unembed() {
                let d = t.shape[1];
                let mut data = t.data;
                data.extend(rng.vec_scaled((VOCAB - t.shape[0]) * d, 0.5));
                (t.name, vec![VOCAB as u64, d as u64], data)
            } else {
                (t.name, t.shape.iter().map(|&s| s as u64).collect(), t.data)
            }
        })
        .collect();
    checkpoint::st::save_safetensors_with_metadata(&dir.join("model.safetensors").to_string_lossy(), &tensors, Default::default()).unwrap();
    let vocab: serde_json::Map<String, Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let added: Vec<Value> = SPECIALS.iter().enumerate().map(|(i, c)| json!({"content": c, "id": 256 + i, "special": true})).collect();
    std::fs::write(dir.join("tokenizer.json"), json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added}).to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), json!({"chat_template": TEMPLATE}).to_string()).unwrap();
    std::fs::write(dir.join("generation_config.json"), json!({"eos_token_id": 256 + SPECIALS.len() - 1}).to_string()).unwrap();
    Some(dir)
}

fn resident(dir: &std::path::Path) -> Inkling2Resident {
    let sizing = Qwen35Sizing { ctx: 256, ctx_explicit: true, max_batch: 2, ..Qwen35Sizing::default() };
    Inkling2Resident::from_hf_dir(dir.to_str().unwrap(), "local/inkling2", sizing).expect("the pulled directory is served")
}

fn ask(text: &str) -> Invocation {
    Invocation::new().set("messages", json!(json!([{"role": "user", "content": text}]).to_string())).set("temperature", json!(0.0))
}

#[test]
fn the_resident_serves_a_typed_message_chat_within_its_context() {
    let Some(dir) = pulled_dir("serve") else { return };
    let r = resident(&dir);
    let m = r.manifest();
    assert_eq!(m.max_context_tokens, Some(256), "the context is the asked cap, inside the model's positions");
    let generate = m.actions.iter().find(|a| a.name == "generate").expect("generate");
    assert!(generate.cancellable, "the generate action declares cancellation");
    assert!(generate.params.iter().any(|p| p.name == "reasoning_effort"), "reasoning_effort is a served param");
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let format = qwen3::chat::ChatFormat::own_template(&dir).unwrap();
    let inv = ask("hi").set("reasoning_effort", json!("high")).set("max_new", json!(3));
    let rendered = qwen3::chat::render_prompt_as(&format, &inv).unwrap();
    assert!(rendered.text.starts_with("<|message_system|><|content_text|>effort high<|end_message|>"), "{}", rendered.text);
    gpu_core::devices::with_host_tier(|| {
        let mut instance = r.activate(&InstanceKey::new("local/inkling2", "default"), Device::Cpu).expect("the resident activates");
        let out = instance.run("generate", &inv, &mut |_| {}).expect("served").outputs;
        assert_eq!(out["prompt_tokens"].as_u64(), Some(data::tokenizer::Tokenizer::encode(&tok, &rendered.text).len() as u64), "{out}");
        let e = instance.run("generate", &ask("hi").set("reasoning_effort", json!("1.5")), &mut |_| {}).unwrap_err();
        assert!(e.contains("reasoning_effort"), "an effort outside 0..1 is refused by name: {e}");
        let picture = json!([{"role": "user", "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {}}]}]);
        let e = instance.run("generate", &Invocation::new().set("messages", json!(picture.to_string())), &mut |_| {}).unwrap_err();
        assert!(e.contains("no vision tower"), "a checkpoint without a tower refuses an image part, never reads it as its placeholder token: {e}");
        let long = || ask("count").set("max_new", json!(24)).set("ignore_eos", json!(true));
        let instance = std::cell::RefCell::new(instance);
        // The token fires on the first event (the request's admission); the prefill already queued with it still
        // answers its first token, then the next scheduler step sees the token: two events at most.
        certify("inkling2 generate", &long, &|inv, p| instance.borrow_mut().run("generate", inv, p), Bound { fire_after: 1, within: 2 });
    });
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_footprint_holds_the_weights_and_the_whole_visit_pool() {
    let Some(dir) = pulled_dir("estimate") else { return };
    let r = resident(&dir);
    let cfg = inkling2::Inkling2Config::from_dir(&dir).unwrap();
    let layout = inkling2::serve::Engine::layout(&cfg, model::kv_tier::KvTier::F32).unwrap();
    let s = r.options().sizing;
    assert_eq!((s.live_rows, s.snapshot_rows, s.num_blocks), (2, 2, 2 * 256 / s.block_size), "two sequences of the context, as many cached prompts");
    let pool = u64::from(s.num_blocks) * layout.block_bytes(s.block_size) + 4 * layout.row_bytes() + layout.zero_row_bytes();
    let weights = std::fs::metadata(dir.join("model.safetensors")).unwrap().len();
    let reserved = r.estimate(&InstanceKey::new("local/inkling2", "default")).vram;
    assert!(reserved > weights + pool, "the estimate holds the weights ({weights}), the pool ({pool}) and the step buffers: {reserved}");
    std::fs::remove_dir_all(&dir).ok();
}

/// `--kv-cache bf16` keeps every K/V row of the pool in bf16 (half the bytes per token) and still serves; a tier the
/// engine does not offer is refused by name at the scan, never served as f32.
#[test]
fn the_kv_cache_choice_sizes_the_pool_and_an_unoffered_tier_is_refused() {
    let Some(dir) = pulled_dir("kv") else { return };
    let sized = |kv| Qwen35Sizing { ctx: 256, ctx_explicit: true, max_batch: 2, kv_cache: kv, ..Qwen35Sizing::default() };
    let f32 = Inkling2Resident::from_hf_dir(dir.to_str().unwrap(), "local/inkling2", sized(None)).unwrap();
    let bf16 = Inkling2Resident::from_hf_dir(dir.to_str().unwrap(), "local/inkling2", sized(Some(KvTier::BF16))).unwrap();
    assert_eq!((f32.options().kv_tier, bf16.options().kv_tier), (KvTier::F32, KvTier::BF16));
    let cfg = inkling2::Inkling2Config::from_dir(&dir).unwrap();
    let block = |kv| inkling2::serve::Engine::layout(&cfg, kv).unwrap().block_bytes(f32.options().sizing.block_size);
    assert_eq!(2 * block(KvTier::BF16), block(KvTier::F32), "a bf16 block is half an f32 one");
    let key = InstanceKey::new("local/inkling2", "default");
    assert!(bf16.estimate(&key).vram < f32.estimate(&key).vram, "the estimate follows the tier");
    for refused in [KvTier::INT8, KvTier::FP8] {
        let e = Inkling2Resident::from_hf_dir(dir.to_str().unwrap(), "local/inkling2", sized(Some(refused))).err().expect("refused");
        assert!(e.contains(refused.as_str()) && e.contains("bf16"), "the refusal names the tier and what is offered: {e}");
    }
    gpu_core::devices::with_host_tier(|| {
        let mut instance = bf16.activate(&key, Device::Cpu).expect("the bf16 resident activates");
        let out = instance.run("generate", &ask("hi").set("max_new", json!(3)), &mut |_| {}).expect("served at bf16").outputs;
        assert!(out["completion_tokens"].as_u64().is_some_and(|n| n > 0), "{out}");
    });
    std::fs::remove_dir_all(&dir).ok();
}

/// A tower on the decoder's width in the published naming, random but well scaled: one layer, regions of 40 pixels
/// (20-pixel patches merged 2x2, as the processor requires).
fn tiny_vision() -> Value {
    json!({
        "model_type": "inkling2_vision", "text_hidden_size": 64, "temporal_patch_size": 2, "native_patch_size": 20,
        "adapter_merge_size": 2, "projection_hidden_dim": 32, "projector_bias": false, "hidden_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4, "mlp_ratio": 2.0, "posemb_grid_size": 4, "layer_norm_eps": 1e-5
    })
}

/// [`pulled_dir`] with a vision tower: `vision_config`, the tower's tensors beside the decoder's and the processor's
/// configuration, as the released directory carries them.
fn pulled_dir_with_tower(tag: &str) -> Option<PathBuf> {
    let dir = pulled_dir(tag)?;
    let mut config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    config["vision_config"] = tiny_vision();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let vision: inkling2::config::VisionConfig = serde_json::from_value(tiny_vision()).unwrap();
    // seed: any tower serves; the test compares the served rows with the uncached forward over the same tower.
    let mut rng = Lcg::new(23);
    let path = dir.join("model.safetensors");
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = checkpoint::safetensors::read(&path.to_string_lossy()).unwrap().into_iter().map(|t| (t.name, t.shape.iter().map(|&s| s as u64).collect(), t.data)).collect();
    for e in inkling2::names::vision_manifest_of(&vision) {
        let n: u64 = e.shape.iter().product();
        let gain = e.name.ends_with("ln_1.weight") || e.name.ends_with("ln_2.weight") || e.name.ends_with("ln_pre.weight") || e.name.ends_with("gamma");
        let data: Vec<f32> = rng.vec_scaled(n as usize, if gain { 0.2 } else { 0.1 }).into_iter().map(|x| if gain { 1.0 + x } else { x }).collect();
        tensors.push((e.name, e.shape, data));
    }
    checkpoint::st::save_safetensors_with_metadata(&path.to_string_lossy(), &tensors, Default::default()).unwrap();
    let processor = json!({
        "image_processor": {"do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "Inkling2ImageProcessor", "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.00392156862745098,
            "size": {"height": 40, "width": 40}},
        "image_token": "<|unused_200054|>", "processor_class": "Inkling2Processor"
    });
    std::fs::write(dir.join("processor_config.json"), processor.to_string()).unwrap();
    Some(dir)
}

/// A picture of `w x h` pixels with random content.
fn picture(rng: &mut Lcg, w: u32, h: u32) -> imaging::pixels::Rgb8 {
    let px: Vec<u8> = rng.vec_scaled((w * h * 3) as usize, 1.0).into_iter().map(|x| ((x + 1.0) * 127.5) as u8).collect();
    imaging::pixels::Rgb8::new(w, h, px).unwrap()
}

fn with_images(inv: Invocation, images: &[&imaging::pixels::Rgb8]) -> Invocation {
    images.iter().enumerate().fold(inv, |inv, (i, img)| inv.blob(&capability::blob::image_key(i), capability::blob::image_blob(&img.to_hwc_unit(), img.w, img.h, 3)))
}

fn log_softmax_at(row: &[f32], at: usize) -> f64 {
    let max = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
    let lse = max + row.iter().map(|&x| (f64::from(x) - max).exp()).sum::<f64>().ln();
    f64::from(row[at]) - lse
}

/// An image+text chat through the resident's request path: the image blobs and the messages' image parts become the
/// template's placeholders, one token per 40x40 region, with the tower's rows standing there; the served tokens and
/// their log-probabilities are the uncached forward's over the same prompt and the same spliced rows. Every bound on
/// the new input is refused by name as the client's error.
#[test]
fn an_image_chat_is_served_with_the_uncached_forwards_logits_over_the_spliced_rows() {
    let Some(dir) = pulled_dir_with_tower("image") else { return };
    let r = resident(&dir);
    let generate = r.manifest().actions.into_iter().find(|a| a.name == "generate").unwrap();
    let declared: Vec<&str> = generate.inputs.iter().filter(|b| b.media == capability::Media::Image).map(|b| b.name.as_str()).collect();
    assert_eq!(declared.len(), capability::blob::MAX_IMAGES, "a checkpoint with a tower declares its image inputs: {declared:?}");

    let mut rng = Lcg::new(5);
    let (wide, small) = (picture(&mut rng, 90, 50), picture(&mut rng, 40, 40));
    let text = |s: &str| json!({"type": "text", "text": s});
    let image = json!({"type": "image_url", "image_url": {}});
    // The literal placeholder typed by hand is text: escaped, it never becomes an image position.
    let messages = json!([{"role": "user", "content": [text("compare <|unused_200054|>"), image.clone(), text("with"), image.clone()]}]);
    let chat = |messages: &Value| Invocation::new().set("messages", json!(messages.to_string())).set("temp", json!(0.0)).set("max_new", json!(4)).set("ignore_eos", json!(true));
    let inv = with_images(chat(&messages).set("logprobs", json!(true)).set("return_token_ids", json!(true)), &[&wide, &small]);

    // The prompt the request should become, built apart from the front end: the template's text, one placeholder per
    // image expanded to the image's regions (3x2 and 1x1), and the tower's rows over the processor's pixels.
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let format = qwen3::chat::ChatFormat::own_template(&dir).unwrap();
    let escaped = inv.without_blobs().escape_client_text(&|t| data::tokenizer::Tokenizer::escape_untrusted(&tok, t));
    let rendered = qwen3::chat::render_prompt_as(&format, &escaped).unwrap();
    let ids = data::tokenizer::Tokenizer::encode(&tok, &rendered.text);
    assert_eq!(ids.iter().filter(|&&t| t == IMAGE_TOKEN).count(), 2, "one placeholder per image part, none from the typed text");
    let prompt = inkling2::vision::expand_placeholders(&ids, IMAGE_TOKEN, &[6, 1]).unwrap();

    gpu_core::devices::with_host_tier(|| {
        let ops = inkling2::serve::device_ops().unwrap();
        let vision: inkling2::config::VisionConfig = serde_json::from_value(tiny_vision()).unwrap();
        let tower_gpu = ops.gpu().new_like(inkling2::vision::pipelines());
        let tower = inkling2::vision::VisionTower::new(&tower_gpu, &vision, &inkling2::vision::VisionWeights::open(&dir, &vision).unwrap()).unwrap();
        let tiling = inkling2::vision::ImageProcessorSpec::from_json(&std::fs::read_to_string(dir.join("processor_config.json")).unwrap()).unwrap().tiling(&vision).unwrap();
        let regions: Vec<imaging::RegionImage> = [&wide, &small].iter().map(|img| tiling.preprocess(&imaging::Decoded::Rgb((*img).clone())).unwrap()).collect();
        let features = tower.encode(&tower_gpu, &regions).unwrap();
        let embeds = inkling2::vision::splice_embeds(&prompt, IMAGE_TOKEN, &features).unwrap();
        let model = inkling2::Inkling2::load(ops, &dir).unwrap();

        let mut instance = r.activate(&InstanceKey::new("local/inkling2", "default"), Device::Cpu).expect("the resident activates with its tower");
        // Greedy: every served token is the argmax of the uncached forward over the prompt, the rows and the tokens
        // served before it. Sampled over the whole vocabulary at temperature 1: every served log-probability is the
        // uncached forward's log-softmax at the token drawn.
        let mut seq = Vec::new();
        for sampled in [false, true] {
            let inv = if sampled { inv.clone().set("temp", json!(1.0)).set("top_k", json!(0)).set("top_p", json!(1.0)).set("seed", json!(3)) } else { inv.clone() };
            let out = instance.run("generate", &inv, &mut |_| {}).expect("an image chat is served").outputs;
            let served_prompt: Vec<u32> = serde_json::from_value(out["prompt_token_ids"].clone()).unwrap();
            assert_eq!(served_prompt, prompt, "the served prompt is the template's, each placeholder expanded to its image's regions");
            let completion: Vec<u32> = serde_json::from_value(out["completion_token_ids"].clone()).unwrap();
            let logprobs: Vec<f64> = serde_json::from_value(out["completion_logprobs"].clone()).unwrap();
            assert_eq!(completion.len(), 4, "{out}");
            seq = prompt.iter().chain(&completion[..3]).copied().collect();
            let logits = model.forward_embedded(&seq, &embeds).unwrap();
            for (i, (&t, &lp)) in completion.iter().zip(&logprobs).enumerate() {
                let row = &logits[(prompt.len() - 1 + i) * VALID..(prompt.len() + i) * VALID];
                if sampled {
                    let want = log_softmax_at(row, t as usize);
                    assert!((lp - want).abs() < 1e-4, "token {i}: served log-probability {lp}, uncached forward {want}");
                } else {
                    let argmax = row.iter().enumerate().fold((0, f32::MIN), |b, (j, &x)| if x > b.1 { (j, x) } else { b }).0 as u32;
                    assert_eq!(t, argmax, "token {i}: the served greedy token is the uncached forward's");
                }
            }
        }
        let (v, logits) = (VALID, model.forward_embedded(&seq, &embeds).unwrap());
        // Without the spliced rows the forward reads the placeholder's own embedding: a different prompt, so the
        // comparison above is about the images.
        let plain = model.forward(&seq).unwrap();
        assert_ne!(&plain[(prompt.len() - 1) * v..prompt.len() * v], &logits[(prompt.len() - 1) * v..prompt.len() * v]);

        let mut refused = |inv: &Invocation, needle: &str| {
            let e = instance.run("generate", inv, &mut |_| {}).unwrap_err();
            assert!(matches!(capability::failure::classify(&e), Some((capability::FailureClass::InvalidRequest, _))), "a client's error: {e}");
            assert!(e.contains(needle), "{needle:?} in {e}");
        };
        // An image part without its image, an image without its part.
        refused(&chat(&json!([{"role": "user", "content": [text("x"), image.clone()]}])), "carries none");
        refused(&with_images(chat(&json!([{"role": "user", "content": "x"}])), &[&small]), "placeholder");
        refused(&with_images(chat(&json!([{"role": "user", "content": [image.clone(), image.clone()]}])), &[&small]), "placeholder");
        // A side past the processor's 4096 pixels, refused before any pixel is processed.
        refused(&with_images(chat(&json!([{"role": "user", "content": [image.clone()]}])), &[&picture(&mut rng, 4097, 1)]), "4096");
        // More regions than the context (256 tokens) holds: 17 x 16 regions.
        refused(&with_images(chat(&json!([{"role": "user", "content": [image.clone()]}])), &[&picture(&mut rng, 17 * 40, 16 * 40)]), "context");
        // A raw prompt cannot place an image.
        refused(&with_images(Invocation::new().set("prompt", json!("x")).set("chat", json!(false)), &[&small]), "raw prompt");
        // A blob that is not an RGB image of its declared size.
        let bad = Invocation::new().set("prompt", json!("x")).blob("image", capability::Blob::new(capability::Media::Image, vec![0u8; 12]).with_meta(json!({"w": 2, "h": 2})));
        refused(&bad, "image");
    });
    std::fs::remove_dir_all(&dir).ok();
}
