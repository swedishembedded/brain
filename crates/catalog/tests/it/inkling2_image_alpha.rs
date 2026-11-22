// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the vision-language models its clients serve. If your team needs
// expertise in getting a picture from an HTTP request to a vision tower exactly as the model was trained to see it
// then you can procure our services by sending an email to info@swedishembedded.com.

//! A transparent PNG sent over the OpenAI surface reaches the tower as the reference processor's pixels: the image keeps
//! its alpha channel from the HTTP decode to the processor, which composites it onto its checkerboard (the model's image
//! inputs declare that they read alpha). The served greedy tokens and their log-probabilities are the uncached forward's
//! over the tower's rows of the reference processor's own `pixel_values` for that file; with the alpha dropped (what an
//! RGB decode gives) the rows, and the logits, are different.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use residency::{Device, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::inkling2_served::{log_softmax_at, pulled_dir_with_tower, resident, tiny_vision, IMAGE_TOKEN, VALID};

#[tokio::test(flavor = "multi_thread")]
async fn a_transparent_png_over_the_api_reaches_the_tower_as_the_reference_processors_pixels() {
    let preproc = brain_testutil::testdata_path("golden/inkling2/vision/preproc");
    if !preproc.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/vision/preproc absent (tools/goldens/inkling2_vision_dump.py preproc)");
        return;
    }
    let Some(dir) = pulled_dir_with_tower("alpha-http") else { return };
    let png = std::fs::read(preproc.join("rgba_mixed.png")).unwrap();
    assert!(matches!(imaging::decode_alpha_bounded(&png, u64::MAX).unwrap(), imaging::Decoded::Rgba(_)), "the fixture is transparent");
    let golden = brain_testutil::parity::load(&preproc.join("golden.safetensors"));
    let grid = &golden["rgba_mixed.grid"].data;
    let reference = imaging::RegionImage { rows: grid[1] as u32, cols: grid[2] as u32, pixels: golden["rgba_mixed.pixels"].data.clone() };

    let mut budgets = residency::budget::Budgets::new();
    budgets.set(Device::Cpu, 64 << 30, 0);
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(resident(&dir))];
    let exec = residency::Executor::start(models, budgets, residency::Policy::default());
    let app = apiserve::router(apiserve::AppState::new(exec.clone(), "k", apiserve::Provider::OpenAI));
    let picture = ["data:image/png;base64,", &events::base64::encode(&png)].concat();
    let ask = |sampling: Value| {
        let mut body = json!({
            "model": "local/inkling2", "max_tokens": 3, "logprobs": true, "return_token_ids": true,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": picture.clone()}}, {"type": "text", "text": "what is this"}]}]
        });
        body.as_object_mut().unwrap().extend(sampling.as_object().unwrap().clone());
        Request::builder().method(Method::POST).uri("/v1/chat/completions").header(header::AUTHORIZATION, "Bearer k").header(header::CONTENT_TYPE, "application/json").body(Body::from(body.to_string())).unwrap()
    };
    // Greedy, then sampled over the whole vocabulary: the tokens of the first are the forward's argmax, the
    // log-probabilities of the second its log-softmax at the tokens drawn.
    let mut served = Vec::new();
    for sampling in [json!({"temperature": 0.0}), json!({"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 3})] {
        let response = app.clone().oneshot(ask(sampling)).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        let out: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, StatusCode::OK, "{out}");
        let prompt: Vec<u32> = serde_json::from_value(out["prompt_token_ids"].clone()).unwrap();
        let completion: Vec<u32> = serde_json::from_value(out["choices"][0]["token_ids"].clone()).unwrap();
        let logprobs: Vec<f64> = out["choices"][0]["logprobs"]["content"].as_array().unwrap().iter().map(|t| t["logprob"].as_f64().unwrap()).collect();
        assert!(!completion.is_empty() && completion.len() == logprobs.len(), "{out}");
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE_TOKEN).count(), reference.regions(), "one placeholder per region of the 70x45 image");
        served.push((prompt, completion, logprobs));
    }
    exec.shutdown();

    gpu_core::devices::with_host_tier(|| {
        let ops = inkling2::serve::device_ops().unwrap();
        let vision: inkling2::config::VisionConfig = serde_json::from_value(tiny_vision()).unwrap();
        let tower_gpu = ops.gpu().new_like(inkling2::vision::pipelines());
        let tower = inkling2::vision::VisionTower::new(&tower_gpu, &vision, &inkling2::vision::VisionWeights::open(&dir, &vision).unwrap()).unwrap();
        let model = inkling2::Inkling2::load(ops, &dir).unwrap();
        let forward = |image: &imaging::RegionImage, seq: &[u32], prompt: &[u32]| {
            let features = tower.encode(&tower_gpu, std::slice::from_ref(image)).unwrap();
            model.forward_embedded(seq, &inkling2::vision::splice_embeds(prompt, IMAGE_TOKEN, &features).unwrap()).unwrap()
        };
        for (sampled, (prompt, completion, logprobs)) in served.iter().enumerate() {
            let seq: Vec<u32> = prompt.iter().chain(&completion[..completion.len() - 1]).copied().collect();
            let logits = forward(&reference, &seq, prompt);
            for (i, (&t, &lp)) in completion.iter().zip(logprobs).enumerate() {
                let row = &logits[(prompt.len() - 1 + i) * VALID..(prompt.len() + i) * VALID];
                if sampled == 1 {
                    let want = log_softmax_at(row, t as usize);
                    assert!((lp - want).abs() < 1e-4, "token {i}: served log-probability {lp}, forward over the reference pixels {want}");
                } else {
                    let argmax = row.iter().enumerate().fold((0, f32::MIN), |b, (j, &x)| if x > b.1 { (j, x) } else { b }).0 as u32;
                    assert_eq!(t, argmax, "token {i}: the served greedy token is the forward's over the reference pixels");
                }
            }
        }
        let (prompt, _, _) = &served[0];
        let logits = forward(&reference, prompt, prompt);
        // The comparison is about the transparency: the same file with its alpha dropped gives other rows and logits.
        let processor = std::fs::read_to_string(dir.join("processor_config.json")).unwrap();
        let tiling = inkling2::vision::ImageProcessorSpec::from_json(&processor).unwrap().tiling(&vision).unwrap();
        let dropped = tiling.preprocess(&imaging::Decoded::Rgb(imaging::codec::decode_bounded(&png, u64::MAX).unwrap())).unwrap();
        assert_ne!(dropped.pixels, reference.pixels, "an RGB decode is not the reference's composite");
        let other = forward(&dropped, prompt, prompt);
        let at = (prompt.len() - 1) * VALID;
        let moved = other[at..at + VALID].iter().zip(&logits[at..at + VALID]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-4, "the dropped-alpha rows move the logits ({moved:.3e})");
    });
    std::fs::remove_dir_all(&dir).ok();
}
