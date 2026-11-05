// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The OpenRouter `/videos` contract against a fake video model: listing,
//! submit, poll, download (whole and by range), idempotent resubmission, an
//! edit fed the previous result, cancellation, and every refusal a client can
//! provoke, each before the model runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use capability::video::{Audio, Fps, VideoContract};
use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-brain-test-key";
const MODEL: &str = "brain/fake-video";

fn contract() -> VideoContract {
    VideoContract::new("frames", (1, 4), Fps::Param { name: "fps".into(), default: 8 }, ("width", "height"), 16, 864 * 480, (1.0, 4.0))
        .audio(Audio::Param { name: "audio".into(), default: false })
        .seed("seed")
}

/// Frame `i`'s flat colour: far enough apart to survive lossy coding.
fn red(i: usize) -> u8 {
    ((i * 40) % 256) as u8
}

fn clip(first: usize, n: usize, w: u32, h: u32) -> Vec<(Vec<f32>, u32, u32)> {
    (first..first + n).map(|i| ([f32::from(red(i)) / 255.0, 0.5, 0.25].into_iter().cycle().take((w * h * 3) as usize).collect(), w, h)).collect()
}

struct FakeVideo(Arc<AtomicUsize>);
struct FakeVideoInst(Arc<AtomicUsize>);

impl ResidentModel for FakeVideo {
    fn manifest(&self) -> Manifest {
        let t2v = ActionSpec::new("t2v", "text to video")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("frames", ParamType::Int, "frames").default(json!(5)))
            .param(ParamSpec::new("width", ParamType::Int, "width").default(json!(32)))
            .param(ParamSpec::new("height", ParamType::Int, "height").default(json!(32)))
            .param(ParamSpec::new("fps", ParamType::Int, "rate").default(json!(8)))
            .param(ParamSpec::new("seed", ParamType::Int, "seed"))
            .param(ParamSpec::new("audio", ParamType::Bool, "sound").default(json!(false)))
            .param(ParamSpec::new("hold", ParamType::Bool, "run until cancelled").default(json!(false)))
            .output(BlobSpec::new("video", Media::Video, "the clip").required())
            .video(contract());
        let extend = ActionSpec::new("extend", "continue a clip")
            .streaming()
            .input(BlobSpec::new("video", Media::Video, "the clip").required())
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("frames", ParamType::Int, "frames to add").default(json!(4)))
            .output(BlobSpec::new("video", Media::Video, "the longer clip").required())
            .video(contract().clip_input("video"));
        let smoke = ActionSpec::new("smoke", "a random-weight smoke test").streaming().param(ParamSpec::new("prompt", ParamType::Str, "p")).output(BlobSpec::new("video", Media::Video, "v")).video(contract().sellable(false));
        Manifest::new(MODEL, "a fake video model", vec![t2v, extend, smoke])
    }
    fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }
    fn estimate(&self, _k: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(FakeVideoInst(self.0.clone())))
    }
}

impl Instance for FakeVideoInst {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        let frames = match action {
            "t2v" => {
                if inv.get_bool("hold").unwrap_or(false) {
                    for _ in 0..1000 {
                        if inv.cancel.is_cancelled() {
                            return Err("cancelled".into());
                        }
                        progress(Progress::step(1, 2, "holding"));
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                let (n, w, h) = (inv.get_i64("frames").unwrap() as usize, inv.get_i64("width").unwrap() as u32, inv.get_i64("height").unwrap() as u32);
                clip(0, n, w, h)
            }
            "extend" => {
                let mut c = capability::blob::decode_video(inv, "video")?;
                let (w, h) = (c[0].1, c[0].2);
                let start = c.len();
                c.extend(clip(start, inv.get_i64("frames").unwrap() as usize, w, h));
                c
            }
            other => return Err(format!("no action {other}")),
        };
        progress(Progress::step(frames.len() as u32, frames.len() as u32, "done"));
        let mut video = capability::blob::video_blob(&frames)?;
        video.meta["fps"] = json!(8);
        Ok(Outcome::new().blob("video", video))
    }
}

fn app_counting(provider: Provider) -> (Router, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(FakeVideo(runs.clone()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    let spool = std::env::temp_dir().join(format!("brain-video-test-{}", uuid::Uuid::new_v4()));
    let state = AppState::new(Executor::start(models, budgets, Policy::default()), KEY.to_string(), provider).with_video_spool(spool, 1 << 30);
    (router(state), runs)
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>, headers: &[(&str, &str)]) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let mut req = Request::builder().method(method).uri(uri).header(header::AUTHORIZATION, format!("Bearer {KEY}"));
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let req = match body {
        Some(b) => req.header(header::CONTENT_TYPE, "application/json").body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    };
    let resp = app.clone().oneshot(req.unwrap()).await.unwrap();
    let (status, headers) = (resp.status(), resp.headers().clone());
    (status, headers, axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap().to_vec())
}

async fn json_call(app: &Router, method: &str, uri: &str, body: Option<Value>, headers: &[(&str, &str)]) -> (StatusCode, Value) {
    let (status, _, bytes) = call(app, method, uri, body, headers).await;
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn wait(app: &Router, id: &str) -> Value {
    for _ in 0..1500 {
        let (status, v) = json_call(app, "GET", &format!("/v1/videos/{id}"), None, &[]).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        if !matches!(v["status"].as_str(), Some("pending" | "in_progress")) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("job {id} never finished");
}

async fn submit(app: &Router, body: Value) -> String {
    let (status, v) = json_call(app, "POST", "/v1/videos", Some(body), &[]).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["status"], "pending");
    assert_eq!(v["polling_url"], format!("/v1/videos/{}", v["id"].as_str().unwrap()));
    v["id"].as_str().unwrap().to_string()
}

/// Decode an MP4 back into frames through brain's own reader.
fn frames_of(mp4: &[u8]) -> Vec<(Vec<f32>, u32, u32)> {
    let path = std::env::temp_dir().join(format!("brain-video-test-{}.mp4", uuid::Uuid::new_v4()));
    std::fs::write(&path, mp4).unwrap();
    let blob = imaging::video::clip_from_file(&path, 1000).unwrap();
    std::fs::remove_file(&path).ok();
    capability::blob::decode_video(&Invocation::new().blob("video", blob), "video").unwrap()
}

fn mean_red(frame: &(Vec<f32>, u32, u32)) -> f32 {
    let px: Vec<f32> = frame.0.iter().step_by(3).copied().collect();
    px.iter().sum::<f32>() / px.len() as f32 * 255.0
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_video_model_is_listed_with_what_it_accepts() {
    let (app, _) = app_counting(Provider::OpenRouter);
    let (status, v) = json_call(&app, "GET", "/v1/videos/models", None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let m = &v["data"][0];
    assert_eq!(m["id"], MODEL);
    assert_eq!(m["supported_durations"], json!([1, 2, 3, 4]));
    assert_eq!(m["seed"], json!(true));
    // The smoke-test action is runnable but never offered.
    let actions: Vec<&str> = m["actions"].as_array().unwrap().iter().map(|a| a["action"].as_str().unwrap()).collect();
    assert_eq!(actions, ["t2v", "extend"]);

    let (_, cards) = json_call(&app, "GET", "/v1/models", None, &[]).await;
    let card = cards["data"].as_array().unwrap().iter().find(|c| c["id"] == MODEL).unwrap();
    assert_eq!(card["architecture"]["output_modalities"], json!(["video"]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clip_is_generated_polled_and_downloaded_whole_and_by_range() {
    let (app, _) = app_counting(Provider::OpenRouter);
    let id = submit(&app, json!({"model": MODEL, "prompt": "a ball", "duration": 1, "size": "32x16", "seed": 4})).await;
    let done = wait(&app, &id).await;
    assert_eq!(done["status"], "completed", "{done}");
    // 1 s at 8 fps on the 1 + 4k grid is 9 frames.
    assert_eq!(done["usage"]["frames"], json!(9));
    assert_eq!(done["usage"]["seconds"], json!(9.0 / 8.0));
    assert_eq!((done["usage"]["width"].as_u64(), done["usage"]["height"].as_u64()), (Some(32), Some(16)));
    let url = done["unsigned_urls"][0].as_str().unwrap().to_string();

    let (status, headers, mp4) = call(&app, "GET", &url, None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "video/mp4");
    assert_eq!(mp4.len() as u64, done["content"][0]["bytes"].as_u64().unwrap());
    assert_eq!(&mp4[4..8], b"ftyp");
    let frames = frames_of(&mp4);
    assert_eq!(frames.len(), 9);
    for (i, f) in frames.iter().enumerate() {
        assert!((mean_red(f) - f32::from(red(i))).abs() < 12.0, "frame {i}: {} vs {}", mean_red(f), red(i));
    }

    let (status, headers, part) = call(&app, "GET", &url, None, &[("range", "bytes=4-7")]).await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(part, b"ftyp");
    assert_eq!(headers[header::CONTENT_RANGE], format!("bytes 4-7/{}", mp4.len()));
    let (status, ..) = call(&app, "GET", &url, None, &[("range", &format!("bytes={}-", mp4.len()))]).await;
    assert_eq!(status, StatusCode::RANGE_NOT_SATISFIABLE);
    let (status, ..) = call(&app, "GET", &format!("/v1/videos/{id}/content?index=1"), None, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Deleting a finished job forgets it and its file.
    let (status, v) = json_call(&app, "DELETE", &format!("/v1/videos/{id}"), None, &[]).await;
    assert_eq!((status, v["status"].as_str()), (StatusCode::OK, Some("deleted")));
    let (status, ..) = call(&app, "GET", &url, None, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_same_idempotency_key_never_renders_twice() {
    let (app, runs) = app_counting(Provider::OpenAI);
    let body = json!({"model": MODEL, "prompt": "once"});
    let key = [("idempotency-key", "job-1")];
    let (s1, a) = json_call(&app, "POST", "/videos", Some(body.clone()), &key).await;
    assert_eq!(s1, StatusCode::ACCEPTED);
    wait(&app, a["id"].as_str().unwrap()).await;
    let (s2, b) = json_call(&app, "POST", "/videos", Some(body), &key).await;
    assert_eq!((s2, &b["id"]), (StatusCode::OK, &a["id"]));
    assert_eq!(b["status"], "completed");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let (s3, _) = json_call(&app, "POST", "/videos", Some(json!({"model": MODEL, "prompt": "other"})), &key).await;
    assert_eq!(s3, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_finished_clip_can_be_extended_as_an_edit() {
    let (app, _) = app_counting(Provider::OpenRouter);
    let id = submit(&app, json!({"model": MODEL, "prompt": "start"})).await;
    let url = wait(&app, &id).await["unsigned_urls"][0].as_str().unwrap().to_string();
    let (_, _, mp4) = call(&app, "GET", &url, None, &[]).await;
    let data = format!("data:video/mp4;base64,{}", events::base64::encode(&mp4));
    let edit = submit(
        &app,
        json!({"model": MODEL, "prompt": "more", "input_video": {"type": "video_url", "video_url": {"url": data}},
               "provider": {"options": {"brain": {"action": "extend", "params": {"frames": 4}}}}}),
    )
    .await;
    let done = wait(&app, &edit).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["usage"]["frames"], json!(9));
    let (_, _, longer) = call(&app, "GET", done["unsigned_urls"][0].as_str().unwrap(), None, &[]).await;
    let frames = frames_of(&longer);
    assert!((mean_red(&frames[7]) - f32::from(red(7))).abs() < 12.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_running_job_is_cancelled() {
    let (app, _) = app_counting(Provider::OpenRouter);
    let id = submit(&app, json!({"model": MODEL, "prompt": "slow", "provider": {"options": {"brain": {"params": {"hold": true}}}}})).await;
    let (status, v) = json_call(&app, "DELETE", &format!("/v1/videos/{id}"), None, &[]).await;
    assert_eq!((status, v["status"].as_str()), (StatusCode::OK, Some("cancelled")));
    assert_eq!(wait(&app, &id).await["status"], "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_requests_are_refused_before_the_model_runs() {
    let (app, runs) = app_counting(Provider::OpenRouter);
    let data = "data:video/mp4;base64,AAAA";
    for (body, want) in [
        (json!({"model": MODEL, "prompt": "x", "duration": 9}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "size": "33x16"}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "colour": "red"}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "callback_url": "https://example.com"}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "frame_images": [{"type": "image_url", "image_url": {"url": data}, "frame_type": "first_frame"}]}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "input_video": {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}, "provider": {"options": {"brain": {"action": "extend"}}}}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "provider": {"options": {"brain": {"action": "extend"}}}}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "provider": {"options": {"brain": {"params": {"weights": "/etc/passwd"}}}}}), StatusCode::BAD_REQUEST),
        (json!({"model": MODEL, "prompt": "x", "seed": 1, "provider": {"options": {"brain": {"params": {"seed": 2}}}}}), StatusCode::BAD_REQUEST),
        (json!({"model": "brain/nothing", "prompt": "x"}), StatusCode::NOT_FOUND),
    ] {
        let (status, v) = json_call(&app, "POST", "/v1/videos", Some(body.clone()), &[]).await;
        assert_eq!(status, want, "{body} -> {v}");
    }
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let (status, _) = json_call(&app, "GET", "/v1/videos/not-a-job", None, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = json_call(&app, "GET", &format!("/v1/videos/{}", uuid::Uuid::new_v4()), None, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn video_routes_need_the_key_and_are_not_on_the_anthropic_surface() {
    let (app, _) = app_counting(Provider::OpenRouter);
    let req = Request::get("/v1/videos/models").body(Body::empty()).unwrap();
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    let (app, _) = app_counting(Provider::Anthropic);
    let req = Request::get("/v1/videos/models").header("x-api-key", KEY).body(Body::empty()).unwrap();
    assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
}
