// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for putting one inference endpoint
// in front of every card of a cluster for its clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The router (`apiserve::route`) in front of mock upstream servers on loopback.
//!
//! Each mock answers `/v1/health` and `/v1/models` like `brain serve` and records every request it is sent, so a test
//! asserts where the router sent what: prefix affinity, load relative to slots, health, drain, the merged listing,
//! streaming pass-through and cancel-on-disconnect, the at-most-once retry rule, overload and the security boundary.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use apiserve::route::{self, Fleet, FleetConfig, UpstreamUrl};
use apiserve::{Authenticator, CallerSpec, Callers, ConnectionLimits, Role, StaticKey};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, Semaphore};

const CLIENT_KEY: &str = "sk-client";
const UPSTREAM_KEY: &str = "sk-upstream";

/// What one mock upstream was sent.
#[derive(Clone, Debug)]
struct Hit {
    path: String,
    body: Value,
    headers: HeaderMap,
}

/// A mock `brain serve`: fixed slots and models, a record of what it was sent, and switches a test flips.
struct Mock {
    name: String,
    slots: u64,
    edge: u64,
    models: Vec<String>,
    hits: Mutex<Vec<Hit>>,
    healthy: AtomicBool,
    /// A request with `"hold": true` waits for a permit before it answers.
    release: Semaphore,
    /// A streamed request sends its first event, then waits for this before the rest.
    stream_go: Notify,
    /// Set when a streamed response is dropped before it ended (its client went away).
    stream_dropped: AtomicBool,
    /// The server task: aborting it closes the listener, so connecting is refused.
    server: Mutex<Option<tokio::task::AbortHandle>>,
}

impl Mock {
    fn hits(&self) -> Vec<Hit> {
        self.hits.lock().unwrap().clone()
    }
    fn count(&self) -> usize {
        self.hits.lock().unwrap().len()
    }
    /// Stop listening: a connection to it is refused from now on.
    fn stop(&self) {
        if let Some(server) = self.server.lock().unwrap().take() {
            server.abort();
        }
    }
}

struct DropFlag(Arc<Mock>, bool);
impl Drop for DropFlag {
    fn drop(&mut self) {
        if !self.1 {
            self.0.stream_dropped.store(true, Ordering::SeqCst);
        }
    }
}

async fn mock_health(State(m): State<Arc<Mock>>, headers: HeaderMap) -> Response {
    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some(&format!("Bearer {UPSTREAM_KEY}")) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if !m.healthy.load(Ordering::SeqCst) {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let completion = 10 * m.count() as u64;
    let served = json!({"requests": m.count(), "prompt_tokens": 2 * completion, "completion_tokens": completion, "cached_prompt_tokens": completion});
    Json(json!({"status": "ok", "decode_slots": m.slots, "edge_concurrency": m.edge, "queue_depth": 0, "served": served})).into_response()
}

async fn mock_models(State(m): State<Arc<Mock>>) -> Json<Value> {
    let data: Vec<Value> = m.models.iter().map(|id| json!({"id": id, "object": "model", "owned_by": m.name})).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn mock_generate(State(m): State<Arc<Mock>>, uri: axum::http::Uri, headers: HeaderMap, body: Bytes) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    m.hits.lock().unwrap().push(Hit { path: uri.path().to_string(), body: body.clone(), headers });
    if body["hold"] == json!(true) {
        m.release.acquire().await.expect("release").forget();
    }
    if body["stream"] == json!(true) {
        let m2 = m.clone();
        let events = async_stream::stream! {
            let mut flag = DropFlag(m2.clone(), false);
            yield Ok::<_, std::convert::Infallible>(Bytes::from("data: one\n\n"));
            m2.stream_go.notified().await;
            yield Ok(Bytes::from("data: two\n\n"));
            yield Ok(Bytes::from("data: [DONE]\n\n"));
            flag.1 = true;
        };
        return Response::builder().header("content-type", "text/event-stream").header("x-internal", "secret").body(Body::from_stream(events)).unwrap();
    }
    let mut resp = Json(json!({"upstream": m.name})).into_response();
    resp.headers_mut().insert("x-internal", "secret".parse().unwrap());
    resp.headers_mut().insert("set-cookie", "session=1".parse().unwrap());
    resp
}

/// Start a mock upstream on loopback.
async fn mock(name: &str, slots: u64, edge: u64, models: &[&str]) -> (Arc<Mock>, String) {
    let m = Arc::new(Mock {
        name: name.into(),
        slots,
        edge,
        models: models.iter().map(|s| s.to_string()).collect(),
        hits: Mutex::default(),
        healthy: AtomicBool::new(true),
        release: Semaphore::new(0),
        stream_go: Notify::new(),
        stream_dropped: AtomicBool::new(false),
        server: Mutex::default(),
    });
    let app = Router::new()
        .route("/v1/health", get(mock_health))
        .route("/v1/models", get(mock_models))
        .route("/v1/chat/completions", post(mock_generate))
        .route("/chat/completions", post(mock_generate))
        .route("/v1/completions", post(mock_generate))
        .route("/v1/embeddings", post(mock_generate))
        .with_state(m.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    *m.server.lock().unwrap() = Some(server.abort_handle());
    (m, format!("http://{addr}"))
}

/// An upstream that answers its probes, then takes a POST and closes the connection without answering: a request the
/// router sent, which may have run.
async fn black_hole(models: &[&str]) -> (Arc<AtomicUsize>, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let posts = Arc::new(AtomicUsize::new(0));
    let (seen, models) = (posts.clone(), models.iter().map(|id| json!({"id": id})).collect::<Vec<_>>());
    tokio::spawn(async move {
        loop {
            let Ok((mut s, _)) = listener.accept().await else { return };
            let (seen, models) = (seen.clone(), models.clone());
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let mut n = 0;
                while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                    match s.read(&mut buf[n..]).await {
                        Ok(0) | Err(_) => return,
                        Ok(k) => n += k,
                    }
                }
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                let answer = if head.starts_with("GET /v1/health") {
                    json!({"decode_slots": 4, "edge_concurrency": 8}).to_string()
                } else if head.starts_with("GET /v1/models") {
                    json!({"data": models}).to_string()
                } else {
                    seen.fetch_add(1, Ordering::SeqCst);
                    return;
                };
                let _ = s.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}", answer.len()).as_bytes()).await;
            });
        }
    });
    (posts, format!("http://{addr}"))
}

/// The test fleet's configuration over `upstreams`: probed only when a test says so.
fn config(upstreams: &[String]) -> FleetConfig {
    let urls = upstreams.iter().map(|u| UpstreamUrl::parse(u).unwrap()).collect();
    FleetConfig::new(urls).with_upstream_key(Some(UPSTREAM_KEY.into())).with_health_interval(Duration::from_secs(3600)).with_connect_timeout(Duration::from_secs(2))
}

/// A router over `upstreams`, probed once, serving on loopback.
async fn router_with(upstreams: &[String], auth: Arc<dyn Authenticator>) -> (SocketAddr, Arc<Fleet>, brain_shutdown::Trigger) {
    router_over(config(upstreams), auth).await
}

async fn router_over(config: FleetConfig, auth: Arc<dyn Authenticator>) -> (SocketAddr, Arc<Fleet>, brain_shutdown::Trigger) {
    let fleet = Fleet::new(config).unwrap();
    fleet.probe().await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (trigger, shutdown) = brain_shutdown::channel();
    tokio::spawn(route::run(fleet.clone(), listener, auth, ConnectionLimits::default(), None, shutdown));
    (addr, fleet, trigger)
}

async fn router(upstreams: &[String]) -> (SocketAddr, Arc<Fleet>, brain_shutdown::Trigger) {
    router_with(upstreams, Arc::new(StaticKey::new(CLIENT_KEY))).await
}

/// Send one request to the router and open its response.
async fn send(addr: SocketAddr, method: Method, path: &str, key: Option<&str>, body: &str, extra: &[(&str, &str)]) -> hyper::Response<hyper::body::Incoming> {
    let stream = TcpStream::connect(addr).await.unwrap();
    let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, Body>(TokioIo::new(stream)).await.unwrap();
    tokio::spawn(conn);
    let mut req = Request::builder().method(method).uri(path).header("host", "router");
    if let Some(key) = key {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    if !body.is_empty() {
        req = req.header("content-type", "application/json");
    }
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    sender.send_request(req.body(Body::from(body.to_string())).unwrap()).await.unwrap()
}

/// Send one request and read the whole answer.
async fn call(addr: SocketAddr, method: Method, path: &str, key: Option<&str>, body: &str) -> (StatusCode, HeaderMap, Value) {
    let resp = send(addr, method, path, key, body, &[]).await;
    let (parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(Body::new(body), 1 << 24).await.unwrap();
    (parts.status, parts.headers, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn chat(addr: SocketAddr, body: Value) -> (StatusCode, HeaderMap, Value) {
    chat_key(addr, CLIENT_KEY, body).await
}

fn conversation(model: &str, system: &str, turns: usize) -> Value {
    let mut messages = vec![json!({"role": "system", "content": system})];
    for t in 0..turns {
        messages.push(json!({"role": "user", "content": format!("question {t}")}));
        if t + 1 < turns {
            messages.push(json!({"role": "assistant", "content": format!("answer {t}")}));
        }
    }
    json!({"model": model, "messages": messages})
}

/// Wait (bounded) until `cond` holds.
async fn until(what: &str, cond: impl Fn() -> bool) {
    for _ in 0..500 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_conversation_returns_to_the_upstream_holding_its_prefix_and_new_ones_go_to_the_least_loaded() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (b, ub) = mock("b", 8, 16, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[ua, ub]).await;
    let sys_one = "You are agent one. ".repeat(200);
    let sys_two = "You are agent two. ".repeat(200);

    // Conversation one starts while nothing is in flight: the first upstream on a tie. It is held there, so
    // conversation two finds `a` busy and goes to `b`, the least loaded.
    let mut held = conversation("m", &sys_one, 1);
    held["hold"] = json!(true);
    let first = tokio::spawn(chat(addr, held));
    until("the held turn to reach a", || a.count() == 1).await;
    let (code, _, body) = chat(addr, conversation("m", &sys_two, 1)).await;
    assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("b")));
    a.release.add_permits(1);
    assert_eq!(first.await.unwrap().2["upstream"], "a");

    // Both idle now, so load alone would pick `a` for both: the prefix each holds decides instead.
    for turns in 2..=4 {
        assert_eq!(chat(addr, conversation("m", &sys_two, turns)).await.2["upstream"], "b", "turn {turns} of conversation two");
        assert_eq!(chat(addr, conversation("m", &sys_one, turns)).await.2["upstream"], "a", "turn {turns} of conversation one");
    }
    // The upstream sees the router's key, never the client's.
    for hit in a.hits().into_iter().chain(b.hits()) {
        assert_eq!(hit.headers.get("authorization").unwrap(), &format!("Bearer {UPSTREAM_KEY}"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn load_is_weighed_against_each_upstreams_slots() {
    let (small, us) = mock("small", 1, 8, &["m"]).await;
    let (big, ub) = mock("big", 4, 8, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[us, ub]).await;
    let mut held = Vec::new();
    for _ in 0..5 {
        let mut body = json!({"model": "m", "messages": []});
        body["hold"] = json!(true);
        held.push(tokio::spawn(chat(addr, body)));
        let sent = held.len();
        until("the request to land", || small.count() + big.count() == sent).await;
    }
    assert_eq!((small.count(), big.count()), (1, 4), "four slots take four times the requests of one");
    small.release.add_permits(8);
    big.release.add_permits(8);
    for h in held {
        assert_eq!(h.await.unwrap().0, StatusCode::OK);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unhealthy_upstream_is_out_of_rotation_until_it_answers_again() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (b, ub) = mock("b", 8, 16, &["m"]).await;
    let (addr, fleet, _stop) = router(&[ua, ub]).await;
    a.healthy.store(false, Ordering::SeqCst);
    fleet.probe().await;
    for _ in 0..3 {
        assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "b");
    }
    let (code, _, status) = call(addr, Method::GET, "/v1/health", Some(CLIENT_KEY), "").await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!((status["upstreams"][0]["healthy"].clone(), status["upstreams"][1]["healthy"].clone()), (json!(false), json!(true)));
    assert!(!status.to_string().contains("127.0.0.1"), "the health report names no address: {status}");
    a.healthy.store(true, Ordering::SeqCst);
    fleet.probe().await;
    assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "a", "back in rotation, and first on a tie");
    b.healthy.store(false, Ordering::SeqCst);
    a.healthy.store(false, Ordering::SeqCst);
    fleet.probe().await;
    let (code, headers, _) = chat(addr, conversation("m", "s", 1)).await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert!(headers.contains_key("retry-after"));
    assert_eq!(call(addr, Method::GET, "/v1/health", Some(CLIENT_KEY), "").await.0, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_draining_upstream_takes_nothing_new_and_finishes_what_it_holds() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (b, ub) = mock("b", 8, 16, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[ua, ub]).await;
    let mut body = conversation("m", "s", 1);
    body["hold"] = json!(true);
    let in_flight = tokio::spawn(chat(addr, body));
    until("the held request to reach a", || a.count() == 1).await;
    let (code, _, answer) = call(addr, Method::POST, "/v1/upstreams/0/drain", Some(CLIENT_KEY), "").await;
    assert_eq!((code, answer["draining"].clone()), (StatusCode::OK, json!(true)));
    for _ in 0..3 {
        assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "b");
    }
    a.release.add_permits(1);
    let (code, _, answer) = in_flight.await.unwrap();
    assert_eq!((code, answer["upstream"].as_str()), (StatusCode::OK, Some("a")), "the request it held finished");
    assert_eq!(call(addr, Method::POST, "/v1/upstreams/0/resume", Some(CLIENT_KEY), "").await.0, StatusCode::OK);
    assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "a");
    assert_eq!(call(addr, Method::POST, "/v1/upstreams/9/drain", Some(CLIENT_KEY), "").await.0, StatusCode::NOT_FOUND);
    assert_eq!(b.count(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stream_passes_through_unbuffered_and_a_client_that_hangs_up_closes_the_upstream() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[ua]).await;
    let body = json!({"model": "m", "messages": [], "stream": true}).to_string();
    let resp = send(addr, Method::POST, "/v1/chat/completions", Some(CLIENT_KEY), &body, &[]).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("content-type").unwrap(), "text/event-stream");
    assert!(resp.headers().get("x-internal").is_none(), "an upstream header the API does not define stays behind");
    let mut events = Body::new(resp.into_body()).into_data_stream();
    // The upstream sends the rest only once the client has the first event: a buffering router would deadlock here.
    let first = tokio::time::timeout(Duration::from_secs(5), events.next()).await.expect("the first event arrives while the upstream still streams");
    assert_eq!(first.unwrap().unwrap(), "data: one\n\n");
    a.stream_go.notify_one();
    let mut rest = Vec::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(5), events.next()).await.expect("the rest") {
        rest.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(String::from_utf8(rest).unwrap(), "data: two\n\ndata: [DONE]\n\n");
    assert!(!a.stream_dropped.load(Ordering::SeqCst));

    // A client that goes away mid-stream: the upstream's response is dropped (its generation would be cancelled).
    let resp = send(addr, Method::POST, "/v1/chat/completions", Some(CLIENT_KEY), &body, &[]).await;
    let mut events = Body::new(resp.into_body()).into_data_stream();
    assert_eq!(events.next().await.unwrap().unwrap(), "data: one\n\n");
    drop(events);
    until("the upstream stream to be dropped", || a.stream_dropped.load(Ordering::SeqCst)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_moves_on_only_when_it_was_never_sent() {
    // Sent to the black hole (the first on a tie), which takes it and closes: the router answers 502 and never sends
    // it to `b`, because it may have run.
    let (b, ub) = mock("b", 8, 16, &["m"]).await;
    let (posts, hole) = black_hole(&["m"]).await;
    let (addr, fleet, _stop) = router(&[hole, ub.clone()]).await;
    assert_eq!(fleet.status()["upstreams"][0]["healthy"], json!(true), "the black hole answers its probes");
    let (code, headers, body) = chat(addr, conversation("m", "s", 1)).await;
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert!(!headers.contains_key("retry-after"), "nothing invites a blind retry of a request that may have run");
    assert_eq!(body["error"]["code"], "upstream_error");
    assert_eq!((posts.load(Ordering::SeqCst), b.count()), (1, 0));

    // An upstream in rotation that has stopped listening: nothing could be sent there, so the same request goes to
    // `b`, and the dead one is out of rotation at once.
    let (first, uf) = mock("first", 8, 16, &["m"]).await;
    let (addr, fleet, _stop2) = router(&[uf, ub]).await;
    assert_eq!(fleet.status()["upstreams"][0]["healthy"], json!(true));
    first.stop();
    let (code, _, body) = chat(addr, conversation("m", "s", 1)).await;
    assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("b")));
    assert_eq!(fleet.status()["upstreams"][0]["healthy"], json!(false), "a failed connect takes the upstream out at once");
    assert_eq!(first.count(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_listing_merges_every_upstream_and_a_model_goes_only_where_it_is_served() {
    let (a, ua) = mock("a", 8, 16, &["only-a", "shared"]).await;
    let (b, ub) = mock("b", 8, 16, &["only-b", "shared"]).await;
    let (addr, _fleet, _stop) = router(&[ua, ub]).await;
    let (code, _, list) = call(addr, Method::GET, "/v1/models", Some(CLIENT_KEY), "").await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(list["object"], "list");
    let ids: Vec<&str> = list["data"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["only-a", "only-b", "shared"], "each model once");
    assert_eq!(call(addr, Method::GET, "/v1/models/only-b", Some(CLIENT_KEY), "").await.2["owned_by"], "b");
    assert_eq!(call(addr, Method::GET, "/models/nope", Some(CLIENT_KEY), "").await.0, StatusCode::NOT_FOUND);
    for _ in 0..2 {
        assert_eq!(chat(addr, conversation("only-b", "s", 1)).await.2["upstream"], "b");
        assert_eq!(chat(addr, conversation("only-a", "s", 1)).await.2["upstream"], "a");
    }
    let (code, _, body) = chat(addr, conversation("nope", "s", 1)).await;
    assert_eq!((code, body["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("model_not_found")));
    let (code, _, body) = call(addr, Method::POST, "/v1/completions", Some(CLIENT_KEY), &json!({"model": "shared", "prompt": "hi"}).to_string()).await;
    assert_eq!(code, StatusCode::OK);
    assert!(body["upstream"].is_string());
    assert_eq!(a.hits().iter().chain(b.hits().iter()).filter(|h| h.path == "/v1/completions").count(), 1, "the path is kept");
}

#[tokio::test(flavor = "multi_thread")]
async fn past_the_fleets_slots_requests_wait_in_order_and_past_the_queue_bound_are_refused() {
    let (a, ua) = mock("a", 1, 256, &["m"]).await;
    let (addr, fleet, _stop) = router_over(config(&[ua]).with_queue(2, Duration::from_secs(30)), Arc::new(StaticKey::new(CLIENT_KEY))).await;
    let held_body = |n: usize| {
        let mut body = conversation("m", "s", 1);
        body["hold"] = json!(true);
        body["n"] = json!(n);
        body
    };
    // One slot: the first runs, the next two wait in the router (the upstream sees only one).
    let first = tokio::spawn(chat(addr, held_body(1)));
    until("the first request to land", || a.count() == 1).await;
    let second = tokio::spawn(chat(addr, held_body(2)));
    until("the second to queue", || fleet.status()["queued"] == json!(1)).await;
    let third = tokio::spawn(chat(addr, held_body(3)));
    until("the third to queue", || fleet.status()["queued"] == json!(2)).await;
    assert_eq!(a.count(), 1, "the upstream is sent no more than its decode slots");
    let status = fleet.status();
    assert_eq!((status["capacity"].clone(), status["in_flight"].clone(), status["upstreams"][0]["capacity"].clone()), (json!(1), json!(1), json!(1)));

    // The queue is full: refused at once.
    let started = std::time::Instant::now();
    let (code, headers, body) = chat(addr, conversation("m", "s", 1)).await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert!(headers.contains_key("retry-after"));
    assert!(body["error"]["message"].as_str().unwrap().contains("queue"));
    assert!(started.elapsed() < Duration::from_secs(2), "refused, not queued");

    // Slots free one at a time, and the waiting requests take them in arrival order.
    a.release.add_permits(1);
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    until("the second to be sent", || a.count() == 2).await;
    assert_eq!(a.hits()[1].body["n"], 2, "first come, first served");
    a.release.add_permits(2);
    assert_eq!(second.await.unwrap().0, StatusCode::OK);
    assert_eq!(third.await.unwrap().0, StatusCode::OK);
    assert_eq!(a.hits()[2].body["n"], 3);
    assert_eq!(fleet.status()["queued"], json!(0));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_queued_request_leaves_at_its_deadline_or_when_its_client_goes_away() {
    let (a, ua) = mock("a", 1, 256, &["m"]).await;
    let (addr, fleet, _stop) = router_over(config(&[ua]).with_queue(8, Duration::from_millis(300)), Arc::new(StaticKey::new(CLIENT_KEY))).await;
    let mut held = conversation("m", "s", 1);
    held["hold"] = json!(true);
    let first = tokio::spawn(chat(addr, held));
    until("the first request to land", || a.count() == 1).await;
    let (code, headers, _) = chat(addr, conversation("m", "s", 1)).await;
    assert_eq!(code, StatusCode::TOO_MANY_REQUESTS, "no slot within the deadline");
    assert!(headers.contains_key("retry-after"));
    assert_eq!(fleet.status()["queued"], json!(0));

    // A client that hangs up while queued leaves the queue and is never sent.
    let gone = tokio::spawn(chat(addr, conversation("m", "s", 1)));
    until("it to queue", || fleet.status()["queued"] == json!(1)).await;
    gone.abort();
    until("it to leave the queue", || fleet.status()["queued"] == json!(0)).await;
    a.release.add_permits(1);
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    assert_eq!(a.count(), 1, "the abandoned request was never sent");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_key_keeps_a_conversation_on_one_upstream() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (b, ub) = mock("b", 8, 16, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[ua, ub]).await;
    // Short prompts share no useful prefix, so without a key every idle request goes to the first upstream on a tie.
    let mut held = conversation("m", "s", 1);
    held["hold"] = json!(true);
    let busy = tokio::spawn(chat(addr, held));
    until("a to be busy", || a.count() == 1).await;
    let mut keyed = conversation("m", "s", 1);
    keyed["prompt_cache_key"] = json!("session-7");
    assert_eq!(chat(addr, keyed.clone()).await.2["upstream"], "b", "least loaded");
    a.release.add_permits(1);
    busy.await.unwrap();
    for _ in 0..3 {
        assert_eq!(chat(addr, keyed.clone()).await.2["upstream"], "b", "the session stays where it started");
        assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "a");
    }
    keyed["prompt_cache_key"] = json!(7);
    assert_eq!(chat(addr, keyed).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(b.hits().iter().filter(|h| h.body["prompt_cache_key"] == "session-7").count(), 4, "the key is forwarded");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_adapter_is_routed_only_to_the_upstreams_serving_it_and_the_status_reports_rates() {
    let (a, ua) = mock("a", 8, 16, &["base", "base:tuned-a"]).await;
    let (b, ub) = mock("b", 8, 16, &["base", "base:tuned-b", "base:tuned-a"]).await;
    let (addr, fleet, _stop) = router(&[ua, ub]).await;
    for _ in 0..3 {
        assert_eq!(chat(addr, conversation("base:tuned-b", "s", 1)).await.2["upstream"], "b");
        assert!(chat(addr, conversation("base:tuned-a", "s", 1)).await.2["upstream"].is_string());
    }
    let ids: Vec<String> = call(addr, Method::GET, "/v1/models", Some(CLIENT_KEY), "").await.2["data"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids, ["base", "base:tuned-a", "base:tuned-b"]);
    assert_eq!(a.hits().iter().filter(|h| h.body["model"] == "base:tuned-b").count(), 0);

    // Two probes apart, the status carries each upstream's completion rate and prefix hit rate from its counters.
    tokio::time::sleep(Duration::from_millis(50)).await;
    fleet.probe().await;
    let (code, _, status) = call(addr, Method::GET, "/v1/metrics", Some(CLIENT_KEY), "").await;
    assert_eq!(code, StatusCode::OK);
    let busiest = &status["upstreams"][1];
    assert!(busiest["tokens_per_s"].as_f64().unwrap() > 0.0, "{status}");
    assert_eq!(busiest["prefix_hit_rate"], json!(0.5), "the mock serves half of every prompt from cache");
    assert!(status["tokens_per_s"].as_f64().unwrap() >= busiest["tokens_per_s"].as_f64().unwrap());
    assert_eq!((status["capacity"].clone(), status["queued"].clone(), status["in_flight"].clone()), (json!(16), json!(0), json!(0)));
    let _ = b;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_router_is_attack_surface_keys_bodies_and_headers_are_checked() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let callers = Callers::new(vec![
        CallerSpec { name: "app".into(), role: Role::Inference, key: "sk-app".into(), max_in_flight: 1 },
        CallerSpec { name: "ops".into(), role: Role::Admin, key: "sk-ops".into(), max_in_flight: 4 },
    ])
    .unwrap();
    let (addr, _fleet, _stop) = router_with(&[ua], Arc::new(callers)).await;

    // Every route, the fallback included, needs a key: no route can be discovered without one.
    for (method, path) in [(Method::GET, "/v1/models"), (Method::GET, "/v1/health"), (Method::POST, "/v1/chat/completions"), (Method::POST, "/v1/upstreams/0/drain"), (Method::GET, "/no/such/route")] {
        for key in [None, Some("wrong"), Some("")] {
            let (code, _, body) = call(addr, method.clone(), path, key, "{}").await;
            assert_eq!(code, StatusCode::UNAUTHORIZED, "{method} {path} with {key:?}");
            assert!(!body.to_string().contains("sk-"), "no key in an error: {body}");
        }
    }
    assert_eq!(call(addr, Method::GET, "/no/such/route", Some("sk-app"), "").await.0, StatusCode::NOT_FOUND);
    assert_eq!(call(addr, Method::GET, "/v1/jobs", Some("sk-app"), "").await.0, StatusCode::NOT_FOUND, "stateful routes are not forwarded");

    // Draining is the operator's.
    assert_eq!(call(addr, Method::POST, "/v1/upstreams/0/drain", Some("sk-app"), "").await.0, StatusCode::FORBIDDEN);
    assert_eq!(call(addr, Method::POST, "/v1/upstreams/0/resume", Some("sk-ops"), "").await.0, StatusCode::OK);

    // The body is checked at the boundary before anything is sent.
    let key = Some("sk-app");
    for bad in ["[1,2]", "not json", r#"{"messages": []}"#, r#"{"model": 7}"#, r#"{"model": ""}"#, r#"{"model": "m", "model": "other"}"#] {
        let (code, _, body) = call(addr, Method::POST, "/v1/chat/completions", key, bad).await;
        assert_eq!(code, StatusCode::BAD_REQUEST, "{bad}");
        assert!(body["error"]["message"].is_string());
    }
    let long = json!({"model": "x".repeat(300)}).to_string();
    assert_eq!(call(addr, Method::POST, "/v1/chat/completions", key, &long).await.0, StatusCode::BAD_REQUEST);
    let huge = json!({"model": "m", "messages": [{"role": "user", "content": "x".repeat(apiserve::MAX_BODY_BYTES)}]}).to_string();
    assert_eq!(call(addr, Method::POST, "/v1/chat/completions", key, &huge).await.0, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(a.count(), 0, "no refused request reached the upstream");

    // Only the headers the router sets reach the upstream, and only API headers come back.
    let body = json!({"model": "m", "messages": []}).to_string();
    let resp = send(addr, Method::POST, "/v1/chat/completions", key, &body, &[("x-forwarded-for", "10.9.9.9"), ("cookie", "c=1"), ("x-custom", "1"), ("accept", "application/json")]).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get("set-cookie").is_none() && resp.headers().get("x-internal").is_none());
    let client_id = resp.headers().get("x-request-id").expect("the client sees the request id").to_str().unwrap().to_string();
    let hit = a.hits().pop().unwrap();
    let names: Vec<&str> = hit.headers.keys().map(|k| k.as_str()).collect();
    for name in &names {
        assert!(["host", "content-type", "content-length", "authorization", "accept", "x-request-id"].contains(name), "the upstream was sent {name}");
    }
    let id = hit.headers.get("x-request-id").expect("the router's request id").to_str().unwrap().to_string();
    assert!(id.starts_with("route-") && id == client_id, "one id for the request on both sides");
    assert_eq!(hit.headers.get("authorization").unwrap(), &format!("Bearer {UPSTREAM_KEY}"));
    assert_eq!(hit.body, serde_json::from_str::<Value>(&body).unwrap(), "the body is forwarded as sent");

    // A named caller holds at most its own share in flight.
    let mut held = conversation("m", "s", 1);
    held["hold"] = json!(true);
    let first = tokio::spawn(chat_key(addr, "sk-app", held));
    until("the held request to land", || a.count() == 2).await;
    let (code, headers, _) = chat_as(addr, "sk-app").await;
    assert_eq!(code, StatusCode::TOO_MANY_REQUESTS);
    assert!(headers.contains_key("retry-after"));
    a.release.add_permits(1);
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    assert_eq!(chat_as(addr, "sk-app").await.0, StatusCode::OK);
}

async fn chat_as(addr: SocketAddr, key: &'static str) -> (StatusCode, HeaderMap, Value) {
    chat_key(addr, key, conversation("m", "s", 1)).await
}

async fn chat_key(addr: SocketAddr, key: &'static str, body: Value) -> (StatusCode, HeaderMap, Value) {
    call(addr, Method::POST, "/v1/chat/completions", Some(key), &body.to_string()).await
}
