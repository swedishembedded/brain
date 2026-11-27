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
    /// The router's side of the connection the request came on.
    peer: SocketAddr,
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
    /// Free KV tokens its health reports (none when `None`).
    kv_free: Mutex<Option<u64>>,
    /// The server task: aborting it closes the listener, so connecting is refused.
    server: Mutex<Option<tokio::task::AbortHandle>>,
    /// The process identity its health reports: a test changes it to stand for a restart.
    instance: Mutex<String>,
    /// Jobs its health reports waiting on it for a lane.
    queue_depth: AtomicUsize,
    /// Requests it finished: what its served counters count.
    answered: AtomicUsize,
    /// Its device hangs: a request is taken and never finished (a stream sends its first event first), while its
    /// health still answers.
    wedged: AtomicBool,
    /// Fields its card for a model carries beyond the id (its decode slots for that model, whether it is resident).
    card_extra: Mutex<std::collections::HashMap<String, Value>>,
    /// Its health says every card runs a workload that has not lent it.
    serving_held: AtomicBool,
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
    let answered = m.answered.load(Ordering::SeqCst);
    let completion = 10 * answered as u64;
    let served = json!({"requests": answered, "prompt_tokens": 2 * completion, "completion_tokens": completion, "cached_prompt_tokens": completion});
    let kv_free = *m.kv_free.lock().unwrap();
    let instance = m.instance.lock().unwrap().clone();
    let queue_depth = m.queue_depth.load(Ordering::SeqCst);
    let held = m.serving_held.load(Ordering::SeqCst);
    let free_slots = if held { 0 } else { m.slots };
    Json(json!({"status": "ok", "instance": instance, "decode_slots": m.slots, "free_slots": free_slots, "serving_held": held, "kv_free_tokens": kv_free, "edge_concurrency": m.edge, "queue_depth": queue_depth, "served": served})).into_response()
}

async fn mock_models(State(m): State<Arc<Mock>>) -> Json<Value> {
    let extra = m.card_extra.lock().unwrap().clone();
    let data: Vec<Value> = m
        .models
        .iter()
        .map(|id| {
            let mut card = json!({"id": id, "object": "model", "owned_by": m.name});
            if let Some(Value::Object(fields)) = extra.get(id) {
                card.as_object_mut().unwrap().extend(fields.clone());
            }
            card
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn mock_generate(State(m): State<Arc<Mock>>, axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>, uri: axum::http::Uri, headers: HeaderMap, body: Bytes) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    m.hits.lock().unwrap().push(Hit { path: uri.path().to_string(), peer, body: body.clone(), headers });
    if m.wedged.load(Ordering::SeqCst) {
        if body["stream"] != json!(true) {
            std::future::pending::<()>().await;
        }
        let m2 = m.clone();
        let events = async_stream::stream! {
            let _flag = DropFlag(m2, false);
            yield Ok::<_, std::convert::Infallible>(Bytes::from("data: one\n\n"));
            std::future::pending::<()>().await;
        };
        return Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(events)).unwrap();
    }
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
            m2.answered.fetch_add(1, Ordering::SeqCst);
        };
        return Response::builder().header("content-type", "text/event-stream").header("x-internal", "secret").body(Body::from_stream(events)).unwrap();
    }
    m.answered.fetch_add(1, Ordering::SeqCst);
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
        kv_free: Mutex::default(),
        instance: Mutex::new("mock0".into()),
        queue_depth: AtomicUsize::new(0),
        answered: AtomicUsize::new(0),
        wedged: AtomicBool::new(false),
        card_extra: Mutex::default(),
        serving_held: AtomicBool::new(false),
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
    let server = tokio::spawn(async move { axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await });
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
    assert_eq!(chat(addr, conversation("m", "a new conversation", 1)).await.2["upstream"], "a", "back in rotation, and first on a tie");
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
    // Another conversation than the one `a` holds, so nothing but load and the key decide.
    let mut keyed = conversation("m", "the keyed session", 1);
    keyed["prompt_cache_key"] = json!("session-7");
    assert_eq!(chat(addr, keyed.clone()).await.2["upstream"], "b", "least loaded");
    a.release.add_permits(1);
    busy.await.unwrap();
    for _ in 0..3 {
        assert_eq!(chat(addr, keyed.clone()).await.2["upstream"], "b", "the session stays where it started");
        assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "a", "the unkeyed conversation stays where it is cached");
    }
    keyed["prompt_cache_key"] = json!(7);
    assert_eq!(chat(addr, keyed).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(b.hits().iter().filter(|h| h.body["prompt_cache_key"] == "session-7").count(), 4, "the key is forwarded");
}

/// A server's `decode_slots` is an admission bound, not what its card decodes at speed: a card advertising thousands
/// of slots is busy long before they fill. A batch on one upstream therefore spreads onto idle upstreams where the
/// model is just as resident, even when the busy one holds every request's prefix and the conversation's session: a
/// cached prompt is worth a few requests of load, never a whole idle card. Moved, a session stays where it went.
#[tokio::test(flavor = "multi_thread")]
async fn a_batch_on_one_upstream_spreads_onto_idle_ones_whatever_prefix_or_session_it_holds() {
    let mut mocks = Vec::new();
    for name in ["a", "b", "c", "d"] {
        mocks.push(mock(name, 3466, 4096, &["m"]).await);
    }
    let urls: Vec<String> = mocks.iter().map(|(_, u)| u.clone()).collect();
    for (m, _) in &mocks[1..] {
        m.healthy.store(false, Ordering::SeqCst);
    }
    let (addr, fleet, _stop) = router(&urls).await;
    let a = &mocks[0].0;
    let system = "You are the agent. ".repeat(400);
    let session = |turns: usize| {
        let mut body = conversation("m", &system, turns);
        body["prompt_cache_key"] = json!("agent-0");
        body
    };
    // While `a` is the only upstream, the session starts there and a batch of 64 sharing its system prompt runs there.
    assert_eq!(chat(addr, session(1)).await.2["upstream"], "a");
    let mut batch = Vec::new();
    for agent in 0..64 {
        let mut body = conversation("m", &system, 1);
        body["messages"][1]["content"] = json!(format!("task {agent}"));
        body["hold"] = json!(true);
        batch.push(tokio::spawn(chat(addr, body)));
    }
    until("the batch to run on a", || a.count() == 65).await;
    // The other cards come up idle, with the model resident (every card serves it alike).
    for (m, _) in &mocks[1..] {
        m.healthy.store(true, Ordering::SeqCst);
    }
    fleet.probe().await;
    // The session follows the idle card instead of queueing its turn behind the batch, and stays there.
    assert_eq!(chat(addr, session(2)).await.2["upstream"], "b", "the session leaves the loaded upstream");
    assert_eq!(fleet.status()["upstreams"][1]["balance_moves"], 1, "the move is counted on the upstream it went to");
    assert_eq!(chat(addr, session(3)).await.2["upstream"], "b", "and stays where it went");
    assert_eq!(fleet.status()["upstreams"][1]["balance_moves"], 1, "a session at home is no move");
    // A new conversation sharing the batch's system prompt goes to an idle card too.
    let fresh = chat(addr, conversation("m", &system, 1)).await.2["upstream"].clone();
    assert_ne!(fresh, "a", "a cached prompt does not outweigh an idle card");
    assert_eq!(a.count(), 65, "nothing more was sent to the loaded upstream");
    assert_eq!(fleet.status()["upstreams"][0]["balance_moves"], 0, "nothing moved onto the loaded upstream");
    a.release.add_permits(64);
    for request in batch {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
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

#[tokio::test(flavor = "multi_thread")]
async fn connections_are_kept_alive_and_reused() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (addr, fleet, _stop) = router(&[ua]).await;
    for _ in 0..6 {
        assert_eq!(chat(addr, conversation("m", "s", 1)).await.0, StatusCode::OK);
        until("the connection to go back to the pool", || fleet.status()["upstreams"][0]["idle_connections"] == json!(1)).await;
    }
    let peers: std::collections::HashSet<SocketAddr> = a.hits().iter().map(|h| h.peer).collect();
    assert_eq!(peers.len(), 1, "six requests in turn share one connection");
    // A streamed answer read to its end gives its connection back too.
    let body = json!({"model": "m", "messages": [], "stream": true}).to_string();
    let resp = send(addr, Method::POST, "/v1/chat/completions", Some(CLIENT_KEY), &body, &[]).await;
    let mut events = Body::new(resp.into_body()).into_data_stream();
    events.next().await.unwrap().unwrap();
    a.stream_go.notify_one();
    while events.next().await.is_some() {}
    until("the streamed connection to go back", || fleet.status()["upstreams"][0]["idle_connections"] == json!(1)).await;
    assert_eq!(a.hits().iter().map(|h| h.peer).collect::<std::collections::HashSet<_>>().len(), 1);
}

/// An upstream that answers every request on a kept-alive connection, then closes it shortly after: the router must
/// not send a request down a connection the upstream has closed.
async fn closes_after_answering(models: &[&str]) -> (Arc<AtomicUsize>, String) {
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
                loop {
                    match s.read(&mut buf[n..]).await {
                        Ok(0) | Err(_) => return,
                        Ok(k) => n += k,
                    }
                    let Some(end) = buf[..n].windows(4).position(|w| w == b"\r\n\r\n") else { continue };
                    let head = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                    let length: usize = head.lines().find_map(|l| l.strip_prefix("content-length:")).map_or(0, |v| v.trim().parse().unwrap());
                    if n < end + 4 + length {
                        continue;
                    }
                    let answer = if head.starts_with("get /v1/health") {
                        json!({"decode_slots": 4, "edge_concurrency": 8}).to_string()
                    } else if head.starts_with("get /v1/models") {
                        json!({"data": models}).to_string()
                    } else {
                        seen.fetch_add(1, Ordering::SeqCst);
                        json!({"upstream": "closer"}).to_string()
                    };
                    let _ = s.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{answer}", answer.len()).as_bytes()).await;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    return;
                }
            });
        }
    });
    (posts, format!("http://{addr}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_connection_the_upstream_closed_is_not_reused() {
    let (posts, url) = closes_after_answering(&["m"]).await;
    let (addr, _fleet, _stop) = router(&[url]).await;
    for n in 1..=4 {
        let (code, _, body) = chat(addr, conversation("m", "s", 1)).await;
        assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("closer")), "request {n}");
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    assert_eq!(posts.load(Ordering::SeqCst), 4, "each sent once");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drain_deadline_cuts_off_what_the_upstream_still_runs() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[ua]).await;
    let body = json!({"model": "m", "messages": [], "stream": true}).to_string();
    let resp = send(addr, Method::POST, "/v1/chat/completions", Some(CLIENT_KEY), &body, &[]).await;
    let mut events = Body::new(resp.into_body()).into_data_stream();
    assert_eq!(events.next().await.unwrap().unwrap(), "data: one\n\n");
    let (code, _, answer) = call(addr, Method::POST, "/v1/upstreams/0/drain", Some(CLIENT_KEY), r#"{"deadline_ms": 200}"#).await;
    assert_eq!((code, answer["deadline_ms"].clone()), (StatusCode::OK, json!(200)));
    let ended = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(chunk) = events.next().await {
            if chunk.is_err() {
                return true;
            }
        }
        false
    })
    .await
    .expect("the stream ends at the deadline");
    assert!(ended, "the cut stream ends in an error, not a silent end");
    until("the upstream to see its client go", || a.stream_dropped.load(Ordering::SeqCst)).await;
    assert_eq!(call(addr, Method::POST, "/v1/upstreams/0/drain", Some(CLIENT_KEY), r#"{"deadline_ms": 0}"#).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(call(addr, Method::POST, "/v1/upstreams/0/drain", Some(CLIENT_KEY), r#"{"deadline": 5}"#).await.0, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_upstream_that_queues_work_without_running_it_gets_none_while_another_has_room() {
    let (a, ua) = mock("a", 8, 16, &["m"]).await;
    let (_b, ub) = mock("b", 8, 16, &["m"]).await;
    let (addr, fleet, _stop) = router(&[ua, ub]).await;
    let answered = |body: Value| async move { tokio::time::timeout(Duration::from_secs(5), chat(addr, body)).await.expect("answered, not stranded behind a queue that does not move") };

    // A conversation cached on `a` (first on a tie); then `a` reports jobs waiting while its slots are free, and takes
    // requests it never answers. The prefix it holds must not pull the next turn into that queue.
    let cached = "You are agent one. ".repeat(200);
    assert_eq!(chat(addr, conversation("m", &cached, 1)).await.2["upstream"], "a");
    a.queue_depth.store(5, Ordering::SeqCst);
    a.wedged.store(true, Ordering::SeqCst);
    fleet.probe().await;
    let (code, _, body) = answered(conversation("m", &cached, 2)).await;
    assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("b")));

    // A session whose upstream restarted and came back with such a queue: the session follows the free upstream.
    a.queue_depth.store(0, Ordering::SeqCst);
    a.wedged.store(false, Ordering::SeqCst);
    fleet.probe().await;
    let keyed = |turns: usize| {
        let mut body = conversation("m", &"You are agent two. ".repeat(200), turns);
        body["prompt_cache_key"] = json!("agent-2");
        body
    };
    assert_eq!(chat(addr, keyed(1)).await.2["upstream"], "a");
    *a.instance.lock().unwrap() = "mock1".into();
    a.queue_depth.store(10, Ordering::SeqCst);
    a.wedged.store(true, Ordering::SeqCst);
    fleet.probe().await;
    let status = fleet.status();
    assert_eq!((status["upstreams"][0]["healthy"].clone(), status["upstreams"][0]["queue_depth"].clone()), (json!(true), json!(10)));
    let hung = a.count();
    let (code, _, body) = answered(keyed(2)).await;
    assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("b")));
    assert_eq!(a.count(), hung, "nothing more was sent into the queue");

    // Its queue drained, `a` takes new work again.
    a.queue_depth.store(0, Ordering::SeqCst);
    a.wedged.store(false, Ordering::SeqCst);
    fleet.probe().await;
    assert_eq!(chat(addr, conversation("m", "a new conversation", 1)).await.2["upstream"], "a");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_upstream_that_stops_making_progress_fails_what_it_holds_and_leaves_rotation_until_it_restarts() {
    let (w, uw) = mock("w", 8, 16, &["m"]).await;
    let (b, ub) = mock("b", 8, 16, &["m"]).await;
    let stall = Duration::from_millis(300);
    let (addr, fleet, _stop) = router_over(config(&[uw, ub]).with_stall_timeout(stall), Arc::new(StaticKey::new(CLIENT_KEY))).await;
    // Its device hangs: it takes requests and never finishes them, while its health still says every slot is free.
    w.wedged.store(true, Ordering::SeqCst);
    let keyed = |stream: bool| {
        let mut body = conversation("m", "s", 1);
        body["prompt_cache_key"] = json!("agent");
        body["stream"] = json!(stream);
        body
    };
    let waiting = tokio::spawn(chat(addr, keyed(false)));
    until("the first request to reach w", || w.count() == 1).await;
    let resp = send(addr, Method::POST, "/v1/chat/completions", Some(CLIENT_KEY), &keyed(true).to_string(), &[]).await;
    assert_eq!(resp.status(), StatusCode::OK, "the session's stream went to w and started");
    let mut events = Body::new(resp.into_body()).into_data_stream();
    assert_eq!(events.next().await.unwrap().unwrap(), "data: one\n\n");
    assert_eq!(w.count(), 2);

    // Probes go on answering; past the stall bound with nothing finished or streamed, w is taken out of rotation and
    // what it holds is failed loudly.
    fleet.probe().await;
    assert_eq!(fleet.status()["upstreams"][0]["stalled"], json!(false), "within the bound");
    tokio::time::sleep(stall + Duration::from_millis(100)).await;
    fleet.probe().await;
    let (code, _, body) = tokio::time::timeout(Duration::from_secs(5), waiting).await.expect("failed, not stranded").unwrap();
    assert_eq!(code, StatusCode::GATEWAY_TIMEOUT);
    assert!(body["error"]["message"].as_str().unwrap().contains("no progress"), "{body}");
    let ended = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(chunk) = events.next().await {
            if chunk.is_err() {
                return true;
            }
        }
        false
    })
    .await
    .expect("the stream is cut");
    assert!(ended, "the cut stream ends in an error, not a silent end");
    let status = fleet.status();
    assert_eq!((status["upstreams"][0]["stalled"].clone(), status["upstreams"][0]["in_flight"].clone()), (json!(true), json!(0)));
    assert_eq!((status["upstreams"][0]["stalls"].clone(), status["upstreams"][1]["stalls"].clone()), (json!(1), json!(0)), "the stall is counted");
    assert_eq!(status["upstreams"][0]["healthy"], json!(true), "its health still answers");

    // Answering probes do not put it back; the session moves to b.
    fleet.probe().await;
    for _ in 0..2 {
        assert_eq!(chat(addr, keyed(false)).await.2["upstream"], "b");
    }
    assert_eq!(w.count(), 2);

    // A restart (another instance) does.
    w.wedged.store(false, Ordering::SeqCst);
    *w.instance.lock().unwrap() = "mock1".into();
    fleet.probe().await;
    assert_eq!(fleet.status()["upstreams"][0]["stalled"], json!(false));
    assert_eq!(chat(addr, conversation("m", "a new conversation", 1)).await.2["upstream"], "w");
    let _ = b;
}

/// A tokenizer for the tests: one token per word.
struct Words;
impl route::PromptTokenizer for Words {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace().map(|w| w.bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32))).collect()
    }
}
struct WordsFor(&'static str);
impl route::TokenizerSource for WordsFor {
    fn tokenizer(&self, base: &str) -> Option<Arc<dyn route::PromptTokenizer>> {
        (base == self.0).then(|| Arc::new(Words) as Arc<dyn route::PromptTokenizer>)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn prompts_are_keyed_by_tokens_and_kv_headroom_steers_the_choice() {
    let (a, ua) = mock("a", 8, 16, &["m", "plain"]).await;
    let (b, ub) = mock("b", 8, 16, &["m", "plain"]).await;
    let config = config(&[ua, ub]).with_tokenizers(Some(Arc::new(WordsFor("m"))));
    let (addr, fleet, _stop) = router_over(config, Arc::new(StaticKey::new(CLIENT_KEY))).await;
    let long = |tag: &str| format!("{tag} ") + &"word ".repeat(400);

    // Conversation one is held on `a`, so conversation two starts on `b`.
    let mut held = conversation("m", &long("one"), 1);
    held["hold"] = json!(true);
    let first = tokio::spawn(chat(addr, held));
    until("a to be busy", || a.count() == 1).await;
    assert_eq!(chat(addr, conversation("m", &long("two"), 1)).await.2["upstream"], "b");
    a.release.add_permits(1);
    first.await.unwrap();

    // Turn two of conversation two, written with other key order and spacing: the same tokens, so the same upstream.
    let turn = conversation("m", &long("two"), 2);
    let messages: Vec<String> = turn["messages"].as_array().unwrap().iter().map(|m| format!(r#"{{ "content" : {}, "role" : {} }}"#, m["content"], m["role"])).collect();
    let respelled = format!(r#"{{"messages": [ {} ], "model": "m"}}"#, messages.join(" , "));
    let (code, _, body) = call(addr, Method::POST, "/v1/chat/completions", Some(CLIENT_KEY), &respelled).await;
    assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("b")));

    // `b` holds conversation two, but reports too little free KV for it: `a` takes it.
    *b.kv_free.lock().unwrap() = Some(8);
    *a.kv_free.lock().unwrap() = Some(1 << 20);
    fleet.probe().await;
    assert_eq!(chat(addr, conversation("m", &long("two"), 3)).await.2["upstream"], "a");

    // A model without a tokenizer is keyed by text, and the metrics say how each was keyed.
    assert_eq!(chat(addr, conversation("plain", "s", 1)).await.0, StatusCode::OK);
    let keying = fleet.status()["keying"].clone();
    assert_eq!((keying["token_keyed_requests"].clone(), keying["text_keyed_requests"].clone()), (json!(4), json!(1)), "{keying}");
    assert!(keying["memo_tokens"].as_u64().unwrap() > 0, "resent messages came from the memo: {keying}");
}

#[test]
fn overload_past_the_listeners_default_cap_is_answered_not_reset() {
    // The router's own listener must hold every request its upstreams run and its queue holds: a connection over
    // the cap is closed unanswered. With a cap of 2, one slot and a queue of 2, the fourth request is still answered.
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    let (a, ua) = rt.block_on(mock("a", 1, 256, &["m"]));
    let fleet = Fleet::new(config(&[ua]).with_queue(2, Duration::from_secs(30))).unwrap();
    let addr = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let (trigger, shutdown) = brain_shutdown::channel();
    let limits = ConnectionLimits { max_connections: 2, ..ConnectionLimits::default() };
    let opts = route::RouteOpts { shutdown: Some(shutdown), limits, ..route::RouteOpts::default() };
    let serving = std::thread::spawn(move || route::serve(fleet, vec![addr], Arc::new(StaticKey::new(CLIENT_KEY)), opts));
    rt.block_on(async {
        let mut up = false;
        for _ in 0..200 {
            if TcpStream::connect(addr).await.is_ok() {
                up = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(up, "the router listens");
        let mut held = Vec::new();
        for n in 0..3 {
            let mut body = conversation("m", "s", 1);
            body["hold"] = json!(true);
            held.push(tokio::spawn(chat(addr, body)));
            if n == 0 {
                until("the first to run", || a.count() == 1).await;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        let (code, headers, _) = chat(addr, conversation("m", "s", 1)).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "the overflow is answered");
        assert!(headers.contains_key("retry-after"));
        a.release.add_permits(3);
        for h in held {
            assert_eq!(h.await.unwrap().0, StatusCode::OK);
        }
    });
    trigger.fire();
    serving.join().unwrap().unwrap();
}

/// A server decodes a fixed number of sequences of each model it serves, and its health's `decode_slots` sums them
/// over every model it could load: an upstream is sent at most a model's own slots of that model (its card's
/// `decode_slots`), so a conversation whose prefix it holds goes elsewhere once that model is full there.
#[tokio::test(flavor = "multi_thread")]
async fn an_upstream_takes_at_most_its_models_own_slots_of_that_model() {
    let (a, ua) = mock("a", 100, 200, &["m", "other"]).await;
    let (_b, ub) = mock("b", 100, 200, &["m", "other"]).await;
    for mock in [&a, &_b] {
        mock.card_extra.lock().unwrap().insert("m".into(), json!({"decode_slots": 1, "resident": true}));
    }
    let (addr, _fleet, _stop) = router(&[ua, ub]).await;
    let system = "You are the agent. ".repeat(200);
    let mut held = conversation("m", &system, 1);
    held["hold"] = json!(true);
    let first = tokio::spawn(chat(addr, held));
    until("the held turn to reach a", || a.count() == 1).await;
    // `a` holds this conversation's prefix and has 99 free slots in all, but none for `m`.
    let (code, _, body) = chat(addr, conversation("m", &system, 2)).await;
    assert_eq!((code, body["upstream"].as_str()), (StatusCode::OK, Some("b")));
    a.release.add_permits(1);
    assert_eq!(first.await.unwrap().2["upstream"], "a");
}

/// A server holds the models it serves on its cards by demand, and loading another one evicts what it held: a request
/// goes to an upstream where its model is resident while one has a free slot for it, so two models sharing a fleet
/// settle on cards of their own instead of each request swapping a card's model.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_goes_where_its_model_is_resident() {
    let (a, ua) = mock("a", 8, 16, &["m1", "m2"]).await;
    let (b, ub) = mock("b", 8, 16, &["m1", "m2"]).await;
    a.card_extra.lock().unwrap().extend([("m1".to_string(), json!({"decode_slots": 4, "resident": true})), ("m2".to_string(), json!({"decode_slots": 4, "resident": false}))]);
    b.card_extra.lock().unwrap().extend([("m1".to_string(), json!({"decode_slots": 4, "resident": false})), ("m2".to_string(), json!({"decode_slots": 4, "resident": true}))]);
    let (addr, _fleet, _stop) = router(&[ua, ub]).await;
    for turn in 1..=3 {
        assert_eq!(chat(addr, conversation("m2", &format!("agent {turn}"), 1)).await.2["upstream"], "b", "m2, turn {turn}");
        assert_eq!(chat(addr, conversation("m1", &format!("agent {turn}"), 1)).await.2["upstream"], "a", "m1, turn {turn}");
    }
}

/// Residency must not cost a fleet its cards: a model whose resident upstreams carry a real batch (at least 8 requests
/// for it, however many slots they advertise) also goes to an upstream where it is not loaded once that upstream is
/// completely idle, since loading it there only evicts a model nobody is using. An upstream busy with another model is never used for it,
/// whatever load the model's own upstreams carry: that would evict running work.
#[tokio::test(flavor = "multi_thread")]
async fn a_loaded_model_spills_onto_an_idle_upstream_but_never_onto_a_busy_one() {
    let models = ["m1", "m2", "m3"];
    // Advertised slots are an admission figure, far more than a card decodes at speed: the bound must not read them.
    let (a, ua) = mock("a", 3466, 4096, &models).await;
    let (b, ub) = mock("b", 3466, 4096, &models).await;
    let (c, uc) = mock("c", 3466, 4096, &models).await;
    for (mock, resident) in [(&a, "m1"), (&b, "m3"), (&c, "m2")] {
        mock.card_extra.lock().unwrap().extend(models.iter().map(|m| (m.to_string(), json!({"decode_slots": 3466, "resident": *m == resident}))));
    }
    let (addr, fleet, _stop) = router(&[ua, ub, uc]).await;
    let held = |model: &str, agent: usize| {
        let mut body = conversation(model, &format!("agent {agent}"), 1);
        body["hold"] = json!(true);
        tokio::spawn(chat(addr, body))
    };
    // `b` and `c` each run their own model.
    let mut running = vec![held("m3", 100), held("m2", 101)];
    until("b and c to run their own models", || b.count() == 1 && c.count() == 1).await;
    // `m1` loads `a` to the spill load and past it: every upstream where it is not loaded is busy, so it stays.
    for agent in 0..9 {
        running.push(held("m1", agent));
        until("the m1 batch to reach a", || a.count() == agent + 1).await;
    }
    assert_eq!((b.count(), c.count()), (1, 1), "a busy upstream took a model it does not hold");
    // `b` finishes its own work and is idle: the next `m1` request spills there, while `c` stays out of it.
    b.release.add_permits(1);
    assert_eq!(running.remove(0).await.unwrap().2["upstream"], "b");
    until("b to hold nothing", || fleet.status()["upstreams"][1]["in_flight"] == 0).await;
    running.push(held("m1", 9));
    until("the spilled m1 request to reach b", || b.count() == 2).await;
    assert_eq!(b.hits()[1].body["model"], "m1");
    let spills: Vec<Value> = fleet.status()["upstreams"].as_array().unwrap().iter().map(|u| u["spills"].clone()).collect();
    assert_eq!(spills, [json!(0), json!(1), json!(0)], "the router counts the spill on the upstream it went to");
    assert_eq!((a.count(), c.count()), (9, 1));
    a.release.add_permits(9);
    b.release.add_permits(1);
    c.release.add_permits(1);
    for request in running {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
}

/// Where `requests` held requests for `model` land on `mocks`, counted per upstream: they are sent at once and the
/// count is read once all of them have reached an upstream. They stay held until the test releases them.
async fn burst(addr: std::net::SocketAddr, mocks: &[Arc<Mock>], model: &str, requests: usize) -> (Vec<usize>, Vec<tokio::task::JoinHandle<(StatusCode, HeaderMap, Value)>>) {
    let before: Vec<usize> = mocks.iter().map(|m| m.count()).collect();
    let mut held = Vec::new();
    for agent in 0..requests {
        let mut body = conversation(model, &format!("agent {agent}"), 1);
        body["hold"] = json!(true);
        held.push(tokio::spawn(chat(addr, body)));
    }
    until("every request of the burst to reach an upstream", || mocks.iter().zip(&before).map(|(m, b)| m.count() - b).sum::<usize>() == requests).await;
    (mocks.iter().zip(&before).map(|(m, b)| m.count() - b).collect(), held)
}

/// The production pattern of 2026-11-27 (knowledge #609): 16 upstreams advertising 3466 slots each, a 256-request
/// burst for the 27B, which is resident on one of them, while every other card works for another router (each reports
/// jobs waiting, the higher-index cards fewer). With the model's own bound on its card (`decode_slots`, what a server
/// advertises for every model), the hot upstream takes that bound and the rest is loaded where it costs least: whole
/// batches on the least-loaded cards (each load evicts what a card holds, so as few as the burst needs), never on
/// the lowest-index cards whatever they run, and never all onto the hot card.
#[tokio::test(flavor = "multi_thread")]
async fn a_burst_for_a_model_hot_on_one_upstream_of_a_busy_fleet_goes_to_its_least_loaded_cards() {
    const UPSTREAMS: usize = 16;
    const HOT: usize = 1;
    const BOUND: u64 = 64;
    const BURST: usize = 256;
    let mut mocks = Vec::new();
    let mut urls = Vec::new();
    for i in 0..UPSTREAMS {
        let (m, u) = mock(&format!("u{i}"), 3466, 8192, &["big", "small"]).await;
        m.card_extra.lock().unwrap().insert("big".into(), json!({"decode_slots": BOUND, "resident": i == HOT, "slot_group": "big"}));
        if i != HOT {
            m.queue_depth.store(10 + 3 * (UPSTREAMS - i), Ordering::SeqCst);
        }
        mocks.push(m);
        urls.push(u);
    }
    let (addr, _fleet, _stop) = router(&urls).await;
    let (spread, held) = burst(addr, &mocks, "big", BURST).await;
    eprintln!("in-flight spread of a {BURST}-request burst over {UPSTREAMS} upstreams, hot upstream {HOT}: {spread:?}");
    let mut expected = vec![0; UPSTREAMS];
    for i in [HOT, 13, 14, 15] {
        expected[i] = BOUND as usize;
    }
    assert_eq!(spread, expected, "the hot upstream and the three least-loaded take a batch each");
    for m in &mocks {
        m.release.add_permits(BURST);
    }
    for request in held {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
}

/// A model loaded nowhere is loaded where that costs least: on an idle upstream, not on the lowest-index one while it
/// runs another model (which loading would evict), and failing an idle one, on the upstream holding the fewest
/// requests.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_resident_nowhere_goes_to_an_idle_upstream_not_the_first_open_one() {
    let mut mocks = Vec::new();
    let mut urls = Vec::new();
    for name in ["a", "b", "c"] {
        let (m, u) = mock(name, 3466, 8192, &["cold", "warm"]).await;
        m.card_extra.lock().unwrap().extend([("cold".to_string(), json!({"decode_slots": 64, "resident": false})), ("warm".to_string(), json!({"decode_slots": 64, "resident": name != "c"}))]);
        mocks.push(m);
        urls.push(u);
    }
    let (addr, _fleet, _stop) = router(&urls).await;
    // `a` and `b` run the model they hold, three requests between them; `c` is idle.
    let (spread, mut held) = burst(addr, &mocks, "warm", 3).await;
    assert_eq!(spread[2], 0, "{spread:?}");
    let idle = 2;
    // The cold model goes to the idle upstream, whatever its index.
    let (cold, more) = burst(addr, &mocks, "cold", 1).await;
    assert_eq!(cold.iter().position(|&n| n == 1), Some(idle), "the cold model went to the idle upstream: warm {spread:?}, cold {cold:?}");
    held.extend(more);
    // With none idle, to the upstream holding the fewest requests.
    let (again, more) = burst(addr, &mocks, "cold", 1).await;
    held.extend(more);
    let lightest = (0..3).filter(|&i| i != idle).min_by_key(|&i| (spread[i], i)).unwrap();
    let placed = again.iter().position(|&n| n == 1).unwrap();
    assert!(placed == lightest || placed == idle, "the second went to a lightest upstream: warm {spread:?}, cold {again:?}");
    for m in &mocks {
        m.release.add_permits(8);
    }
    for request in held {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
}

/// An upstream decodes one batch of a base however many adapters of it it lists: adapters whose cards name the same
/// `slot_group` count against one bound there, not one each (53 adapters on a card were 53 batches of 64).
#[tokio::test(flavor = "multi_thread")]
async fn adapters_of_one_base_share_one_load_bound_on_an_upstream() {
    let (a, ua) = mock("a", 3466, 8192, &["base:x", "base:y"]).await;
    a.card_extra.lock().unwrap().extend(["base:x", "base:y"].map(|id| (id.to_string(), json!({"decode_slots": 2, "resident": true, "slot_group": "base"}))));
    let (addr, _fleet, _stop) = router_over(config(&[ua]).with_queue(16, Duration::from_millis(300)), Arc::new(StaticKey::new(CLIENT_KEY))).await;
    let (spread, held) = burst(addr, std::slice::from_ref(&a), "base:x", 2).await;
    assert_eq!(spread, [2]);
    let (code, _, _) = chat(addr, conversation("base:y", "s", 1)).await;
    assert_eq!(code, StatusCode::TOO_MANY_REQUESTS, "the base's batch is full whichever adapter asks");
    a.release.add_permits(2);
    for request in held {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
    assert_eq!(chat(addr, conversation("base:y", "s", 1)).await.0, StatusCode::OK);
}

/// A server whose every card runs a fine-tune that holds it serves nothing until the run ends: its health says
/// `serving_held` and the router sends it nothing, whatever it has resident or cached, and takes it back once the run
/// is over. Training cards held 13-41 queued requests, producing no token, while they stayed in rotation (knowledge #611).
#[tokio::test(flavor = "multi_thread")]
async fn an_upstream_whose_cards_a_workload_holds_takes_no_request_until_it_ends() {
    let (a, ua) = mock("a", 3466, 8192, &["m"]).await;
    let (b, ub) = mock("b", 3466, 8192, &["m"]).await;
    a.card_extra.lock().unwrap().insert("m".into(), json!({"decode_slots": 64, "resident": true}));
    b.card_extra.lock().unwrap().insert("m".into(), json!({"decode_slots": 64, "resident": false}));
    a.serving_held.store(true, Ordering::SeqCst);
    let (addr, fleet, _stop) = router(&[ua, ub]).await;
    for turn in 0..3 {
        assert_eq!(chat(addr, conversation("m", "s", turn + 1)).await.2["upstream"], "b", "turn {turn}");
    }
    assert_eq!(a.count(), 0, "the held upstream was sent nothing although the model is resident there");
    assert_eq!(fleet.status()["upstreams"][0]["serving_held"], json!(true));
    a.serving_held.store(false, Ordering::SeqCst);
    fleet.probe().await;
    assert_eq!(chat(addr, conversation("m", "s", 1)).await.2["upstream"], "a", "back in rotation once the run ended");
}

/// The production burst of 2026-11-27 with its training cards (knowledge #611): the 27B burst of 256 over 16
/// upstreams, resident on upstream 1, while upstreams 2, 4, 8 and 12 each run a fine-tune that holds the card and
/// report nothing waiting (their queue is the requests they will never run), and the rest work for another router.
/// No request goes to a held card.
#[tokio::test(flavor = "multi_thread")]
async fn a_burst_never_lands_on_cards_a_fine_tune_holds() {
    const UPSTREAMS: usize = 16;
    const HELD: [usize; 4] = [2, 4, 8, 12];
    let mut mocks = Vec::new();
    let mut urls = Vec::new();
    for i in 0..UPSTREAMS {
        let (m, u) = mock(&format!("u{i}"), 3466, 8192, &["big"]).await;
        m.card_extra.lock().unwrap().insert("big".into(), json!({"decode_slots": 64, "resident": i == 1, "slot_group": "big"}));
        if HELD.contains(&i) {
            m.serving_held.store(true, Ordering::SeqCst);
        } else if i != 1 {
            m.queue_depth.store(10 + 3 * (UPSTREAMS - i), Ordering::SeqCst);
        }
        mocks.push(m);
        urls.push(u);
    }
    let (addr, _fleet, _stop) = router(&urls).await;
    let (spread, held) = burst(addr, &mocks, "big", 256).await;
    eprintln!("in-flight spread with fine-tunes holding upstreams {HELD:?}: {spread:?}");
    assert!(HELD.iter().all(|&i| spread[i] == 0), "a held card took requests: {spread:?}");
    for m in &mocks {
        m.release.add_permits(256);
    }
    for request in held {
        assert_eq!(request.await.unwrap().0, StatusCode::OK);
    }
}

/// A stall is worth detecting only while the requests it holds are still waited for. Agent runtimes bound one step
/// (one model call) at minutes: 480 s in the fleet of 2026-11-27, where a 600 s default stall bound fired only after
/// every client it would have saved had timed out, so held requests vanished instead of failing over (knowledge
/// #612). The default bound, plus one probe to notice it, leaves such a client a third of its step to retry.
#[test]
fn the_default_stall_bound_fires_while_an_agent_step_still_waits() {
    const AGENT_STEP: Duration = Duration::from_secs(480);
    let config = FleetConfig::new(Vec::new());
    assert!(config.stall_timeout + config.health_interval <= AGENT_STEP * 2 / 3, "stall {:?} + probe {:?} against a {AGENT_STEP:?} step", config.stall_timeout, config.health_interval);
}

/// Conformance: the router's `/health` (`brain.router.health` v1) has exactly the fields its `.fields` file pins (each
/// documented, `make check/metrics-fields`); a field that goes or is renamed fails here.
#[tokio::test(flavor = "multi_thread")]
async fn the_routers_health_has_exactly_the_pinned_documented_fields() {

    let (_a, ua) = mock("a", 8, 16, &["m"]).await;
    let (addr, _fleet, _stop) = router(&[ua]).await;
    let (code, _, health) = call(addr, Method::GET, "/v1/health", Some(CLIENT_KEY), "").await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!((health["schema"].as_str(), health["schema_version"].as_u64()), (Some(route::HEALTH_SCHEMA), Some(1)));
    crate::metrics::assert_pinned("brain.router.health", &crate::metrics::field_paths(&health), include_str!("fields/router-health-v1.fields"));
}
