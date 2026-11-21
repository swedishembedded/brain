// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for putting one inference endpoint
// in front of every card of a cluster for its clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! One OpenAI-compatible surface in front of several `brain serve` endpoints (`brain serve --upstream URL ...`).
//!
//! ```no_run
//! # fn main() -> Result<(), String> {
//! use apiserve::route::{Fleet, FleetConfig, UpstreamUrl};
//! let upstreams = ["http://127.0.0.1:8790", "http://127.0.0.1:8791"].iter().map(|u| UpstreamUrl::parse(u)).collect::<Result<_, _>>()?;
//! let fleet = Fleet::new(FleetConfig::new(upstreams))?;
//! # let _ = fleet; Ok(()) }
//! ```
//!
//! A client sees one URL and one key. Each upstream is a [`Member`] of the [`Fleet`]; a request goes to one of them,
//! chosen by the policy the in-process dispatcher uses for replicas of a model (`residency::affinity`): the upstream
//! whose prefix cache holds the longest useful prefix of the prompt, among those with a free slot, else the one with the
//! lowest load relative to its slots. Requests and responses pass through unbuffered (a streamed completion reaches the
//! client token by token), and a client that hangs up closes its upstream connection, which cancels the generation.
//!
//! What the router guarantees:
//!
//! - **Only routable upstreams take requests.** An upstream is in rotation while its `/health` and `/models` answer
//!   (probed every [`FleetConfig::health_interval`]) and it is not draining. A drained upstream ([`Fleet::drain`], `POST
//!   /v1/upstreams/{n}/drain`) takes nothing new and finishes what it holds.
//! - **A request is sent at most once.** It moves to another upstream only when the connection to the chosen one could
//!   not be opened, so no byte of it was sent. Once sent, a failure is the client's answer (502), because the
//!   generation may have run.
//! - **Load is bounded, not queued.** An upstream takes at most the requests its own edge admits (`edge_concurrency`
//!   in its `/health`); past that on every upstream the request is refused at once with 503 and `Retry-After`.
//! - **Only what is known is forwarded.** The JSON body is forwarded byte for byte after its top level is checked; the
//!   upstream sees only `Host`, `Content-Type`, `Content-Length`, `Accept` and the router's own upstream key, never the
//!   client's key or any other header. The client sees only the upstream's status, `Content-Type`, `Cache-Control`,
//!   `Retry-After` and body.

mod upstream;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Extension, Path, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use residency::affinity::{choose_replica, text_request_chain, PrefixChain, PrefixDirectory, ReplicaView, TextPrompt, PROMPT_SHAPING};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{json, Value};

pub use upstream::{check_upstream_exposure, UpstreamTls, UpstreamUrl};

use crate::auth::{Authenticator, Caller, Principal, Role};
use crate::error::ApiError;
use crate::listener::{self, ConnectionLimits};
use crate::surface::Provider;
use crate::tls::Tls;
use crate::MAX_BODY_BYTES;

/// The dialect the router speaks.
const PROVIDER: Provider = Provider::OpenAI;

/// The most upstreams one router fronts.
pub const MAX_UPSTREAMS: usize = 256;

/// The longest model id a request may name.
const MAX_MODEL_ID_BYTES: usize = 256;

/// The most models one upstream may list, and the most bytes its listing and its health answer may hold.
const MAX_LISTED_MODELS: usize = 4096;
const MAX_LISTING_BYTES: usize = 4 << 20;
const MAX_HEALTH_BYTES: usize = 64 << 10;

/// The most decode slots, and requests at once, an upstream may claim.
const MAX_ADVERTISED_SLOTS: u64 = 1 << 20;

/// How long one health probe may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Everything a [`Fleet`] is built from.
#[derive(Clone, Debug)]
pub struct FleetConfig {
    /// The upstream `brain serve` endpoints, in the order their indices name them.
    pub upstreams: Vec<UpstreamUrl>,
    /// The key the router presents to every upstream (`Authorization: Bearer`); `None` sends none.
    pub upstream_key: Option<String>,
    /// The authorities an `https://` upstream's certificate must chain to.
    pub upstream_tls: Option<UpstreamTls>,
    /// How long opening a connection (TCP, TLS and HTTP handshakes) may take before the upstream is passed over.
    pub connect_timeout: Duration,
    /// How long an upstream may take to answer a sent request with its status line. Its body then streams for as long
    /// as the upstream sends it.
    pub response_timeout: Duration,
    /// How often every upstream's health and model listing are read.
    pub health_interval: Duration,
    /// Prompts remembered per upstream for prefix affinity: the bound of the router's record of each upstream's
    /// prefix cache.
    pub cached_prompts: usize,
}

impl FleetConfig {
    /// `upstreams` with the default timeouts: 5 s to connect, 31 min for an answer (an upstream may queue a request up
    /// to its own admit deadline before it starts), health every 2 s, 256 prompts remembered per upstream.
    pub fn new(upstreams: Vec<UpstreamUrl>) -> FleetConfig {
        FleetConfig {
            upstreams,
            upstream_key: None,
            upstream_tls: None,
            connect_timeout: Duration::from_secs(5),
            response_timeout: Duration::from_secs(31 * 60),
            health_interval: Duration::from_secs(2),
            cached_prompts: 256,
        }
    }
    pub fn with_upstream_key(mut self, key: Option<String>) -> FleetConfig {
        self.upstream_key = key;
        self
    }
    pub fn with_upstream_tls(mut self, tls: Option<UpstreamTls>) -> FleetConfig {
        self.upstream_tls = tls;
        self
    }
    pub fn with_connect_timeout(mut self, timeout: Duration) -> FleetConfig {
        self.connect_timeout = timeout;
        self
    }
    pub fn with_response_timeout(mut self, timeout: Duration) -> FleetConfig {
        self.response_timeout = timeout;
        self
    }
    pub fn with_health_interval(mut self, interval: Duration) -> FleetConfig {
        self.health_interval = interval;
        self
    }
}

/// What the last good probe of an upstream said.
#[derive(Default)]
struct Advertised {
    /// Requests it decodes at once.
    slots: usize,
    /// Requests it admits at once before shedding.
    ceiling: usize,
    /// The model cards it lists, and their ids.
    cards: Vec<Value>,
    ids: HashSet<String>,
}

/// One upstream endpoint and what the router knows of it.
pub struct Member {
    url: UpstreamUrl,
    healthy: AtomicBool,
    /// Probed at least once: the first failure is reported, later ones only when the upstream was in rotation.
    probed: AtomicBool,
    draining: AtomicBool,
    in_flight: AtomicUsize,
    advertised: RwLock<Advertised>,
}

impl Member {
    fn advertised(&self) -> std::sync::RwLockReadGuard<'_, Advertised> {
        self.advertised.read().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// One request held by an upstream: its load, given back when the response ends or is dropped.
struct InFlight(Arc<Member>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The outcome of choosing an upstream.
enum Pick {
    Chosen(usize, InFlight),
    /// No upstream that answers lists the model.
    UnknownModel,
    /// Upstreams list the model, but none is in rotation now.
    NoneAvailable,
    /// Every upstream that could take it holds all it admits.
    Saturated,
}

/// The upstreams one router fronts, their state, and the record of which prompts went where.
pub struct Fleet {
    members: Vec<Arc<Member>>,
    /// Per upstream, the prompts routed to it: what its prefix cache holds, as far as this router knows.
    directories: Mutex<Vec<PrefixDirectory>>,
    connector: upstream::Connector,
    key: Option<String>,
    response_timeout: Duration,
    health_interval: Duration,
}

impl std::fmt::Debug for Fleet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fleet").field("upstreams", &self.members.iter().map(|m| m.url.to_string()).collect::<Vec<_>>()).finish_non_exhaustive()
    }
}

impl Fleet {
    /// A fleet over `config.upstreams`. Every upstream starts out of rotation until its first probe ([`Fleet::probe`])
    /// answers.
    ///
    /// # Errors
    /// No upstream, more than [`MAX_UPSTREAMS`], one named twice, an empty key, or a zero timeout.
    pub fn new(config: FleetConfig) -> Result<Arc<Fleet>, String> {
        if config.upstreams.is_empty() {
            return Err("no upstream is given".into());
        }
        if config.upstreams.len() > MAX_UPSTREAMS {
            return Err(format!("at most {MAX_UPSTREAMS} upstreams"));
        }
        let mut seen = HashSet::new();
        if let Some(twice) = config.upstreams.iter().find(|u| !seen.insert(u.to_string())) {
            return Err(format!("upstream {twice} is given twice"));
        }
        if config.upstream_key.as_deref().is_some_and(|k| k.is_empty() || k.chars().any(|c| c.is_control() || c.is_whitespace())) {
            return Err("the upstream key is empty or holds whitespace".into());
        }
        if config.connect_timeout.is_zero() || config.response_timeout.is_zero() || config.health_interval.is_zero() {
            return Err("timeouts and the health interval are positive".into());
        }
        let members: Vec<Arc<Member>> = config
            .upstreams
            .into_iter()
            .map(|url| Arc::new(Member { url, healthy: AtomicBool::new(false), probed: AtomicBool::new(false), draining: AtomicBool::new(false), in_flight: AtomicUsize::new(0), advertised: RwLock::default() }))
            .collect();
        let directories = Mutex::new(members.iter().map(|_| PrefixDirectory::new(config.cached_prompts)).collect());
        Ok(Arc::new(Fleet {
            members,
            directories,
            connector: upstream::Connector::new(config.upstream_tls, config.connect_timeout),
            key: config.upstream_key,
            response_timeout: config.response_timeout,
            health_interval: config.health_interval,
        }))
    }

    /// Stop sending new requests to upstream `index`; what it holds finishes. `false` when there is no such upstream.
    pub fn drain(&self, index: usize) -> bool {
        self.members.get(index).map(|m| m.draining.store(true, Ordering::SeqCst)).is_some()
    }

    /// Put a drained upstream `index` back in rotation. `false` when there is no such upstream.
    pub fn resume(&self, index: usize) -> bool {
        self.members.get(index).map(|m| m.draining.store(false, Ordering::SeqCst)).is_some()
    }

    /// Whether some upstream is in rotation.
    pub fn available(&self) -> bool {
        self.members.iter().any(|m| m.healthy.load(Ordering::SeqCst) && !m.draining.load(Ordering::SeqCst))
    }

    /// The fleet as `GET /health` reports it: per upstream (by index; addresses stay with the operator) whether it is
    /// healthy or draining, what it holds and what it takes.
    pub fn status(&self) -> Value {
        let upstreams: Vec<Value> = self
            .members
            .iter()
            .enumerate()
            .map(|(index, m)| {
                let adv = m.advertised();
                json!({
                    "index": index,
                    "healthy": m.healthy.load(Ordering::SeqCst),
                    "draining": m.draining.load(Ordering::SeqCst),
                    "in_flight": m.in_flight.load(Ordering::SeqCst),
                    "decode_slots": adv.slots,
                    "models": adv.ids.len(),
                })
            })
            .collect();
        let routable = |m: &&Arc<Member>| m.healthy.load(Ordering::SeqCst) && !m.draining.load(Ordering::SeqCst);
        let slots: usize = self.members.iter().filter(routable).map(|m| m.advertised().slots).sum();
        let ceiling: usize = self.members.iter().filter(routable).map(|m| m.advertised().ceiling).sum();
        let status = if self.available() { "ok" } else { "unavailable" };
        json!({"status": status, "decode_slots": slots, "edge_concurrency": ceiling, "upstreams": upstreams})
    }

    /// Read every upstream's health and model listing once, concurrently, and put each in or out of rotation by the
    /// answer.
    pub async fn probe(&self) {
        futures::future::join_all((0..self.members.len()).map(|i| self.probe_one(i))).await;
    }

    async fn probe_one(&self, index: usize) {
        let member = &self.members[index];
        let key = self.key.as_deref();
        let read = async {
            let health = upstream::get_json(&self.connector, &member.url, "/v1/health", key, MAX_HEALTH_BYTES, PROBE_TIMEOUT).await?;
            let listing = upstream::get_json(&self.connector, &member.url, "/v1/models", key, MAX_LISTING_BYTES, PROBE_TIMEOUT).await?;
            parse_advertised(&health, &listing)
        };
        match read.await {
            Ok(advertised) => {
                *member.advertised.write().unwrap_or_else(std::sync::PoisonError::into_inner) = advertised;
                member.probed.store(true, Ordering::SeqCst);
                if !member.healthy.swap(true, Ordering::SeqCst) {
                    eprintln!("brain route: upstream {index} ({}) is in rotation", member.url);
                }
            }
            Err(why) => self.mark_down(index, &why),
        }
    }

    /// Take upstream `index` out of rotation until a probe answers again. Its prefix cache may not survive what
    /// happened to it, so the record of it is dropped.
    fn mark_down(&self, index: usize, why: &str) {
        let member = &self.members[index];
        let was_healthy = member.healthy.swap(false, Ordering::SeqCst);
        if was_healthy || !member.probed.swap(true, Ordering::SeqCst) {
            eprintln!("brain route: upstream {index} ({}) is out of rotation: {why}", member.url);
        }
        self.directories.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[index].clear();
    }

    /// Probe every upstream every [`FleetConfig::health_interval`] until `shutdown` fires.
    async fn keep_probing(self: Arc<Fleet>, shutdown: brain_shutdown::Shutdown) {
        let stop = shutdown.wait();
        tokio::pin!(stop);
        loop {
            self.probe().await;
            tokio::select! {
                _ = tokio::time::sleep(self.health_interval) => {}
                _ = &mut stop => return,
            }
        }
    }

    /// The upstream that runs a request for `model` keyed by `chain` (see the module doc), with one of its slots
    /// taken. `excluded` upstreams failed to connect for this request.
    fn pick(&self, model: &str, chain: &PrefixChain, excluded: &[bool]) -> Pick {
        let mut directories = self.directories.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut known, mut open, mut candidates) = (false, Vec::new(), 0usize);
        for (i, m) in self.members.iter().enumerate() {
            let adv = m.advertised();
            let serves = adv.ids.contains(model);
            known |= serves;
            if !serves || excluded[i] || !m.healthy.load(Ordering::SeqCst) || m.draining.load(Ordering::SeqCst) {
                continue;
            }
            candidates += 1;
            let load = m.in_flight.load(Ordering::SeqCst);
            if load < adv.ceiling {
                open.push((i, ReplicaView { load, slots: adv.slots, active: true, can_run: true, matched: directories[i].matched(chain) }));
            }
        }
        if candidates == 0 {
            return if known || !self.available() { Pick::NoneAvailable } else { Pick::UnknownModel };
        }
        if open.is_empty() {
            return Pick::Saturated;
        }
        let views: Vec<ReplicaView> = open.iter().map(|&(_, v)| v).collect();
        let chosen = open[choose_replica(&views)].0;
        directories[chosen].record(chain);
        let member = self.members[chosen].clone();
        member.in_flight.fetch_add(1, Ordering::SeqCst);
        Pick::Chosen(chosen, InFlight(member))
    }
}

/// The upstream's answers to `/v1/health` and `/v1/models`, checked: they come from another process and bound what
/// this one allocates.
fn parse_advertised(health: &Value, listing: &Value) -> Result<Advertised, String> {
    let count = |field: &str| health.get(field).and_then(Value::as_u64).filter(|&n| (1..=MAX_ADVERTISED_SLOTS).contains(&n)).map(|n| n as usize);
    let slots = count("decode_slots").ok_or("its /v1/health names no decode_slots in range")?;
    let ceiling = count("edge_concurrency").unwrap_or(slots.saturating_mul(2)).max(1);
    let data = listing.get("data").and_then(Value::as_array).ok_or("its /v1/models has no data list")?;
    if data.len() > MAX_LISTED_MODELS {
        return Err(format!("it lists more than {MAX_LISTED_MODELS} models"));
    }
    let mut adv = Advertised { slots, ceiling, ..Advertised::default() };
    for card in data {
        let Some(id) = card.get("id").and_then(Value::as_str).filter(|id| valid_model_id(id)) else {
            return Err("it lists a model whose id is not a plain string".into());
        };
        if adv.ids.insert(id.to_string()) {
            adv.cards.push(card.clone());
        }
    }
    Ok(adv)
}

fn valid_model_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_MODEL_ID_BYTES && !id.chars().any(char::is_control)
}

/// The top level of a request body: its fields in order, each value as the client wrote it. A field named twice is
/// refused, so the router and the upstream cannot read two different requests out of one body.
struct Head(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for Head {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Head, D::Error> {
        struct Fields;
        impl<'de> Visitor<'de> for Fields {
            type Value = Head;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Head, A::Error> {
                let mut fields: Vec<(String, Box<RawValue>)> = Vec::new();
                let mut names = HashSet::new();
                while let Some((name, value)) = map.next_entry::<String, Box<RawValue>>()? {
                    if !names.insert(name.clone()) {
                        return Err(serde::de::Error::custom(format!("the field {name:?} is given twice")));
                    }
                    fields.push((name, value));
                }
                Ok(Head(fields))
            }
        }
        deserializer.deserialize_map(Fields)
    }
}

impl Head {
    fn get(&self, name: &str) -> Option<&RawValue> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_ref())
    }
}

/// How a forwarded route keys its requests for prefix affinity.
#[derive(Clone, Copy)]
enum Keyed {
    /// A chat: the conversation's messages.
    Chat,
    /// A completion: the raw prompt.
    Completion,
    /// Nothing worth routing by: load alone decides.
    Load,
}

/// The upstream routes a fleet client calls, each with how it is keyed. A path the router does not list here is a 404:
/// background jobs and video jobs live on the upstream that started them, which a stateless router cannot find again.
const FORWARDED: [(&str, Keyed); 10] = [
    ("/v1/chat/completions", Keyed::Chat),
    ("/chat/completions", Keyed::Chat),
    ("/v1/completions", Keyed::Completion),
    ("/completions", Keyed::Completion),
    ("/v1/embeddings", Keyed::Load),
    ("/embeddings", Keyed::Load),
    ("/v1/images/generations", Keyed::Load),
    ("/images/generations", Keyed::Load),
    ("/v1/audio/speech", Keyed::Load),
    ("/audio/speech", Keyed::Load),
];

#[derive(Clone)]
struct RouteState {
    fleet: Arc<Fleet>,
    authenticator: Arc<dyn Authenticator>,
}

/// The router's axum [`Router`]: every forwarded route, the merged `/models`, `/health`, and the drain routes, all
/// behind `authenticator` (fallback included) and the body limit. Exposed so a test can drive it without a socket.
pub fn router(fleet: Arc<Fleet>, authenticator: Arc<dyn Authenticator>) -> Router {
    let state = RouteState { fleet, authenticator };
    let mut app = Router::new()
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models))
        .route("/models/{id}", get(one_model))
        .route("/v1/models/{id}", get(one_model))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/upstreams/{index}/drain", post(drain))
        .route("/v1/upstreams/{index}/resume", post(resume));
    for (path, keyed) in FORWARDED {
        app = app.route(path, post(move |state: State<RouteState>, principal: Extension<Principal>, headers: HeaderMap, body: Bytes| forward(state, principal, headers, body, path, keyed)));
    }
    app.fallback(|| async { ApiError::not_found(PROVIDER, "no such route") })
        .layer(axum::middleware::from_fn_with_state(state.clone(), authenticate))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

async fn authenticate(State(state): State<RouteState>, mut req: Request, next: Next) -> Result<Response, ApiError> {
    let principal = state.authenticator.authenticate(PROVIDER, req.headers())?;
    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}

async fn list_models(State(state): State<RouteState>) -> Json<Value> {
    Json(crate::models::envelope(PROVIDER, merged_cards(&state.fleet).into_values().collect()))
}

async fn one_model(State(state): State<RouteState>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    merged_cards(&state.fleet).remove(&id).map(Json).ok_or_else(|| ApiError::model_not_found(PROVIDER, &id))
}

/// Every model an upstream in rotation lists, once, in id order; the first upstream's card for an id wins.
fn merged_cards(fleet: &Fleet) -> std::collections::BTreeMap<String, Value> {
    let mut cards = std::collections::BTreeMap::new();
    for m in fleet.members.iter().filter(|m| m.healthy.load(Ordering::SeqCst) && !m.draining.load(Ordering::SeqCst)) {
        let adv = m.advertised();
        for card in &adv.cards {
            if let Some(id) = card.get("id").and_then(Value::as_str) {
                cards.entry(id.to_string()).or_insert_with(|| card.clone());
            }
        }
    }
    cards
}

async fn health(State(state): State<RouteState>) -> Response {
    let code = if state.fleet.available() { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (code, Json(state.fleet.status())).into_response()
}

/// Draining moves load between machines, so it is the operator's: with named callers only an admin may.
fn operator_only(principal: &Principal) -> Result<(), ApiError> {
    match Caller::of(Some(principal)) {
        Some(caller) if caller.role < Role::Admin => Err(ApiError::forbidden(PROVIDER, "only an admin caller may drain or resume an upstream")),
        _ => Ok(()),
    }
}

async fn drain(State(state): State<RouteState>, Extension(principal): Extension<Principal>, Path(index): Path<String>) -> Result<Json<Value>, ApiError> {
    operator_only(&principal)?;
    let index = index.parse::<usize>().ok().filter(|&i| state.fleet.drain(i)).ok_or_else(|| ApiError::not_found(PROVIDER, "no such upstream"))?;
    Ok(Json(json!({"index": index, "draining": true})))
}

async fn resume(State(state): State<RouteState>, Extension(principal): Extension<Principal>, Path(index): Path<String>) -> Result<Json<Value>, ApiError> {
    operator_only(&principal)?;
    let index = index.parse::<usize>().ok().filter(|&i| state.fleet.resume(i)).ok_or_else(|| ApiError::not_found(PROVIDER, "no such upstream"))?;
    Ok(Json(json!({"index": index, "draining": false})))
}

/// The model a request names, checked.
fn requested_model(head: &Head) -> Result<String, ApiError> {
    let raw = head.get("model").ok_or_else(|| ApiError::invalid_request(PROVIDER, "the request names no model"))?;
    serde_json::from_str::<String>(raw.get()).ok().filter(|m| valid_model_id(m)).ok_or_else(|| ApiError::invalid_request(PROVIDER, format!("model must be a string of 1 to {MAX_MODEL_ID_BYTES} bytes")))
}

/// The prefix chain a request is routed by: the model is its namespace, the prompt-shaping fields its root.
fn request_chain(head: &Head, model: &str, keyed: Keyed) -> PrefixChain {
    let shaping = PROMPT_SHAPING.iter().filter_map(|&key| head.get(key).map(|v| (key, std::borrow::Cow::Borrowed(v.get()))));
    match keyed {
        Keyed::Load => PrefixChain::default(),
        Keyed::Chat => text_request_chain(model, shaping, TextPrompt::Messages(head.get("messages").map_or("", RawValue::get))),
        Keyed::Completion => {
            let raw = head.get("prompt").map_or("", RawValue::get);
            let text = serde_json::from_str::<String>(raw).ok();
            text_request_chain(model, shaping, TextPrompt::Raw(text.as_deref().unwrap_or(raw)))
        }
    }
}

async fn forward(State(state): State<RouteState>, Extension(principal): Extension<Principal>, headers: HeaderMap, body: Bytes, path: &'static str, keyed: Keyed) -> Response {
    let head: Head = match serde_json::from_slice(&body) {
        Ok(head) => head,
        Err(e) => return ApiError::invalid_request(PROVIDER, format!("the request body is not a JSON object: {e}")).into_response(),
    };
    let model = match requested_model(&head) {
        Ok(model) => model,
        Err(e) => return e.into_response(),
    };
    let chain = request_chain(&head, &model, keyed);
    drop(head);
    // A named caller holds at most its own share in flight, as on a serving surface.
    let caller_slot = match Caller::of(Some(&principal)) {
        Some(caller) => match Arc::downcast::<Caller>(principal.clone()).ok().and_then(|c| c.begin()) {
            Some(slot) => Some(slot),
            None => return ApiError::rate_limited(PROVIDER, format!("caller {} has its whole share of the server in flight; retry when a call ends", caller.name)).into_response(),
        },
        None => None,
    };
    let accept = headers.get(header::ACCEPT).filter(|v| v.to_str().is_ok_and(|s| s.len() <= 256)).cloned();
    let fleet = &state.fleet;
    let mut excluded = vec![false; fleet.members.len()];
    loop {
        let (index, in_flight) = match fleet.pick(&model, &chain, &excluded) {
            Pick::Chosen(index, in_flight) => (index, in_flight),
            Pick::UnknownModel => return ApiError::model_not_found(PROVIDER, &model).into_response(),
            Pick::NoneAvailable => return ApiError::saturated(PROVIDER, "no upstream serving this model is available").into_response(),
            Pick::Saturated => return ApiError::saturated(PROVIDER, "every upstream serving this model is at capacity").into_response(),
        };
        let member = &fleet.members[index];
        let conn = match fleet.connector.open(&member.url).await {
            Ok(conn) => conn,
            Err(not_sent) => {
                // Nothing was sent: the request may go to another upstream, and this one waits for its next probe.
                fleet.mark_down(index, &not_sent.0);
                excluded[index] = true;
                continue;
            }
        };
        let sent = tokio::time::timeout(fleet.response_timeout, conn.send(Method::POST, path, fleet.key.as_deref(), accept.as_ref(), body.clone())).await;
        return match sent {
            Ok(Ok((response, task))) => relay(response, (task, in_flight, caller_slot)),
            Ok(Err(why)) => {
                eprintln!("brain route: upstream {index} ({}) failed a sent request: {why}", member.url);
                ApiError::bad_gateway(PROVIDER, "the upstream failed after the request was sent; it may have run").into_response()
            }
            Err(_) => ApiError::failure(PROVIDER, capability::FailureClass::Timeout, "the upstream did not answer in time; the request may have run").into_response(),
        };
    }
}

/// The upstream's answer as the client's: its status, the headers a client of the API reads, and its body streamed
/// through. `held` (the connection, the upstream slot, the caller's slot) lives until the body ends or the client hangs
/// up.
fn relay<H: Send + 'static>(response: hyper::Response<hyper::body::Incoming>, held: H) -> Response {
    const PASSED: [header::HeaderName; 3] = [header::CONTENT_TYPE, header::CACHE_CONTROL, header::RETRY_AFTER];
    let (parts, incoming) = response.into_parts();
    let mut upstream_body = Body::new(incoming).into_data_stream();
    let body = async_stream::stream! {
        let _held = held;
        while let Some(chunk) = upstream_body.next().await {
            yield chunk;
        }
    };
    let mut out = Response::new(Body::from_stream(body));
    *out.status_mut() = parts.status;
    for name in PASSED {
        if let Some(value) = parts.headers.get(&name) {
            out.headers_mut().insert(name, value.clone());
        }
    }
    out
}

/// What [`serve`] needs beyond the fleet and the addresses.
#[derive(Default)]
pub struct RouteOpts {
    /// Stops serving; `None` installs one from this process's signals.
    pub shutdown: Option<brain_shutdown::Shutdown>,
    /// Notified once per address, after it binds.
    pub ready: brain_shutdown::ready::Gate,
    /// What one peer may hold of the surface.
    pub limits: ConnectionLimits,
    /// Serve over TLS with this certificate; plaintext only on loopback ([`crate::tls::check_exposure`]).
    pub tls: Option<Tls>,
    /// Where to publish the connection record once every address is bound, and the key it names (empty for none).
    pub record: Option<(PathBuf, String)>,
}

/// Serve `fleet` on `listener` under `authenticator` until `shutdown` fires, probing the upstreams meanwhile.
pub async fn run(fleet: Arc<Fleet>, listener: tokio::net::TcpListener, authenticator: Arc<dyn Authenticator>, limits: ConnectionLimits, tls: Option<Tls>, shutdown: brain_shutdown::Shutdown) {
    let probing = tokio::spawn(fleet.clone().keep_probing(shutdown.clone()));
    listener::serve(listener, router(fleet, authenticator), limits, tls, shutdown).await;
    probing.abort();
}

/// Serve `fleet` on every address in `addrs` until the shutdown fires. Blocks the calling thread. The upstreams are
/// probed once before any address binds, so a client that finds the server ready finds the upstreams that answered
/// already in rotation.
///
/// # Errors
/// An address that cannot be bound, or a runtime that cannot start.
pub fn serve(fleet: Arc<Fleet>, addrs: Vec<SocketAddr>, authenticator: Arc<dyn Authenticator>, opts: RouteOpts) -> anyhow::Result<()> {
    let shutdown = opts.shutdown.unwrap_or_else(brain_shutdown::Shutdown::from_signals);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let result = rt.block_on(async {
        fleet.probe().await;
        let up = fleet.members.iter().filter(|m| m.healthy.load(Ordering::SeqCst)).count();
        eprintln!("brain route: {up} of {} upstreams in rotation", fleet.members.len());
        let mut bound = Vec::new();
        let mut set = tokio::task::JoinSet::new();
        let scheme = if opts.tls.is_some() { "https" } else { "http" };
        for addr in addrs {
            let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
            let addr = listener.local_addr().unwrap_or(addr);
            eprintln!("brain apiserve: {PROVIDER} on {scheme}://{addr} (routing to {} upstreams)", fleet.members.len());
            opts.ready.bound(PROVIDER.as_str());
            bound.push(addr);
            set.spawn(run(fleet.clone(), listener, authenticator.clone(), opts.limits, opts.tls.clone(), shutdown.clone()));
        }
        if let Some((path, key)) = &opts.record {
            let entries: Vec<(Provider, SocketAddr, String)> = bound.iter().map(|&a| (PROVIDER, a, key.clone())).collect();
            let record = crate::surface::record_json(std::process::id(), scheme, &entries).to_string();
            if let Err(e) = path.parent().map_or(Ok(()), crate::surface::private_dir).and_then(|()| crate::surface::write_private(path, record.as_bytes())) {
                eprintln!("brain serve: the connection record {} could not be written ({e}); local clients will not find this server", path.display());
            }
        }
        let drain_expired = async {
            shutdown.wait().await;
            tokio::time::sleep(crate::SHUTDOWN_DRAIN).await;
        };
        tokio::pin!(drain_expired);
        loop {
            tokio::select! {
                joined = set.join_next() => if joined.is_none() { break },
                _ = &mut drain_expired => {
                    set.abort_all();
                    break;
                }
            }
        }
        Ok(())
    });
    rt.shutdown_background();
    if let Some((path, _)) = &opts.record {
        std::fs::remove_file(path).ok();
    }
    result
}

/// The upstream URLs in `text`: one per line; blank lines and lines starting with `#` are skipped.
///
/// # Errors
/// The first line that is not an upstream URL, with its line number.
pub fn parse_upstreams_file(text: &str) -> Result<Vec<UpstreamUrl>, String> {
    text.lines()
        .enumerate()
        .map(|(n, line)| (n + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .map(|(n, line)| UpstreamUrl::parse(line).map_err(|e| format!("line {n}: {e}")))
        .collect()
}
