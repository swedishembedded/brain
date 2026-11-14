// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The async→sync seam between the HTTP handlers (Tokio tasks) and the shared,
//! synchronous [`residency::Executor`]. Mirrors `crates/dbus/src/service.rs`:
//! a handler only builds a [`capability::Invocation`], submits a [`residency::Job`],
//! and waits for the reply/progress — the model never runs on the HTTP task.
//!
//! Two shapes:
//! - [`submit`] — one-shot: arm a cancel token, submit, await the single reply
//!   (non-streaming chat).
//! - [`stream`] — streaming: the job's `on_progress`/`reply` callbacks push
//!   [`StreamMsg`]s onto an unbounded channel, returned as a [`futures::Stream`].
//!   The closures only ever touch the channel (`Send`), NEVER the model/GPU. A
//!   [`CancelGuard`] rides inside the returned stream so a dropped SSE response
//!   (client disconnect) cancels the running generation and clears the registry.

use std::pin::Pin;
use std::task::{Context, Poll};

use capability::{Invocation, Outcome};
use futures::Stream;
use residency::{Job, Supply};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::error::ApiError;
use crate::hooks::{Call, CallResult, Ticket};
use crate::state::{AppState, JobRegistry};
use crate::surface::Provider;

/// Refuse a training action on a surface that does not train. Every dispatch
/// passes here ([`Pending::begin`]), so no route - `/v1/run`, `/v1/jobs`, a
/// dialect's own - can reach one around it.
/// A named caller below [`Role::Training`](crate::Role::Training) is refused one
/// even where training is served.
fn refuse_training(state: &AppState, model: &str, action: &str) -> Result<(), ApiError> {
    let trains = state.exec.manifests().iter().filter(|m| m.model == model).flat_map(|m| &m.actions).any(|a| a.name == action && a.training);
    if !trains {
        return Ok(());
    }
    if !state.training {
        return Err(ApiError::forbidden(state.provider, format!("{model} {action} trains, and this server does not serve training (start it with --training)")));
    }
    if let Some(caller) = crate::auth::Caller::of(state.caller.as_ref()).filter(|c| c.role < crate::Role::Training) {
        return Err(ApiError::forbidden(state.provider, format!("{model} {action} trains, and caller {} may not train", caller.name)));
    }
    Ok(())
}

/// A call that has begun (the embedder's [`RequestHooks`](crate::RequestHooks) let
/// it run) and not yet been settled.
///
/// Settling consumes it, and dropping it unsettled settles it as
/// [`CallResult::Refused`], so every path out of a call -- a reply, a failure, a
/// job that was cancelled before it ran and whose reply closure was dropped
/// unfired -- reaches the embedder exactly once; a named caller's in-flight slot
/// is held as long. Without hooks or a named caller it holds nothing.
struct Pending {
    ticket: Option<Box<dyn Ticket>>,
    /// The caller's in-flight slot, given back when the call is dropped.
    _in_flight: Option<crate::auth::InFlight>,
}

impl Pending {
    /// Asks the hooks whether this call may run.
    fn begin(state: &AppState, model: &str, action: &str, invocation: &Invocation) -> Result<Pending, ApiError> {
        refuse_training(state, model, action)?;
        refuse_unread_media(&state.exec, state.provider, model, action, invocation)?;
        // A named caller holds at most its own share in flight, for as long as
        // this call is unsettled - a background job's whole run included.
        let in_flight = match state.caller.as_ref().and_then(|p| p.clone().downcast::<crate::auth::Caller>().ok()) {
            None => None,
            Some(caller) => Some(caller.begin().ok_or_else(|| ApiError::rate_limited(state.provider, format!("caller {} has its whole share of the server in flight; retry when a call ends", caller.name)))?),
        };
        match &state.hooks {
            None => Ok(Pending { ticket: None, _in_flight: in_flight }),
            Some(hooks) => hooks.begin(&Call { provider: state.provider, caller: state.caller.as_ref(), model, action, invocation }).map(|ticket| Pending { ticket: Some(ticket), _in_flight: in_flight }),
        }
    }

    /// Settles the call. An error means the settlement could not be recorded.
    fn settle(mut self, result: CallResult<'_>) -> Result<(), ApiError> {
        match self.ticket.take() {
            Some(ticket) => ticket.finish(result),
            None => Ok(()),
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            let _ = ticket.finish(CallResult::Refused);
        }
    }
}

/// Refuses a request that carries a medium the model's `action` does not read.
///
/// Images (`image`, `image1`, ...) and the audio clip are attached to every chat invocation as blobs, and a model that
/// declares no such input would never read them: it would answer the text alone, as if it had looked at the picture
/// or listened to the recording. So every attached medium must match a declared input; a model that takes one image
/// refuses a request with two rather than reading the first. A model not yet known to the executor (one still to be
/// fetched) is checked again once it is, by the same call.
pub fn refuse_unread_media(exec: &residency::Executor, provider: Provider, model: &str, action: &str, inv: &Invocation) -> Result<(), ApiError> {
    let images = (0..).take_while(|&i| inv.get_blob(&capability::blob::image_key(i)).is_some()).count();
    let audio = inv.get_blob("audio").is_some();
    if images == 0 && !audio {
        return Ok(());
    }
    let manifests = exec.manifests();
    let Some(spec) = manifests.iter().find(|m| m.model == model).and_then(|m| m.actions.iter().find(|a| a.name == action)) else {
        return Ok(());
    };
    let declares = |name: &str| spec.inputs.iter().any(|b| b.name == name);
    let readable = (0..images).take_while(|&i| declares(&capability::blob::image_key(i))).count();
    let refuse = |msg: String| Err(ApiError::invalid_request(provider, msg));
    if images > 0 && readable == 0 {
        return refuse(format!(
            "model '{model}' does not read images: its '{action}' action takes no image input (a text-only model, or a checkpoint loaded without vision weights); resend without image parts"
        ));
    }
    if readable < images {
        return refuse(format!("model '{model}' reads at most {readable} image(s) per request on its '{action}' action, and this request carries {images}; resend with fewer images"));
    }
    if audio && !declares("audio") {
        return refuse(format!("model '{model}' does not read audio: its '{action}' action takes no audio input; resend without input_audio parts"));
    }
    Ok(())
}

/// Auto-fetch entry point: `model` didn't resolve against `resolve` (the caller's
/// `catalog::resolve_*` predicate, already tried once). If `state.supplier` can
/// classify `model` as fetchable, blocks (off the async runtime, via
/// `spawn_blocking` — a cold fetch can take minutes) until it's on disk and
/// registered, then retries `resolve`. Returns a `model_not_found` for every other
/// outcome (no supplier configured, `Unknown`, or the fetch itself failing) —
/// deliberately the SAME error as an unknown model, so a failed/refused fetch
/// tells a client nothing more than "this model isn't available" (hub URLs,
/// filesystem paths, and other fetch-internal detail are logged server-side,
/// never reflected to the caller).
pub async fn ensure_and_recheck<T>(state: &AppState, provider: Provider, model: &str, resolve: impl Fn(&str) -> Option<T>) -> Result<T, ApiError> {
    let Some(supplier) = state.supplier_for_caller() else {
        return Err(ApiError::model_not_found(provider, model));
    };
    match supplier.classify(model) {
        Supply::Fetchable => {}
        Supply::Resident | Supply::Unknown(_) => return Err(ApiError::model_not_found(provider, model)),
    }
    let exec = state.exec.clone();
    let model_owned = model.to_string();
    let outcome = tokio::task::spawn_blocking(move || exec.ensure_model(&model_owned, supplier.as_ref(), &mut |_, _, _| {})).await;
    match outcome {
        Ok(Ok(())) => resolve(model).ok_or_else(|| ApiError::model_not_found(provider, model)),
        Ok(Err(reason)) => {
            eprintln!("apiserve: auto-fetch {model}: {reason}");
            Err(ApiError::model_not_found(provider, model))
        }
        Err(join_err) => {
            eprintln!("apiserve: auto-fetch {model}: blocking task failed: {join_err}");
            Err(ApiError::model_not_found(provider, model))
        }
    }
}

/// Read the contract's `generate` [`Outcome`] into `(text, prompt_tokens,
/// completion_tokens, finish_reason)`, the part of [`read_chat_outcome`] every
/// generation reports. `text` is the `text` blob (full assistant text); the
/// counts + `finish_reason` come from the `outputs` object. Missing fields
/// degrade gracefully (empty text, zero counts, `"stop"`).
fn read_outcome(o: &Outcome) -> (String, i64, i64, String) {
    let text = o.blobs.get("text").map(|b| String::from_utf8_lossy(&b.bytes).into_owned()).unwrap_or_default();
    let prompt = o.outputs.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let completion = o.outputs.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
    let finish = o.outputs.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("stop").to_string();
    (text, prompt, completion, finish)
}

/// A generation outcome's full chat shape, read from the contract `Outcome`:
/// everything [`read_outcome`] reads, plus `reasoning` (the model's
/// `<think>…</think>` text, empty when it didn't reason) and `tool_calls` (one
/// JSON object per call — `{"id","name","arguments"}`, `arguments` a raw JSON-text
/// string, never re-parsed — from `outputs.tool_calls`'s JSON-array-string;
/// empty when absent). `text` is VISIBLE content only — the resident layer
/// (`crates/catalog/src/resident_llm.rs::QwenInstance::run`,
/// `resident_mock.rs::generate_tool_call`) guarantees `<think>`/`<tool_call>`
/// markup never reaches the `text` blob.
pub struct ChatOutcome {
    pub text: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    /// Of `completion_tokens`, those spent inside think blocks.
    pub reasoning_tokens: i64,
    /// Of `prompt_tokens`, those the model's prefix cache served; `None` when the model does not report it.
    pub cached_tokens: Option<i64>,
    pub finish: String,
    pub reasoning: String,
    pub tool_calls: Vec<serde_json::Value>,
    /// The sampling seed the model decoded with, when it reports one.
    pub seed: Option<u64>,
    /// The `(id, digest, version)` of the adapter that answered, when one did. The version is the policy version
    /// of that release under its id.
    pub adapter: Option<(String, String, u64)>,
    /// `sha256:<hex>` of the base checkpoint that answered, when the model knows it.
    pub base_digest: Option<String>,
    /// The backend the decode ran on (`cuda`, `cpu`, ...).
    pub backend: Option<String>,
    /// The engine build that decoded (`brain <version>`).
    pub engine: Option<String>,
    /// The ids of the prompt as the model read it, when the request asked.
    pub prompt_token_ids: Option<Vec<i64>>,
    /// The ids of the tokens the model sampled, when the request asked.
    pub completion_token_ids: Option<Vec<i64>>,
    /// The log-probability of each sampled token under the distribution it was
    /// drawn from, when the request asked.
    pub logprobs: Option<Vec<f64>>,
    /// The model's policy trace (prompt ids, generated ids and their full-distribution log-probabilities), when the
    /// request asked ([`Self::policy_trace`] adds the policy that served it).
    pub trace: Option<serde_json::Value>,
}

impl ChatOutcome {
    /// The reply's `policy_trace` object: what the model scored, plus the policy that generated it (the base
    /// checkpoint's digest and the adapter's id and digest, each only when the model reported it), why the
    /// generation ended, and whether it was cut short by the token budget (`truncated`), which a trainer masks.
    /// `None` when the request did not ask.
    ///
    /// `generation_log_probs[i]` is `log softmax(logits / log_prob_temperature)[generation_token_ids[i]]` over the
    /// whole vocabulary: no top-k, top-p or min-p, however the token was drawn. The sampler's own figure is the
    /// `logprobs` choice field and is a different quantity (the truncated, temperature-scaled distribution).
    pub fn policy_trace(&self) -> Option<serde_json::Value> {
        let mut trace = self.trace.clone()?;
        let policy = serde_json::json!({
            "base_digest": self.base_digest,
            "adapter": self.adapter.as_ref().map(|(id, digest, version)| serde_json::json!({ "id": id, "digest": digest, "version": version })),
            "policy_version": self.policy_version(),
            "tokenizer_digest": trace.get("tokenizer_digest"),
            "chat_template_digest": trace.get("chat_template_digest"),
        });
        trace["policy"] = policy;
        trace["finish_reason"] = serde_json::json!(self.finish);
        trace["truncated"] = serde_json::json!(self.finish == "length");
        trace["log_prob_distribution"] = serde_json::json!("full_softmax_at_temperature");
        Some(trace)
    }

    /// The policy version this reply was generated under: the version of the adapter release that answered under its id
    /// (1 for the first, +1 for each release with different weights), 0 for the base model.
    pub fn policy_version(&self) -> u64 {
        self.adapter.as_ref().map_or(0, |(_, _, version)| *version)
    }

    /// The reply's `brain` extension object: the seed that reproduces the
    /// decode, the adapter that answered, the `policy_version` and, when the
    /// request asked for log-probabilities, their `logprob_kind` (`policy`: the
    /// full distribution of a policy trace; `sampler`: the sampler's truncated
    /// figure) and, for a policy trace, the `tokenizer_digest` and
    /// `chat_template_digest` of the prompt path; each present only when the
    /// model reported it. `None` when it reported none, so no empty object is
    /// sent.
    pub fn served(&self) -> Option<serde_json::Value> {
        let mut served = serde_json::Map::new();
        if let Some(seed) = self.seed {
            served.insert("seed".into(), serde_json::json!(seed));
        }
        if let Some((id, digest, version)) = &self.adapter {
            served.insert("adapter".into(), serde_json::json!({ "id": id, "digest": digest, "version": version }));
        }
        for (key, value) in [("base_digest", &self.base_digest), ("backend", &self.backend), ("engine", &self.engine)] {
            if let Some(value) = value {
                served.insert(key.into(), serde_json::json!(value));
            }
        }
        if !served.is_empty() {
            served.insert("policy_version".into(), serde_json::json!(self.policy_version()));
        }
        if self.trace.is_some() || self.logprobs.is_some() {
            served.insert("logprob_kind".into(), serde_json::json!(if self.trace.is_some() { "policy" } else { "sampler" }));
        }
        for key in ["tokenizer_digest", "chat_template_digest"] {
            if let Some(v) = self.trace.as_ref().and_then(|t| t.get(key)).filter(|v| !v.is_null()) {
                served.insert(key.into(), v.clone());
            }
        }
        (!served.is_empty()).then_some(serde_json::Value::Object(served))
    }
}

/// Read a [`ChatOutcome`] from the contract `Outcome`.
pub fn read_chat_outcome(o: &Outcome) -> ChatOutcome {
    let (text, prompt_tokens, completion_tokens, finish) = read_outcome(o);
    let reasoning = o.outputs.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let tool_calls = o
        .outputs
        .get("tool_calls")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    let seed = o.outputs.get("seed").and_then(|v| v.as_u64());
    let adapter = o.outputs.get("adapter").and_then(|a| Some((a.get("id")?.as_str()?.to_string(), a.get("digest")?.as_str()?.to_string(), a.get("version")?.as_u64()?)));
    let numbers = |key: &str| o.outputs.get(key).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|n| n.as_i64()).collect::<Vec<_>>());
    let logprobs = o.outputs.get("completion_logprobs").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|n| n.as_f64()).collect::<Vec<_>>());
    let text_of = |key: &str| o.outputs.get(key).and_then(|v| v.as_str()).map(str::to_string);
    ChatOutcome { text, prompt_tokens, completion_tokens, reasoning_tokens: o.outputs.get("reasoning_tokens").and_then(|v| v.as_i64()).unwrap_or_default(), cached_tokens: o.outputs.get("cached_tokens").and_then(|v| v.as_i64()), finish, reasoning, tool_calls, seed, adapter, base_digest: text_of("base_digest"), backend: text_of("backend"), engine: text_of("engine"), prompt_token_ids: numbers("prompt_token_ids"), completion_token_ids: numbers("completion_token_ids"), logprobs, trace: o.outputs.get("policy_trace").cloned() }
}

/// Map an executor reply error string to a provider-shaped [`ApiError`], through
/// the sorting every surface shares ([`capability::ReplyError`]): a model's own
/// classed failure, a fault, an unknown model, a FIM refusal and a capacity
/// rejection keep their meaning; the raw text of anything else is NEVER sent
/// (it can carry paths, panic text, backend internals) - it is logged here.
pub fn map_reply_err(provider: Provider, model: &str, e: &str) -> ApiError {
    let reply = capability::ReplyError::of(e);
    if reply.hides_detail() {
        eprintln!("apiserve: model '{model}' request failed: {e}");
    }
    let message = reply.client_message(model);
    match reply {
        capability::ReplyError::Classified(class, _) => ApiError::failure(provider, class, message),
        capability::ReplyError::Faulted => ApiError::failure(provider, capability::FailureClass::BackendFailure, message),
        capability::ReplyError::NoModel => ApiError::model_not_found(provider, model),
        capability::ReplyError::Capacity { .. } => ApiError::context_length_exceeded(provider, message),
        // Only a draining server cancels under an open request (a caller's
        // own cancel is answered by the job API), so it is retryable elsewhere.
        capability::ReplyError::Cancelled => ApiError::failure(provider, capability::FailureClass::Unavailable, "the request was cancelled"),
        capability::ReplyError::NoFim | capability::ReplyError::Failed => ApiError::invalid_request(provider, message),
    }
}

/// The admission race every submit/stream path below runs: wait for `admit_rx`
/// to fire (the job got claimed onto a lane), gated by a two-tier deadline
/// rather than one flat `admit_deadline`. Up to `admit_deadline`, this is a
/// plain wait — the common case, where admission is near-instant even for a
/// cold model (claiming a lane happens BEFORE the slow part,
/// `ResidentModel::activate`, ever runs). Only once that short deadline has
/// already elapsed does it pay for `residency::admission::model_is_cold_building`
/// (one blocking round-trip to the dispatcher, so run off the async runtime via
/// `spawn_blocking`): if `model` itself is already running elsewhere (same-key
/// jobs serialize onto one lane — this is almost always that model's own
/// first-ever cold activation), the wait extends to `cold_build_admit_deadline`
/// instead of shedding — a legitimate cold start is not overload. Any OTHER
/// reason admission is still pending (every lane genuinely busy with unrelated
/// models, nothing evictable) keeps shedding at the short deadline, unchanged.
/// Returns `Ok` if admitted before whichever deadline applies; a shed request is
/// logged (see `Refusal::record`) and returned as the `Err`.
/// Takes the executor/deadlines by value/reference rather than `&AppState` so
/// [`stream_with_autofetch`]'s `'static` background task can call it too.
async fn wait_for_admission(
    exec: &residency::Executor,
    deadlines: residency::admission::Deadlines,
    model: &str,
    admit_rx: &mut oneshot::Receiver<()>,
) -> Result<(), residency::admission::Refusal> {
    let start = std::time::Instant::now();
    let tick = std::time::Duration::from_millis(250).min(deadlines.admit).max(std::time::Duration::from_millis(1));
    loop {
        tokio::select! {
            _ = &mut *admit_rx => return Ok(()),
            _ = tokio::time::sleep(tick) => {}
        }
        let elapsed = start.elapsed();
        if elapsed < deadlines.admit {
            continue;
        }
        let exec_owned = exec.clone();
        let model_owned = model.to_string();
        let building = tokio::task::spawn_blocking(move || residency::admission::model_is_cold_building(&exec_owned, &model_owned)).await.unwrap_or(false);
        let limit = if building { deadlines.cold_build } else { deadlines.admit };
        if elapsed >= limit {
            return Err(residency::admission::Refusal::new(exec, model, deadlines, building, elapsed).record());
        }
    }
}

/// The 429 for a request that was not admitted in time.
fn refused(provider: Provider, refusal: &residency::admission::Refusal) -> ApiError {
    ApiError::overloaded(provider, refusal.client_message())
}

/// Submit one `generate` job and block (async) for its single reply. Arms a fresh
/// cancel token, registers it, then runs the **admission race** ([`wait_for_admission`]):
/// a bounded wait for the job to be ADMITTED (claimed onto a lane — its `on_admit`
/// signal fires). If it is not admitted in time the job is cancelled (dropped from
/// the queue so it never runs wastefully), unregistered, and the caller gets a 429.
/// Once admitted, the wait for the (possibly long) reply is UNBOUNDED — only the
/// wait-to-start is deadlined.
pub async fn submit(state: &AppState, model: &str, action: &str, mut inv: Invocation) -> Result<Outcome, ApiError> {
    let provider = state.provider;
    // Before anything is registered or queued: a refused call costs nothing.
    inv.scope = state.scope();
    let pending = Pending::begin(state, model, action, &inv)?;
    residency::log::info(&format!("{provider:?} request: {action} {model}"));
    let (id, token) = state.register();
    inv.cancel = token.clone();
    // A one-shot request is one handler future, dropped when the client hangs up: the guard then cancels the generation instead of
    // leaving it to run to its token limit for nobody (a streamed request has the same guard in its `EventStream`).
    let _guard = CancelGuard { token: token.clone(), jobs: state.jobs.clone(), id };
    let (tx, rx) = oneshot::channel();
    let (admit_tx, mut admit_rx) = oneshot::channel::<()>();
    let job = Job::new(model.to_string(), action.to_string(), inv)
        .on_progress(|_| {})
        .on_admit(move || {
            let _ = admit_tx.send(());
        })
        .reply(move |r| {
            let _ = tx.send(r);
        });
    state.exec.submit(job);

    if let Err(refusal) = wait_for_admission(&state.exec, state.deadlines(), model, &mut admit_rx).await {
        token.cancel();
        state.finish(&id);
        let _ = pending.settle(CallResult::Refused);
        return Err(refused(provider, &refusal));
    }

    // Admitted — await the reply (unbounded; a running job may take long).
    let res = rx.await;
    state.finish(&id);
    match res {
        Ok(Ok(outcome)) => {
            log_request_tokens(&state.exec, model, &outcome);
            pending.settle(CallResult::Done(&outcome))?;
            Ok(outcome)
        }
        Ok(Err(e)) => {
            let _ = pending.settle(CallResult::Failed(&e));
            Err(map_reply_err(provider, model, &e))
        }
        Err(_) => {
            let _ = pending.settle(CallResult::Failed("executor dropped the reply"));
            Err(ApiError::overloaded(provider, "executor dropped the reply"))
        }
    }
}

/// Info-level per-request token accounting: this request's own prompt/
/// completion counts (from `outcome.outputs`, exact) alongside the model's
/// CURRENT cumulative prefix-cache counters (`Executor::stats().metrics`,
/// refreshed on the dispatcher's ~250ms cadence — a snapshot at request-
/// completion time, not a delta scoped to just this one request; labeled
/// "cumulative" so that distinction is never implied to be more precise than
/// it is). Not every model publishes the KV metrics (only the paged-KV
/// serving engine does), so their absence is silently skipped, not an error.
/// Takes `&Executor` (not `&AppState`) so streaming's `'static` reply
/// closures, which only ever clone `state.exec`, can call it too.
fn log_request_tokens(exec: &residency::Executor, model: &str, outcome: &Outcome) {
    if residency::log::verbosity() < 2 {
        return; // skip the stats() round-trip entirely below info level
    }
    let prompt = outcome.outputs.get("prompt_tokens").and_then(|v| v.as_i64());
    let completion = outcome.outputs.get("completion_tokens").and_then(|v| v.as_i64());
    let mut line = format!("{model}: request done");
    if let Some(p) = prompt {
        line += &format!(", prompt_tokens={p}");
    }
    if let Some(c) = completion {
        line += &format!(", completion_tokens={c}");
    }
    let metrics = exec.stats().metrics;
    // A model that runs one instance per card has one set of counters per instance: the cache totals are their sum.
    let instances: Vec<&Vec<(String, serde_json::Value)>> = metrics.iter().filter(|(k, _)| k.model == model).map(|(_, m)| m).collect();
    if let Some(m) = instances.first() {
        let get = |name: &str| m.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone());
        let total = |name: &str| instances.iter().filter_map(|m| m.iter().find(|(n, _)| n == name)).map(|(_, v)| v.as_u64()).sum::<Option<u64>>();
        if let (Some(hit), Some(looked)) = (total("kv_prefix_hit_tokens"), total("kv_prefix_lookup_tokens")) {
            line += &format!(", kv_cache_hit_tokens={hit} (cumulative), kv_cache_lookup_tokens={looked} (cumulative)");
        }
        if let (Some(rounds), Some(accept), Some(per_round)) = (get("spec_rounds"), get("spec_accept_rate"), get("spec_tokens_per_round")) {
            if rounds.as_u64().unwrap_or(0) > 0 {
                line += &format!(", spec_rounds={rounds} spec_accept_rate={accept:.3} spec_tokens_per_round={per_round:.2}");
                for name in ["spec_verify_steps", "spec_mixed_verify_steps", "spec_plain_steps", "spec_gated_steps"] {
                    if let Some(v) = get(name) {
                        line += &format!(" {name}={v}");
                    }
                }
                line += " (cumulative)";
            }
        }
    }
    residency::log::info(&line);
}

/// One `StreamMsg::Fetching` tick: a human-readable phase description plus the
/// raw byte counts (mirrors `residency::Executor::ensure_model`'s progress
/// callback shape: `name`, `got`, `total`). Rendered as an SSE COMMENT line
/// (`: …`, RFC-legal, ignored by every conformant parser) so a cold auto-fetch
/// keeps the connection alive and visibly progressing through a multi-minute
/// download instead of holding the response back in silence — without adding
/// a new `data:` event shape that would break the vendored-OpenAPI/Anthropic
/// conformance tests.
pub struct FetchProgress {
    pub message: String,
    pub got: u64,
    pub total: Option<u64>,
}

impl FetchProgress {
    /// Render as SSE COMMENT-line text (the caller wraps it in
    /// `axum::response::sse::Event::comment`) -- e.g. `BRAIN fetching Qwen/
    /// Qwen3-0.6B: model.safetensors 34% (512/1503 KiB)` with a known total,
    /// `BRAIN fetching …: model.safetensors (512 KiB)` without one (some hub
    /// responses omit `Content-Length`).
    pub fn comment_text(&self) -> String {
        match self.total {
            Some(total) if total > 0 => {
                let pct = (self.got.min(total).saturating_mul(100) / total).min(100);
                format!("BRAIN {} {pct}% ({}/{} KiB)", self.message, self.got / 1024, total / 1024)
            }
            _ => format!("BRAIN {} ({} KiB)", self.message, self.got / 1024),
        }
    }
}

/// One item on a streaming generation: an incremental text piece, a structured
/// out-of-band event (reasoning/tool-call progress — see [`capability::Progress::event`]
/// and `resident_llm.rs::emit_chat_events`'s neutral `{"kind":...}` shapes; chat
/// streaming renders these into `delta.reasoning_content`/`delta.tool_calls`,
/// NEVER into `delta.content` — that structural separation is what keeps raw
/// `<think>`/`<tool_call>` markup from ever leaking into plain content deltas), a
/// coarse `(step, total)` progress tick (used by image generation's denoise loop;
/// chat streaming ignores it), a cold-auto-fetch progress tick (see
/// [`FetchProgress`] — only ever produced by [`stream_with_autofetch`]), the
/// terminal outcome (full text/image + counts), or an error.
pub enum StreamMsg {
    Delta(String),
    Event(serde_json::Value),
    Progress(u32, u32),
    Fetching(FetchProgress),
    Done(Outcome),
    Err(ApiError),
}

/// Dropped when the returned [`EventStream`] is dropped (SSE finished OR the client
/// disconnected): cancels the job's token and clears its registry entry, so a
/// disconnect aborts the running generation (cancel-on-disconnect).
pub struct CancelGuard {
    token: capability::CancelToken,
    jobs: JobRegistry,
    id: Uuid,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.token.cancel();
        self.jobs.remove(&self.id);
    }
}

/// The receiver side of a streaming generation plus its [`CancelGuard`]. Yields
/// [`StreamMsg`]s until the reply's `Done`/`Err`; dropping it cancels the job.
pub struct EventStream {
    rx: mpsc::UnboundedReceiver<StreamMsg>,
    _guard: CancelGuard,
    /// The first message, when [`EventStream::settle_start`] read it to see whether the
    /// request had already failed; handed out before anything else.
    head: Option<StreamMsg>,
}

impl Stream for EventStream {
    type Item = StreamMsg;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<StreamMsg>> {
        if let Some(first) = self.head.take() {
            return Poll::Ready(Some(first));
        }
        self.rx.poll_recv(cx)
    }
}

/// How long a streamed request is given to fail before its response headers go out. A
/// request the model refuses outright (too long for its context, an unknown adapter, a
/// malformed message) is refused within milliseconds; the client then gets the HTTP status
/// an unstreamed request would have got, instead of a 200 whose body is one error event.
/// A request that is merely slow to produce its first token is not held up by this: its
/// stream opens as soon as the window ends.
const EARLY_FAILURE_WINDOW: std::time::Duration = std::time::Duration::from_millis(300);

impl EventStream {
    /// Wait up to [`EARLY_FAILURE_WINDOW`] for the request's first message. A failure that
    /// arrives within it is returned as the error; anything else (or nothing yet) leaves
    /// the stream as it was.
    pub async fn settle_start(mut self) -> Result<EventStream, ApiError> {
        match tokio::time::timeout(EARLY_FAILURE_WINDOW, self.rx.recv()).await {
            Ok(Some(StreamMsg::Err(e))) => Err(e),
            Ok(first) => {
                self.head = first;
                Ok(self)
            }
            Err(_elapsed) => Ok(self),
        }
    }
}

/// Submit one `generate` job whose progress/reply fan into a channel, running the
/// same **admission race** as [`submit`] BEFORE returning the stream: a bounded wait
/// (`state.admit_deadline`) for the job to be claimed onto a lane. If it is not
/// admitted in time the job is cancelled + unregistered and a 429 [`ApiError`] is
/// returned — so the caller returns a plain 429 body, NOT an SSE stream that then
/// errors. Once admitted, the returned [`EventStream`] yields a [`StreamMsg::Delta`]
/// per generated token (progress updates whose `delta` is set — coarse step-only
/// progress is ignored), then one terminal [`StreamMsg::Done`]/[`StreamMsg::Err`].
/// The `on_progress`/`reply` closures only touch the (`Send`) channel — never the
/// model.
pub async fn stream(state: &AppState, model: &str, action: &str, inv: Invocation) -> Result<EventStream, ApiError> {
    stream_inner(state, model, action, inv, false).await?.settle_start().await
}

/// Like [`stream`], but also forwards coarse `(step, total)` progress ticks as
/// [`StreamMsg::Progress`] (a denoise loop reports these with no `delta`). Used by
/// image generation, whose "progress" is denoise steps, not token deltas.
pub async fn stream_progress(state: &AppState, model: &str, action: &str, inv: Invocation) -> Result<EventStream, ApiError> {
    stream_inner(state, model, action, inv, true).await
}

/// Shared implementation of [`stream`]/[`stream_progress`]. `forward_steps` decides
/// whether a `delta`-less progress update becomes a [`StreamMsg::Progress`] tick
/// (image denoise) or is dropped (chat, which streams only token deltas).
async fn stream_inner(state: &AppState, model: &str, action: &str, mut inv: Invocation, forward_steps: bool) -> Result<EventStream, ApiError> {
    let provider = state.provider;
    inv.scope = state.scope();
    let pending = Pending::begin(state, model, action, &inv)?;
    residency::log::info(&format!("{provider:?} request: {action} {model} (stream)"));
    let model_owned = model.to_string();
    let (id, token) = state.register();
    inv.cancel = token.clone();
    // Held from here, not from admission: a client that hangs up while its job is still waiting for a lane
    // drops this future, and the guard then cancels the queued job instead of leaving it to run, later, for nobody.
    let guard = CancelGuard { token: token.clone(), jobs: state.jobs.clone(), id };
    let (tx, rx) = mpsc::unbounded_channel::<StreamMsg>();
    let (admit_tx, mut admit_rx) = oneshot::channel::<()>();
    let tx_progress = tx.clone();
    let exec_for_log = state.exec.clone();
    let job = Job::new(model.to_string(), action.to_string(), inv)
        .on_progress(move |p| {
            if let Some(piece) = p.delta {
                let _ = tx_progress.send(StreamMsg::Delta(piece));
            } else if let Some(ev) = p.event {
                let _ = tx_progress.send(StreamMsg::Event(ev));
            } else if forward_steps {
                let _ = tx_progress.send(StreamMsg::Progress(p.step, p.total));
            }
        })
        .on_admit(move || {
            let _ = admit_tx.send(());
        })
        .reply(move |r| {
            let msg = match r {
                Ok(outcome) => {
                    log_request_tokens(&exec_for_log, &model_owned, &outcome);
                    // Settled here, on the job's own reply, so it happens once
                    // whether or not the client is still reading.
                    match pending.settle(CallResult::Done(&outcome)) {
                        Ok(()) => StreamMsg::Done(outcome),
                        Err(refusal) => StreamMsg::Err(refusal),
                    }
                }
                Err(e) => {
                    let _ = pending.settle(CallResult::Failed(&e));
                    StreamMsg::Err(map_reply_err(provider, &model_owned, &e))
                }
            };
            let _ = tx.send(msg);
        });
    state.exec.submit(job);

    // Admission race BEFORE returning the SSE body: a shed request yields a plain
    // 429, never an event-stream that immediately errors.
    if let Err(refusal) = wait_for_admission(&state.exec, state.deadlines(), model, &mut admit_rx).await {
        // Dropping the guard cancels the queued job and unregisters it.
        return Err(refused(provider, &refusal));
    }

    Ok(EventStream { rx, _guard: guard, head: None })
}

/// Like [`stream`]/[`stream_progress`], but for a `model` that is NOT already
/// resident: returns the [`EventStream`] IMMEDIATELY (before any network I/O),
/// then — in a background task — classifies and (if `Fetchable`) fetches the
/// model with its progress forwarded as [`StreamMsg::Fetching`] ticks, and only
/// once it's ready submits the real job under the normal admission race,
/// continuing to forward `Delta`/`Progress`/`Done`/`Err` into the SAME stream.
/// The caller (e.g. `openai::handle_generate`) is expected to have
/// already confirmed `state.supplier.is_some()` and classified `Fetchable`
/// with zero I/O (`ModelSupplier::classify`) BEFORE calling this — an `Unknown`
/// or no-supplier model must still be a plain 404 with no SSE body opened at
/// all, exactly like [`crate::bridge::ensure_and_recheck`]'s non-streaming
/// sibling.
///
/// A 429 (admission failed after the fetch completed) becomes a terminal
/// [`StreamMsg::Err`] here, not an HTTP 429 status — the SSE body's headers
/// are already committed by the time admission is even attempted.
pub fn stream_with_autofetch(state: &AppState, supplier: std::sync::Arc<dyn residency::ModelSupplier>, model: &str, action: &str, mut inv: Invocation, forward_steps: bool) -> Result<EventStream, ApiError> {
    let provider = state.provider;
    // Before the download starts: a caller who may not run the call must not be
    // able to make this server fetch a model first.
    inv.scope = state.scope();
    let pending = Pending::begin(state, model, action, &inv)?;
    residency::log::info(&format!("{provider:?} request: {action} {model} (stream, not yet resident -- auto-fetching)"));
    let (id, token) = state.register();
    let (tx, rx) = mpsc::unbounded_channel::<StreamMsg>();
    let exec = state.exec.clone();
    let jobs = state.jobs.clone();
    let deadlines = state.deadlines();
    let model_owned = model.to_string();
    let action_owned = action.to_string();
    let mut inv = inv;
    inv.cancel = token.clone();
    let guard_token = token.clone();

    tokio::spawn(async move {
        let tx_fetch = tx.clone();
        let fetch_model = model_owned.clone();
        let fetch_exec = exec.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            fetch_exec.ensure_model(&fetch_model, supplier.as_ref(), &mut |name, got, total| {
                let _ = tx_fetch.send(StreamMsg::Fetching(FetchProgress {
                    message: format!("fetching {fetch_model}: {name}").replace(['\n', '\r'], " "),
                    got: got as u64,
                    total: if total == 0 { None } else { Some(total as u64) },
                }));
            })
        })
        .await;
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => {
                eprintln!("apiserve: auto-fetch {model_owned}: {reason}");
                let _ = tx.send(StreamMsg::Err(ApiError::model_not_found(provider, &model_owned)));
                jobs.remove(&id);
                return;
            }
            Err(join_err) => {
                eprintln!("apiserve: auto-fetch {model_owned}: blocking task failed: {join_err}");
                let _ = tx.send(StreamMsg::Err(ApiError::model_not_found(provider, &model_owned)));
                jobs.remove(&id);
                return;
            }
        }
        // The fetched model's manifest is known now: images it cannot read are refused before the job is queued.
        if let Err(e) = refuse_unread_media(&exec, provider, &model_owned, &action_owned, &inv) {
            let _ = tx.send(StreamMsg::Err(e));
            jobs.remove(&id);
            return;
        }

        // Model is ready -- submit the real job, forwarding into the SAME
        // channel the fetch ticks just used, so the client sees one
        // continuous stream (fetch progress, then generation deltas). A
        // dedicated clone for the admission-timeout branch below: the other
        // two are moved into the job's closures and won't be available if
        // admission times out before either ever fires.
        let tx_timeout = tx.clone();
        let (admit_tx, mut admit_rx) = oneshot::channel::<()>();
        let tx_progress = tx.clone();
        let tx_reply = tx;
        let model_for_admit = model_owned.clone();
        let exec_for_log = exec.clone();
        let job = Job::new(model_owned.clone(), action_owned, inv)
            .on_progress(move |p| {
                if let Some(piece) = p.delta {
                    let _ = tx_progress.send(StreamMsg::Delta(piece));
                } else if let Some(ev) = p.event {
                    let _ = tx_progress.send(StreamMsg::Event(ev));
                } else if forward_steps {
                    let _ = tx_progress.send(StreamMsg::Progress(p.step, p.total));
                }
            })
            .on_admit(move || {
                let _ = admit_tx.send(());
            })
            .reply(move |r| {
                let msg = match r {
                    Ok(outcome) => {
                        log_request_tokens(&exec_for_log, &model_owned, &outcome);
                        match pending.settle(CallResult::Done(&outcome)) {
                            Ok(()) => StreamMsg::Done(outcome),
                            Err(refusal) => StreamMsg::Err(refusal),
                        }
                    }
                    Err(e) => {
                        let _ = pending.settle(CallResult::Failed(&e));
                        StreamMsg::Err(map_reply_err(provider, &model_owned, &e))
                    }
                };
                let _ = tx_reply.send(msg);
            });
        exec.submit(job);

        if let Err(refusal) = wait_for_admission(&exec, deadlines, &model_for_admit, &mut admit_rx).await {
            token.cancel();
            jobs.remove(&id);
            let _ = tx_timeout.send(StreamMsg::Err(refused(provider, &refusal)));
        }
    });

    Ok(EventStream { rx, _guard: CancelGuard { token: guard_token, jobs: state.jobs.clone(), id }, head: None })
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── failure classes ─────────────────────────────────────────────────────

    #[test]
    fn a_classed_failure_maps_to_its_own_status_and_code_with_its_message() {
        use capability::{tag, FailureClass};
        let cases = [
            (FailureClass::NotInstalled, 404, "not_installed"),
            (FailureClass::FailedToLoad, 500, "failed_to_load"),
            (FailureClass::UnsupportedCapability, 422, "unsupported_capability"),
            (FailureClass::Timeout, 504, "timeout"),
            (FailureClass::InvalidRequest, 422, "invalid_request"),
            (FailureClass::BackendFailure, 500, "backend_failure"),
            (FailureClass::Unavailable, 503, "unavailable"),
        ];
        for (class, status, code) in cases {
            let e = map_reply_err(Provider::OpenAI, "m", &tag(class, "the message"));
            assert_eq!(e.kind.status().as_u16(), status, "{code}");
            let body = e.body();
            assert_eq!(body["error"]["code"], code);
            assert_eq!(body["error"]["message"], "the message");
        }
    }

    #[test]
    fn a_full_card_is_a_503_that_tells_the_client_when_to_retry() {
        use axum::response::IntoResponse;
        use capability::{tag, FailureClass};
        let e = map_reply_err(Provider::OpenAI, "m", &tag(FailureClass::Unavailable, "qwen35: the device ran out of memory"));
        let response = e.into_response();
        assert_eq!(response.status().as_u16(), 503);
        assert!(response.headers().contains_key(axum::http::header::RETRY_AFTER), "a retryable refusal says so");
    }

    #[test]
    fn a_scheduler_prefix_does_not_hide_the_class() {
        let e = map_reply_err(Provider::OpenAI, "m", "activate acme/model/default: [failed-to-load] the checkpoint is corrupt");
        assert_eq!(e.body()["error"]["code"], "failed_to_load");
        assert_eq!(e.body()["error"]["message"], "the checkpoint is corrupt");
    }

    #[test]
    fn a_lane_panic_is_a_backend_failure_that_leaks_nothing() {
        let e = map_reply_err(Provider::OpenAI, "m", "m/default: panicked while running: index out of bounds at /home/secret/weights.rs");
        let body = e.body();
        assert_eq!(body["error"]["code"], "backend_failure");
        assert!(!body["error"]["message"].as_str().unwrap().contains("secret"));
    }

    /// A job cancelled under a request (the server draining at shutdown) is
    /// a 503 the client may retry elsewhere, not its own invalid request.
    #[test]
    fn a_cancelled_job_is_unavailable_not_a_client_error() {
        let e = map_reply_err(Provider::OpenAI, "m", "cancelled");
        assert_eq!(e.kind.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(e.body()["error"]["message"], "the request was cancelled");
    }

    #[test]
    fn an_unclassed_failure_is_still_the_generic_refusal() {
        let e = map_reply_err(Provider::OpenAI, "m", "backend panic at /home/secret/weights.gguf: kernel exploded");
        assert_eq!(e.body()["error"]["message"], "the model failed to process the request");
    }

    // ── parse_exceeds_capacity ──────────────────────────────────────────────

    #[test]
    fn parse_exceeds_capacity_matches_the_real_reject_reason_shape() {
        // Exactly what resident_llm.rs produces: format!("qwen: {reason}")
        // where reason is RejectReason::ExceedsCapacity's Display impl.
        let got = capability::failure::parse_exceeds_capacity("qwen: needs 14995 tokens, engine capacity is 2048");
        assert_eq!(got, Some((14995, 2048)));
    }

    #[test]
    fn parse_exceeds_capacity_ignores_unrelated_qwen_errors() {
        // Other qwen error paths share the "qwen:" prefix but are NOT this
        // shape and may embed on-disk paths - must not be misparsed.
        assert_eq!(
            capability::failure::parse_exceeds_capacity("qwen: folding adapter /home/user/lora.safetensors: bad shape"),
            None
        );
    }

    #[test]
    fn parse_exceeds_capacity_ignores_other_reject_reasons() {
        assert_eq!(capability::failure::parse_exceeds_capacity("qwen: rejected by admission_policy"), None);
        assert_eq!(
            capability::failure::parse_exceeds_capacity("qwen: token 999999 is outside the vocabulary (151936)"),
            None
        );
    }

    #[test]
    fn parse_exceeds_capacity_ignores_generic_backend_errors() {
        assert_eq!(capability::failure::parse_exceeds_capacity("backend panic at /home/secret/weights.gguf: kernel exploded"), None);
    }

    // ── map_reply_err ────────────────────────────────────────────────────────

    #[test]
    fn map_reply_err_capacity_rejection_is_context_length_exceeded_400() {
        let err = map_reply_err(Provider::OpenAI, "brain-chat", "qwen: needs 14995 tokens, engine capacity is 2048");
        assert_eq!(err.kind.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(err.body()["error"]["code"], "context_length_exceeded");
        assert!(err.message.contains("14995"), "message must name the safe counts: {}", err.message);
        assert!(err.message.contains("2048"), "message must name the safe counts: {}", err.message);
    }

    #[test]
    fn map_reply_err_generic_backend_error_stays_generic_and_never_leaks() {
        let err = map_reply_err(
            Provider::OpenAI,
            "brain-chat",
            "backend panic at /home/secret/models/weights.gguf: kernel exploded",
        );
        assert_eq!(err.body()["error"]["code"], "invalid_request");
        assert!(!err.message.contains("secret"), "must not leak: {}", err.message);
        assert!(!err.message.contains("weights.gguf"), "must not leak: {}", err.message);
        assert!(!err.message.contains("panic"), "must not leak: {}", err.message);
    }

    #[test]
    fn map_reply_err_no_model_is_model_not_found() {
        let err = map_reply_err(Provider::OpenAI, "brain-chat", "no model resolved for 'brain-chat'");
        assert_eq!(err.kind.status(), axum::http::StatusCode::NOT_FOUND);
    }
}
