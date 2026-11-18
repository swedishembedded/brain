<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 518. A chat request that left a sampling setting out was drawn with the parser's constants, not the defaults the manifest advertised

**Lesson:** A param default in an `ActionSpec` is applied only by `ActionSpec::validate`. A path that hands an
invocation to an instance without validating it (the HTTP chat surfaces) gets whatever the model's own parser falls
back to, while the manifest, `/v1/models` and the docs show the declared value. Echo what the sampler applied, and test
the echo through the surface, not the spec.

## What happened

Since 2026-10-27 a served chat model's `generate` action carries the checkpoint's declared sampling as its param
defaults (`llmserve::with_sampling_defaults`, `catalog::sampling_defaults`): Qwen3.8-27B declares temperature 1.0, top-k
20, top-p 0.95. Those defaults reached a request only through validation, which the generic routes (`/v1/run`,
`/v1/jobs`), D-Bus `Run` and a `ProviderInstance` perform. The OpenAI, OpenRouter and Anthropic chat and completions
routes built the invocation from the request body, forwarded only the fields the client sent (`sampling::apply`) and
queued it unvalidated; a scheduler-backed instance (`llmserve::ScheduledInstance`) parses it with
`qwen3::chat::sampling_params`, whose constants are temperature 0.8, top-k 40, top-p 1.0. Every chat request that
omitted a setting, every rollout that relied on "the model's defaults" among them, was drawn at those constants.

Found by writing the reply's sampler echo (`brain.sampler`, step 13 of the training-regimes plan): the surface test
asked for the model's defaults and the echo read `{"min_p": 0}`.

## Fix

`sampling::settle` runs on every queued job (`bridge::submit`, `StreamedJob::build`): each sampling param the request
did not set takes the default the served action declares; a neutral value the client sent (`min_p: 0`,
`repetition_penalty: 1`, a zero penalty) is kept for a model that has that sampler, so it overrides a declared default,
and dropped for one that does not. `sampling::apply` forwards neutral values for that reason. The reply's `brain.sampler`
(and the outcome's `sampler` over D-Bus) names `temperature`, `top_k`, `top_p` and `min_p` as the scheduler drew with
them (`llmserve::sampler_settings`, from the `SampleParams` handed to the scheduler).

Tests: `apiserve` `every_dialect_echoes_the_sampler_settings_the_reply_was_drawn_with` (OpenAI chat and completions,
OpenRouter, Anthropic, each streamed and not: the defaults when omitted, the request's own when sent),
`sampling::tests::an_omitted_setting_takes_the_default_the_served_action_declares`; `llmserve`
`a_reply_names_the_sampler_settings_its_tokens_were_drawn_with` (the echoed settings reproduce the sampler's reported
log-probability of every drawn token).

## Checked on real weights (2026-11-18)

Qwen3.8-27B-FP8 through `brain serve` (98e3a8024, one GH200): a chat request with no sampling field echoed
`brain.sampler: {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}`, the checkpoint's
`generation_config.json`; one that sent temperature 0.5, top_k 0, top_p 1, min_p 0 echoed those.

## Every param, every route (2026-11-18)

The sampling-only fix left two gaps: other params with spec defaults (`enable_thinking`, `reasoning_effort`,
`template_flavor`) reached a chat instance only as the parser's fallbacks, and a param the instance does not declare
was not refused on the chat routes. Both are closed by validating the call itself:

- `bridge::admit` is the one validation every HTTP route runs before it queues a call - OpenAI, OpenRouter and
  Anthropic chat, completions, embeddings, images, transcription, speech, video, `/v1/run`, `/v1/jobs`, systemone. It
  runs `ActionSpec::check_served` (the typed form of `validate_served`; `validate` is its `Display`) and
  `Executor::check_job`; the dispatchers (`bridge::submit`, `stream`, `stream_progress`) take only its `Admitted`, so
  no route can queue an unvalidated call. `sampling::settle` now only drops a neutral sampler value the model has no
  sampler for; every default comes from validation. D-Bus `Run`/`Subscribe` already validated (`served_invocation`).
  Before this, images, transcription and systemone's `select` queued unvalidated calls too.
- A param the action does not declare is a 400 `unsupported_parameter` naming the field (by the request field that
  sets it on a chat route: `temperature` for `temp`) and the model.
- A top-level body field the dialect does not define is refused the same way (`request_fields::refuse_unknown`), except
  the fields in `request_fields::IGNORED`, each accepted with the reason ignoring it answers the request sent
  (`docs/reference/api/http.md` lists them).
- Declaring every param a front end reads became the specs' job: `llmserve::frontend_params` (what `TextFrontend` and
  the shared `qwen3::chat` parser read: `ignore_eos`, the thinking budget, the extended sampler) and
  `qwen3::chat::template_params(flavor)` (`reasoning_effort`, `preserve_thinking`, `template_flavor` with the model's
  own flavor). The served Qwen3.8-27B GGUF spec lacked the first group, the fp32 Qwen3.8, Qwen3.5-MoE and EdgeRock
  residents the second: with validation on, a request using them would have been refused although the engine honours
  them.

Tests: `apiserve` `every_chat_route_holds_its_call_to_the_served_spec` (each dialect, streamed and not: omitted
sampler and template settings take the declared defaults, an undeclared or undefined field is refused naming it and
the model, an ignored field passes), `request_fields::tests`; `capability`
`a_refused_invocation_names_its_field_and_what_is_wrong`; `dbus`
`run_and_subscribe_refuse_an_undeclared_param_naming_it_and_the_model`.

## Not covered

A model whose chat format is chosen per checkpoint when it is activated (the Qwen3 resident, Huginn, Ouro, Ripple:
ChatML or the checkpoint's own template) declares one static spec before it knows. The Qwen3 resident declares the
ChatML template settings, which a checkpoint with its own template accepts and does not read; `chat_template_kwargs`
(read by a checkpoint template) is declared only by Ripple, so the others refuse it.
