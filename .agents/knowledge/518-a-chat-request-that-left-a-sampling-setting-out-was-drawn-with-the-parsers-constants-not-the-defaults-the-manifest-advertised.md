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

## Not covered

Other params with spec defaults (`enable_thinking`, `reasoning_effort`, `template_flavor`) still reach a chat instance
only as the parser's fallbacks, which match the declared defaults today by hand; a param the instance does not
declare is not refused on the chat routes either. Validating the chat invocation against the served spec would close
both and is a larger change (every chat spec would have to declare every field a surface forwards).
