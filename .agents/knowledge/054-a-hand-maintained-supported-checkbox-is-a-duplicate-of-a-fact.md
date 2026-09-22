<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 54. A hand-maintained "Supported" checkbox is a duplicate of a fact the code already derives, and three of them were wrong in two directions

Every `docs/models/<model>.md` opens with a support table:

```
| HTTP API               | [x] |
| D-Bus                  | [x] |
```

Nothing computes those. But `crates/apiserve/src/catalog.rs::api_caps` DOES
compute the HTTP half, from the manifest's action *shape* rather than from any
per-model list:

- **chat** (`/v1/chat/completions`, `/v1/messages`): an action named `generate`
  that **is `.streaming()`**, takes `prompt`/`messages`/`text`, and outputs
  `Media::Text`.
- **embeddings** (`/v1/embeddings`): an action named `embed` with a `text` param.
- **image** (`/v1/images/generations`): an action outputting `Media::Image`,
  taking `prompt`, with no required input blob.

Checked against that rule, three tables were wrong, and not all in the same
direction:

- **`gpt2.md` claimed HTTP `[x]`.** `resident_llm::GptResident` passes
  `generate_spec(..., chat=false)`, and `generate_spec` only calls
  `.streaming()` in the `chat` branch - so the char-level GPT baseline has
  never been on the HTTP dialects. Overstated.
- **`glmdsa.md` claimed HTTP `[x]`** - written *in this session*, by reasoning
  "GLM is registered in `build_executor`, and D-Bus and HTTP are both backed by
  the same `Executor`, so both are reachable". The premise is true and the
  conclusion is false: the same executor backs both, but the HTTP dialects
  additionally filter by action shape and D-Bus `Run` does not. Overstated.
- **`qwen35moe.md` claimed HTTP `[ ]` and D-Bus `[ ]`.** Its resident passes
  `chat=true` and is registered in `build_executor`. Understated - the more
  dangerous direction, because nobody goes looking for a feature the docs say
  does not exist.

Two things to carry:

1. **"Served" is not one fact.** D-Bus `Run` dispatches any capability action
   generically; the OpenAI/Anthropic routes dispatch three specific *shapes*. A
   model can be resident, scheduled, and D-Bus-reachable while being invisible
   to `/v1/chat/completions` - and the thing that decides is one bool
   (`streaming`) buried in an `ActionSpec` builder, not anything in the model
   crate.
2. **Do not "fix" the bool to fix the docs.** Adding `.streaming()` to a spec
   whose action generates its whole output then returns makes the manifest lie
   to the router. Per-token `Progress::delta` emission comes first; the flag
   then describes what is true. (Recorded as an open item on
   `.agents/roadmap/glmdsa.md`.)

The durable fix is a test that derives each table's HTTP row from `api_caps`
over the real catalog instead of trusting the checkbox - the same reasoning as
`make kernels-table/check`, which exists because a hand-written table of a
derivable fact always drifts. Not yet written; the model-id-to-doc-page mapping
it needs is itself drift-prone and wants designing rather than guessing.
