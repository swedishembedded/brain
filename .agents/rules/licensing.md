# Licensing & copyright provenance (mandatory reading before porting any model)

**Invariant.** brain claims a single blanket `Apache-2.0` for the whole
workspace (`Cargo.toml:140`). That claim is only true if every crate's Rust
is actually brain's own copyrightable expression, and every place it isn't
(a transcribed table, a ported formula) carries an attribution entry in
`/NOTICE.md`. Two axes, never conflate them:

1. **Code license**  -  is `crates/<model>`'s Rust brain's own work? Tracked
   in `/NOTICE.md`.
2. **Weight/checkpoint license**  -  what terms does the *released checkpoint*
   carry? A completely independent question  -  a checkpoint can be
   non-commercial-only even when the code that loads it is clean, and vice
   versa. Tracked in `docs/compliance/third-party-models.md`.

A 2026-09-03 full-repo audit (`.agents/roadmap/licensing-audit.md`) checked
this against real upstream source for ~60 model crates. The result: the
methodology below, where actually followed, produces genuinely independent
code  -  confirmed by direct comparison against real upstream files, not
assumed. Keep following it; it is both the correct engineering practice
(`.agents/rules/porting.md`) and the thing that keeps brain's copyright
claim real. **We want compliance, not retreat**  -  don't over-attribute or
delete work that's genuinely brain's own out of excess caution; do attribute
precisely, and only, what's actually transcribed.

## What keeps code brain's own

Per `.agents/rules/porting.md` §0: consult an upstream reference for
**facts**  -  the paper's algorithm, the checkpoint's own tensor names/shapes,
published hyperparameters, a config file. Then **write the Rust yourself**
against brain's own kernel/tensor-op engine (`crates/kernels`, `crates/vae`,
`crates/model`, ...), and prove correctness with a **numerical** parity/
gradient-check gate against the reference's real output  -  never by having
the reference's source file open while writing the Rust function it
corresponds to.

Facts you may freely use, because they aren't copyrightable expression:
- Tensor/parameter names and shapes, straight from the checkpoint header or
  the reference's `state_dict`  -  you need these verbatim to load the real
  weights at all.
- Published hyperparameters, formulas, and algorithm names (CIoU, DFL,
  task-aligned assignment, YaRN, RoPE, SwiGLU, ...)  -  these come from papers,
  not from any one implementation.
- The *order of operations* a model performs, when it's dictated by the
  architecture itself (there is only one way to compute a published residual
  block that matches the checkpoint's numbers).

## What does NOT keep code brain's own  -  attribute it

If you read a **specific named upstream file** and your Rust function's
decomposition, control flow, or a **literal table** (constants, config
values, lookup tables) traces to it beyond what the architecture's facts
require, that's a translation of expression, not just facts, regardless of
how small. Two remediation paths, same in every case:

1. Add a one-line provenance comment at the point of transcription, naming
   the exact upstream file/function and license.
2. Add (or extend) the matching entry in `/NOTICE.md` at the repo root, with
   the upstream's copyright holder and license text.

This applies even under a **compatible** license (MIT/Apache-2.0)  -  MIT and
Apache-2.0 §4(c) both have a real, textual attribution-retention
requirement; "we're also Apache-2.0" does not waive it. The audit found this
gap on `crates/model/src/yarn.rs` (HF `transformers`), `crates/checkpoint/
src/gguf.rs` + `crates/gguf/src/kquant.rs` + `crates/qwen3/src/
gguf_import.rs` (ggml/llama.cpp), `crates/zipdepth`, `crates/rrdbnet`,
`crates/cosyvoice`, `crates/s3tokenizer`, and `crates/kronos`  -  all now fixed
in `/NOTICE.md`, all under permissive upstream licenses, none requiring a
rewrite.

## Never consult a restrictively-licensed upstream's source while writing

Do not open the source of an AGPL/GPL, "non-commercial", or "research-only"
licensed reference implementation while writing brain's Rust for that
architecture. Use instead: the paper, the checkpoint's own tensor manifest,
and  -  if you want a second implementation to cross-check conventions
against  -  a **permissively-licensed** reimplementation of the same published
algorithm (this is exactly how `crates/yolov8`'s Task-Aligned Assigner
stayed clean of Ultralytics' AGPL `tal.py`: the algorithm is from the TOOD
paper, independently reimplemented, verified only against the published
formula). If no permissive reference exists and the restrictive one is the
only source of the needed facts, stop and escalate  -  don't proceed on the
assumption that "I'll just look at the shapes."

## Before wiring up any new checkpoint auto-fetch

1. Look up the actual checkpoint's license (not just the code repo's  -  they
   can differ; see FastVLM's Apple code license vs. its separate
   `LICENSE_MODEL`).
2. Add a row to `docs/compliance/third-party-models.md`.
3. If it's anything other than Apache-2.0/MIT/BSD, add the restriction
   **inline** in `docs/models/<name>.md` too  -  the matrix alone is not
   enough, per the FastVLM/CodeFormer/YOLOv8 gaps the audit found (silent
   one-command fetch, zero warning).
4. If it's non-commercial, revenue-capped, field-of-use-restricted, or
   territory-excluded, gate the auto-fetch behind an explicit opt-in env var
   (`BRAIN_<MODEL>_ALLOW_NC`, following `flux2`'s `BRAIN_FLUX2_ALLOW_NC`
   precedent)  -  never a silent one-command default.
5. If the restriction is a **contract** rather than a copyright license
   (e.g. TimesFM-3's Non-Commercial License, accepted by download/use, with
   its own broad "Derivative" definition), flag it for an actual legal read
   before shipping  -  a from-scratch independently-written port is not
   automatically outside a contract's own definition of "Derivative," even
   though it would be outside copyright's.

## Re-audit cadence

Re-run this checklist's spirit whenever: a new model crate lands (add its
row to `docs/compliance/third-party-models.md` in the same PR that adds the
crate  -  same discipline as updating `AGENTS.md`'s model table), or roughly
annually as a full sweep. Ledger findings in
`.agents/roadmap/licensing-audit.md`, the same pattern as `.agents/rules/
lessons.md`.
