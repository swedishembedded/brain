# Porting playbook — from reference model to parity-proven implementation

The distilled process for taking a new architecture from zero to proven
forward parity and a gradchecked trainer, minimizing the number of real bugs
along the way. Follow it in order; every step exists because skipping it cost
time somewhere. The gates are cheap compared to debugging a many-layer model
that is "mostly right".

## 0. Facts before code

- **Three independent references, one authority.** Fetch the official repo
  (authority on math), a widely-used reimplementation (tensor naming, pipeline
  glue), and a third ecosystem tool if one exists (a third opinion +
  conventions like LoRA key layouts). Where they disagree, say so in writing
  and trust the authority — but *verify the disagreement empirically before
  building on it*. A shift/scale chunk order that differs between two
  reference implementations is exactly the kind of disagreement an import
  test should settle, not a coin flip.
- **Checkpoint headers are free architecture docs.** A safetensors JSON header
  or a GGUF header (range-request the first few MB) yields the complete tensor
  name/shape manifest without downloading weights — enough to derive hidden
  sizes, block counts, fused layouts, and (by absence) missing features.
- **Secondhand write-ups lie in the details.** A blog/walkthrough can get a
  sigma schedule or chat template subtly wrong. Use them for orientation only;
  annotate known errors where you store them.

## 1. Dump reference goldens FIRST (before any Rust)

One Python script (`tools/<model>_dump_reference.py`), CPU + fp32, fixed seeds,
everything saved as **f32** (brain's safetensors reader is F32/F16/BF16-only —
ints cast exactly):

- **Stage taps**: tokenizer output + text-encoder hidden states; every VAE
  boundary (moments → mean → packed/normalized latent → decode); schedule
  sigma vectors for several (resolution, steps) combos.
- **Transformer I/O via forward hooks during a REAL pipeline run.** Do not
  hand-assemble model inputs — hook the module and capture
  `hidden_states/ctx/timestep/ids` + output exactly as the pipeline produced
  them. This freezes every convention you would otherwise re-derive wrong
  (packing order, id layout, timestep scaling), and the Rust parity test
  becomes a pure replay.
- **Per-step latents via the step-end callback** — enables composed-loop
  parity later *without* reproducing torch's RNG (replay from the first
  captured post-step latent).
- **Self-validate inside the dumper**: compute the same quantity two ways
  (e.g. a manual text path vs the pipeline's own encode helper) and assert
  they agree.
- Write a `manifest.json` (shapes + sha256 + run params + library versions).

## 2. Import with two-way coverage, split fused weights at the boundary

- Derive a **canonical tensor manifest** from the config: every name + shape,
  counted against the real checkpoint (assert the count).
- Import validates **both directions**: any missing tensor errors *by name*;
  any unused source tensor errors. Never zero-fill, never skip.
- **Split fused weights on the host at import time** (qkv thirds, gate/up MLP
  halves, column-split output projections). Every device matmul then reads a
  whole buffer — no offset-view gymnastics in the hot path, and the trainer
  reuses the same splits.

## 3. Know the kernel contract before dispatching it

Assumption-vs-contract mismatches on *existing* kernels are a recurring class
of real bug, and both of the ones below have shipped more than once across
different ports:

1. An offset/length param that is **f32 elements, not bytes** (→ SIGSEGV).
2. A `Params` struct that is a single combined field where the caller assumed
   several separate ones (→ silently computes over the wrong slice; forward
   cosine can still look plausible while being badly wrong).

Rules that catch both up front:
- Read the kernel's **header comment and Params struct** for every kernel you
  dispatch (`sed -n '/struct Params/,/^};/p' crates/kernels/wgsl/<k>.wgsl`).
- Find one **existing call site** and copy its params/threads idiom
  (`grep -rn 'K_<KERNEL>\|<kernel_name>' crates/<model>/src`).
- New kernels only when semantics genuinely don't exist — check first per
  `.agents/rules/kernels.md` §A.

## 4. Tiny-config smoke test before real weights

A weight-free forward at toy dims (small hidden size, a couple of blocks, fake
tensors from the manifest) exercising **every step kind** runs in under a
second and catches buffer-sizing/binding bugs that a many-minute real-weights
parity run would hit opaquely. Make it bisectable from day one:
- block counts overridable by env,
- a step-truncation env (submits only the first k steps) — this kind of knob
  has localized a units bug (element-vs-byte offsets) in a handful of runs.

## 5. Climb the parity ladder — never skip a rung

Each rung gates the next; failures localize to the rung that introduced them.

1. **Mapping units**: manifest counts, name remaps, fusion order, on synthetic
   tensors (seconds).
2. **Stage parity**: text features / VAE stages / schedule vectors vs goldens
   — *exact* for pure math (schedule within a couple of ULPs), cosine ≥ 0.9999
   for networks.
3. **Single-forward parity**: replay the hooked reference inputs through the
   full model. Include the *variant* cases (editing/ref-tokens) here.
4. **Composed-loop parity**: replay from the first captured latent through
   your own Euler/schedule/decode — proves the system composition with RNG
   removed.
5. **Real run** (CLI, own RNG, own text encoder): "statistically equivalent,
   not bit-identical" is the documented expectation for seeded noise.

Report cosine AND max_abs; break out sub-populations when semantics differ
per row (content vs pad rows is a common place a masking requirement hides).

## 6. Settle semantic questions with 2-minute experiments, not debate

When a convention is ambiguous, measure it in the reference framework *before*
implementing: running a reference text encoder with vs without the attention
mask, for example, can show content rows bit-identical and pad rows
meaningfully off — turning "do we need masked attention?" from an opinion into
a requirement. The experiment costs minutes; guessing wrong can cost an
e2e-parity debugging session many layers deep.

## 7. Exploit structure, but differentiate the unfolded form

Token-independent modulation often folds into a norm's affine params
(`(1+scale)·LN(x)+shift` ≡ LayerNorm with `gamma=1+scale, beta=shift`; an
RMSNorm fold is the same move) — this can turn per-block modulation work into
a handful of LN-param pairs and gates computed once per forward. Two
conditions: (a) prove token-independence first (a KV-cache variant can break
it — document it as out of scope if so), (b) the backward oracle must
differentiate the **unfolded** reference form, not the folded trick.

## 8. Training: one implementation, FD-gated at f64

- Write the block/model math **generic over the float type**: f64
  instantiation = the FD gradcheck oracle (FD shares no code with the
  analytic backward it checks), f32 instantiation = the trainer. One
  implementation, no oracle/trainer drift.
- Gate every layer: block FD < 1e-4, model FD < 1e-3, host-trainer forward vs
  the parity-proven GPU forward (cosine 1.0), LoRA = exact no-op at init +
  measured descent + fold-vs-apply bit-equality.
- LoRA on fused checkpoints: one adapter pair per **slice** (q/k/v rows,
  w1/w3 rows, column splits), folded back at the exact fused offsets.

## 9. Process notes that compound across ports

- **Parallelize only after the forward is parity-proven.** Serving surface,
  training stack, and perf target can proceed concurrently as independent
  changes *once* `model.rs` is frozen by its gate; before that, everything is
  serial on the parity ladder.
- **Ledger every found+fixed bug with numbers** (`docs/models/<m>/status.md`):
  the pre-fix cosine is what proves the fix mattered.
- **Params structs must match byte-for-byte, not just field-count.** A struct
  with the right number of fields in the wrong order compiles and dispatches
  cleanly; it is caught only by reading the kernel's own header comment (§3),
  never by field count alone.
- Fetching references: unauthenticated transfers can truncate silently on a
  zero exit code — always compare against the remote's reported size; some
  CLI tools silently reinterpret extra positional arguments after a flag like
  `--include` (verify the file list actually landed).

## 10. Then the PERFORMANCE ladder — a separate climb, same discipline

Parity proves the model is right. It says nothing about whether it is fast,
and the first working version usually will not be. A port that lands correct
at a low percentage of a card's peak, then climbs to a large multiple of its
initial speed through a few profile-driven rounds without moving parity by a
digit, is the expected shape, not an exception.

Read **`.agents/rules/kernels.md` §E** before optimizing anything; the rules are
there, the ladder is here:

1. **Get it correct first, then freeze it.** Every speed change is gated by
   re-running the parity test — which only exists because rung 3–5 of §5 built
   it. Optimizing before parity means you cannot tell a speedup from a bug.
2. **Profile per kernel-kind and publish the table before touching code.**
   Confident guesses about the bottleneck are wrong more often than not (see
   `.agents/rules/kernels.md` §E for named examples). Copy an existing
   `*_bench` binary that replays the dispatch sequence over shape-correct
   scratch — it needs no weights and reproduces the real forward closely.
3. **Attack by share of time, not by suspicion.** The GEMMs often *look* like
   the problem; attention or a different stage entirely can dominate instead.
4. **Re-profile after every fix — the bottleneck moves.** A large win on
   yesterday's bottleneck can be worth very little of the current step once
   the profile has moved on.
5. **Separate a bug from a ceiling.** Achieved vs peak decides it: well under
   10% of both the FLOP and bandwidth rooflines is a defect; a kernel flat at
   its byte/FLOP ratio across every shape is structural and needs an
   algorithmic change (conv → im2col+GEMM), not tuning.
6. **A precision change is not a speed change.** int8 arithmetic can have a
   much higher peak than fp32 without buying anywhere near that multiple in
   practice, because the bottleneck is often a kernel no GEMM precision can
   touch. Quantization pays for *capacity* (single-card residency, freeing a
   second GPU) and only pays for speed once the profile says arithmetic is
   the limiter.
7. **Ledger it** in `docs/models/<m>/status.md` with before/after tables, the
   hardware named exactly, and the hypotheses you killed with their numbers.
   Negative results are the deliverable that stops the next person re-running
   the same dead end.
