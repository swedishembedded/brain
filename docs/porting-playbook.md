# Porting playbook — from reference model to parity-proven brain implementation

The distilled process that took FLUX.2 Klein from zero to **cosine 1.000000**
forward parity and a gradchecked trainer in one pass, with only two real bugs
along the way — both of the same, preventable class. Follow it in order; every
step exists because skipping it cost time somewhere (here or in an earlier
port). The gates are cheap compared to debugging a 40-layer model that is
"mostly right".

## 0. Facts before code

- **Three independent references, one authority.** Fetch the official repo
  (authority on math), diffusers (tensor naming, pipeline glue), and ComfyUI
  (a third opinion + ecosystem conventions like LoRA key layouts). Where they
  disagree, say so in writing and trust the authority — but *verify the
  disagreement empirically before building on it* (the FLUX.2
  `norm_out` shift/scale chunk order differs between BFL and diffusers; the
  import test settled it).
- **Checkpoint headers are free architecture docs.** A safetensors JSON header
  or a GGUF header (range-request the first ~4 MB) yields the complete tensor
  name/shape manifest without downloading weights — enough to derive hidden
  sizes, block counts, fused layouts, and (by absence) features like a missing
  `guidance_in`.
- **Secondhand write-ups lie in the details.** A blog/Medium walkthrough got
  the FLUX.2 sigma schedule and chat template wrong. Use them for orientation
  only; annotate known errors where you store them.

## 1. Dump reference goldens FIRST (before any Rust)

One Python script (`tools/<model>_dump_reference.py`, e.g.
`tools/flux2_dump_reference.py`), CPU + fp32, fixed seeds, everything saved as
**f32** (brain's safetensors reader is F32/F16/BF16-only — ints cast exactly):

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
  parity later *without* reproducing torch's Philox RNG (replay from the first
  captured post-step latent).
- **Self-validate inside the dumper**: compute the same quantity two ways
  (manual text path vs `pipeline.encode_prompt`) and assert they agree.
- Write a `manifest.json` (shapes + sha256 + run params + library versions).

## 2. Import with two-way coverage, split fused weights at the boundary

- Derive a **canonical tensor manifest** from the config
  (`Flux2Config::tensor_manifest` pattern): every name + shape, counted
  against the real checkpoint (149/201 tensors — assert the number).
- Import validates **both directions**: any missing tensor errors *by name*;
  any unused source tensor errors. Never zero-fill, never skip
  (`qwen::import::brain_init_from_hf` discipline).
- **Split fused weights on the host at import time** (qkv thirds, gate/up MLP
  halves, column-split output projections). Every device matmul then reads a
  whole buffer — no offset-view gymnastics in the hot path, and the trainer
  reuses the same splits.

## 3. Know the kernel contract before dispatching it

Both real bugs of the FLUX.2 port were assumption-vs-contract mismatches on
*existing* kernels:

1. `step_sliced` offsets/lengths are **f32 elements, not bytes** (→ SIGSEGV).
2. `silu_mul`'s Params is a single `total`, not `[rows, cols]` (→ silently
   computed 1/9216th of the MLP; forward cosine 0.504).

Rules that would have caught both up front:
- Read the kernel's **header comment and Params struct** for every kernel you
  dispatch (`sed -n '/struct Params/,/^};/p' crates/kernels/wgsl/<k>.wgsl`).
- Find one **existing call site** and copy its params/threads idiom
  (`grep -rn 'K_SILU\|step_sliced' crates/{zimage,gpt,qwen}/src`).
- New kernels only when semantics genuinely don't exist (`gqa_scores_kmask`
  was the port's single new kernel — an anticipated need from another
  model's ledger).

## 4. Tiny-config smoke test before real weights

A weight-free forward at toy dims (hidden 16, 2+2 blocks, fake tensors from
the manifest) exercising **every step kind** runs in 0.2 s and catches
buffer-sizing/binding bugs that a 12-minute real-weights parity run would hit
opaquely. Make it bisectable from day one:
- block counts overridable by env (`SMOKE_DBL`/`SMOKE_SGL`),
- a step-truncation env (`SMOKE_STEPS=k` submits only the first k steps) —
  this located the element-vs-byte bug in three runs.

## 5. Climb the parity ladder — never skip a rung

Each rung gates the next; failures localize to the rung that introduced them.

1. **Mapping units**: manifest counts, name remaps, fusion order, on synthetic
   tensors (seconds).
2. **Stage parity**: text features / VAE stages / schedule vectors vs goldens
   — *exact* for pure math (schedule <2e-6), cosine ≥ 0.9999 for networks.
3. **Single-forward parity**: replay the hooked reference inputs through the
   full model. Include the *variant* cases (editing/ref-tokens) here.
4. **Composed-loop parity**: replay from the first captured latent through
   your own Euler/schedule/decode — proves the system composition with RNG
   removed.
5. **Real run** (CLI, own RNG, own text encoder): "statistically equivalent,
   not bit-identical" is the documented expectation for seeded noise.

Report cosine AND max_abs; break out sub-populations when semantics differ
per row (content vs pad rows caught the masking requirement).

## 6. Settle semantic questions with 2-minute experiments, not debate

When a convention is ambiguous, measure it in torch *before* implementing:
running the reference text encoder with vs without the attention mask showed
content rows **bit-identical** and pad rows off by up to ~6e3 — turning "do we
need masked attention?" from an opinion into a requirement (and justifying the
one new kernel). The experiment cost 2 minutes; guessing wrong would have cost
an e2e-parity debugging session at 40 layers deep.

## 7. Exploit structure, but differentiate the unfolded form

Token-independent modulation folds into norm affine params
(`(1+scale)·LN(x)+shift` ≡ LayerNorm with `gamma=1+scale, beta=shift`;
zimage's RMSNorm fold is the same move) — the whole FLUX.2 model needs 6
LN-param pairs + 5 gates per forward instead of per-block modulation work.
Two conditions: (a) prove token-independence first (the KV-cache variant
breaks it — documented as out of scope), (b) the backward oracle must
differentiate the **unfolded** reference form, not the folded trick.

## 8. Training: one implementation, FD-gated at f64

- Write the block/model math **generic over the float type**: f64
  instantiation = the FD gradcheck oracle (AGENTS exception 1 — FD shares no
  code with the analytic backward it checks), f32 instantiation = the
  trainer. One implementation, no oracle/trainer drift.
- Gate every layer: block FD < 1e-4, model FD < 1e-3 (measured: 4e-6 / 9e-8),
  host-trainer forward vs the parity-proven GPU forward (cosine 1.0), LoRA =
  exact no-op at init + measured descent + fold-vs-apply bit-equality.
- LoRA on fused checkpoints: one adapter pair per **slice** (q/k/v rows,
  w1/w3 rows, column splits), folded back at the exact fused offsets.

## 9. Process notes that compounded

- **Parallelize only after the forward is parity-proven.** Serving surface,
  training stack, and perf target proceeded concurrently as independent
  changes *once* model.rs was frozen by its gate; before that, everything is
  serial on the parity ladder.
- **Ledger every found+fixed with numbers** (`docs/models/<m>/status.md`):
  the pre-fix cosine (0.504) is what proves the fix mattered.
- Fetching references: unauthenticated HF transfers can truncate with curl
  exit 0 — always compare against the remote `x-linked-size`; the `hf`
  CLI silently reinterprets extra positional args after `--include`
  (verify the file list landed).
