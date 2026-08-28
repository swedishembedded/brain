# Lessons — defects this engine has actually paid for

Cross-cutting engineering findings, kept in one place because each was expensive
to learn and none belongs to a single model. Add to this file whenever a
non-obvious defect is found, and prefer a one-line entry with the number that
proved it over a paragraph of advice.

Model-specific ledgers stay in `docs/models/<model>/status.md`; kernel-authoring
rules stay in `.agents/rules/kernels.md`; porting order stays in
`.agents/rules/porting.md`. This file is for the things that generalise.

---

## 1. A gate that never runs is worse than no gate

It reports green and nobody looks again. Four separate instances, all found in
one workstream:

| Gate | Why it never ran |
|---|---|
| `sam2` parity | `scripts/fetch-testdata.sh` had no sam2 entry; the checkpoints had been hard-linked in by hand, so the test self-skipped on every machine but the one that created them |
| `flux2` host/device parity | `hidden: 16` put a modulation slice at byte offset 192, violating the 256-byte storage-binding alignment — it failed on every GPU, and had simply stopped being run |
| `cargo clippy` | aborts at the first deny-by-default lint and then reports nothing about everything after it — a 123-file backlog stayed invisible **twice in one day** |
| `wgsl-cpu` `compile_all` | one kernel out of the whole set would not compile on the CPU JIT, so int8 quantization had no working CPU path at all |

**Rules that follow.** A test that self-skips must say so loudly and its fixture
must be provisioned by `make fetch/testdata`. A tool's **exit code** is the
signal, not its stdout (`make clippy` checks both, separately). And when a suite
reports "N passed", check how many of those N were skips — an implementer
reported 9 vqgan tests passing when 7 of them were `SKIP: set BRAIN_VQGAN_WEIGHTS`.

## 2. Cosine is scale-invariant, so it cannot see a dropped scale factor

`got = 1.05 * want` scores cosine 1.0. A dropped `output_scale_factor`, a
doubled attention scale, or a mis-read GroupNorm gain all pass a cosine-only
ladder. Gate `rel_l2` (or `max_abs`) alongside it — `crates/unet`'s parity does,
after a reviewer noticed the ladder computed `rel_l2` at every stage and never
asserted it.

## 3. Finite differences gate the backward against whatever forward is emitted

So a **mis-weighted objective is self-consistent and passes**. `check_vqgan`
cannot see which term `beta` multiplies; that is pinned by reading the reference
implementation's source, where `beta` sits on the **codebook** term — not on the
commitment term a stale local comment claimed. Finite differences prove the
derivative, never the objective.

## 4. Degenerate test dims hide whole bug classes

At T5-XXL, `heads == d_kv == 64` **and** `heads*d_kv == d_model == 4096`. Both
toy configs shared that degeneracy, so a head-count/head-width swap, or a
`d_model`-vs-attention-inner-width swap, would have passed a 19 GB gate at cosine
1.0. Choose toy dims where every dimension **differs** — `heads=2, d_kv=64,
d_model=64` was the fix, plus a checkpoint-free tiny gate at those dims.

The same rule catches the `kv_rows == num_queries + 1` trap in the IP-Adapter
resampler: sizing k/v at `num_queries` is shape-legal only at one query.

It applies to *which fixtures you gate*, not only to their dims. InstantID's 70
cross-attention sites come in two widths (640 and 1280) and `heads = hidden / 64`
differs between them — the first draft dumped sites 1 and 3, which are **both
640**, so a width-dependent bug would have passed. The dumper now picks one site
per distinct width and the test **discovers the indices from the goldens** and
asserts it saw ≥2 widths, so a re-dump cannot silently stop gating one.

## 5. Run every gradcheck on BOTH backends

A `var<workgroup>` + `workgroupBarrier()` reduction with no barrier-free sibling
returns **all zeros** on `backend-cpu` — no crash, no error, a trainable
parameter whose gradient is silently dead and a loss curve that still looks
plausible. `backend-cpu` reports `workgroup_reductions: false` and its
split-at-barrier JIT mis-executes the construct. A GPU-only gate passes it
completely.

Any per-channel or per-row reduction in a backward therefore needs the
**barrier-free + cooperative pair**, selected on the *queried*
`DeviceCaps::workgroup_reductions` (`prelu_bwd`/`prelu_bwd_wg`,
`gn_stats`/`gn_stats_wg`). That selection is a correctness gate, not a perf tweak.

## 6. Timing without `poll_wait()` measures the host

`WgpuBackend::submit` with an empty clear list only appends to `pending` — it
encodes and queues nothing, so a loop of bare `submit`s times host-side
bind-group construction and reports it as device bandwidth. It produced
**377 GB/s on a ~346 GB/s card**. Compute the roof first: a result above the
physical limit means you measured the CPU. See `.agents/rules/kernels.md` §E.0.

## 7. Storage bindings must respect the 256-byte alignment

`min_storage_buffer_offset_alignment` is 256 bytes = **64 floats**. Slicing a
buffer at a row offset requires `r0 * d` to be a multiple of 64. Violating it is a
wgpu **validation error**, not a wrong number — which sounds safe until it
silently disables a parity gate nobody re-runs. Where offsets are genuinely
ragged (windowed attention), pass them in the kernel's own `Params` instead of
binding a slice.

## 8. "Unused" kernel constants are usually an unwired fast path

A registered-but-never-dispatched kernel still compiles a pipeline at every
`Gpu::new`, and usually means a model never learned about a faster sibling.
`crates/lfm` registered `flash_attn_bidir`, never dispatched it, and never
registered `flash_attn_bidir_split` — a drop-in at cosine 1.0 that measured
**~4×** at lfm's actual `head_dim 64` (1090 ms → 274 ms at T=8192).

Worse, the ledger's reason for not using flash — "measured ≈ naive here" — had
been measured against the *baseline* kernel, which is genuinely **0.3×**, i.e.
slower than the GEMM path. The conclusion never applied to the split kernel.
Before deleting an unused constant, check whether it is dead or merely unwired.

## 9. Preprocessing must be shared, not described as shared

`brain depth calib` letterboxed into a padded square while `Predictor::begin`
does an aspect-preserving resize with no pad — and the loader's doc-comment
claimed they were "the same transform the predictor uses at inference". INT8
scales were fitted to a distribution that never occurs. The fix is structural:
one extracted `preprocess_chw` both call, so the claim cannot drift again.

## 10. Normalisation conventions are asymmetric and silent

PuLID's `id_cond` is `ArcFace(raw) ‖ EvaClip(L2-normalised)` — the reference
reads insightface's `face_info['embedding']`, *not* `normed_embedding`, and
divides only the CLIP half. brain's `arcface` `embed` action normalises, because
its output is meant to be cosine-ready, so wiring it in is the natural move and
nothing structural catches it: same length, same dtype, same finite values, first
512 components ~20× too small. Verified numerically against the golden
(`‖id_cond[:512]‖ = 20.11`, `‖id_cond[512:]‖ = 1.0000`) before writing code.

## 11. Test RNGs are code too

`(*state >> 33) as f32 / 2^31 - 1.0` yields **[-1, 0)** — every sample negative,
so no test ever exercised the positive branch of an activation. Correct is
`>> 32`. Fixing it made depth's QARep gradcheck fail at rel 0.1119, which was a
*second* real finding: the analytic gradient was right and `eps` was the defect
(a ±1 direction over `n` components is an L2 step of `eps*sqrt(n)`). The knee was
measured across five values, `eps` dropped to 1e-4 and the tolerance **tightened**
8× — not loosened.

## 12. An orphaned `///` mis-documents its neighbour

rustdoc concatenates a doc comment onto the **next item**, even across a blank
line. So a `///` left behind by a function that was deleted or hoisted does not
just sit there — it becomes the first paragraph of whatever follows.
`vision::blocks::Act` documented itself as "A single `Conv` unit. Supports
stride 1/2…"; `chronos2`'s `rmsnorm_bwd` carried the docs of the `rmsnorm` that
had been hoisted away from it. Four instances, all leftovers of the
one-implementation consolidations.

Clippy calls this `empty_line_after_doc_comments` and it is worth fixing rather
than silencing. The fix needs judgment: sometimes the doc belongs to the item
below and only the blank line is wrong, sometimes the doc is an orphan and must
go. An automated pass cannot tell the two apart.

## 13. Kernel indices are per-crate, so one `Gpu` is not interchangeable

Every model resolves its kernel ids as positions in the pipeline list ITS crate
registered. Building `clip::ClipText` on a `Gpu` constructed from
`unet::KERNELS` therefore binds the wrong pipelines. `Gpu::new_like` exists for
this — a different kernel set on the same device — and is what a pipeline
assembling several crates must use.

That one surfaced as `Number of bindings in bind group descriptor (4) does not
match the bind group layout (5)`, which was luck: a wrong index is *silently
wrong output*, and only crashed because those two kernels happened to differ in
arity.

## 14. Fit the card by lifetime, not by hope

SDXL is ~3.5 B params across four models (~14 GB at fp32), and — under the
default wgpu backend — a non-ReBAR Pascal card carries roughly 2x resident
overhead per uploaded storage buffer (see #35: this turned out to be specific
to wgpu's Vulkan HAL, not the hardware — `--device vulkan` measures 1x on the
same card). Holding the UNet, both text encoders and the VAE at once does not
fit 24 GB.

The fix is scheduling, not precision: the text encoders are needed ONCE per
generation and the VAE once at the end, while the UNet runs every step. Build
the encoders, encode every prompt (conditional AND unconditional) in one pass,
drop them; run the loop; then decode.

The failure mode is worth the entry: the first version OOMed **at the VAE decode
after all 24 steps had already run** — the most expensive possible moment to
discover a memory problem. Prefer decoding on the CPU when the resident model
still owns the card; it is one pass.

## 15. A kernel-selection GUARD can be the bottleneck

`block::pick_gemm` fell back to the naive GEMM when `m < 128 || n < 128`, on the
reading that a partial 128x128 tile is wasted. The tiled kernel bounds-guards its
tile, so a short `M` costs only the idle rows — while the naive kernel gives one
thread per output element, each walking `k` serially, which collapses on a wide
`N`.

SDXL's cross-attention `kv` projection is `[77, 2048, 2560]`. 77 text tokens is
under the old threshold, so **60 of those per forward took the naive path at
43 GFLOP/s — 0.4% of the roof, and 49% of the entire UNet forward**. Measured
crossover (k=2048, n=2560): naive wins to m=4, tiled from m=8, and by m=77 tiled
is **22x** (18.67 ms -> 0.84 ms), bit-identical.

Lowering the guard to `m < 8` took the whole UNet forward from **2609 ms to
1349 ms (1.93x)** and the end-to-end 24-step generation from 174 s to 105.7 s,
with the output **bit-identical** at the same seed.

Two general points:

* This is the third instance of #8 in one workstream — a fast
  kernel that existed and was not reached. Twice it was an unregistered sibling;
  here it was a *selection rule* that excluded it. Profile the selection, not
  just the kernels.
* At `m = 1` neither is right: naive beat tiled 0.19 ms vs 0.48 ms, and the
  correct kernel is `matmul_gemv` (one workgroup per output column), which
  `gemm_variant` selects for models that register it. `crates/unet` does not, so
  its conditioning path still runs naive — small (~6 ms) but real.

## 16. A config default must mirror the REFERENCE's default, not "off"

diffusers' `AutoencoderKL.__init__` declares `use_quant_conv: bool = True,
use_post_quant_conv: bool = True`, and a `config.json` only carries the keys it
**overrides** — so the entire SDXL/SD1.x family omits both and means *true*.
`VaeConfig::from_json` defaulted them to **false**, which silently dropped
SDXL's `post_quant_conv`: a 1x1 mixing of the four latent channels, applied
before `decoder.conv_in`.

The failure is the instructive part. Dropping a channel mixing leaves the
decode **in a perfectly plausible `[-1, 1]` range** and **uncorrelated with the
truth** — cosine **-0.03** against the reference. The picture kept its
structure (a fox, a forest, snow) and had unusable colour, because structure
comes from the latent and the ruined part was a linear recombination of it.

Three things let it ship:

* No gate. `crates/vae`'s only decode parity covers *Z-Image's* 16-channel
  `AutoencoderKL` and skips unless `BRAIN_S3DIT_VAE` is set, so nothing gated
  a VAE decode of this kind at all (#1, again).
* The UNet's own 165-tap parity was green — and stayed green. It ends at the
  latent; the defect is downstream of everything it checks.
* The sanity check applied to the output was **mean gradient magnitude**, which
  "photo range" confirms for any textured image, including a wrong one. A
  statistic that a broken result also satisfies is not a check.

Read the reference's constructor signature, not just the checkpoint's JSON. Any
model that genuinely wants these off says so explicitly (FLUX.1-Kontext and
Z-Image both write `false`); the ones that say nothing are the ones that need
`true`. Gated by `crates/vae/tests/sdxl_decode_parity.rs`, now at cosine
1.000000 / PSNR 121 dB.

## 17. `matmul_reg3` supersedes `matmul_reg2` — everywhere

They are the same kernel: identical `Params`, identical `@workgroup_size(256)`,
identical dispatch arithmetic, bit-identical output. `reg3` is `reg2` with the
shared-memory bank conflicts removed. Swept across twelve shapes from
`[1,4096,4096]` to `[8192,320,320]`, `reg3` won **every one** by 1.08x-1.30x,
with `max|Δ| = 0` at each. There is no shape where preferring `reg2` is correct.

`crates/unet` registered `matmul_reg2` *beside* the `matmul_reg3` it already
carried through `vae::blocks`, and sent every `nn.Linear` to the slower of the
two it was holding. Dropping its own slot and pointing `pick_gemm` at
`vae::blocks::MATMUL_REG3_SLOT` took the SDXL UNet forward from 1349 ms to
1261 ms with one fewer pipeline compiled.

This is the **fourth** instance of #8 in one workstream, and the second where
the fast kernel was already registered in the same process. When two kernels
differ only by an optimisation, the slow one should not be selectable — the
place to encode that is the shared slot the block set exports, so a caller
cannot re-derive the wrong answer. Eleven other crates still register `reg2`.

## 18. A constant tuned on the toy fixture can be orders out on the real model

`crates/upscale`'s tile halo is the cost/quality knob for super-resolution
tiling: too small and each tile is computed as if the image ended at its border,
which shows as a grid of seams. The first draft picked **16**, and the
checkpoint-free 2-block gate agreed — max |seam| **9.2e-4**, four times below an
8-bit quantisation step, zero visible pixels. It looked measured, because it was.

On the released 23-block `x4plus` the same halo measures **7.3e-1**: three
orders of magnitude worse, 45 676 visibly wrong pixels. The reason is
structural — a 3x3-conv net's receptive radius grows with DEPTH (~`1 + 15*blocks
+ 1`, so ~32 input pixels at 2 blocks and ~347 at 23) — so the toy could not have
predicted the real number no matter how carefully it was measured.

Two things follow, neither specific to upscaling:

* A **checkpoint-free gate is necessary and not sufficient.** It runs everywhere
  and catches wiring, shapes and algebra; it cannot calibrate anything whose
  scale depends on the real model's depth or width. Any constant of that kind
  needs a measurement on the released weights, even when that gate has to skip
  on most machines.
* **Report the sweep, not the chosen value.** The table in `TILE_HALO`'s doc
  comment shows both configs at every halo tried, so the next person can see
  that the number is a trade-off with a known cost.

And a corollary earned the hard way, in the same file: **the obvious remedy was
wrong.** "Hard-cropped tiling leaves a seam, so blend the overlap instead" is
the standard move, and it was written into the doc comment as the planned fix
before anyone measured it. Blending is *worse* — 2.1e-2 against cropping's
3.3e-6 on the toy, 2.0e-1 against 1.6e-1 on the released net — because it mixes
each tile's halo pixels, the least accurate ones it computed, back into the
output, where cropping throws them away and keeps the well-conditioned interior.
Blending trades the error's magnitude for its continuity. A planned fix recorded
in a comment is a claim like any other; this one is now recorded as refuted, with
the numbers, so it is not attempted a third time.

The comparison also had to be set up correctly to mean anything: tiled-vs-whole
image is NOT a seam measurement, because the whole-image path lets the
convolutions zero-pad at the image border while any tiled path replicate-pads
it. Holding the border regime fixed — tiled vs ONE tile covering everything —
is what isolates the seam.

## 19. Registration split across N lists is a defect waiting for its turn

Adding a served model meant editing three lists with no link between them —
`caps_cli::static_manifests()` (what `brain caps` lists),
`caps_cli::build_registry()` (what `brain do` can run) and
`resident::build_executor()` (what the transports serve). Each omission fails
SILENTLY and differently: undiscoverable, or listed-then-"unknown model", or
invisible to D-Bus.

It caught a model that had everything else right. `Real-ESRGAN` shipped with a
manifest, a provider, a residency adapter, a parity ladder at cosine 1.0 and 16
green tests, and `brain caps <id>` still said "unknown model" — because only the
third list had been edited. No test could see it, because no test related the
lists to each other.

The fix is not "remember to edit three places", it is to make three places
impossible: one `ModelEntry` per model holding the manifest, the provider
constructor and the residency adapter, with the other lists DERIVED. Then the
invariant becomes testable, and the test that matters is the one that reproduces
the original failure — *every model this lists must be constructible by name*.

Generalises past this repo: whenever a thing must be declared in more than one
place for a feature to work, the duplication is not a style problem, it is an
unexploded defect. Look for it wherever a "registry", a "catalog" and a
"dispatch table" name the same set.

## 20. A fallback path is a path — measure it too

`vae::blocks::gn` picks a cooperative GroupNorm reduction when the device has
workgroup reductions and a serial one otherwise. The cooperative branch was
measured and tuned; the FALLBACK was never measured at all, and `backend-cpu`
reports `workgroup_reductions: false`, so every conv-autoencoder in the tree —
vae, vqgan, restore, unet, flux1 — ran the unmeasured branch on the CPU JIT.

It was the serial kernel: `g` = 32 invocations for up to 33 M elements. A
barrier-free two-stage reduction measured **~3x** faster at every VAE decoder
shape, and it already existed — `crates/wm-diamond` had written it privately
after measuring the serial one at 77% of its frame time.

Three things generalise:

* **Profile the branch your hardware does NOT take.** A capability-gated
  fallback is invisible on the machine that never takes it, which is exactly
  where an unmeasured path hides.
* **Faster can be more accurate.** Summing 33 M elements in one lane loses
  precision a two-stage reduction keeps: SDXL's VAE decode parity went from
  PSNR 121.39 dB to 127.95 dB at the same cosine. Speed and accuracy are not
  always a trade.
* **A/B harnesses need their own sanity check.** The first run of this one
  reported the three kernels disagreeing by 1.7 — which was the harness
  dispatching `gn_stats_wg` at 64 threads when it is `@workgroup_size(256)`.
  It was caught only because the VAE parity gate was green, so the harness had
  to be the wrong one. Compare against a HOST oracle, not just kernel-to-kernel.

## 21. The per-kernel table is an upper bound — the whole pass is the truth

A profiler that times contiguous groups has to drain the queue between them, so
each group's number carries a round-trip AND loses whatever overlap that kernel
would have had with its neighbours in the real submit. On the VQGAN backward the
grouped sum was **855 ms against a 574 ms whole pass — a 49% inflation.**

That is fine for RANKING (the order is right, which is what §F.1 uses it for)
and dangerous for CREDIT. Two changes, both looking like clear wins in the
table:

| change | grouped | whole pass |
|---|---|---|
| `gn_dsum` → two-stage | 229 → 21 ms | 835 → 716 ms ✅ |
| `gn_dgamma`+`gn_dbeta` → fused pair | 170 → 16 ms | 716 → 574 ms ✅ |
| `bias_grad` → two-stage | 99 → 30 ms | 574 → 575 ms ❌ |

The third was **reverted**: two kernels and 88 extra dispatches for no
end-to-end gain. The difference is parallelism already available — `gn_dsum` ran
32 lanes and `gn_dgamma` one per channel, so they genuinely serialised the
device and nothing could overlap them; `bias_grad` at 512 features already had
enough lanes to interleave with its neighbours, and its grouped 99 ms was mostly
drain and lost overlap rather than work.

**A fix does not count until the WHOLE-pass number moves**, measured more than
once — the 574 that made `bias_grad` look like a 1 ms regression and the 573.67
it was compared against were the same measurement, one sample apart.

## 22. Check disk headroom before a wide build

`cargo build` is a few GB of `target/`; adding `--tests --examples` across the
workspace can be **an order of magnitude more**. That kind of jump can fill a
disk and hard-block every tool that writes output to the same filesystem —
with no recovery path from inside a running session once it happens. Check
available disk space before a wide build.

## 23. A training-time config field the checkpoint doesn't persist makes the whole feature a no-op

`QwenConfig::to_json` never emitted the `lora` field; `from_json` hardcoded
`lora: None`. The LoRA forward/backward were correct and gradient-checked
(`gradcheck::check_qwen_lora`) — `Qwen::save` wrote the trained `*.lora_a`/
`*.lora_b` tensors to disk, exactly as designed. But the very next
`load_inference` rebuilt the param list from the reloaded config, which had
no `lora` entry, so it never allocated slots for those tensors: `lora_for()`
returned `None`, `lora_fwd` never dispatched, and the reloaded model was
bit-for-bit the untrained base. Every fine-tune "worked" (the training loop
ran, loss went down) and then silently did nothing the moment the process
that trained it exited.

Nobody noticed because the only test of it asserted `exact_match_after >=
exact_match_before` — "did not regress below baseline", which an unchanged
base model also satisfies exactly. This is #16 again, with a twist: the
statistic didn't just fail to distinguish success from failure, it couldn't
have distinguished them even in principle, because the two are
byte-identical once you skip the reload. **A test of a save/load feature
that never actually closes the save→load round trip is not testing the
feature.**

The fix is a two-line `Option` round-trip
(`crates/qwen3/src/config.rs`), but the gate that would have caught it needed
three things together, none optional: (1) take a few real optimizer steps so
`B ≠ 0` (a fresh LoRA init is `B = 0`, making the delta zero regardless of
whether it loads — see #4 on degenerate test setups hiding a whole bug
class), (2) **save, then load in a fresh process/struct** rather than
comparing against the live trained model still sitting in memory, (3) assert
the reloaded logits *differ from the base* by a real margin, not merely
"didn't get worse". `crates/qwen3/tests/lora_roundtrip.rs` does all three;
`crates/qwen3/tests/lora_learning_gate.rs` (Gate A) goes one further and
reintroduces this exact defect on purpose as its own verification that the
gate has teeth, rather than trusting the story that it would have caught it.

## 24. A benchmark that measures a path no request takes is a healthy number about the wrong thing

Every serving benchmark before `perf::targets::HttpTarget` drove
`qwen3::serve::Scheduler`/`residency::Executor` directly — real kernels, real
batching, genuinely fast. Meanwhile `crates/cli/src/resident_llm.rs`, the
ONLY code an actual `/v1/chat/completions` request ever reaches, called a
single-sequence decode loop that touched none of it: no paged KV, no
scheduler, no batching. The benchmark suite was green and fast while a real
agentic client saw 600+ seconds, because "the engine is fast" and "the
request reaches the engine" are two different claims, and only the second
one is what a user experiences. Nothing forced the benchmark to prove the
second claim — it was structurally impossible for it to be wrong about the
first while being catastrophically wrong about the second, and it stayed
that way for as long as no target actually drove the transport layer.

The fix generalizes past this one bug: a target that measures a scheduler,
an engine, or a codec directly is answering "is the fast path fast", never
"does a request reach the fast path" — those need a DIFFERENT harness that
goes in through the same door a client does (here, `apiserve::router()` via
`tower::Service::oneshot`, no socket, but every layer a real HTTP request
passes through: auth, admission, JSON parsing, chat-template rendering).
Keep both kinds of target — the direct one is cheaper and still useful for
kernel-level regressions — but never let the direct one stand in for "is the
served path fast," because the gap between them is exactly where a
serving-path regression like this one hides.

## 25. "Batched prefill" that batches the readback, not the dispatch is not batched

`Qwen::prefill`'s fix replaced a per-token `decode_submit` loop with one call
to the batched primitive — but an earlier attempt at "fixing" prefill
performance batched only the readback (one `map`/fence at the end) while
still issuing one GPU submit per token underneath. It measured faster than
the naive `step()`-per-token loop (fewer host↔device round trips), so it read
as progress, but it was still `O(T)` submits for a `T`-token prompt — the
same defect class in a lighter disguise. "Faster than before" and "actually
O(1) [per chunk]" are different claims, and a wall-clock-only benchmark
cannot tell them apart — only a device-op COUNT (`gpu_core::DeviceStats.
submits`) can, because it is insensitive to how fast any individual submit
happens to run on the current machine. The gate this lesson names,
`prefill_submits_scale_with_chunks_not_with_token_count`, is what survives.

## 26. A barrier kernel on `backend-cpu` corrupts memory; it does not refuse

`DeviceCaps::workgroup_reductions` is false on the CPU JIT because it cannot
compile `workgroupBarrier`. What it does with one is worse than an error:
`crates/gradcheck/tests/layernorm2d_kernels.rs` recorded `layernorm_rows`
(2 barriers) there and the process died with

    munmap_chunk(): invalid pointer
    signal: 6, SIGABRT

no test name, no kernel name, no backtrace into the offending dispatch. The
whole-suite runner only said `FAIL: gradcheck suite — CPU backend`; finding
which of many test binaries aborted took longer than the fix.

Two things follow, and the second is the one that bit:

* The capability branch must wrap the **recording**, not just the submit —
  `gpu.step()` on a barrier kernel is already too late. Registering it in the
  kernel table is fine (`prelu_kernels.rs` does, and gates the dispatch), which
  is what made the guard look sufficient when it was not.
* A new kernel test is not done when it is green on the GPU. This one shipped
  measured, host-oracled, and red on the other backend, exactly the way
  lesson #5 says not to — and the A/B harness is the *easiest* place to make
  this mistake, because the fast path is the one you are excited about and the
  reference path is the one carrying the barrier.

Running the fused kernel alone on the CPU was not a consolation prize: it
matched the host oracle **exactly** (0.0e0, against 7.2e-7 on the GPU), which
is a stronger correctness result than the GPU run produced.

## 27. A "% of peak" divided by a hardcoded peak is not a measurement

Every profiler in this tree reported utilisation against a literal:
`PEAK_TFLOPS`/`PEAK_GBPS`/`PEAK_FP32` constants hardcoded in `vqgan_bench`,
`unet_bench`, `zimage_bench` and several microbenches — one card's spec-sheet
numbers. Three separate problems follow, and the third is the one that matters:

1. **On any other device every number is wrong**, silently and by an unbounded
   factor. `DeviceCaps::peak_bandwidth_gbs` existed for exactly this and was
   `None` on all three backends, with nothing anywhere filling it.
2. **Spec-sheet peak is not achievable peak.** Measured with a
   dependency-free FMA probe and a STREAM-triad probe, achieved throughput came
   in at roughly 85-90% of the spec-sheet number. Grading kernels against a
   roof nothing can reach builds a permanent 10-17% pessimism into every row.
3. **The whole method depends on the denominator.** `.agents/rules/kernels.md`
   §F is "rank against the roof, fix the top row, re-profile". A wrong roof does
   not produce an obviously wrong answer — it produces a plausible one, and
   quietly invalidates the ranking that everything else is built on.

`gpu_core::roof` measures both roofs once per adapter (persisted with
`gpu_core::tune`'s key discipline, so editing a probe invalidates old numbers by
construction) and `Gpu::caps()` overlays them. The probe measures the
**silicon**, deliberately not "the best GEMM we have written" — a roof derived
from `matmul_reg3` would hide precisely the gap the workstream exists to close.

Corollary worth stating separately: **do not measure a roofline under
contention.** Two probes sharing a device measure the contended device and
disagree by more than 25%, which broke a reproducibility test under heavy
parallel test execution and is not a bug in the probe.

## 28. A partial FLOP numerator over a full denominator under-reports in silence

`vqgan_bench`'s `WHOLE PASS` row summed FLOPs across every row regardless of
whether `gpu_core::cost` had a formula for it, then divided by the whole-pass
time. A third of the VQGAN backward's kernel kinds had no formula, so the
published "backward = 5.4% of peak" was computed from a numerator missing a
third of the graph.

This is the mirror image of the failure `cost` was designed to prevent. An
uncovered kernel already reported `-` per row (never a zero that reads as slow),
but the pass-level total had no such guard — and a *pass* rate is the number a
reader quotes. The fix is that a partly covered pass reports **no rate at all**
and names the kinds it is missing:

    WHOLE PASS   457.42 ms   1404   (rate unavailable — no cost formula for: mse_grad, masked_l1_grad)

which turned an invisible accounting hole into a two-line work item. With every
kind covered, the honest numbers are **forward 356.8 GFLOP/s (3.4%) and backward
638.1 GFLOP/s (6.0%) of the measured roof.**

Two structural notes:

* The coverage test was a hand-maintained *list* of kernel names, so it could
  only fail when someone remembered to extend it — it could not stop a new
  kernel landing unmeasurable. It is now backed by a **ratchet** over the whole
  kernel table that fails when coverage falls.
* Name the uncovered kinds in the profile output, never just count them. They
  are usually cheap enough to fall outside the printed top rows, which is
  exactly where a missing formula hides.

## 29. `make kernels-regen` had been broken since the script moved

`scripts/build/kernels-regen.sh` computed its repo root relative to its own
location, which was right when it lived directly under `scripts/` and became
wrong once it moved a directory deeper. Every invocation died with a missing
path.

Nobody noticed because the failure mode is *silent at the level that matters*:
you add a `.wgsl`, the regen fails, and you hand-append the two lines to
`crates/kernels/src/lib.rs` instead. The registry stays correct, so no test ever
fails — it just stops being mechanically derivable. Re-running the fixed script
produced a nontrivial diff over kernels added since the breakage: hand-written
doc comments replaced by the generated form, and several consts out of sort
order.

The lesson is about the class, not the typo: **a generator whose output can be
produced by hand will be, and its breakage is invisible until someone needs it
to be authoritative.** `make check/scripts` verifies every script *parses*; it
cannot verify one still *works*. A generator wants a regen-is-a-no-op check.

## 30. Two of three backends never implemented `stats()`, and a caller papered over it

`Backend::stats()` defaults to `None` — "this backend does not count device
ops" — and the documented consumer contract is *report null, never zero*.
`backend-cpu` and `backend-vulkan` both took the default. Nothing noticed,
because the single in-tree consumer wrote:

```rust
let before = e.device_stats().map(|s| s.submits).unwrap_or(0);
```

which turns "not counted" into "zero". The test built on it
(`prefill_submits_scale_with_chunks_not_with_token_count`) then compared
`0 == 0` in its first two assertions and passed **vacuously** on the CPU
backend; only its third assertion — `submits > 0` — ever noticed, and it read
as a plain failure rather than as "this backend cannot answer".

Three things worth separating, because only the first is obvious:

1. **The fix is the counters, not the caller.** Making the test skip when
   `stats()` is `None` was the first thing tried and it is a workaround: it
   leaves two backends unable to answer a question the trait says they may be
   asked, and it makes the *next* consumer rediscover this. Implementing the
   counters removes the ambiguity at its source; `None` then becomes an
   assertion failure at the call site rather than a shrug.
2. **`stats()` is per HANDLE, not per device** — the trait says so, and it
   matters. The counters were first put on `backend-cpu`'s `Arc`-shared
   `CpuShared`, which compiles, passes serially, and fails under a
   multi-threaded test run: every test in the binary shares one pooled device,
   so a neighbour's submits land inside your delta. Per-handle counters make
   concurrent measurement correct by construction; a mutex around the
   measurement only serialises the test against *itself* and does not help.
   Note the failure mode of getting this wrong is a **flaky** test, which is
   worse than a wrong one.
3. **A default trait method is a silent opt-out.** `fn stats(&self) -> Option<_>
   { None }` reads as a courtesy to backends that cannot count; in practice it
   is how two backends went a long time without counting while the API looked
   complete. `crates/gpu-core/tests/device_stats.rs` now asserts the contract on
   whatever backend `BRAIN_DEVICE` selects, so the default cannot creep back.

## 31. Timing a kernel by host wall-clock around a drained slice is not a measurement

The per-kernel-kind profiler (`gpu_core::profile`, and the benches that
preceded it) times a group by submitting that slice alone and bracketing it with
`poll_wait`. That is honest about *drain* — it prints the group-sum vs
whole-pass inflation, lesson #21 — but it is not a measurement of the kernel. It
measures **launch + execute + fence**, and the floor of that is roughly constant,
so the error is inversely proportional to how small the kernel is.

Measured both ways on the same workload, host-timed slices against the
backend's **GPU timestamp queries** (`BRAIN_PROFILE=1`, which brackets each
dispatch with `beginning_of_pass_write_index`/`end_of_pass_write_index`):

| kernel | host-timed ms/call | timestamped ms/call | inflation |
|---|---:|---:|---:|
| `matmul_reg3` | 0.605 | 0.400 | 1.5× |
| `rmsnorm_rows` | 0.240 | 0.0122 | **19.7×** |
| `paged_decode_scores_batched` | 0.349 | 0.0205 | **17×** |
| `paged_decode_apply_batched` | 0.443 | 0.0152 | **29×** |

**The ranking the host method produces is wrong, not merely imprecise.** By
device time `matmul_reg3` is **94.8%** of the pass; the host-timed table put it at
53.8% and promoted `rmsnorm_rows` (1.7% of real GPU time) and `add2` to "DEFECT"
rows at 12.3% and 6.2%. Optimising either would have been work against an
artifact — the same failure §E already records twice (`bias_grad`,
`matmul_gemv`), except those were caught by the whole-pass number afterwards
rather than by the profile being right in the first place.

It also explains an "impossible rate" that three rounds of fixing the byte model
never closed. `paged_decode_scores_batched` was flagged at 250-292% of the roof;
with the true (shorter) device time the same byte model gives **6537 GB/s**, i.e.
the byte model is wrong by more than 20×, not by the 2-3× the successive
corrections were chasing. Two independent errors were being tuned against each
other, and the impossible-rate guard was the only thing that stopped a plausible
wrong answer from being published.

**The rule:** rank with **device** time. A host-bracketed slice is a legitimate
measurement of *a submit*, and the whole-pass number it produces is still the
thing a change is judged by (#21 stands) — but it must not be used to attribute
time *between* kernels.

**RESOLVED** — `gpu_core::profile` now uses device time wherever the backend can
give it. `Backend::set_kernel_timing`/`kernel_times` expose per-kernel device
totals, and `backend-wgpu` implements them with `TIMESTAMP_QUERY_INSIDE_PASSES`:
`n+1` timestamps written *between dispatches inside the production single
compute pass*, so the pass structure being measured is the one that ships. The
validation is that the two numbers now account for each other — **kernel device
time 80.28 ms against a whole pass of 80.85 ms** (0.7% apart), where the
per-dispatch-pass mode reported 330.7 ms against 114.9 ms.

It immediately inverted the ranking it was built to fix: `rmsnorm_rows` and
`add2`, previously "DEFECT" rows at 12.3% and 6.2% of the pass, are 0.8% and
0.3% at **31% and 54% of the bandwidth roof** — not defects at all.

And it exposed a much larger harness bug underneath, recorded as #32.

**Where this left the tooling before the fix.** Neither source was clean:

* the host method has the floor above;
* `BRAIN_PROFILE`'s timestamps measure real device execution, but only in a mode
  that puts **one compute pass per dispatch**, which changes the execution the
  production single-pass flush actually performs — so its *absolute* times are
  not production times either.

The fix is timestamp queries **inside the production single-pass flush** — one
query pair per dispatch, resolved once at the end — so per-kernel attribution and
the whole-pass number come from the same execution. Treat every per-kernel
*share* recorded before this fix as suspect for small kernels, and re-derive any
ranking from `BRAIN_PROFILE`'s distribution. Whole-pass numbers, and every
speedup in this document that was judged by one, are unaffected.

## 32. A profiler that drives the graph wrong measures a graph that does nothing

With device timing correct, `paged_decode_scores_batched` still reported
**7060 GB/s**. The timing was right and the kernel really was that fast — because
it was attending to nothing.

`qwen_bench serve` drove the served tape with `Input::Resident`. That is the
on-device decode-window mode: it deliberately performs **no host writes**,
because `decode_feed`/`decode_advance` are supposed to have produced the token
ids *and* the paged metadata on the device already. Driven from a profiler,
nothing had, so `seq_lens` stayed zero, every attention thread early-returned,
and **61% of the pass did no work**.

The damage was not confined to the attention rows. The measured "≈27–29× serve
prefill speedup" from registering the tiled GEMM was published from that harness
and is wrong; corrected, it is **11.8×** (2413.18 → 204.4 ms, 53 → 626 rows/s).
The *ratio* was stable across four runs precisely because both arms ran the same
no-op attention — a like-for-like comparison against a pass missing most of its
work. **Reproducibility is not validity**, and a stable ratio is exactly the
shape of evidence that makes this kind of error survive review.

Three defences, in order of how much they would have caught:

1. **Make the profiler drive the model the way production does.** `Resident`
   exists for a mode with a device-side producer; a profiler has none. This is
   the whole bug.
2. **Cross-check the pass against its own roofline.** The impossible-rate guard
   (#31) is what refused to publish 7060 GB/s and forced the question. Without
   it the number would have been printed as a percentage and believed.
3. **Sanity-check the shape of the answer.** A suspiciously cheap rows/s figure
   for a small model should prompt "against what ceiling?" — the weight-bandwidth
   budget the same tool already prints says a served step cannot be that cheap.

## 33. A checker that cannot tell code from prose fails on the best-documented file

The kernel catalogue cross-checks each `@cpu` declaration against the file's
`workgroupBarrier()` count, because a kernel with two or more corrupts memory on
the CPU JIT (#26). It counted the word over the **raw source**, so it also
counted every mention in a comment — and the kernels most likely to discuss
their barrier discipline are exactly the cooperative ones the check exists for.

It fired on a correct new kernel (`paged_decode_scores_wg`: one barrier in code,
green on `backend-cpu`) whose header said "Exactly ONE top-level
`workgroupBarrier()`". Counting code only then revealed the seeded catalogue had
been wrong about **four** kernels all along:

| kernel | published | actual | barriers in code |
|---|---|---|---:|
| `layernorm_rows` | ✗ CPU | ✓ | 1 |
| `gradnorm_part` | ✗ CPU | ✓ | 1 |
| `prelu_bwd_wg` | ✗ CPU | ✓ | 1 |
| `conv2d_tiled` | native only | native | 1 |

Every one had a comment mentioning the barrier. Verified after correcting:
`compile_all` and `make gradcheck` are both green on `BRAIN_DEVICE=cpu`, so the
CPU claims now hold.

Two things worth keeping:

* **The failure mode is inverted from the usual one.** A checker with false
  positives does not merely annoy; it trains you to distrust it, and this one
  fired first on a *correct* file. Had the new kernel not been documented, the
  four wrong rows would have shipped indefinitely.
* **A derived value seeded from a buggy derivation stays buggy after the source
  of truth moves.** The `@` blocks are hand-maintained now, but they were
  *seeded* by the same comment-counting code, so fixing the checker was not
  enough — the seeds had to be recomputed too. Any bootstrap-then-hand-maintain
  migration carries this: the bootstrap's bugs are baked into the data.
* **The bug was in two files because the derivation was.** The seeder and the
  checker each had their own copy of "count the barriers, read the workgroup
  size, decide the tier", so one defect needed two fixes and could be
  half-fixed. They now share `scripts/build/kernelmeta.py`, and the property
  that matters is asserted: for every kernel, what the seeder would *propose*
  for the mechanical fields is exactly what the checker *demands*. Without
  that, seeding a new kernel could emit a block that immediately fails the
  build.

The same comment-blindness was in **production** code:
`backend_api::workgroup_size_of` took the first `@workgroup_size` anywhere in
the source, and several in-repo kernels document the attribute in their header
prose, which sits above the declaration. All of them happen to state the right
number, so nothing was broken — but a single stale or aspirational comment would
have laid out every backend's dispatch grid with the wrong size, while the
kernel reconstructed its flat invocation id from a different one. Its own doc
comment admitted it was relying on the parity tests to notice. It now scans code
only, with a test that pins it.

## 34. A memory saving is not measured by anything unless someone measures it

int8 paged KV had a real, careful quality gate before this workstream
started: a 3-way `fp32`/`int8`/`int8-calib` loss and token-accuracy table on
the real checkpoint. It had NO memory gate at all. The headline claim — "int8
KV is ~4x smaller" — lived as a single `/ 4` inside the allocator and a code
comment next to it. `crates/perf`'s `memory` block was hardcoded `null` for the
KV pool the whole time; the residency budget (`QwenResident::estimate`) only
ever counted checkpoint file size, so switching KV dtype changed the real
footprint by nearly 4x and changed the number `estimate()`/`crates/stats`/
braintop reported by exactly zero. Nothing was lying — there was simply no
path from "the allocator does `/4` here" to any artifact, gate, or UI a human
or a CI job would ever look at.

This is the memory-saving twin of lesson #1 (a gate that never runs is worse
than none): a saving that is only ever a mental derivation from reading the
allocator code is not verified, is not regression-tested, and cannot be
quoted in a doc without someone re-deriving it by hand each time — exactly
the failure mode `kv_pool_bytes_identity_holds_at_the_real_shape` exists to
close, by making the `4·head_dim / (head_dim + 4)` ratio an assertion instead
of a comment, and asserting it is DIFFERENT at `QwenConfig::tiny()`'s
`head_dim=8` (2.667x) than at the real `head_dim=128` (3.8788x) — a toy-fitted
number cannot stand in for the real one (lesson #18), including when the
"number" is a memory-savings ratio rather than a numeric result.

The fix generalizes: any optimization whose entire pitch is "uses less X"
needs X measured through the same artifact/gate path its quality claim
already gets, in the same commit — not as a follow-up, because the
follow-up is exactly the step that silently never happened here for however
many commits this shipped without one.

## 35. The "2x resident overhead" was real, but it was wgpu's, not the hardware's

§14 and `crates/qwen3/src/q8.rs` both stated the fact as a property of
non-ReBAR Pascal: "each storage buffer carries ~2x resident overhead." That
was measured correctly but attributed too broadly — it is a property of
**the wgpu backend's Vulkan HAL on this class of hardware**, not of the
hardware or of Vulkan itself.

Direct measurement (`crates/gpu-core/tests/vram_overhead.rs`, `nvidia-smi`
deltas around known allocations):

* Allocation alone (`gpu.storage(n)`, never written): **1.00x**, every time.
* Allocation + any upload (`write_f32`/`storage_init`, any `BufUsage`
  combination, `COPY_SRC` present or absent): **exactly 2.00x**, every time.
* Splitting the same upload into 64 MiB chunks via `write_at`/
  `write_f32_chunked` (added to `backend_api::Backend` and all three native
  backends for this investigation): still **2.00x**. So it is not "wgpu's
  staging belt sized to the biggest single call" — the resident cost tracks
  *cumulative bytes ever written to that buffer*, not any call's size, and
  chunking cannot bound it.
* The same probe on **brain's own native Vulkan backend**
  (`crates/backend-vulkan`, ash + naga, selected via `--device vulkan` /
  `Gpu::try_new_vulkan`) instead of the default wgpu backend: **1.00x**. Its
  `with_staging` (`crates/vulkan/src/context.rs`) reuses one shared, bounded
  staging buffer across every upload; wgpu-core evidently keeps a per-upload
  (or per-buffer) shadow instead and never frees it.

**The fix is `--device vulkan`, not a wgpu-level change**, and it is
model-agnostic: every model that quantizes to int8 for capacity reasons
(`crates/qwen3/src/q8.rs`, `zimage`, `flux2`) gets roughly double the effective
VRAM budget for free by preferring the native Vulkan backend over the wgpu
default on this class of hardware. It does **not** relax the "fp32 arithmetic
only" invariant or change any kernel — it is purely a device/backend
selection at residency-placement time. The native Vulkan backend's own
coop-matrix (tensor-core) kernel is unavailable without `glslc`/
`glslangValidator` on `$PATH` (falls back to the scalar kernel), so this trade
needs a real throughput measurement per model before it is made a *default*;
today it is a documented, available override, exercised by one model's
residency plan to fit int8 weights across two GPUs without a CPU-expert
fallback.

**Do not re-derive "2x on non-ReBAR" as a hardware fact in a new model's
docs** — cite this lesson and, if it matters for that model's memory budget,
measure with `vram_overhead.rs` on the backend that model actually plans to
run.

## 35b. A `const` bump without its array literal is a silent out-of-bounds write

`crates/kernels/wgsl/router_gate.wgsl` and `router_gate_train.wgsl` declare
`const MAX_EXPERTS: u32 = 128u;` — bumped up from 64 in an earlier commit,
with a comment explaining why (128-expert MoE routers were coming). But the
shader body's actual scratch storage, `var prob: array<f32, 64>` / `var used:
array<bool, 64>`, was a second, independent literal that the bump did not
touch. Both compiled cleanly (WGSL fixed-size arrays are just a length
literal, unrelated to `MAX_EXPERTS` unless something cross-checks them), and
every existing test kept passing, because `crates/model/tests/moe_sparse_parity.rs`
exercises `router_gate.wgsl` at a synthetic `n_experts: 8` — nowhere near 64.

The result: for any router with more than 64 experts, every expert index ≥ 64
read and wrote past the end of a 64-element array during the per-token
softmax/top-k/renormalize loop. WGSL has no bounds-checking panic for this —
it is UB that manifests as silently wrong probabilities for roughly the top
half of the expert set, not a crash, not a validation error, nothing that
shows up in a smoke test. Found only because one model's Thinker decoder
(128 experts) was validated against a real top-k-id/weight golden
end-to-end, not just via a same-composition sparse-vs-dense oracle test.

**The general shape to watch for**: any `const` that exists specifically to
size a fixed-length local/array, where the array's own literal is a second,
separately-typed place the same number has to be repeated. Grep for the
`const`'s name is not enough — grep for `array<`/`[N;` sitting near it and
confirm the literal actually reads the const (WGSL has no way to do that, so
"confirm by eye, then add a test at a size between the old and new bound" is
the actual mitigation, not a language fix). The existing `moe_sparse_parity.rs`
oracle test is exactly this class of gap: it proves the sparse and dense
paths agree with EACH OTHER, which is blind to a bug present identically in
both — a real-weight/real-golden test is what caught this one, not the
same-composition oracle.

**Recurrence, caught by a later audit of this lesson's own fix**: the
fix above touched `router_gate.wgsl`/`router_gate_train.wgsl` (`MAX_EXPERTS`
64→128) but never reached `router_gate_sigmoid.wgsl` — a THIRD kernel with
its OWN independent `const MAX_E: u32 = 64u;` and six of its own
`array<f32, 64>` scratch locals, never bumped. Worse: a status doc's own
written record of the fix above stated one model was "unaffected (uses the
separate `router_gate_sigmoid.wgsl`)" — true as far as it went, but the
person who wrote that line checked that the model used a DIFFERENT file, not
that the different file had its OWN safe cap. That model's published config
declares 256 routed experts through that router, so this sat as a live,
undetected out-of-bounds write behind a status doc that actively said the
opposite. Same root cause as this lesson's own `router_bwd.wgsl` (a SEPARATE
kernel from the pair above, `array<f32, 64>` `pr`/`dp` scratch, hard-capped at
exactly the same 64): every kernel that copy-pasted this softmax/top-k
scratch-array shape needed its own audit — fixing the first two instances did
not imply the third and fourth were safe, and the written record repeating
"unaffected" without re-deriving it made this LESS likely to be caught, not
more. Fixed by rewriting `router_bwd.wgsl` array-free (mirroring
`router_bwd_sigmoid.wgsl`'s already-unbounded style — that one was NEVER
capped, its own header comment claiming "E <= 64" was simply stale/wrong) and
adding a loud `assert!(n_routed_experts <= 64, ...)` in the model's
constructor for the `router_gate_sigmoid.wgsl` forward path specifically,
since an array-free top-k rewrite there is real kernel work (needs `top_k`
passes over `E`), not a literal bump — deliberately NOT re-bumping the
constant a second time, which is this lesson's own failure mode repeated.
**The sharper general shape this recurrence adds**: when a status/lessons
doc records "X is unaffected because it uses Y instead," that claim is only
as strong as whoever wrote it having ALSO checked Y's own limit — a written
record that sounds like verification but is actually just "different code
path, assumed safe" is worse than no record at all, because it stops the
next reader from re-deriving what was never actually checked.

## 35c. The softmax top-k router's cap outlived its own bug-class writeup

`35b`'s "recurrence" paragraph fixed `router_bwd.wgsl` array-free and added a
guard `assert!` to `router_gate_sigmoid.wgsl`'s forward path, but left
`router_gate.wgsl`/`router_gate_train.wgsl` — the plain-softmax top-k
router, used by every model that isn't the sigmoid-`noaux_tc` scheme —
still hard-capped at `MAX_EXPERTS = 128` via `var prob: array<f32, 128>` /
`var used: array<bool, 128>`. Nobody had needed more than 128 experts yet, so
the cap never got exercised past its own boundary and the writeup moved on
to the other two kernels.

A later model declares **256** routed experts through exactly this router
(plain softmax top-k, not sigmoid), which is the first real model in this
repo to hit it. Fixed the same array-free way `router_bwd.wgsl` already was:
neither kernel now caches anything in an array sized by `n_experts` — the
softmax numerator is stashed in the `gate`/`probs` OUTPUT buffer itself
(already sized `[rows, n_experts]`, so it doubles as scratch) instead of a
second buffer, and the only remaining `var<function>` array
(`sel_idx: array<u32, 32>`) is bounded by `top_k`, never by `n_experts` — the
exact fix shape `35b` names but a bound the `top_k`-side needed too (a
top-k selection loop that excludes already-picked experts via an `n_experts`-
sized `used[]` bool array has the identical failure shape as caching the
probabilities themselves; both are "an array sized by the wrong dimension").

Caught by `crates/model/tests/router_gate_expert_cap.rs`, mirroring
`router_bwd_expert_cap.rs`'s own shape: a real (non-same-composition) host
oracle at `n_experts` values below, one past, and far past the former cap
(8, 129, 256) — the class of test `35b` itself says is the only thing that
actually catches this, since a same-composition sparse-vs-dense oracle is
blind to a bug present identically on both sides of the comparison.

**`router_gate_sigmoid.wgsl` (the sigmoid-`noaux_tc` router) is still
behind its documented `assert!(n_routed_experts <= 64)` guard, unchanged by
this fix** — the model above doesn't use that router kind, and a full
array-free rewrite there needs its own group-limited top-k pass structure
(`n_group`/`topk_group`), which is real kernel work, not a bound swap. Anyone
raising that router past 64 routed experts needs to do that work first, not
bump the `assert!`.

## 36. `testdata/`'s "restorable from a mirror" claim needs auditing against a real machine, not assumed from the script

`scripts/data/fetch-testdata.sh`'s own header states the design goal
plainly: `testdata/` is disposable, gitignored, and `make fetch/testdata`
repopulates it from local mirrors (`BRAIN_*_MIRROR` env vars, each defaulting
to a fixed path — the ONE place a machine-specific path may appear in this
repo, per AGENTS.md). An audit of a real machine found several of the
referenced mirror roots simply absent, and several newer fixture trees (an
imaging workstream's fixtures among them) with no entry in the script at all,
mirror or otherwise.

Net: on that machine, most of `testdata/`'s contents were not actually
recoverable by running the documented recovery command. The tree LOOKS
disposable (gitignored, a script claims to repopulate it) but can function as
irreplaceable local state whenever the mirrors it depends on were either
never copied to that machine or were never wired into the script for newer
fixtures. `rm -rf testdata/` under that condition would be silent, uncontested
data loss, not the "idempotent, re-fetchable" operation the script's own doc
comment promises.

**The general shape to watch for**: any "regenerate me" claim (a script, a
doc comment, a design note) that depends on an external resource (a mirror
directory, a network endpoint, a service) is a claim about a DIFFERENT
machine's state until it's been exercised on the one you're sitting at. The
fix isn't code; it's process: before treating any gitignored/"disposable"
tree as safe to clear, run the stated recovery path for real (or audit
mirror-existence + script-coverage per subtree) rather than trusting the
tree's own claimed disposability.

## 37. A roofline probe calibrated for GPU throughput reads as a hang on the CPU backend

`gpu_core::roof::measure_compute`'s self-calibrating loop starts at a small
iteration count and doubles (or jumps straight to a computed target) until a
dispatch's wall-clock time clears `MIN_PROBE_SECONDS` (50 ms), capping at
`1 << 20` iterations over `FMA_THREADS = 1 << 20` parallel lanes. Every
existing caller of `gpu_core::roof::ensure` (`qwen_bench` and its siblings)
had only ever been run against a real GPU backend, where that calibration
converges in a handful of doublings. The first CPU-backend run of it sat
alive for hours of wall-clock time — negligible RSS, negligible CPU, tens of
seconds of total CPU time accumulated — before being killed. That is not a
slow computation (which would show high utilization across the whole thread
pool the whole time); it is a blocked/deadlocked one. Bisected by killing the
process and checking where its stdout had stopped: it never got past the
banner print, i.e. it was stuck inside the probe itself, before any real
per-layer work began.

The root cause (somewhere in `measure_compute`'s calibration loop,
`crates/backend-cpu`'s rayon dispatch, or their interaction under this
kernel set / thread count) was not tracked down at the time — see #38 for the
follow-up. `gpu_core::roof` already ships the escape hatch, `BRAIN_NO_ROOF=1`,
which skips the probe outright and makes callers report "roofline unmeasured"
instead of guessing. **Any new bench or profiler that calls
`gpu_core::roof::ensure` (directly, or via a shared banner-style helper) and
might run on the CPU backend should default to skipping the probe there, or
at minimum document the escape hatch loudly** — the existing benches never
hit this because they only ever ran on GPU.

## 38. Two unrelated unbounded waits, not one bug, were behind "the GPU hangs"

Lesson #37 above named the CPU-side symptom (a roofline probe stuck for
hours) but explicitly left the root cause untracked. Revisiting it: the
kernel itself was innocent (`roof_fma.wgsl` takes its iteration count
through a *uniform*, not a specialization constant, so there is no per-rung
recompile; the first rung is only a few GFLOP, trivially fast on any real
device) — the "hours at near-zero CPU" was always a BLOCK, not slow
arithmetic, confirming #37's own reasoning without yet finding what was
blocking.

Reading `gpu_core::roof`'s three calibration loops
(`measure_compute`/`measure_int8`/`measure_bandwidth`) and both GPU
backends' wait implementations side by side surfaced **two independent
defects**:

1. **The calibration loop itself had no ceiling of any kind** — no
   wall-clock deadline, no total-work budget, only the opt-in
   `BRAIN_NO_ROOF=1` escape hatch #37 already documented. A backend that
   stalls on ANY single dispatch inside that loop (for any reason) blocks
   forever, because nothing bounds how long the loop is willing to wait
   before giving up and reporting "unmeasured."
2. **Both GPU backends' waits were LITERALLY unbounded** —
   `backend-vulkan`'s `wait_for_fences(&[fence], true, u64::MAX)` and
   `backend-wgpu`'s five `poll(PollType::wait_indefinitely())` call sites
   had no timeout and no device-lost handling. A wedged queue therefore
   blocks the calling thread forever BY CONSTRUCTION — this is why past
   hangs presented as unkillable-without-SIGKILL rather than as a reported
   error the caller could catch and act on.

Fix: a wall-clock deadline on every calibration loop (`BRAIN_ROOF_BUDGET_S`,
returning "unmeasured" on expiry — the SAME contract `ensure`'s doc already
promised for an unprobeable device, so no caller needed to change); the
probe now defaults OFF on the CPU device class specifically, since #37's own
root cause was never fully bisected and CPU is the one backend that
reproduced it; both backends' waits now have a finite deadline
(`BRAIN_GPU_WAIT_S`) and report which submit wedged rather than retrying
silently. Validated on real GPU hardware: the full gradcheck suite ran clean
after the fix, zero hangs.

**A separate, related finding surfaced while validating this fix, and later
corrected once a debugger was actually attached.** Running the roofline
test battery hung, and was initially assumed to be a known
concurrent-device-creation deadlock (`gpu_core::testgpu`'s own module doc
warns about a "every thread parked in `futex_do_wait`" failure mode) —
plausible given `gpu.new_like(PROBE_KERNELS)` runs per `measure()` call,
`/proc/<pid>/task/*/wchan` showed `futex_do_wait`, and `BRAIN_GPU_WAIT_S=5`
did not resolve it. That diagnosis was WRONG. The general shape held —
"the GPU hangs" is not always one bug, and `wchan` inspection correctly
ruled OUT the bounded-wait class — but `wchan` alone was not enough to find
the real cause, because a `Mutex::lock()` self-deadlock and a driver worker
thread both park in `futex_do_wait`; distinguishing them needs an actual
backtrace.

Environments that restrict `ptrace` to a process's own parent (`ptrace_scope=1`)
block attaching to an already-running process — but that restriction only
gates non-parent attach; launching the debugger as the process's own parent
(`gdb ./binary`, sending it commands, interrupting the child mid-hang to force
a prompt) sidesteps it entirely and is available even when live-attach is
blocked. The resulting backtrace showed `roof::ensure` holding `CACHE`'s
lock across a call into `measure(gpu)`, and `measure()` unconditionally
calling `gpu.caps()` (to gate the int8 probe), which locks the SAME `CACHE`
via `roof::known()` — `std::sync::Mutex` is not reentrant, so this was a
guaranteed, deterministic, single-threaded deadlock with zero dependency on
concurrency, device count, or the driver at all. It reproduced specifically
on the first test in the roofline suite to call `ensure()` (not `measure()`
directly) against a cold cache, and is masked in almost all ordinary use
because the on-disk persist store is normally already warm from a prior run,
so `ensure()` returns before ever reaching the self-deadlocking branch.
Fixed: `ensure()` now takes and releases `CACHE`'s lock at each access point,
never across `measure()`.

Re-validating Vulkan device sharing once the deadlock no longer masked it
found a REAL, separate, third defect: `VulkanBackend` had never implemented
`Backend::share`/`new_like`/`downgrade` (all default to `None`/no-op), so
`gpu_core::testgpu::dev`'s "one shared device per process" pool never
actually shared anything on Vulkan — every call silently built a whole new
device. Fixing that (proper `Arc`-shared `VkContext`/`VkPipelineSet`, see
`crates/backend-vulkan/src/lib.rs`) then surfaced a FOURTH defect one layer
down: `VulkanBackend`'s command-buffer recording and `VkContext::run_cmd`/
`dispatch` all touch a SHARED `command_pool`/`queue` with no synchronization,
and the Vulkan spec requires host access to both to be externally
synchronized — two threads sharing a device via the newly-working `share()`
reproduced a REAL `ERROR_DEVICE_LOST` within seconds
(`crates/gpu-core/tests/device_sharing.rs::concurrent_shared_handles_do_not_deadlock`,
unreliable before a fix, clean after adding `VkContext::queue_lock`, held
across every allocate-record-submit-wait-free sequence).

**The lesson under the lesson**: `wchan`-only triage can correctly rule OUT
one hypothesis (here, the bounded-wait class) while still pointing at the
WRONG mechanism for what remains — a self-deadlock and a driver stall
produce the identical `futex_do_wait` signature. When a debugger is even
partially available (here: blocked for attach, but not for launch-as-child),
use it before writing down a driver-level conclusion; a wrong "it's the
driver's fault, needs a kernel-level fix" diagnosis can stand undisturbed
for an entire investigation if nothing pushes back on it.

## 39. A `write_*` call with no matching read on that code path is a silent no-op, not a bug the type system catches

`qwenvl::Qwen3Vl::generate()` (KV-cache generation) called
`self.decoder.write_deepstack(level, &data)` during prefill — a real
function, on a real buffer, compiling cleanly, doing exactly what its name
says. The bug: `qwen3::Qwen`'s incremental decode path
(`decode_steps`/`step`/`step_mrope`) never READS `deepstack_bufs` — the one
dispatch that adds them into the residual (`SPLICE_ADD`) lives only inside
`forward_steps()`, the BATCHED training-graph builder, which `generate()`
never calls. So the write happened, the data sat in a real GPU buffer,
correctly, and was never consulted by anything — for any checkpoint enabling
DeepStack (`VisionConfig::deepstack_indexes` non-empty, which a real
production config is), every generated token was silently missing a real
architectural contribution. No panic, no wrong-shape error, no type error:
`write_deepstack` and `decode_steps` are both individually correct functions
that simply never talk to each other on this path. The existing test
(`generate_is_deterministic_and_respects_eos`) passed throughout, because it
only checks determinism and EOS-stopping, never a numerical value that would
reveal a missing contribution — the same shape of gap this file's own
"gates that lie" pattern describes, just for a hand-written test instead of
an automated gate.

Found by asking a narrow, specific question while wiring an UNRELATED
follow-up: "does the fast (incremental) path apply the same architectural
pieces as the slow (batched) reference path?" — not by a test failing. A
targeted grep across the file in question is what actually answered it
(every reference was setup or the one batched-only consumption site). **The
general shape**: when a model has TWO forward implementations for the same
architecture (a batched training/reference path and an incremental/decode
fast path — extremely common for any KV-cache-capable model), a feature added
to one must be independently verified to exist in the other, not assumed
from "the setter function got called." A parity test between the two paths
(`deepstack_step_matches_full_recompute`) is the gate that would have caught
this before it shipped, not after.

## 40. A tiny gradcheck config can numerically starve a normalization op, producing a hollow — but *passing* — gradcheck

`gradcheck::check_qwen35` (qwen35moe's model-level backward integration -
renamed to `check_qwen35moe` once the dense `qwen35` sibling arch took the
plain name; this entry is left as originally written) passed
clean on first run: every parameter's finite-difference check fell inside
the workspace's `(4e-3, 8e-2)` gate. Every Gated-DeltaNet-layer parameter
(`A_log`, `dt_bias`, `in_proj_a.weight`, `in_proj_b.weight`) reported its
finite-difference numeric gradient as *exactly* `0.0`, not merely small —
and the tolerance's `denom.max(1e-3)` floor swallowed the distinction, since
the tiny *analytic* values (`~1e-13` to `~1e-17`) were dwarfed by the same
floor. A wholly wrong or entirely-disconnected gradient looks identical to
a genuinely tiny one once both round to the floor.

Confirmed by direct probing (bypass the checker, call `forward()` by hand):
perturbing `A_log` by ±10 — enormous relative to its own scale — left the
loss **bit-identical**, while the raw decay gate `g` it feeds was confirmed
to vary correctly (-0.001 to -157) under the same perturbation. The root
cause was three stages downstream: `Qwen35Config::tiny()`'s standard
`std=0.02` init, applied to `in_proj_qkv.weight` then a *depthwise* causal
conv1d (`groups=conv_dim`, so each output channel sums only `kernel=4`
terms — far less central-limit averaging than a `d_model`-wide matmul) then
SiLU, collapses `query`/`key`'s pre-normalization magnitude to `~1e-6` by
the time they reach `l2norm_scale.wgsl`. That kernel's `eps=1e-6` (tuned for
the REAL model's much wider `d_model=2048`, not a toy config) then
*dominates* the normalization denominator instead of the vector's own norm,
so the "normalized" output is just `x/sqrt(eps)` — still proportional to
the collapsed input, not an actual unit vector — and the entire chunked
recurrence downstream is swamped to `~1e-11` regardless of any decay/beta/
gate parameter. The standalone `model::gdn` unit tests
(`crates/model/tests/gdn_chunk_{fwd,bwd}.rs`) never hit this: they feed
`q`/`k`/`v`/`g`/`beta` directly at `std=1.0` with no upstream conv chain, so
this normalization/init interaction never had a chance to manifest there.

Fix: **do not touch `model::gdn`'s `eps` or the model's production
`std=0.02` init** — both are correct for the real model's scale. Instead, the
gradcheck harness itself (`qwen35_gradcheck_harness`, now
`qwen35moe_gradcheck_harness`, in
`crates/gradcheck/src/lib.rs`) overrides just `in_proj_qkv.weight`/
`conv1d.weight` to `std=1.0` post-init, restoring a real, non-degenerate
finite-difference signal (confirmed: `blocks.N.linear_attn.conv1d.weight`
and `A_log` both now show close analytic/numeric agreement instead of a
floor-dominated `0.0`). **The general shape**: whenever a tiny gradcheck
config feeds a fresh, small-std init through more than one cascaded
small-scale stage before an operation with a fixed additive epsilon
(L2-norm, RMSNorm, any `/(x+eps)` normalization), check whether the
pre-normalization magnitude has collapsed to epsilon's own scale — a
gradcheck that passes because the *true* gradient is unresolvably small is
indistinguishable, from the report alone, from one that passes because the
implementation is correct. When a parameter reports numeric `0.0` across
every check in a report, perturb it by hand outside the checker and confirm
the *loss itself* moves before trusting the pass.

## 41. A kernel's uniform offset parameter usually offsets only ONE named side — check which before reusing it in the other direction

`qwen3::Qwen::decode_steps`'s per-token DeepStack add needed to read
`deepstack_bufs[level]` (a compact `[n_rows,d]` block) starting at
`local_row*d`, while writing into `res[l+1]` (already a single `[d]`-sized
row) at offset 0 — the opposite offset direction from `splice_add.wgsl`'s
OTHER caller (prefill's whole-range add, which reads the compact block from
its own index 0 and writes it into the big sequence buffer at `row0*d`). The
existing code used `step_sliced` to put `local_row*d` on the SOURCE as an
actual bind-group buffer offset — which produced a real, deterministic wgpu
validation failure ("Buffer offset N does not respect
`min_storage_buffer_offset_alignment`") on hardware enforcing the full 256B
limit, since `local_row*d*4` has no reason to be a multiple of 256.

The fix that looked obvious — switch to a plain `g.step` and pass
`local_row*d` as `splice_add.wgsl`'s own `base` uniform, since uniform
offsets have no alignment constraint — was *wrong* and initially replaced a
crash with a silent wrong-value bug that a numeric parity test caught
(`deepstack_step_matches_full_recompute` went from a wgpu panic to a
`maxabs=0.499` mismatch). Reading the kernel's own `dst[p.base+idx] +=
src[idx]` body shows why: `base` was written to parameterize the
DESTINATION offset only, matching prefill's need exactly — it has no way to
express a SOURCE offset. Decode needed the other direction. The real fix was
a new sibling kernel (`splice_add_offset_src.wgsl`, `dst[dst_base+idx] +=
src[src_base+idx]`) rather than reusing the existing one's single offset
knob backwards.

**The general shape**: before reusing an existing kernel's uniform offset
parameter for a new call site, read the kernel body (not just its
signature) to confirm which of its several buffer operands that parameter
actually addresses — a `base`/`offset` name gives no guarantee it's
symmetric across every operand, and assuming so produces a bug the
compiler, the alignment validator, and a coarse test can all miss (only a
value-level parity check caught this one).

## 42. Streaming through mmap fixes the peak but not the sum, and a checkpoint's index filename is a naming convention, not a guarantee

Fixing Z-Image's OOM on a memory-constrained box took three independent
bugs, each invisible until a real large checkpoint ran on a real machine:

1. `checkpoint::weightio::WeightReader::open_hf_dir` and
   `safetensors::read_model_dir` both hardcoded the HF-transformers index
   filename (`model.safetensors.index.json`). A diffusers-style checkpoint
   (`diffusion_pytorch_model.safetensors.index.json` — what Z-Image's
   `transformer/` dir actually ships) didn't fail to parse; it silently took
   the "no index found → there must be exactly one shard" fallback and opened
   only the alphabetically-first file. Two-thirds of the checkpoint was never
   even mapped. The failure surfaced as a late, unrelated-looking panic
   ("missing tensor X") deep in a totally different layer, not as an error
   at open time — a **wrong-but-plausible fallback path masquerading as the
   single-shard case** is the general shape here, and it will recur for any
   reader that treats "recognized index absent" as "no index exists" rather
   than "check for the other convention, then fail loudly."
2. Streaming a tensor from an mmap and dropping the returned `Vec<f32>`
   bounds *heap* allocation to one tensor, but says nothing about the
   mapped file's *page-cache* footprint — every page touched during a
   whole-checkpoint scan stays resident (counted in RSS) until the kernel
   feels enough pressure to reclaim it, and under swap exhaustion that
   reclaim can lose the race against the OOM killer. `MADV_DONTNEED` existed
   in this repo for exactly this and had zero callers. **A "streaming" read
   path is only actually bounded in RSS if something explicitly evicts what
   it just consumed — mmap alone bounds the wrong resource.**
3. A per-model default that was correct reasoning for a multi-GPU box (keep
   the encoder on CPU so it doesn't compete with the DiT for a card's VRAM)
   was pure pessimization on a single-GPU box, because there the CPU
   alternative isn't cheaper — it's a permanently-resident fp32 copy instead
   of a smaller, on-demand, auto-dropped int8 one. **A safe default on one
   topology is not a safe default on every topology; branch the default on
   what's actually true of the machine (here: GPU count), not on what's
   convenient to assume.**

None of the three had a unit test written ahead of time that would have
caught it — each needed the real checkpoint's actual shard layout, the real
machine's actual memory pressure, or the real machine's actual GPU count to
manifest. The fix in every case was cheap once found; the finding required
actually running the thing end to end, not reasoning about it in the
abstract.

## 43. A config field that is parsed but never read is a silent architecture gap, and it can affect more than one caller at once

`crates/codec/src/config.rs`'s `CodecConfig::sliding_window` (real value 72)
has existed since the Code2Wav port landed — parsed straight from the real
`decoder_config.sliding_window` JSON key, correctly, every time. Nothing in
`crates/codec/src/model.rs::transformer` ever read it: the pre-transformer's
attention dispatched plain `model::block::gqa_fwd` (full causal, unbounded),
not a windowed variant. The real reference
(`Qwen3OmniMoeCode2WavAttention`/`MimiAttention`, both built on
`create_sliding_window_causal_mask(sliding_window=config.sliding_window)`)
applies the window on **every** forward call, one-shot included — not only a
chunked/streaming path — so this was a plain correctness bug, not a missing
optimization: `Codec::decode_omni`'s output has been wrong for any T > 72
since the day it shipped. It was invisible because the only Omni Code2Wav
test uses T=8, and the standalone-codec unit test uses T=24 — both comfortably
under the window, so both passed at cosine ~1.0 while silently exercising the
plain-causal fallback that happens to equal the windowed answer at small T.

**Two compounding lessons, not one:**

1. A config field with no reader is not evidence the field is unused — it is
   evidence nobody checked whether it's used *yet*. `grep`ping for a config
   field's read sites (not just its parse site) before trusting "this model's
   forward pass is complete" would have caught this immediately; nothing in
   this codebase's own review process does that grep automatically.
2. **The bug was shared by two callers who look independent.** `Codec::transformer`
   is dispatched by both `Codec::decode` (the standalone Qwen3-TTS codec) and
   `Codec::decode_omni` (Qwen3-Omni's Code2Wav) — one function, one config
   field, two model families. Finding it while working on Omni's chunking
   prerequisites fixed the standalone TTS codec's identical, previously
   unknown bug for free; conversely, `crates/codec`'s ENCODE-side transformer
   (`enc_transformer`) has the *exact same* unapplied-`sliding_window` shape
   (`EncoderConfig::sliding_window`, 250) and was deliberately left unfixed
   here (filed as `.todo/codec-encoder-sliding-window.md`) — a shared-code
   bug rarely has exactly one call site, and finding one instance is a reason
   to grep siblings, not a reason to stop at the first one found.

Fixed with a new kernel (`gqa_scores_win.wgsl`, `crates/kernels/wgsl/`) and a
new shared primitive (`model::block::gqa_fwd_win`,
`crates/model/src/block.rs`) that degenerates to `gqa_fwd`'s plain causal mask
exactly when `window >= t` — so `Codec::transformer` now dispatches it
unconditionally, with no `Option<u32>` branch and no risk of the old
plain-causal call site quietly coming back at the next edit. Proven two ways
in `crates/model/tests/gqa_fwd_win.rs` (no real checkpoint available in this
environment — see that file's own doc for why): window-covers-sequence
matches plain causal exactly, and window-narrower-than-sequence matches an
independent host-computed masked-attention oracle AND provably diverges from
the old unwindowed output at the row where it must (a mutation-style check
that the window is load-bearing, not merely wired and ignored).

## 44. Destroying the last Vulkan instance can permanently blind a process to its own GPUs

`brain ltxv t2v` with real weights ran all 8 denoise steps, then died the
instant VAE decode opened its device:

```
brain: wgpu enumerated 0 adapters while looking for "Tesla P40" (pci ...);
       falling back to wgpu's own adapter request
limits: max_buffer_size 2047 MiB, max_storage_buffer_binding_size 128 MiB
wgpu error: Validation Error
  Buffer binding 2 range 452984832 exceeds `max_*_buffer_binding_size` limit 134217728
```

Every plausible reading of that is wrong. It is not VRAM exhaustion (the card
was under 2 GB of 24 GB throughout), not driver corruption (`nvidia-smi` and
`brain devices` enumerate perfectly the moment the process exits), and not a
placement bug in the identity matching. The process had simply stopped being
able to see hardware that never went anywhere - and then silently continued on
a software rasteriser, whose 128 MiB binding limit is what actually raised the
error, several layers away from the cause.

**The cause: instance churn, not device churn.** Creating a `wgpu::Instance`
(or an `ash` one) makes the Vulkan loader `dlopen` every installed ICD;
destroying the last one makes it `dlclose` them. The NVIDIA ICD does not
survive that cycle. With `VK_LOADER_DEBUG=error` the tenth round says exactly
what happened:

```
loader_scanned_icd_add: Could not get 'vkCreateInstance' via
    'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
```

The library still loads; it just no longer has a working entry point. From
then on the process enumerates zero physical devices, forever. brain hit this
because "open a fresh device per forward call" is a deliberate and measured
design in several models, and each open built its own instance - nine were
enough.

**Rules that follow.** A Vulkan instance is process-scoped state: create one,
never destroy it. Logical devices are still created and destroyed per use;
only the instance is permanent, and it owns no VRAM. And an adapter-enumeration
fallback must be *loud*, because a software adapter computes correct answers
slowly right up until an allocation exceeds its much smaller limits - which is
why the churn tests that already existed
(`crates/gpu-core/tests/device_churn.rs`) passed all the way through this bug:
they asserted the arithmetic was right and never asserted **which card ran it**.
A device-lifecycle test has to gate placement, not just results.

Two earlier investigations recorded confident, wrong conclusions from this same
symptom - "whatever wedges the enumeration is not brain's device lifecycle",
and a "slow-reclaim driver-side handle pool" theory measured to a deterministic
count. A deterministic failure count is a strong hint to go find the mechanism
(here: one `VK_LOADER_DEBUG=error` run), not a phenomenon to characterise
further.

## 45. A stage absent from the timing struct is absent from every conclusion

`ltxv::pipeline::Timings` carries `build_dit`, `denoise` and `vae`. It does
not carry the text encode. So a real `brain ltxv t2v` run printed
`964.3s total (build 10.84s, denoise 440.4s, vae 21.1s)` - three numbers that
sum to 472s of a 964s run - and every performance discussion built on that
line spent its effort on `denoise`, the second-largest stage, while the
largest one (the text encoder, **51% of the wall clock**) had never been
measured at all, by anyone, once.

Nothing was wrong. Each printed number was correct. The line simply did not
claim to be exhaustive, and no reader checked whether it added up.

Two things generalise, and the second is the one that cost the time:

* **A timing breakdown must either account for its own total or say what it
  is missing.** The same rule this file already records for a *rate*
  (#28: a partial FLOP numerator over a full denominator under-reports in
  silence) applies to a *timeline*. The fix shape is the same too: print the
  unattributed remainder as its own row rather than letting it be invisible.
  A breakdown whose parts sum to 49% of its own total is not a rounding
  issue, it is a missing stage.
* **Optimizing the stage that IS instrumented is the predictable
  consequence.** Two prior optimization passes on this model attributed,
  extrapolated and optimized inside `denoise` in real detail - correctly,
  and to real effect - because that is where the numbers were. Instrumenting
  the pipeline end to end (`--trace-ltxv`, one span per stage) is what made
  the largest stage visible, and it was visible immediately, on the first
  run, with no analysis at all.

## 46. A bench with a warm page cache measures the CPU where the real run measures the disk

The isolated harness for one model's streamed forward predicted ~275s of
denoise for a real generation. The real generation measured 440s, and the
per-stage split disagreed in BOTH directions - the harness under-predicted
the first (cold) step by 2.1x and over-predicted every later step by 1.4x.

The harness was not wrong about anything it measured. It had simply been run
several times in a row over the same four transformer blocks, so those blocks
were resident in page cache and its "checkpoint read + dequantize" stage was
timing the dequantize. The real run reads ~50 GB of checkpoint that nothing
has touched, on storage measured at **58-68 MB/s cold** (against 4.3 GB/s for
the same bytes once cached - a ~70x cliff), and its same-named stage is
timing the disk.

Consequences worth keeping:

* **Check whether the resource you are measuring is the one the real run
  hits.** `free`, the process state (`D` vs `R`), and CPU-time-versus-wall
  -clock all answer it cheaply: the run that looked like 964s of computation
  spent 7m12s of CPU across 16m14s of wall clock, which says "waiting" before
  any profiler is opened.
* **Measure the device, do not assume it.** A `dd` from an uncached region
  took under three minutes and turned a kernel-optimization brief into an
  I/O finding. Sixteen parallel readers measured the SAME throughput as one,
  which additionally killed the natural follow-up hypothesis (deepen the
  queue) before any code was written for it.
* **A warm-cache number is still the right number for the CPU work inside
  the stage** - it is how the parallel dequantize and quantize wins were
  isolated from I/O noise at all. Keep both, and label which is which; the
  mistake is letting one stand in for the other in an extrapolation.

## 47. A tap measured downstream of a normalization cannot gate a scale error upstream of it

The int8 tier added to the Gemma-4 text encoder was gated by comparing a real
layer's fp32 and int8 outputs on the real checkpoint, asserting cosine AND
`rel_l2` on the layer output and cosine on the self-attention tap. That gate
was then mutation-verified by halving every per-output-channel weight scale -
the single most likely packing bug in a per-channel int8 tier, and one that
makes the arithmetic wrong by a factor of two.

**The gate passed.** Two independent reasons, and both generalise:

* On the self-attention tap, cosine came back **bit-identical** to the
  unmutated run (0.998157036 either way). That is #2 restated with a real
  number: cosine is scale-invariant, so the one error a per-channel *scale*
  can make is exactly the error cosine cannot see. Only `rel_l2` moved, and it
  moved enormously - 0.061 to 0.503.
* On the layer OUTPUT, *nothing* moved outside tolerance, `rel_l2` included
  (0.067 to 0.082). The attention branch is followed by
  `post_attention_layernorm` and then added to a residual: the norm divides
  the injected factor straight back out, and the residual dominates what
  survives. A metric that can see a scale error still cannot see it if it is
  measured after something that removes scale.

So the gate needed `rel_l2` **on the pre-normalization tap specifically**, and
that assertion is what now fails the mutation. The general rule: when choosing
where to tap a quantized-vs-reference comparison, ask what lies between the
error's entry point and the tap. Any normalization in between (RMSNorm,
LayerNorm, a softmax, an L2 normalize) makes everything downstream blind to a
whole error class, and a residual add makes it blind to magnitude generally.
Tap the sublayer's own output, not the block's.

The corollary is about process rather than numerics: this gate looked complete
and was written by someone who had just read #2. Only deliberately breaking
the thing it guards showed which of its four assertions were load bearing
(one) and which were decoration (three). A gate nobody has seen fail is a
hypothesis - including when the person writing it was actively trying to
avoid this exact defect.

## 48. A parity gate built from the pipeline's own simplified positions cannot catch the pipeline getting positions wrong

A real-weight video DiT produced finite, well-behaved, non-NaN output at
every shape tried - and looked like fine-grained textured noise every time,
regardless of resolution, regardless of `eta` (ancestral vs. deterministic),
regardless of a from-scratch real-image-conditioning feature built and
verified against the reference source line by line. Every other component on
the path - the VAE decode (bit-exact against the reference at real width),
the text encoder (cosine 1.0 through 6 real layers, both attention types),
the embeddings connector (cosine 0.999999973 on real weights), the streamed
vs. eager forward (bit-identical) - had already been proven correct against
the actual reference, independently, several different ways. None of it
mattered, because the RoPE position VALUES the whole stack was built on were
wrong.

**The bug:** the pipeline built RoPE positions as raw latent-grid integers
(`token → [i, i+1)` per axis). The reference builds them in **pixel space**:
`get_pixel_coords` multiplies each latent bound by the VAE's own downsample
factor (time×8, height×32, width×32), applies a causal fix to the temporal
axis (the first latent frame covers exactly one pixel frame, every later one
covers eight), and only then divides the temporal axis by fps. Latent index 3
on the height axis is `[3, 4)` in one convention and `[96, 128)` in the
other - the SAME formula shape (`precompute_freqs_cis` over `theta`/
`max_pos`), fed a value 32x off. No crash follows from that: the model just
runs its trained frequency table over the wrong slice of it, per token,
uniformly, which is indistinguishable from "producing noise" until you know
to look for a scale, and is exactly why `positional_embedding_max_pos:
[20, 2048, 2048]` (unmistakably pixel-scale) sat unremarked next to a
latent-integer position builder for an entire porting pass.

**Why the existing real-weight gates could not have caught it.** Every
reduced-depth real-weight DiT test in this port replayed the reference's own
golden dumper - which itself builds positions from a raw `torch.meshgrid`,
not from `get_pixel_coords`/`VideoLatentTools`, as a deliberate, documented
scope cut ("prove the op sequence," not "prove the pipeline's own position
construction"). Brain's Rust replay used the identical raw-meshgrid
convention to match. Both sides agreed, at cosine ≥ 0.999999, because both
sides were self-consistently wrong the same way - a parity gate cannot see a
divergence that its own golden was built to not exercise. This is a
different shape of blindness than #47's (a real tap made invisible by
something downstream of it): here the tap was fine, the INPUT the gate fed
it was never the one the real pipeline actually constructs.

**A second bug hid in the same reference method, same root cause.** The
reference marks `keyframes_mask` on the first latent frame's tokens
*unconditionally* - the causal VAE gives that frame its own narrower pixel
window, "the same token class as a generated keyframe slot," independent of
whether any real image conditioning is present. The port had reasoned "no
image conditioning → every token is genuinely noise → mask stays zero,"
which is true and irrelevant: the mask marks a structural token CLASS, not
"this token is externally supplied." A real per-token positional-embedding
add was silently skipped on every real generation.

**Consequences worth keeping:**

* **A parity gate that builds its own inputs is only as strong as how those
  inputs were built.** Ask, specifically: does this golden exercise the
  pipeline's REAL position/coordinate/shape-construction code, or a
  simplified stand-in that happens to be self-consistent across both sides?
  "Both sides use the same formula" is not evidence the formula is the real
  one.
* **Cosine similarity (and even a tight `max_abs`) cannot catch a systematic
  input error that both the implementation and its own test were built
  around identically.** The fix is not a better metric - it's exercising the
  ACTUAL pipeline construction path in the gate, end to end, not a purpose-
  built substitute.
* **A finite, non-NaN, well-behaved-looking wrong output is the hardest kind
  to root-cause by staring at pixels.** Every visual symptom here (fine
  mesh texture, no global coherence, indifferent to resolution/eta/image
  conditioning) was consistent with "the model just needs more tokens" -
  a plausible, false lead that survived several independent, correctly-
  reasoned investigations before the actual cause surfaced from re-reading
  the reference's own coordinate-construction code end to end, not from
  another round of comparing pixels.
* **A config field's own stated range is a clue.** `positional_embedding_
  max_pos: [20, 2048, 2048]` had been read and repeated in this port's own
  documentation for an entire phase without anyone asking why a max position
  of 2048 would make sense for a token grid that never exceeds a few hundred
  entries. It would not - because the position space was never meant to be
  the token grid's own index space.
* **When adding position/coordinate handling for any new model, verify units
  against the reference's actual coordinate-construction function (not its
  RoPE math, which is downstream and unit-agnostic), and build at least one
  gate that exercises that construction for real** - a synthetic/simplified
  golden proves the op sequence, never the units.

## 49. A frozen token still has to be TOLD it is frozen - masking the output is only half the mechanism

Image conditioning on a real 22B video DiT (`--start-frame`/`--end-frame`:
hold a real still fixed at the clip's first and/or last pixel frame and
generate the motion in between) produced a video whose conditioned frame was
a flawless reproduction of the source image and whose every OTHER frame was
a fine horizontal-line mesh texture with no recognizable structure. The
same prompt, seed, resolution and schedule with NO image conditioning
produced clean, coherent, temporally consistent video. The token geometry
had already been derived from the reference line by line and unit-tested:
positions (`VideoConditionByKeyframeIndex`'s `get_pixel_coords` + `+=
frame_idx` + single-pixel-frame narrowing), the frozen `denoise_mask`, the
`clean_latent` content, the append-vs-overwrite split between `frame_idx ==
0` and `frame_idx != 0`. All of it was correct.

**The bug:** the denoise loop fed the model `vec![sigma; t]` - the
schedule's current sigma broadcast to every token - as its timesteps. The
reference builds `Modality.timesteps` as `timesteps_from_mask(denoise_mask,
sigma)`, i.e. the per-token *product*. A frozen conditioning token holds
clean VAE content and therefore has timestep `0`, not `sigma`.

**Why that is catastrophic rather than cosmetic.** The DiT's AdaLN builds
one shift/scale/gate row PER TOKEN out of that per-token timestep. A token
carrying clean image content but labelled "as noisy as everything else" gets
modulated as if it were pure noise, and self-attention then mixes that
mis-modulated token into every genuinely-noisy token in the sequence. The
conditioning block, meant to be the strongest signal in the clip, becomes
the strongest *corruption* in it.

**Why the symptom pointed away from the cause.** `post_process_latent`
re-pins frozen tokens to `clean` after every step regardless, so the
conditioned frame decodes perfectly no matter how badly the model was
misled about it. "The frame I pinned is perfect, everything else is
garbage" reads as *the conditioning works and the model cannot cope with
what is left* - a capacity story (too few latent frames, too much temporal
compression, an over-constrained loop) that is entirely plausible, entirely
wrong, and expensive to disprove by generating at larger shapes.

**A second bug of the same family, same commit.** The append path rebuilt
`keyframes_mask` from scratch as all-zeros over the whole (base + appended)
sequence. The reference's `extend_keyframes_mask` *extends* the existing
mask: the appended conditioning tokens are `marked=false`, but the base
video keeps `_first_frame_keyframes_mask`'s unconditional first-latent-frame
ones. Rebuilding instead of extending meant turning image conditioning ON
also silently turned OFF a positional-embedding term the same clip had in
the unconditioned path - a per-feature regression of the very fix in #48.

**Consequences worth keeping:**

* **Conditioning has two halves: what the sampler does with the output, and
  what the model is told about the input.** Freezing a token by masking the
  x0 estimate makes the token come out right; it does nothing to stop that
  token from poisoning everything else. When porting any masked/partial
  denoising feature, find every place the reference derives something from
  `denoise_mask` - not just the one that writes the latent back.
* **"The conditioned part is perfect, the generated part is broken" is
  evidence about the model's INPUTS, not about its capacity.** A pinned
  output is pinned by the sampler and tells you nothing about whether the
  model understood it. Resist the capacity hypothesis until the per-token
  inputs have been diffed against the reference field by field.
* **A per-token tensor passed as a scalar-broadcast is invisible to shape
  checks and to every weight-free geometry test.** `vec![sigma; t]` has the
  right length, the right dtype and the right values on the code path that
  had existed until then (nothing frozen). Where a reference computes a
  tensor as a product of two things, port the product - not the
  simplification that happens to be equal while the other factor is all
  ones.
* **When an append-style conditioning item builds a new state, audit every
  field it reconstructs.** The reference's items use `dataclasses.replace`
  or explicit `extend_*` helpers precisely because rebuilding a field from
  scratch silently discards whatever earlier stages put there. A field the
  port rebuilds as a constant is a field the port has stopped tracking.

## 50. A pixel-delta gate with no calibrated threshold passes on decode dither, and "the conditioned clip does not move" turned out to be the request rather than the bug

`--start-frame X --end-frame X` on the real 22B distilled checkpoint - the
same still pinned at both ends, i.e. a seamless loop - produced a clip in
which every frame is the pinned still. The dog is in the identical
fully-extended pose in all 25 frames; the flying card does not move a pixel.
The immediately preceding fix (#49) had verified itself with "mean
consecutive frame delta went from 14.77 to 2.64-4.59", read as garbage ->
coherent. It was neither.

**The metric was the first bug.** Mean consecutive-frame delta on this
pipeline, 384x192, 9 frames: a visually frozen clip scores **2.2**, a real
one **5.4**. VAE decode dither on static content is not zero, so "nonzero"
proves nothing, and 2.4x reads as "less motion, but motion". The metric that
separates them is **peak excursion from frame 0**, `max_i mean|frame_i -
frame_0|`: the same two runs score **5.1** and **26.6**. A static clip cannot
accumulate - its wobble is uncorrelated noise around one image - while real
motion walks away from frame 0 and stays away. (Not `|frame_last - frame_0|`
either: a loop is supposed to come back.) `crates/ltxv/tests/motion_real.rs`.

**Then localize by neighbouring request, not by reading the port.** Sixteen
real-weight runs, one variable each, peak excursion, seed 42:

| request | 384x192 9f | 384x192 25f | 640x320 25f | 640x320 49f |
|---|---|---|---|---|
| text-to-video, no stills | 26.6 | 18.1 | 20.5 | - |
| one still at frame 0 | 35.1 | - | - | - |
| two DIFFERENT stills | 35.2 | - | 40.0 | - |
| the SAME still at both ends | 4.6 | 32.4 | 7.3 | 7.2 |

Everything at or above 18.1 visibly animates; everything at or below 8.8
reproduces the pinned still in every frame. One row is static. The engine
generates motion from nothing, from one anchor, and between two different
anchors - at the SAME shape, seed, prompt, sampler and conditioning code.
That row also survives every lever the reference exposes: `strength`
1.0/0.8/0.5 (4.6/4.8/5.0), the deterministic and the ancestral sampler
(4.6/5.1, and 7.3/8.8 at 640x320), the in-place and the appended frame-0
conditioning mechanism (5.1/4.6), a CRF-33 re-compressed conditioning still
(8.4 vs 7.3), and doubling the clip length (7.2). The decisive control is
the third column: changing ONLY the end still from "the same image" to
"that image mirrored" moves the score from 7.3 to 40.0. "Start at this
image and end at this same image" has a correct trivial solution and the
model returns it. Five plausible root causes, each with a real reference
citation, each refuted by one run.

**Real port defects the hunt did find, none of which was the reported
symptom:**

* **A refusal inherited from the wrong reference function.** `denoise`
  asserted `eta == 0` whenever any token was conditioned, on the grounds
  that a frozen token's ancestral renoise term needs a per-token sigma. That
  is true of `ltx_pipelines.utils.samplers._inject_sde_noise` (the res2s
  loop, which really does build `stack([timesteps_from_mask(mask, sigma),
  timesteps_from_mask(mask, sigma_next)])`) and false of
  `_ancestral_euler_denoising_loop`, which is what
  `euler_ancestral_denoising_loop` - and therefore LTX-2.5 distilled's own
  stage 1 (`ltx_pipelines.distilled`: `ANCESTRAL_SAMPLER_SINCE_VERSION =
  (2, 5)`, `ANCESTRAL_ETA = 1.0`) - actually runs. That loop steps the whole
  latent with the SCALAR schedule and re-applies `post_process_latent` to
  the STEPPED result. The port had deleted the checkpoint's own sampler from
  the conditioned path to satisfy a constraint that path never had.
* **The two loops apply the conditioning mask in different places.**
  `euler_denoising_loop`/`_step_state` masks the x0 ESTIMATE before
  stepping; `_ancestral_euler_denoising_loop` masks `x_next` AFTER the
  renoise term (`if draw_noise: x_next = post_process_latent(...)`) and
  short-circuits the terminal sigma to the raw estimate with no mask at all.
  Porting one ordering and using it for both leaves freshly injected noise
  sitting on a token that is supposed to be clean.
* **The wrong conditioning builder for interpolation.** `--start-frame` +
  `--end-frame` is `ltx_pipelines.keyframe_interpolation`, which uses
  `helpers.image_conditionings_by_adding_guiding_latent` - every image,
  `frame_idx == 0` included, becomes an APPENDED
  `VideoConditionByKeyframeIndex`. The port reused
  `combined_image_conditionings`' image-to-video branch, whose
  `frame_idx == 0` special case (`VideoConditionByLatentIndex`) OVERWRITES
  the generated video's own first latent frame. At 121 frames that freezes 1
  latent frame in 16; at 9 it freezes half the clip.
* **A required reference argument hardcoded to its most extreme value.**
  `ImageConditioningInput.strength` has no default upstream -
  `--image PATH FRAME_IDX STRENGTH` requires all three and the help text's
  own example is `0.8`. The port pinned it to `1.0` with no knob.
* **A mandatory preprocessing step missing entirely** (still missing, and
  measured not to matter for motion): `helpers.load_image_and_preprocess`
  runs `preprocess(image, crf)` - an H.264 re-encode/decode round trip at
  `DEFAULT_IMAGE_CRF = 33` - before the resize, "to match the compression
  the model was trained on", and raises rather than defaulting when `crf` is
  unresolved. The port feeds the VAE a pristine PNG. It also resizes with
  `resize_exact` (aspect-distorting, Lanczos) where the reference does
  aspect-preserving `resize_and_center_crop` (bilinear).

**Consequences worth keeping:**

* **Calibrate a threshold against a known-BAD run, not only a known-good
  one.** A floor picked from the good run alone has no idea how close the
  bad run gets. Both numbers, from real weights, or the gate is decoration -
  and record them in the test so the next person can see the separation
  instead of re-deriving it.
* **A generative "this looks wrong" is localized fastest by the nearest
  request that looks RIGHT.** Vary one input at a time - drop a
  conditioning, change one anchor, change the shape - and let the boundary
  fall out. Reading the port for a bug that is not there costs more than the
  GPU time; here the decisive evidence was one run with two different stills
  instead of one still twice.
* **A refusal is a feature deletion, and it inherits its justification from
  whichever reference function was open at the time.** When a port refuses
  a combination the reference supports, check that the constraint belongs to
  the code path the reference actually takes for that combination.
* **When the reference makes an argument REQUIRED, the port must not pick a
  value.** A required argument means upstream believes there is no safe
  default. Hardcoding one - especially the extreme end of its range - turns
  a user decision into an invisible property of the port.
* **Look at the frames.** Nine evenly spaced PNGs read by eye settled in one
  minute what the pixel-delta number had been arguing about for a whole
  session.

## 51. Two backends compiling the same WGSL are not running the same code, and only a per-backend roofline can see it

`backend-vulkan` and `backend-wgpu` both feed `crates/kernels`' WGSL through
naga. That made "the kernels are identical, so kernel efficiency must be
identical" feel like a safe assumption, and it was wrong by a factor of two:
the native Vulkan path used `naga::back::spv::Options::default()`
(`force_loop_bounding: true`, bounds policy `Restrict`) while the wgpu path
asks for `ShaderRuntimeChecks::unchecked()`, and `backend-cpu` asks Cranelift
for `MemFlags::trusted()`. Same source, three backends, two rule sets - and
the odd one out measured 5.05 TFLOP/s where its sibling measured 10.62 on the
same card.

Three things this cost, each generalisable:

1. **Nothing had ever rooflined that backend.** `gpu_core::roof` existed and
   worked; every caller just happened to run on the default. A backend with no
   measured roof has no measured anything - the deficit was invisible for as
   long as it went unprobed, and one `roof::measure` call per backend found it.
2. **The roof memo keyed on the adapter description alone**, so once both
   backends were probed in one process the first one's number was served as
   the second's. A roof is a property of the (device, BACKEND) pair. Any cache
   whose key omits a dimension the value actually depends on will eventually
   answer confidently for the wrong thing - and here the wrong thing was off
   by 2x, which is large enough to invert a design decision.
3. **A cross-backend switch has to be spelled once.** The fix reads the same
   `BRAIN_GPU_CHECKED` variable the wgpu backend reads, because a second,
   independently-named knob is how the two drifted apart in the first place.

Corollary for anything comparing backends: measure the roof on the backend you
are about to grade against it, never the roof the other backend left in the
cache; and when a "% of peak" moves after a backend swap, suspect the
compilation options before the silicon.

## 52. Vulkan validation layers are one `apt-get download` away, even with no root and none installed

Two sessions' worth of "GPU device lost" on `backend-vulkan` was diagnosed in a
single run, by the layer, naming the exact VUID
(`VUID-vkCmdDispatch-None-08114`, a descriptor set using a destroyed buffer).
The blocker had looked structural: `/usr/share/vulkan/explicit_layer.d` holds
only Intel and Mesa layers on this box, and there is no root to install more.
Neither fact matters.

```
apt-get download vulkan-validationlayers   # no root needed to DOWNLOAD
dpkg-deb -x vulkan-validationlayers_*.deb vvl
VK_ADD_LAYER_PATH=vvl/usr/share/vulkan/explicit_layer.d \
LD_LIBRARY_PATH=vvl/usr/lib/x86_64-linux-gnu \
VK_LOADER_LAYERS_ENABLE='*validation*'  <any binary>
```

`VK_LOADER_LAYERS_ENABLE` (loader 1.3.234+; this box has 1.4.304) force-enables
a layer from outside the process, so **no debug path has to be added to any
`VkInstance` creation site** to get a diagnosis - which matters when the
instance is created in a crate you were told not to modify, or in a dependency.

The general shape: before reading synchronisation code cold, check whether the
API you are debugging has a validation/sanitizer mode and whether it can be
switched on from the environment. "The layer is not installed" is a statement
about the current filesystem, not about what is reachable.

## 53. Cargo resolves the dependency graph once per invocation, so a `Cargo.toml` edited during a long build produces an "unresolved import" for a dependency that IS there

A full `make test/nextest` (a release build of every test binary, ~1 h on this
box) was launched, and ~46 minutes into it a new crate dependency was added:
`brain-capability` into `crates/glmdsa/Cargo.toml`, alongside a new
`caps.rs` that does `use capability::{...}`.

The run failed with:

```
error[E0432]: unresolved import `capability`
  --> crates/glmdsa/src/caps.rs:45:5
   = help: if you wanted to use a crate named `capability`,
           use `cargo add capability` to add it to your `Cargo.toml`
```

The dependency was in `Cargo.toml`. It was in `Cargo.lock`. Both were
committed. `make build`, `cargo test -p brain-glmdsa`, and a targeted
`cargo test -p brain-glmdsa --release --offline` all succeeded, before and
after.

**Cargo resolves the dependency graph once, at the start of an invocation, and
then compiles from the working tree as it stands when each crate's turn
arrives.** A build long enough to overlap an edit therefore compiles NEW source
against an OLD graph. Source edits alone are harmless-ish (you get a stale
result); a manifest edit is what produces this, because the graph is the one
thing the invocation will not re-read.

What makes it expensive is that the error message is actively misleading: it
names a dependency that is present and correct, and its `help` line tells you
to add something you already added. The instinct is to go looking for a
feature-gate, a `[lib] name` mismatch, or a workspace-inheritance problem -
none of which are wrong in any way you can find, because nothing is wrong.

Two rules:

1. **Do not edit any `Cargo.toml` while a long build or test run is in
   flight.** Source-only edits are recoverable by re-running; a manifest edit
   invalidates the run AND misdiagnoses itself.
2. **Before believing a dependency-resolution error from a long-running build,
   reproduce it in a fresh, short invocation** (`cargo test -p <crate>
   --release --offline --no-run` is enough). If the fresh run is green, the
   failure was the overlap, not the code - re-run the long build rather than
   "fixing" a manifest that was already right.

The same shape applies to any tool that snapshots configuration at startup and
reads inputs lazily afterwards; cargo is just the one that costs an hour.

## 54. A hand-maintained "Supported" checkbox is a duplicate of a fact the code already derives, and three of them were wrong in two directions

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

## 55. `push_step` past a reverse-mode tape is a silent ZERO gradient, and the SDXL UNet had 200 of them

`vae::blocks::Builder::push_step` appends a dispatch to the forward step list
and records **nothing** on the `Op` tape. Its own doc has always said what that
means: `grad::Trace::backward` walks the tape and skips any producer whose
output no consumer claimed, so a pushed step in the middle of a differentiated
chain breaks the chain and *every parameter upstream of it gets a zero gradient
with no error*.

`sdxlunet::model::Rec` emitted its entire transformer half that way - every
LayerNorm, every `nn.Linear`, the GEGLU pair, both attentions and the resnets'
timestep broadcast. Nothing was wrong with the forward, which is why 165/165
forward-parity comparisons passed and nobody noticed; the graph simply was not
differentiable, and the roadmap recorded that as "backward not done yet" rather
than as a hazard.

**The thing to carry**: a builder that offers both a recording API and an escape
hatch has a failure mode where the escape hatch is *quietly* not covered by the
gate. Grep for `push_step` (or its equivalent) before believing a model's
backward is merely missing rather than partially wrong - and prefer to give the
shared builder a real recorder for the stage instead of pushing past it, which
is what closing `check_unet` actually consisted of. No new kernel was needed:
`matmul_dx`/`matmul_dw`, `bias_grad`, `gelu_erf_bwd`, `layernorm_dx`/`_dgamma`/
`_dbeta`, `add_chan_bcast_dv`, `concat_split` and the `attn_bwd_*_cross` quartet
all already existed for the decoder LMs.

Two things train mode has to change in a forward that a reverse walk will read,
both of which run fine and are wrong:

* **Flash attention has no softmax to bind.** `flash_attn_bidir` never
  materialises `probs`, and the adjoint quartet reads exactly that. A recording
  builder must take the materialised path however cooperative the device is.
* **One softmax slab per attention SITE.** The eval graph reused a single
  `(scores, probs)` pair across all sites, which is correct when nothing reads
  them again. Under a reverse walk two sites then differentiate against each
  other's softmax - a plausible number, not a shape error.

## 56. `mse_value` writes one term PER ELEMENT for the host to sum, and a 1-element buffer makes the loss silently mean "element 0"

Closing `check_unet` produced a first run where 239 of 263 tensors failed with
`rel_err` clustered at 0.96-1.00 and `analytic / numeric` ratios scattered
between 1.5x and 190x. The scatter is what made it look like a backward defect:
a uniform factor would have said "normalisation", and the ratios growing with
distance from the output said "contributions being lost or doubled per layer".

Both readings were wrong. The **backward was fine**; the loss head was not.
`mse_value.wgsl` (like `ce_value`) writes `(pred[i] - tgt[i])^2 / n` into
`out[i]`, one invocation per element, and the CALLER sums - the per-element
division is what keeps the host reduction a plain sum. The trainer allocated
`gpu.storage(1)` and dispatched one thread, so `forward()` returned the first
element's term alone. Nothing errors: the buffer is big enough for what was
dispatched, and the value is a real, finite, plausible-looking float. The
finite-difference sweep then measured `d(element 0's term)/d(w)` against an
analytic `d(mean over all 256)/d(w)`, and each parameter influences element 0
by its own arbitrary amount - hence the scatter.

**The thing to carry**: when a finite-difference check fails *broadly* -
including on the tensors closest to the loss, where the chain is shortest -
suspect the objective before the adjoint. `conv_out.weight` failing at
`rel = 1.00` was the tell: one conv from the output, there is almost no chain
left to get wrong. Check that `loss()` computes what the analytic gradient is
the gradient OF, and specifically check the value kernel's own contract - both
`mse_value` and `ce_value` are per-element-plus-host-sum, not scalar-out.
`crates/vqgan/src/train.rs` had it right (`gpu.storage(te)`, `te` threads) and
was the working call site to copy, per the "copy a working dispatch" rule.
## 57. `0` is not a null pipeline index - an unused kernel slot must hold a sentinel, and only `backend-cpu` makes that a silent corruption

Every model in this tree fills an index struct (`model::block::KernelIds`,
`model::vit::VitKernelIds`, `vision::ConvKernelIds`, ...) from its own
`PIPELINES` list, and every such struct has slots the model never dispatches -
backward ids on a forward-only path, GQA ids on a model that rotates through
`rope2d`, RMSNorm ids on a model whose only `block` call is `swiglu_fwd`. The
tempting filler is `0`.

**`0` is a real, registered kernel in every pipeline list in this workspace.**
It was `conv1d` in one file and `matmul` in its sibling - both dispatched on
every single pass. A builder that reaches such a slot does not fail; it runs a
live kernel against a different kernel's bindings and uniform.

What makes it a *silent* defect rather than a loud one is the backend split:

* On a GPU backend the bind-group/binding-count check turns the misroute into a
  panic, so it surfaces the first time anyone takes that branch.
* On `backend-cpu` there is **no buffer-count and no uniform-size check** at
  dispatch. The kernel reads whatever is at those pointers. It is an
  out-of-bounds read that returns plausible numbers, and it is invisible to
  every unit test on that backend - which, for at least one crate here, is
  *every* unit test.

`model::vit::UNREGISTERED = usize::MAX` already existed and already documented
exactly this; `model::block` did not have it, and 13 sites across the workspace
had filled slots with `0` or - the same defect wearing a better disguise - with
a live index for a *different* kernel, commented as a harmless placeholder.
A sentinel that is out of range of `PIPELINES` converts all of it into an
index-out-of-bounds panic at the first dispatch.

**Gate it on DISPATCH, not statically.** "Every unused slot equals the
sentinel" is easy to write and cannot tell you whether getting it wrong would
have mattered. Arm the device's own per-kernel counters
(`Gpu::reset_ops_counters` / `Gpu::ops_counters`), run the real pass, and assert
that no unused slot's index names a kernel that pass actually dispatched. That
version names the kernel a wrong slot would have run - and it goes red on the
one-character mutation, which is the only evidence that a gate is a gate.

The general shape, beyond kernel indices: **a sentinel whose value is a legal
member of the domain is not a sentinel.** Index 0, empty string, epoch zero,
`Duration::ZERO` - each of them means something real somewhere. Pick a value the
consuming layer will reject, and check that the consuming layer really does
reject it.

## 58. A measured number in a doc comment outlives the tree it measured, and the next reader will read it as current

`ltxv::dit::ada_layer_norm_single` carried, in an ordinary `//` comment, "the
bulk of the ~76 s this stage cost per forward call (measured: the table GEMM
alone is ~14 s of it)". A later profiling pass measured the same stage, on the
same command, same shape, same box, at **10.2 s** - table GEMM 8.9 s, embedder
1.4 s. That reads as a 7.5x contradiction on the total and a ~45x one on the
embedder, and it cost a full round trip to resolve.

Both numbers were right. The comment described the tree *before* the two
changes made in the same commit that wrote it (batching the timestep embedder,
and `backend_cpu::host_gemm`'s blocked GEMM); the profile described the tree
after. The comment's own tense said "was", which is exactly as much hedging as
a reader gets - and it is not enough, because nothing beside it says *which*
version of the code the number belongs to.

Three things this actually shows:

* **A number in a comment has no version.** Code is diffed, comments are not
  re-derived. A figure that was a *before* number the day it was written
  becomes an *unmarked* number the day after, and the next profile is compared
  against it as though it were a claim about today.
* **The disagreement points the wrong way.** The reflex on a 7.5x gap is to
  suspect the new measurement - the instrumentation, the scope, the box. Here
  the instrumentation was correct, the scope was correct, and the box was
  quiet; the stale figure was the only wrong input, and it was the one nobody
  thought to date.
* **The workspace already has the right home for it.** Per AGENTS.md, a
  measured number about one model belongs in that model's
  `.agents/roadmap/<model>.md` (dated, phase by phase, with the command) or in
  the assertion of a test that will go red when it stops being true. Both of
  those places have a version. A comment does not.

So: **a comment may say what the shape is and why it costs what it costs; it
must not carry the number.** Where a figure is genuinely load-bearing for
understanding the code, name the harness that reproduces it
(`ltxv_bench streamed <layers> <t> <ctx> 1 1 <distinct_timesteps>`) instead of
the answer it printed once. And when you do have to leave a historical figure
in place, say which commit's behaviour it describes, in the same sentence.

**The nesting corollary, from the same investigation.**
`gpu_core::profile::stage_time` prints one line per call and has no idea that
one bracket encloses another. `forward_q_streamed` brackets the whole adaLN
stage AND its two halves, so the three lines are `parent = child + child`
(measured residual: 5 ms in 10,300). Summing the printed lines double-counts,
and the profile looks like the stage costs twice what it does. The fix is
free and belongs in the label: the enclosing bracket says it is a TOTAL and
names what it nests.
## 59. A feature can be complete, gated, benchmarked and unreachable, and only the RUN'S OWN BANNER says which

Two commits landed a day apart. One built a quantized, device-resident
audio-visual DiT block for `ltxv` - a new block type, a new cached-weight
type, a new residency session, real-weight parity gates, and a measured
several-times speedup at production width. The other made the audio latent
cross a long-form window seam. Both were correct, both were tested, and
`--audio` used neither: the pipeline's denoiser factory still constructed the
old host-fp32 object, one branch above where the new one would have gone.

Nothing in the workspace could see it.

* The **parity suite** was green, because it drives the new forward directly.
* The **benchmark** reported the new numbers, because it drives the new
  forward directly.
* **Nothing tested which arm the pipeline picks**, and there was no assertion
  anywhere that could have.
* The stale routing even carried a **log line asserting its own reason** ("the
  audio stream has no streamed/quantized path yet") that had been false since
  the previous commit, so reading the code near the defect actively confirmed
  it.

What found it was running the feature and reading the first line it printed:
the banner said `(host fp32, both streams)`.

Three things generalize, and the third is the fix:

* **A gate that drives a function directly cannot see whether anything calls
  it.** Coverage of an implementation is not coverage of its adoption. Whenever
  a change adds a faster/cheaper implementation ALONGSIDE an existing one, the
  question "which one does the product take?" is a separate claim needing a
  separate check.
* **File partitioning between concurrent authors creates exactly this seam.**
  Give one worker the pipeline and another the block, and the wiring belongs
  to neither. If work is partitioned that way, someone has to own the join,
  explicitly.
* **Make the banner DERIVED, not written.** The CLI used to spell the arm out
  itself from `(dit_config, audio)`; it now calls the same pure function the
  factory routes on (`ltxv::pipeline::av_arm_label`), so the line a reader
  checks and the object the run builds cannot disagree. A hand-written status
  line is a duplicate of a fact the code already decides, and it goes stale in
  exactly the direction that hides the defect (see also §54).

**Corollary, from the same run: `secs_per_step` in this pipeline's traces is a
RUNNING MEAN, not a per-step cost.** It is `elapsed / steps_done`, so the
first value of a stage carries the whole cold start and the last value times
the step count is the stage's span. Quoting the first value as "the step time"
overstates a warm step by the entire warm-up, and summing the printed values
overstates a stage by several times. A field whose name reads as an
instantaneous quantity and whose value is cumulative will be misread; either
the name or the doc has to say so before anyone quotes it.

## 60. A timing span that ends at a scope's closing brace cannot see the DESTRUCTION its own scope causes, and that is where half the cost was

The LTX-2.5 block forward was instrumented in four buckets - host modulation,
graph record + upload, submit + wait, readback - and each was reported on its
own. Nothing ever compared their SUM against the wall clock of the thing that
contains them, so nobody could see that they did not add up. They did not:
every timing span closed before the function returned, and the last thing
the function does is drop ~70 device buffers, which on a discrete card is ~70
driver frees. That was 10.5 s of a 30 s host half, in a forward whose recorded
"next thing to fix" was the ALLOCATION of the same buffers - measured at
roughly the same size, and treated as if it were the whole cost.

The general shape:

* **Instrument the CALLER of the scope, not just spans inside it.** A bracket
  around `f()` catches what `f`'s own internal spans structurally cannot: its
  drops, its returns, its allocator work. Here the fix was one probe around the
  whole function and one around the loop that calls it, and the residual it
  exposed - 1.9 s per 4 blocks that belonged to no bucket - was the entire
  finding.
* **A stage table that "roughly adds up" is not attribution.** Sum the spans
  and subtract from the enclosing wall clock EVERY time, print the residual as
  its own row, and treat a nonzero residual as a measurement to be made rather
  than as noise. `.agents/roadmap/ltxv.md` phase 33's table has an explicit
  `unaccounted` row for this reason; it reads 0.00 s because the probes were
  added until it did.
* **`free` is a cost, not a cleanup.** Rust makes deallocation invisible at the
  call site, so a profile built from hand-placed spans will systematically
  under-count it. This is the second time device-buffer lifetime has cost this
  workspace real wall clock (the first was staging buffers on non-ReBAR cards,
  `crates/gpu-core/tests/vram_overhead.rs`).

The corroborating check is cheap and was decisive: an external sampling
profiler (`perf record --call-graph dwarf`, `--delay` past the cold phase)
showed `gpu_allocator::vulkan::Allocator::free` and `MemoryBlock::new` at
comparable shares, and both left the profile entirely once the buffers were
pooled. When an in-process attribution says something surprising, one
`perf` run either confirms it from outside or kills it.

## 61. A device-to-host readback can be dominated by the DESTINATION allocation, and a correctness gate cannot see a performance mechanism at all

Two separate findings from one change; they arrived together and each is the
kind that gets re-discovered.

**The readback is not necessarily bus-bound.** `Gpu::read` on a P40 measured
0.82 GB/s for a 206 MiB activation against 5.87 GB/s for the same payload in
the other direction, which reads as a PCIe asymmetry and is not one. Split
into phases, the bus transfer (`copy_buffer_to_buffer` + submit + poll) was
35 ms and the host `get_mapped_range() -> to_vec()` was 150 ms - the *same*
copy into a pre-faulted sink took 22 ms. The cost was the destination `Vec`'s
first-touch page faults: brand-new anonymous pages, zeroed and faulted as the
memcpy writes them. Reading the mapped range was never slow.

The general shape, and why it matters beyond one backend:

* **A transfer path has at least three phases and only one of them is the
  bus.** Allocate/pin, transfer, and copy-out. Quoting an end-to-end GB/s
  attributes all three to whichever one the reader assumes. Split it before
  building anything on the number - the split here was four `Instant`s behind
  a temporary env var and took minutes.
* **Allocation is the recurring culprit in this workspace, in BOTH
  directions.** Upload staging on non-ReBAR cards was the same class
  (`crates/gpu-core/tests/vram_overhead.rs`); this is the read direction, and
  the destination half of it is still open because `read`'s contract is to
  return a fresh `Vec`.
* **A rate that is wildly asymmetric between two directions of the same link
  is a claim about software, not about hardware.** PCIe is symmetric to within
  a few percent.

**And a correctness gate cannot see a performance mechanism.** Reusing the
staging buffer instead of allocating one per call is a pure performance
change, so every correctness test - the ones that pin "a read returns its own
bytes and not the previous read's tail" - passes just as well for a `read`
that allocates every time. Mutating `retain_read_staging` into a no-op, i.e.
silently reverting the whole mechanism, left all four correctness tests
green. The mechanism needed its own OBSERVABLE
(`WgpuBackend::read_staging_allocations`, asserted as "eight reads of one
shape allocate exactly one buffer") before a gate could exist at all.

So: when a change is "the same answer, computed with less work", ask what in
the process state proves the work was skipped, and expose that. A wall clock
is not a gate - nobody watches it, and it is the first thing a noisy machine
takes away. `crates/backend-wgpu/tests/read_staging_reuse.rs`, and the numbers
in `.agents/roadmap/ltxv.md` phase 36.

## 62. A mapped buffer's pointer is only as aligned as the allocator happened to place it, and `cast_slice` on one holds by luck

`WgpuBackend`'s per-kernel timestamp readback did
`bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range())`, and the readback
path did the same to `f32`. Both had worked for the life of the file. Then an
unrelated change - keeping ONE staging buffer alive between reads instead of
allocating a new one per read - moved `gpu-allocator`'s packing, the
timestamp-resolve buffer landed at a suballocation offset that was 4-mod-8,
and a real 22B run died mid-forward with
`cast_slice>TargetAlignmentGreaterAndInputNotAligned` after every stage of the
first forward had already printed.

The general shape:

* **A mapped pointer's alignment is an allocator artifact.** It is
  `memory_block.mapped_ptr + suballocation.offset`, and the offset is chosen
  from the buffer's own `memoryRequirements.alignment` (4 for a plain
  host-visible transfer buffer here) and from whatever else happens to be live.
  Nothing promises it is aligned for the type you want to read.
* **Widen the DESTINATION instead of casting the source.** `&mut [T]` viewed as
  bytes (`cast_slice_mut::<T, u8>`) is always byte-aligned, so
  `copy_from_slice` from an arbitrarily aligned `&[u8]` is well defined, the
  `memcpy` is the same one `to_vec` was doing, and the precondition disappears
  rather than being documented. One helper
  (`backend_wgpu::copy_pod_from_bytes`) now covers all three sites.
* **The bug class is testable even though the trigger is not.** No test can
  force the allocator to hand back a 4-mod-8 offset, and a gate that has never
  been seen fail is a hypothesis. What IS deterministic is the property: copy
  the same payload out of a byte slice at every source offset in `0..8`. That
  gate is red under `cast_slice` with the exact panic the real run produced,
  and it needs no GPU.
* **Watch for this whenever an allocation-lifetime change lands.** Pooling,
  recycling and arenas all move packing, and anything downstream that assumed
  an incidental alignment fails far away from the change - here, in a profiler,
  in a different subsystem, several minutes into a run.

A second, unrelated hazard was confirmed while chasing this and is worth
recording separately: `crates/backend-wgpu/tests/upload_flush.rs` **hangs at
100% CPU** when its two tests run concurrently (`--test-threads` > 1), because
each builds its own `WgpuBackend` and two Vulkan devices in one process is the
documented deadlock this driver has. Reproduced on unmodified `main`, so it is
pre-existing, and the default `TEST_THREADS=8` reaches it.

## 63. A shared "diffusers VAE" builder is diffusers-NAMED, not diffusers-SHAPED - reusing it for a CompVis-named checkpoint of the identical topology still needs a rename pass

`vae::VaeEncoder`/`VaeDecoder` (`crates/vae/src/decoder.rs`) look like the
generic autoencoder graph builder every latent model in the tree should be
able to point at any byte-identical-topology checkpoint through - and
`vae::blocks::Builder` genuinely is that generic, parameterized by
`BlockNames` for exactly this reason (`diffusers()`, `vqgan()`, `diamond()`
already cover three unrelated naming conventions over the SAME block shapes).
But `VaeEncoder::build`/`VaeDecoder::build` themselves do not expose that
parameter: their function BODIES hard-code the diffusers path strings
(`"encoder.conv_in"`, `"encoder.down_blocks.{i}.resnets.{j}"`,
`"encoder.mid_block.attentions.0"`, ...) even though they call into the
generic `Builder` underneath. So a checkpoint whose topology is byte-for-byte
the same VAE encoder but whose KEYS are CompVis-named
(`down.{i}.block.{j}`, `mid.attn_1.{q,k,v,proj_out}`, `nin_shortcut`) - SUPIR's
`denoise_encoder`, byte-identical to the frozen SDXL VAE encoder, only the
weights differ - cannot be handed to `VaeEncoder::from_diffusers` as-is, even
though "same topology, different weights" is exactly the situation that type
exists to cover.

The fix is a plain host-side key rename (`crate::import::denoise_encoder_diffusers_names`
in `crates/supir`): walk the CompVis leaf names into their diffusers
equivalents (`nin_shortcut` -> `conv_shortcut`, `mid.block_1`/`mid.block_2` ->
`mid_block.resnets.0`/`.1`, `mid.attn_1.{q,k,v,proj_out}` ->
`mid_block.attentions.0.{to_q,to_k,to_v,to_out.0}`), plus ONE shape metadata
change: CompVis stores attention q/k/v/proj as `[C,C,1,1]` 1x1-conv weights,
diffusers stores the identical row-major bytes as a `[C,C]` linear - a
`Vec<usize>` edit, not a data transform, easy to miss as "just a reshape,
nothing to verify" and correspondingly easy to get backwards (transposed
instead of flattened) without a test pinning the output rank.

The general shape: before reusing a "generic, parameterized" builder for a
new checkpoint naming convention, check whether the parameterization the type
ADVERTISES (here, `BlockNames`) actually reaches the call site you need, or
whether a higher-level wrapper over that builder hard-coded one specific
naming convention's path strings when it was written for its first (and so
far only) caller. The generic layer being genuinely generic does not mean
every layer built on top of it inherited that property.

## 64. A GGUF's `general.architecture` is UNTRUSTED input to a family-name match - dispatch it in the same `match` as native checkpoints and any collision routes a file to a loader that cannot read it

`crates/cli/src/model_dir.rs` synthesizes a `ModelCard` for a `.gguf` with
`family = general.architecture` **verbatim** - a string the file itself
supplies - and then fed it to the SAME `resident_for` match that dispatches
brain-native checkpoint families. Two namespaces, one match: the moment an
architecture string spells a family name, the raw GGUF is handed to that
family's resident. `qwen35`/`qwen35moe` are the live examples - their serving
`Engine` reads `checkpoint::load`, which is safetensors-only, so a raw
`qwen35moe.gguf` (a real file, produced by llama.cpp, with a registered
importer sitting right there) was routed into a loader that cannot open GGUF
bytes at all and died with a low-level parse error, instead of the one-line
`run \`brain import-gguf ...\`` advice the fallback arm already knew how to
print.

The fallback arm DID have the GGUF advice, and two arms above it had comments
about a previous collision between the two qwen35 crates. That is the tell: a
defect being handled per-arm, with each new arm expected to remember, is a
defect in the dispatch shape. The fix is a gate BEFORE the match - a `.gguf`
reaches a family arm only if that family opted in (`family_reads_gguf`:
`gpt`/`glm`/`qwen`, which load through `checkpoint::weightio::WeightReader`
and so read either container) - and it is an ALLOWLIST, not a list of
exclusions, so the default for the next family somebody adds, and for the next
architecture string llama.cpp invents, is the helpful advice rather than a
mis-route.

The general shape: when one `match` keys on a string that arrives from two
sources - one you control (what your own writer stamps) and one you do not
(what a third-party file declares) - the collision is not hypothetical, it is
scheduled. Separate the two before the match, not inside it.

## 65. A weight scale that spans the whole reduction axis is one outlier away from destroying the row - group it at the block size of the format you interoperate with

`model::int8::quantize_weight` and `model::int4::quantize_weight_q4` computed
ONE scale per output row, over a `k` that reaches 17408 in the models this tier
serves. `AGENTS.md` had already recorded the measurement that condemns that
shape - whole-channel INT8 on the Qwen3 KV-cache decode graph at **cosine
0.994 / max_abs 11.2** against fp32's **1.000000 / 0.0002** - and the NPU-side
quantizer (`npu::topo::linear_quant`, `QUANT_GROUP = 32`) had been fixed for
it. The GPU-serving path had not; the invariant's own text ended "if/when it is
audited for the same granularity", which is what a known defect looks like
while it is still shipping.

Measured on the fixture the fix added (`int8::tests::group_wise_confines_a_row_
outlier_that_whole_channel_spreads_over_the_row` - one 100.0 outlier in a row
of `|w| <= 0.31`, `k = 512`): outside the outlier's own 32-element block, worst
absolute error is **0.001189** group-wise against **0.312000** whole-channel
(**262x**), and the count of elements quantized to worse than 0.1 is **39**
against **687** of 1022. The group-wise figure is bounded by the BLOCK, not by
`k` - that boundedness is the property, and it is what the test asserts rather
than the ratio alone.

Four things the fix turned out to depend on, none obvious from the quantizer:

* **32 is not a tuning parameter.** It is `Q8_0`/`Q4_0`'s block size, so one
  brain group IS one Q8_0 block and the requantization becomes the IDENTITY:
  Q8_0 stores `d = max|x|/127` with `max|q| = 127`, so the group's absmax is
  `127*d`, brain's scale comes out as exactly `d`, and every `q` re-quantizes
  to itself. Pick 16 or 64 and that door closes.
* **The GEMM's k-chunk decides what the group costs.** `matmul_i8_dyn`'s `BKG`
  is 8 packed words = 32 int8 = exactly one group, so the integer accumulators
  still sum a whole chunk exactly and fold once per chunk (64 FMAs per 512
  DP4A, at the price of doubling the accumulator register block). The two GEMV
  kernels stride 64 WORDS, which spans eight groups, so they have no same-group
  run to accumulate over and convert per word instead. Same layout, two
  different right answers - written into each kernel's header, because the
  wrong one costs 3x the inner loop or 2x the registers.
* **The stale scale shape is `[n]`, and it is a valid length.** An old
  checkpoint's whole-channel scale would be read as the first row's worth of
  group scales and produce plausible garbage. `model::int8::check_scale_len`
  recognises exactly that length and names the format change; the on-disk
  convention is versioned (`checkpoint::weightio::PACKED_INT8_LAYOUT`). A
  format change whose old form still parses needs a check that fires on the
  OLD shape by name, not merely a new writer.
* **It is not free, and the storage gate had to be re-derived, not widened.**
  One f32 per 32 weights is 0.125 B/param on top of the packed byte, so the
  int8 storage ceiling is `4 / 1.125 = 3.5556x`, not 4x. The real 22B LTX-2.5
  AV DiT measures **3.3640x** where it used to measure ~3.6x
  (`ltxv/tests/device_bytes_real.rs`). The right response to that gate going
  red was to compute the new ceiling from the layout and assert against it,
  not to loosen the band until the number fit.

The general shape: an invariant documented as "applies equally to X, if/when X
is audited" is not an invariant, it is a TODO with a citation. Either the audit
lands in the same change as the rule, or the rule carries a named exception
with the measurement it is exempting itself from.

## 66. An overfit gate that reads ONE step of a trajectory is only a gate if the trajectory has converged - above the optimizer's stability threshold it reads the limit cycle's phase

`crates/supir/src/finetune.rs::full_backbone_overfits_a_single_sample`
asserted `last < l0 * 0.35` after 120 Adam steps at `lr = 0.03` and failed
deterministically at `7.06e-1 -> 4.68e-1`. Its sibling
`adaptor_only_overfits_a_single_sample`, same helper, same lr, same step
count, only a different freeze predicate, passed. That asymmetry reads like a
backward bug in the parameters only full-backbone trains, and it is not one.

**What the trajectory actually did.** The gate printed every twentieth step,
which showed a clean monotone descent (`3.70e-1, 3.12e-1, 2.80e-1, 2.52e-1,
2.38e-1` at steps 20..100) and hid everything. Printing every step:

| step | 103 | 104 | 105 | 115 | 116 | 117 | 119 | 121 |
|---|---|---|---|---|---|---|---|---|
| loss | 2.29e-1 | 2.40e-1 | 3.88e-1 | 2.90e-1 | **9.85e-1** | 4.83e-1 | 4.68e-1 | 5.76e-1 |

The run is a limit cycle: a slow descent punctuated by excursions that reach
**1.40x `l0`** - above where training started - and step 119 is simply one of
them. Steps 1..5 already read `2.73e0, 1.24e0, 5.64e-1, 2.78e0` against an
`l0` of `7.06e-1`, so it was never in a stable regime at all. The passing
sibling is in the same cycle with smaller excursions (peak 0.44-0.54x `l0`)
and happened to be sampled on a descending step.

**Two things this cost, both worth stating separately.**

* **The backward was ruled out the expensive way, and the ruling-out is a
  reusable gap.** `gradcheck::check_supir` deliberately checks only
  `control_model.*` and `project_modules.*`, on the stated grounds that
  `check_unet` already covers the backbone's own adjoints. True - but it
  means the exact parameter set that distinguishes the two modes is the one
  set no SUPIR gate touches, so "is full-backbone's gradient wrong?" had no
  cheap answer. A directional FD sweep over all **263** unprefixed tensors
  found nothing above the fp32 FD noise floor (worst `rel = 2.36e-1`, on
  `up_blocks.0.resnets.0.conv1.bias` at `|g| = 4.5e-3` where a central
  difference at `eps = 2.5e-4` against a loss of 0.7 is pure rounding), and
  the worst offenders were `up_blocks.*` tensors the PASSING mode trains too.
* **`lr` is not "how fast", it is "how big a step relative to the weight".**
  `model::lora::adam`'s update is scale-free: each entry moves by about `lr`
  per step whatever its gradient's magnitude. `UNetConfig::tiny`'s convs
  initialise at `1/sqrt(fan_in)` ~ 0.06, so `lr = 0.03` moves every weight by
  half its own distribution's width per step. The stability threshold that
  crosses is a property of WHICH parameters are unfrozen, not how many:
  unfreezing the backbone encoder moves the input every later stage is
  conditioned on, so full-backbone leaves the stable regime first *because*
  it has more capacity, not despite it. "Strictly more trainable capacity
  must overfit at least as well" is true at the optimum and says nothing
  about a fixed-`lr` path to it.

**The fix made the gate stronger, not looser.** Re-measured on this box (Tesla
P40, Vulkan, `SupirConfig::tiny`, `H=W=8`, 120 steps, both seeds, both modes):

| lr | adaptor 101 | adaptor 202 | full 101 | full 202 |
|---|---|---|---|---|
| 0.03 | 0.253 | 0.312 | 0.242 | **0.664** (peak 1.395) |
| 0.01 | 4.5e-5 | 4.8e-7 | 5.0e-6 | 9.8e-7 |
| 0.005 | 1.9e-7 | 2.0e-7 | 2.3e-7 | 2.8e-7 |

(each cell `last / l0`.) At `lr = 0.005` every combination reaches a literal
near-zero floor with no post-warm-up step above `0.008 * l0`, so the gate
moved from `last < 0.35 * l0` to `last < 1e-4 * l0` - three and a half orders
of magnitude tighter, on a trajectory that no longer has a phase to sample.
The old comment's framing ("this gate asserts clear, substantial descent
rather than a literal near-zero floor") was describing the divergence, not the
model.

And because "converged" is the property the old gate could not see, the new
one now asserts it directly and cheaply: `overfit_single` returns the WORST
loss after a 20-step Adam warm-up alongside the last one, and every caller
asserts that it stayed below `l0`. That is the check that would have caught
this on the step it happened rather than by luck 14 steps later, it costs one
`max` per step, and it is what makes the endpoint threshold meaningful instead
of a sample. Measured margin on the shipping configuration: worst post-warm-up
loss is `4.6e-3` / `5.4e-3` / `6.3e-2` against an `l0` of `7.4e-1` / `7.1e-1`
/ `7.9e-1`.

**And the calibration named a box that no longer exists.** The comment said it
was measured "on this machine's real Vulkan iGPU backend"; this machine is now
a discrete P40. A single-point assertion on a non-converged trajectory cannot
survive that, because the excursions move - which is lesson #58's "a number in
a comment has no version" with teeth: the number was not just stale, it was
*only ever true for one phase of one run*.

The general shape: before calibrating a threshold against a training run, look
at the run at full resolution and confirm it CONVERGED. A descending
every-20th-step summary is compatible with divergence. If the trajectory
oscillates, the honest options are to fix the hyperparameter until it does not
(preferred - it usually tightens the gate by orders of magnitude, as here) or
to assert on a statistic the phase cannot move (`min` and a hold bound, the
shape `crates/flux2/tests/lora_train.rs` and `supir::lora` already use, and
which each says in-line WHY it is a floor-and-hold rather than an endpoint).
Never re-fit the threshold to the endpoint the unstable run happened to land
on.

