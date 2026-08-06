# Lessons — defects this repo has actually paid for

Cross-cutting engineering findings, kept in one place because each was expensive
to learn and none belongs to a single model. **Add to this file whenever a
non-obvious defect is found** (AGENTS.md makes that a rule), and prefer a
one-line entry with the number that proved it over a paragraph of advice.

Model-specific ledgers stay in `docs/models/<model>/status.md`; kernel-authoring
rules stay in `docs/kernel-checklist.md`; porting order stays in
`docs/porting-playbook.md`. This file is for the things that generalise.

---

## 1. A gate that never runs is worse than no gate

It reports green and nobody looks again. Four separate instances, all found in
one workstream:

| Gate | Why it never ran |
|---|---|
| `sam2` parity | `scripts/fetch-testdata.sh` had no sam2 entry; the checkpoints had been hard-linked in by hand, so the test self-skipped on every machine but the one that created them |
| `flux2` host/device parity | `hidden: 16` put a modulation slice at byte offset 192, violating the 256-byte storage-binding alignment — it failed on every GPU, and had simply stopped being run |
| `cargo clippy` | aborts at the first deny-by-default lint and then reports nothing about everything after it — a 123-file backlog stayed invisible **twice in one day** |
| `wgsl-cpu` `compile_all` | `max_abs_final` was the one kernel of 346 that would not compile on the CPU JIT, so int8 quantization had no working CPU path at all |

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
cannot see which term `beta` multiplies; that is pinned by reading
`vqgan_arch.py:55`, where `beta` sits on the **codebook** term — not on the
commitment term that file's own line-29 comment claims. Finite differences prove
the derivative, never the objective.

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
physical limit means you measured the CPU. See `docs/kernel-checklist.md` §E.0.

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
divides only the CLIP half. brain's `facenet` `embed` action normalises, because
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

SDXL is ~3.5 B params across four models (~14 GB at fp32), and a non-ReBAR
Pascal card carries roughly 2x resident overhead per storage buffer. Holding the
UNet, both text encoders and the VAE at once does not fit 24 GB.

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
43 GFLOP/s — 0.4% of a P40's peak, and 49% of the entire UNet forward**. Measured
crossover (k=2048, n=2560): naive wins to m=4, tiled from m=8, and by m=77 tiled
is **22x** (18.67 ms -> 0.84 ms), bit-identical.

Lowering the guard to `m < 8` took the whole UNet forward from **2609 ms to
1349 ms (1.93x)** and the end-to-end 24-step generation from 174 s to 105.7 s,
with the output **bit-identical** at the same seed.

Two general points:

* This is the third instance of `docs/lessons.md` #8 in one workstream — a fast
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
  `AutoencoderKL` and skips unless `BRAIN_ZIMAGE_VAE` is set, so on this machine
  nothing gated a VAE decode at all (`docs/lessons.md` #1, again).
* The UNet's own 165-tap parity was green — and stayed green. It ends at the
  latent; the defect is downstream of everything it checks.
* The sanity check I ran on the output was **mean gradient magnitude**, which
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

## 22. Disk shape on the dev box

`cargo build` is ~3.7 GB of `target/`; adding `--tests --examples` across the
workspace is **~29 GB**. That 8× jump filled the overlay to 0 bytes and
hard-blocked every tool — the Bash harness writes each command's output to the
same filesystem, so it fails at `open()` before the command runs, with no
recovery from inside the session. Check `df -h /` before a wide build.

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
(`crates/qwen/src/config.rs`), but the gate that would have caught it needed
three things together, none optional: (1) take a few real optimizer steps so
`B ≠ 0` (a fresh LoRA init is `B = 0`, making the delta zero regardless of
whether it loads — see #4 on degenerate test setups hiding a whole bug
class), (2) **save, then load in a fresh process/struct** rather than
comparing against the live trained model still sitting in memory, (3) assert
the reloaded logits *differ from the base* by a real margin, not merely
"didn't get worse". `crates/qwen/tests/lora_roundtrip.rs` does all three;
`crates/qwen/tests/lora_learning_gate.rs` (Gate A, `docs/guides/training.md`)
goes one further and reintroduces this exact defect on purpose as its own
verification that the gate has teeth, rather than trusting the story that it
would have caught it.

## 24. A benchmark that measures a path no request takes is a healthy number about the wrong thing

Every serving benchmark before `perf::targets::HttpTarget` drove
`qwen::serve::Scheduler`/`residency::Executor` directly — real kernels, real
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

`Qwen::prefill`'s fix (this same workstream, earlier) replaced a per-token
`decode_submit` loop with one call to the batched primitive — but the FIRST
attempt at "fixing" prefill performance in this codebase's history batched
only the readback (one `map`/fence at the end) while still issuing one
GPU submit per token underneath. It measured faster than the naive
`step()`-per-token loop (fewer host↔device round trips), so it read as
progress, but it was still `O(T)` submits for a `T`-token prompt — the same
defect class in a lighter disguise. `.todo/serving-performance-audit.md`'s
own audit trail and this workstream's `prefill_submits_scale_with_chunks_
not_with_token_count` gate exist because "faster than before" and "actually
O(1) [per chunk]" are different claims, and a wall-clock-only benchmark
cannot tell them apart — only a device-op COUNT (`gpu_core::DeviceStats.
submits`) can, because it is insensitive to how fast any individual submit
happens to run on this box today.
## 26. A barrier kernel on `backend-cpu` corrupts memory; it does not refuse

`DeviceCaps::workgroup_reductions` is false on the CPU JIT because it cannot
compile `workgroupBarrier`. What it does with one is worse than an error:
`crates/gradcheck/tests/layernorm2d_kernels.rs` recorded `layernorm_rows`
(2 barriers) there and the process died with

    munmap_chunk(): invalid pointer
    signal: 6, SIGABRT

no test name, no kernel name, no backtrace into the offending dispatch. The
whole-suite runner only said `FAIL: gradcheck suite — CPU backend`; finding
which of 40 test binaries aborted took longer than the fix.

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
`PEAK_TFLOPS = 11.76` in `vqgan_bench`, `unet_bench` and `zimage_bench`,
`PEAK_GBPS = 346.0` in four microbenches, `PEAK_FP32 = 11_760.0` in two more.
Those are one card's spec-sheet numbers. Three separate problems follow, and the
third is the one that matters:

1. **On any other device every number is wrong**, silently and by an unbounded
   factor. `DeviceCaps::peak_bandwidth_gbs` existed for exactly this and was
   `None` on all three backends, with nothing anywhere filling it.
2. **Spec-sheet peak is not achievable peak.** Measured on this box with a
   dependency-free FMA probe and a STREAM-triad probe: **10 555 GFLOP/s (89.8%
   of 11.76 TFLOP/s) and 287.6 GB/s (83.1% of 346)**. Grading kernels against a
   roof nothing can reach builds a permanent 10-17% pessimism into every row.
3. **The whole method depends on the denominator.** `docs/kernel-checklist.md`
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
disagree by more than 25%, which broke the reproducibility test at
`--test-threads=8` and is not a bug in the probe.

## 28. A partial FLOP numerator over a full denominator under-reports in silence

`vqgan_bench`'s `WHOLE PASS` row summed FLOPs across every row regardless of
whether `gpu_core::cost` had a formula for it, then divided by the whole-pass
time. **Ten of the VQGAN backward's 26 kernel kinds had no formula**, so the
published "backward = 5.4% of peak" was computed from a numerator missing a
third of the graph.

This is the mirror image of the failure `cost` was designed to prevent. An
uncovered kernel already reported `-` per row (never a zero that reads as slow),
but the pass-level total had no such guard — and a *pass* rate is the number a
reader quotes. The fix is that a partly covered pass reports **no rate at all**
and names the kinds it is missing:

    WHOLE PASS   457.42 ms   1404   (rate unavailable — no cost formula for: mse_grad, masked_l1_grad)

which turned an invisible accounting hole into a two-line work item. With all 26
covered, the honest numbers are **forward 356.8 GFLOP/s (3.4%) and backward
638.1 GFLOP/s (6.0%) of the measured roof.**

Two structural notes:

* The F0 coverage test was a hand-maintained *list* of kernel names, so it could
  only fail when someone remembered to extend it — it could not stop a new
  kernel landing unmeasurable. It is now backed by a **ratchet** over the whole
  of `kernels::ALL` (150/357 today) that fails when coverage falls.
* Name the uncovered kinds in the profile output, never just count them. They
  are usually cheap enough to fall outside the printed top rows, which is
  exactly where a missing formula hides.

## 29. `make kernels-regen` had been broken since the script moved

`scripts/build/kernels-regen.sh` computed its repo root as
`dirname($BASH_SOURCE)/..`, which was right when it lived in `scripts/` and
became `scripts/` itself once it moved into `scripts/build/`. Every invocation
died with `missing .../scripts/crates/kernels/src/lib.rs`.

Nobody noticed because the failure mode is *silent at the level that matters*:
you add a `.wgsl`, the regen fails, and you hand-append the two lines to
`crates/kernels/src/lib.rs` instead. The registry stays correct, so no test ever
fails — it just stops being mechanically derivable. Re-running the fixed script
produced a 23-line diff over kernels added since the breakage: hand-written doc
comments replaced by the generated form, and several consts out of sort order.

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
   `CpuShared`, which compiles, passes serially, and fails under
   `--test-threads=8`: every test in the binary shares one pooled device, so a
   neighbour's submits land inside your delta. Per-handle counters make
   concurrent measurement correct by construction; a mutex around the
   measurement only serialises the test against *itself* and does not help.
   Note the failure mode of getting this wrong is a **flaky** test, which is
   worse than a wrong one.
3. **A default trait method is a silent opt-out.** `fn stats(&self) -> Option<_>
   { None }` reads as a courtesy to backends that cannot count; in practice it
   is how two backends went years without counting while the API looked
   complete. `crates/gpu-core/tests/device_stats.rs` now asserts the contract on
   whatever backend `BRAIN_DEVICE` selects, so the default cannot creep back.

## 31. Timing a kernel by host wall-clock around a drained slice is not a measurement

The per-kernel-kind profiler (`gpu_core::profile`, and the four benches that
preceded it) times a group by submitting that slice alone and bracketing it with
`poll_wait`. That is honest about *drain* — it prints the group-sum vs
whole-pass inflation, lesson #21 — but it is not a measurement of the kernel. It
measures **launch + execute + fence**, and the floor of that is roughly constant,
so the error is inversely proportional to how small the kernel is.

Measured both ways on the same `qwen_bench serve 128` run, host-timed slices
against the backend's **GPU timestamp queries** (`BRAIN_PROFILE=1`, which brackets
each dispatch with `beginning_of_pass_write_index`/`end_of_pass_write_index`):

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
  not production times either (that run's GPU total was 330.7 ms against a
  114.9 ms whole pass).

The fix is timestamp queries **inside the production single-pass flush** — one
query pair per dispatch, resolved once at the end — so per-kernel attribution and
the whole-pass number come from the same execution. Until that lands, treat every
per-kernel *share* in `docs/performance/` recorded before this entry as
suspect for small kernels, and re-derive any ranking from `BRAIN_PROFILE`'s
distribution. Whole-pass numbers, and every speedup in this document that was
judged by one, are unaffected.


## 32. A profiler that drives the graph wrong measures a graph that does nothing

With device timing correct, `paged_decode_scores_batched` still reported
**7060 GB/s**. The timing was right and the kernel really was that fast — because
it was attending to nothing.

`qwen_bench serve` drove the served tape with `Input::Resident`. That is the
on-device decode-window mode (A4): it deliberately performs **no host writes**,
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
3. **Sanity-check the shape of the answer.** 1650 rows/s for a 0.6B model on a
   P40 should have prompted "against what ceiling?" — the weight-bandwidth
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
`compile_all` passes 358/358 and `make gradcheck` is green on `BRAIN_DEVICE=cpu`,
so the CPU claims now hold.

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
