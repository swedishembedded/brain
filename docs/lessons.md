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

## 21. Disk shape on the dev box

`cargo build` is ~3.7 GB of `target/`; adding `--tests --examples` across the
workspace is **~29 GB**. That 8× jump filled the overlay to 0 bytes and
hard-blocked every tool — the Bash harness writes each command's output to the
same filesystem, so it fails at `open()` before the command runs, with no
recovery from inside the session. Check `df -h /` before a wide build.
