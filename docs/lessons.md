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

## 12. Disk shape on the dev box

`cargo build` is ~3.7 GB of `target/`; adding `--tests --examples` across the
workspace is **~29 GB**. That 8× jump filled the overlay to 0 bytes and
hard-blocked every tool — the Bash harness writes each command's output to the
same filesystem, so it fails at `open()` before the command runs, with no
recovery from inside the session. Check `df -h /` before a wide build.
