# completion-plan - finishing the incomplete work

A cross-cutting plan, as opposed to the per-model ledgers next to it. It exists
because the per-model ledgers cannot be read as a work list on their own: five
of them assert gaps that are demonstrably closed (§0), so any prioritisation
built directly on them is prioritising fiction.

Scope: every `- [ ]` in `.agents/roadmap/*`, plus the gaps found by checking the
tree against the invariants in `AGENTS.md` rather than against the ledgers.

---

## How this plan was built

Not by reading the ledgers. Each claim below was checked against the tree:

| check | command |
|---|---|
| capability surface | `ls crates/*/src/caps.rs` |
| residency adapter | `ls crates/cli/src/resident_*.rs`, `resident.rs::build_executor` |
| discovery | `crates/cli/src/catalog.rs::models()` |
| backward gate | `grep 'pub fn check_' crates/gradcheck/src/` + its `tests/` wiring |
| examples | `examples/*/` |
| build health | `make build` |

**Build baseline, measured now:** `make build` is green, **0 warnings**,
24 s incremental. The zero-warning invariant currently holds; this plan must
keep it holding.

---

## The ordering principle

The repo already defines what "done" means. Finish against those definitions in
dependency order - each phase is a prerequisite for trusting the next.

1. **Truth** - the ledger must match the tree, or every estimate is wrong.
2. **Gate** - `make test` must be green in one attempt. `lessons.md` §1: a gate
   that never runs is worse than no gate.
3. **Contract** - a model that cannot be discovered, scheduled, batched and
   driven over D-Bus is incomplete, per the AGENTS.md invariant.
4. **Correctness** - full backward + a `gradcheck` entry point is the stated
   default expectation, not an opt-in.
5. **Capability** - genuinely missing features.
6. **Performance** - last, and only against a published baseline.

---

## Phase 0 - Reconcile the ledger with reality

**DONE.** All four sub-items below are closed; §0.1-0.3 in commit "docs: the
ledgers stop claiming work that is already done", §0.4 measured below. The
descriptions are kept because they record *what was wrong*, which is the part
worth not relearning.

**Blocking. Nothing else in this plan can be scheduled honestly until it's done.
Size: S (hours).**

### 0.1 Five ledgers claim serving-contract gaps that are closed

Each of these has `caps.rs` **and** a `resident_*.rs` **and** a `catalog.rs`
entry **and** (except where noted) an `examples/` client:

| ledger | its claim | reality |
|---|---|---|
| `flux1.md` | "no sampler loop, no VAE glue, no text-encoder call, no CLI subcommand and no serving surface" | `flux1::caps`, `resident_flux1.rs`, catalog entry, `examples/imagegen/flux1_generate.py`, `pipeline::Flux1::generate` |
| `pulid.md` | "Serving contract - no capability manifest, no residency adapter, no batched request handling, no D-Bus surface, no CLI" | `pulid::caps`, `resident_pulid.rs`, catalog entry, `examples/imagegen/pulid_generate.py` |
| `controlnet.md` | "The full serving contract: capability provider, residency adapter, `run_batch`, D-Bus surface, example client" | `controlnet::caps`, `resident_controlnet.rs`, catalog entry, `examples/imagegen/controlnet_generate.py` |
| `sdxlunet.md` | "Full serving contract: a capability provider, a residency adapter, a real batched `run_batch`, D-Bus exposure, and an examples client" | `sdxlunet::caps`, `resident_sdxl.rs`, catalog entry, `examples/imagegen/sdxl_generate.py` |
| `t5encoder.md` | "Serving surface - no `Provider`, no residency adapter, no `run_batch`, no D-Bus exposure, no CLI subcommand" | `t5encoder::caps`, `resident_t5encoder.rs`, catalog entry, `examples/embedding/t5_embed.py` |

`codeformer.md` additionally claims "Backward pass / gradcheck for the
transformer and CFT" is outstanding. It is not: `gradcheck::check_codeformer`
and `check_codeformer_one_layer` exist and are wired in
`crates/gradcheck/tests/imaging_models.rs:66,69`.

The residual truth in each of the five - batching, backward, full-depth parity,
end-to-end fixtures - is real and is carried forward into Phases 2-4 below. Only
the *serving-contract* line is stale.

**Action:** rewrite those six ledgers' "Not yet done" sections to what is
actually outstanding. Do not delete the stale lines silently - the AGENTS.md
serving-status paragraph is the accurate source and should be what they
converge on.

### 0.2 `AGENTS.md` omits nine crates, two of them served models

`arch`, `atif`, `gemma4`, `ltxv`, `memauth`, `minimaxmusic3`, `rl`, `shutdown`,
`weightset` appear nowhere in the routing guide. Two are **served models with
catalog entries and residency adapters** (`ltxv` text-to-video,
`minimaxmusic3` music generation); `arch` is the *canonical architecture
registry* the guide's own naming rule depends on; `memauth` is the memory
authority that supersedes part of what the guide says about `residency`
budgets.

This is the highest-cost documentation defect in the repo: AGENTS.md is the
routing guide, and a model absent from it is a model the next port will
duplicate rather than reuse.

**Action:** add the nine, in the sections they belong to. `ltxv` and
`minimaxmusic3` get Models entries; `arch`, `memauth`, `weightset`, `shutdown`
get Engine-core / serving-stack table rows; `atif` and `rl` get a short
"training-from-trajectories" note; `gemma4` goes under `ltxv` as its text
encoder.

### 0.3 One more stale AGENTS.md claim

"`clip` has no `examples/` entry yet (every other one of the eleven does)" -
`examples/embedding/clip_embed.py` exists. Fix in the same pass.

### 0.4 Operational risk to record before a long campaign

`target/` is **203 GB**; the filesystem is at **88 % (111 GB free)**. Per
`brain-disk-budget`, a full overlay hard-blocks every tool. A multi-week
finishing campaign that builds tests and examples will not survive at this
margin.

**Done, and the surgical fix was the right one.** `make clean` runs
`cargo clean`, which would have discarded all 203 GB including the 119 GB
`debug/deps` dependency graph and forced a cold rebuild. Deleting only
`target/debug/incremental` freed **69 GB** and cost nothing measurable: the
`make build` immediately afterwards finished in **0.75 s** having recompiled
nothing, because `incremental/` is rebuild-state, not artifacts. Filesystem
went 88% -> 80%.

Prefer this over `make clean` whenever space is the problem and the dependency
graph is not.

---

## Phase 1 - Make the gate trustworthy

**Size: M (2-4 days). Everything after this is unverifiable without it.**

**PARTLY DONE, and the first finding changed the shape of the rest.**

### 1.0 `make check/scripts` was red, and had nothing to do with the hangs

`test/full` could not pass regardless of the GPU/NPU hangs below, because
`check/scripts` was failing - and it fails one sub-gate at a time, so each fix
uncovered the next:

1. **check-env-docs** - 19 undocumented `BRAIN_*` variables, every one
   belonging to a model added since the gate last passed (the imaging stack's
   weight roots, Qwen3.8-27B's dir/ctx/batch, Wan and FLUX.1 placement, nine
   LTX-2.5 dump/bench knobs). An undocumented variable is an unreachable
   feature; the face/rrdbnet perf targets already shipped dead from exactly
   this.
2. **check-no-perf-numbers** - one real claim (a duration cited only as
   evidence a run completed; rephrased away) and two false positives inside
   fenced code blocks, where the escape hatch cannot reach because an HTML
   comment renders as page content. Fixed by the gate refinement its own
   header offers: a match preceded by a word character, `_`, `.`, `/` or `%`
   is identifier-internal. Verified to remove exactly 3 of 44 raw hits.
3. **check-golden-source** - 16 dumpers writing a manifest with no `source`
   block, all from the two newest workstreams (ltxv/gemma4, qwen35), i.e. the
   same two that skipped AGENTS.md. Now 22 on the convention, 45 grandfathered.

`make check/scripts` passes end to end as of this writing. **Lesson worth
keeping: a chained gate reports only its first failure, so "the gate is red"
is never one fact.** Re-run it after every fix until it passes rather than
assuming the first fix was the fix.

### 1.1 The hangs

The stated symptom (`npu.md`): `make test` is not reliably green in a single
attempt on this box.

### 1.1a The better gate already existed and had never been installed

`make test/nextest` runs the same tests as `make test` but process-per-test
with a per-test `slow-timeout` (`.config/nextest.toml`: 60 s period,
terminate-after 3). Its Makefile comment already describes exactly the failure
below - "a single wedged GPU test blocks every other test behind it for up to
TEST_TIMEOUT with no attribution" - and says the target is not the default
because it "has not been run against the full workspace end to end, only
smoke-tested".

**`cargo-nextest` was not installed on this box**, which is why that validation
never happened. Installed (`cargo install cargo-nextest --locked`), and a full
`make test/nextest` run is what should produce the attribution the
investigation below has been missing: a wedged test is killed and NAMED after
~190 s instead of hanging the suite.

Note it builds `--release` while `make test` builds debug, so a first run pays
a full release compile of every test binary (~1 h on this box). That difference
is itself worth resolving before it becomes the default gate: the hang was
observed in the debug lane.

### 1.1b The two hangs are probably one bug

`crates/cli/tests/npu_model_parity.rs` hung for 26+ minutes under a full
`make test`, with **exactly one thread pinned at 100 %** and the rest idle -
the signature of a busy-poll on a completion that never signals. That is
frame-for-frame the same signature as the `kernel_timing` hang recorded in
`backend-vulkan.md`, which did *not* reproduce on a clean immediate re-run.

**The discriminating experiment, in order:**
1. Run `npu_model_parity` alone. Hangs alone → isolated-binary bug; done, fix
   it there.
2. Does not hang alone → run the full suite twice back-to-back. Reproduces on
   attempt 1 only → cross-process contention under `--test-threads=8` across
   GPU/NPU-touching binaries.
3. If contention: the shared assumption to audit is device serialisation -
   `gpu_core::testgpu`'s weak pool is per-process, so N test *processes* each
   build a device. `BRAIN_GPU_WAIT_S` (default 30 s) already exists to turn a
   wedged submit into a named panic; find out why it did not fire here. If it
   does not cover the OpenVINO path, that is the fix: no wait in the suite may
   be unbounded.

Instrument first: the hang was killed before the responsible test was
identified because stdout was buffered per-test. Run with
`--nocapture`/`--test-threads=1` on the NPU binary so the next occurrence names
itself.

### 1.2 Record what is *not* fixable here

`vlm.md`'s fastvlm exit segfault is root-caused to NVIDIA driver 570.195.03
(a `[vkps] Update` driver-owned thread, symbol-free stack), with four
mitigations tried and none effective. That is an **accepted, external** defect,
not open work. Move it out of the "not yet done" framing into an explicit
"accepted, external" section so nobody re-spends the day on it - and add it to
`.agents/rules/lessons.md` if it is not already there, since it is exactly the
class that file exists for.

### 1.3 Exit criterion

`make test/full` green twice in a row, from clean, unattended. Until that holds,
treat every "verified" claim produced by later phases as provisional.

---

## Phase 2 - Close the serving contract

**Size: L (2-3 weeks). This is the invariant with the most real violations.**

Verified inventory. Components of composite models (`sam1`, `deepseek2`, `mimi`,
`ecapatdnn`, `campplus`, `s3tokenizer`, `gemma4`) are correctly excluded - they
are not standalone models.

### 2.1 No capability surface at all - the real gaps

| model | state | work |
|---|---|---|
| `moondream3` | no `caps.rs`, no catalog entry, no CLI; reachable only from tests | full contract: caps + resident + catalog + example. Decoder is gradient-checked and import-covered, so the model half is done - this is the serving half only |
| `splat` | no `caps.rs`; `brain splat` CLI only | caps (`render`/`fit`) + resident + D-Bus + example |
| `worldmirror2` | no `caps.rs`; `brain worldmirror2` CLI only | caps (`reconstruct`) + resident + example |
| `diamond` / `genieredux` | no `caps.rs`; AGENTS.md calls `diamond` "the one served world-model architecture", which the tree does not support | decide: either serve it properly or correct the claim. An interactive-play model may genuinely not fit `Run`/`Subscribe` - if so, **extend the D-Bus surface**, per the invariant, and say so |
| `glmdsa` | **DONE.** `glmdsa::caps` + a `catalog.rs` entry; `brain caps` went 36 -> 37 models on a box with no GLM checkpoint | the real gap was narrower than "no serving contract": `GlmResident` was always registered and scheduled, but its manifest only exists when `BRAIN_GLMDSA_WEIGHTS` is set, so GLM was the one model whose *discoverability* depended on deployment state. `GlmResident::manifest` now returns `glmdsa::caps::manifest_resident()`, so the served and direct surfaces are one definition. Remaining: `run_batch`, an `examples/` client |
| `qwen3vl` | catalog entry, `resident: None`, explicitly "no residency adapter yet" | residency adapter, matching `fastvlm`'s stateless `ProviderResident` registration (`resident.rs:179`) |
| `gpt2`, `toymoe`, `toypid`, `toyseq2seq`, `toyautoencoder` | manifest via resident only (`GptResident`), no `caps.rs` | low priority - these are the toy/baseline models; decide explicitly whether the contract applies to them and record the answer rather than leaving it ambiguous |

**Sequencing:** `glmdsa` first (smallest delta - the resident already exists,
only `caps.rs` is missing, and it closes a gap AGENTS.md names explicitly), then
`qwen3vl` (one adapter), then `moondream3` (full contract, but the model is
done), then the three vision/3D/world models (each needs a design decision about
its action shape).

### 2.2 Missing examples

- `qwen3tts` - no D-Bus client example (`qwen3tts.md`), *and* it still carries a
  private socket-based serving side-channel that the same ledger says should be
  consolidated into D-Bus. Do both in one change: consolidating the transport is
  what makes the example writable.
- `lfm2` - no Python D-Bus embedding client (`lfm2.md`).

### 2.3 `run_batch` serial defaults that are not justified

The invariant permits a serial `run_batch` only *with a stated reason*. These
have a batchable forward and no stated reason:

| model | why it should batch |
|---|---|
| `chronos2` + `fincast` | share a batchable transformer core; equal-shape contexts could share one forward (`forecast.md`) |
| `arcface` | input batches trivially; only the graph is pre-allocated at N=1 (`scrfd.md`) |
| `vqgan` / `codeformer` | batch size is hardcoded to 1 in the **shared** `vae::blocks` builder - so the fix lands in the shared builder and pays off for every VAE-family model at once, not just these two |
| `lfm2` | length-bucketed batching + zeroed pad states with an additive key mask (bidirectional attention has no causal mask to hide padding) |
| `s3dit` | listed as an explicit open item |

`scrfd`'s detector (graph pinned N=1), `qwen3tts`/`cosyvoice` (autoregressive),
and the per-request multi-step samplers (`sdxlunet`, `controlnet`, `flux1`,
`pulid`, `flux2`) are legitimately serial and already say so in-file - leave
them, but verify each really does carry the comment.

The `vae::blocks` batch-size fix is the highest-leverage item in this section:
one shared builder, several models.

---

## Phase 3 - Close the backward / gradcheck gaps

**Size: L (3-4 weeks). The stated default expectation, currently unmet in ~8
places.**

`AGENTS.md` is explicit that forward-only requires a *named, recorded*
constraint, and that the existing forward-only models "are a record of what
shipped under real constraints, not a template".

### 3.1 Missing entry points, ordered by cost

| model | missing | note |
|---|---|---|
| `sdxlunet` | `check_unet` | **do this first.** Its own ledger says the graph is built entirely from existing conv/transformer blocks, so the backward composes existing adjoints - no new kernel work. Highest ratio of invariant-closed to effort |
| `controlnet` | `check_controlnet` | the trainable copy *is* the UNet's blocks (recorded by `sdxlunet::model::Rec`), so it follows directly from `check_unet` |
| `flux1` | `check_flux1` | full-depth fp32 does not fit one card; gate at reduced depth as the forward parity already does |
| `pulid` | `check_pulid` | blocked by a real structural issue its ledger names: the forward reuses buffers across layers in an inference shape, so a training-mode forward with per-layer allocation is a prerequisite. Scope that first |
| `chronos2` | `build_backward` + `impl model::Model` + `check_chronos2` | today's path is inference-only, per-op-submit; needs SSA buffers |
| `rrdbnet` | none | forward-only, no recorded justification |
| `deepseek2ocr` | only `check_deepseekocr_relpos*` | the composite has an exact adjoint reaching input pixels but no model-level entry point |
| `instantid` | none | forward is not implemented at all - this is a Phase 5 item, not Phase 3 |

### 3.2 Use the right oracle

`directional_check` alone does **not** catch a partially-wrong gradient on a
folded or shared parameter - measured: deleting T5's cross-block `axpy` fold
leaves `rel_bias.weight` 33 % wrong while every directional check still passes.
Every folded/shared parameter added in this phase needs
`gradcheck::elementwise_check`. This applies directly to `check_unet`
(SDXL's timestep embedding is added into every resnet) and to
`check_controlnet` (the trainable copy shares structure with the frozen UNet).

### 3.3 Explicitly out of scope for this phase

- `splat` - deliberately uses an autograd oracle, not finite differences
  (1/255 output truncation biases FD gradients). Not a gap.
- `glmdsa`'s DSA indexer - forward-only by design (trained by distillation,
  detached from the LM loss). Not a gap.
- `cosyvoice`'s flow decoder and HiFT vocoder - both are pure host scalar-loop
  f32 with no backward *and no float genericity*, and the NSF source generator
  and ISTFT head have no precedent backward in the workspace. Its own ledger
  scoped this out deliberately. Treat as its own project (Phase 5), not as a
  Phase 3 line item.

---

## Phase 4 - Feature completion, per model

**Size: XL. Not one campaign - a backlog to draw from, ordered by how much it
unblocks.**

### 4.1 Unblocks other work

1. **`modelstore` resumable fetch - DONE.** `execute` now skips a
   `Step::Download` whose `dest` exists. The fix turned out simpler than this
   plan proposed (stat + `Content-Length` from a HEAD): `fetch::stream_to_file`
   writes to a `.part` sibling and renames only on full success, so a file at
   `dest` is by construction complete and bare existence is the *correct* test -
   no new `Hub` method, no network round-trip, and no edit to the
   redirect/host-allowlist path, which is the SSRF boundary and has been wrong
   once already. Gated by two tests, the second asserting the skip is per-FILE
   so a partially-fetched repo still completes.
2. **Shared DiT hoist** (`ltxv.md`) - `crates/dit` owns only RoPE despite its
   doc claiming adaLN/patchify/QK-norm, so `wan`/`s3dit`/`flux2` each
   re-implement PixArt timestep embedding and patchify/unpatchify. This is the
   "one implementation" invariant being violated three ways; hoisting it pays
   into every future DiT port.
3. **`vae::blocks` batch axis** - see §2.3.

### 4.2 Per-model completion, largest first

- **`qwen35moe`** - multi-GPU INT8/INT4 residency serving, multi-sequence
  batching, prefix-cache reuse, chunked prefill, speculative decode, INT8 paged
  KV. Also: **numerical parity against the reference is not yet established** -
  only structural forward correctness. That parity gap should be closed *before*
  the serving work, not after.
- **`qwen3omnimoe`** - `converse` and `transcribe` actions, multimodal input
  with `speak`, multi-turn Talker context, streaming Code2Wav, real
  audio-timestamp M-RoPE scaling, DeepStack on the served image path, true
  temporal video patching, `/v1/audio/{speech,transcriptions}`, and the
  OpenAI/Anthropic surfaces currently *dropping* `image_url`/`input_audio`
  content parts. That last one is a silent-data-loss bug, not a feature - treat
  it as Phase 1-grade.
- **`cosyvoice`** - CosyVoice 3 pipeline composition (components are all
  parity-proven; only `pipeline::generate` is CV2-only, and it correctly errors
  rather than silently falling back), streaming chunked token2wav, kaldi fbank
  bit-exact gate, docs.
- **`world-models`** - CoinRun ingest, GenieRedux interactive play loop, and
  moving its forward onto a single on-device graph (it round-trips through the
  host between every op with a naive matmul). The on-device graph is the
  prerequisite for the other two.
- **`chronos2`** - real covariates, long-horizon unrolling, a GPU kernel for
  multivariate group attention (host-computed today).
- **`kronos`** - finish the parity ladder (isolated decoder block, exposure-bias
  sampling, the full `generate` loop are unverified), `ForecastModel` adapter.
- **`splat`** - SH degrees 1-3, densify/prune (required to train a scene from
  scratch), `.splat`/`.spz` I/O.
- **`zipdepth`** - real training data + SSI/gradient loss; INT8 measured against
  real weights; camera path confirmed against real hardware.
- **`minimaxmusic3`** - joint generator+discriminator training,
  multi-resolution discriminator, the real end-to-end WAV gate.
- **`instantid`** - forward is not implemented at all. Decide whether it ships
  or is dropped the way Depth Anything 3 was, and record the decision.

---

## Phase 5 - Performance

**Only against a published baseline. `kernels.md` §F is the ordered loop;
"every confident hypothesis on this engine has been wrong, and the profile has
been right."**

Ranked by measured cost:

1. **`qwen3omnimoe` int8 MoE expert kernel** - with weights resident and decode
   KV-cached, the remaining **2.3 s/token** is almost entirely naive per-thread
   expert dispatches. The blocker is real and named: WGSL has no legal way for
   part of a workgroup to skip a barrier, so a tiled kernel needs routed rows
   gathered into a compact non-ragged batch first. That gather is the actual
   work item.
   - Prerequisite: `BRAIN_PROFILE`'s Vulkan table silently skips batches over
     `MAX_TIMED_DISPATCHES` (8192), which a 48-layer 128-expert forward always
     exceeds - **so there is currently no per-kernel attribution inside a
     layer on this path.** Fix the profiler cap first or this optimisation is
     unmeasurable, which §F forbids.
2. **`wan` self-attention** - three of four named defects are fixed; this one is
   not, and `wan.md` carries the published per-kernel baseline to measure
   against.
3. **`flux2` kernel gaps** - workgroup-per-row LayerNorm (RMSNorm and softmax
   already have it), workgroup-per-row int8 row-max, vec4 shared-memory tile
   loads, a second query row per thread in flash attention, an implicit-GEMM
   conv for VAE decode, and the fused tiled causal+key-mask attention kernel.
   Note its own conclusion: most of a served image's latency is in the
   *unbatched* text encoder and VAE decode, not the transformer - so §2.3's
   batching work may outrank the kernel work here. Profile before choosing.
4. **`genieredux` on-device graph** - see §4.2; performance and capability are
   the same item.
5. **`scrfd`** - unoptimised and unmeasured: excess scratch in conv/PReLU
   blocks, per-frame device→host syncs reading back scalars that never change,
   and every debug tap read back in production. The last one is nearly free to
   fix.

**Do not start any of these before Phase 1.** A perf number taken while
`make test` needs two attempts is a number taken on an unknown machine state.

---

## Accepted, not open work

Record these so they stop reading as backlog:

- The NVIDIA 570.195.03 exit segfault (`vlm.md`, `backend-vulkan.md`) - four
  mitigations tried, two made it worse, one falsified the concurrency theory.
  External.
- `splat`'s FD-free gradient oracle - deliberate.
- `glmdsa`'s DSA-indexer forward-only backward - by design.
- FLUX.2 Klein-9B's cached-reference-attention variant - incompatible with
  folding modulation into LayerNorm.
- Wan FLF2V/VACE - explicitly out of scope for the first landing.
- Full-size GLM-5.2 (78 layers / 256 experts / 155k vocab) - not runnable on
  this hardware; used for import shape validation only.

---

## Working rules for this campaign

Inherited from `AGENTS.md`, restated because a long campaign is exactly where
they erode:

1. **Write the lesson in the same change.** A finding that lives only in a
   commit message is a finding that will be relearned.
2. **Zero warnings.** The baseline measured at the top of this file is 0. Any
   phase that ends above 0 is not finished.
3. **One implementation.** Before writing anything, search for it. Phase 4.1's
   two hoists exist because this rule was missed.
4. **Build through the Makefile.** Never raw `cargo`, never a per-command
   `CARGO_HOME`.
5. **Linear history, self-contained commits.** One commit per closed item, not
   one per phase.
6. **`docs/` only changes when the public contract changes.** Everything in this
   plan that is a status update belongs in `.agents/`.

---

## Tracking

Update this file as phases close; keep the per-model ledgers as the detail
record and this one as the ordering. When a phase's exit criterion is met,
record the *measurement* that proved it, not the assertion - a number nothing
checks is a number that silently goes stale.
