# qwen3tts - roadmap

Qwen3-TTS voice synthesis stack (Talker + MTP + Mimi-style codec + ECAPA
speaker encoder): speaker-free synthesis, voice cloning (x-vector and
in-context), instruct-style voice design, NPU/CPU/GPU backends, and LoRA
fine-tuning. Parity against the reference is verified.

## Not yet done

This list used to duplicate its items against the "Completion plan"'s own
phases and its "Carried over" section below, and the two copies drifted
(one got checked off, the other didn't). Removed the duplication instead of
trying to keep two lists in sync by hand: `run_batch`, D-Bus consolidation +
example client, and codec/speaker training each now have exactly one entry,
in their own phase below. **The generic engine/architecture items
(RMSNorm backward, cancellation, windowed codec mask, MTP fusion, the
runtime stub) live in "Carried over, unchanged priority" near the end of
this file** - that section is the current, authoritative copy.

Codec decoding is sub-real-time on the CPU backend (48 s for 3.36 s of
audio on this box). It is **not** CPU-pinned any more: as of 2026-09-04
`qwen3tts::pipeline::decode_codes` builds it on the ambient `--device`,
where the same decode takes 6.3 s. Still ~1.9x slower than real time
there, so the NPU backend remains the only realtime synthesis path.

## Measured (2026-09-03/04, Intel Arc iGPU box, no discrete GPU; NPU status
## below)

**Correction to this section's own original heading**: it originally said
"CPU-only." That was wrong. `mimi::Codec::load_inference`
(`crates/mimi/src/model.rs:107`) hardcodes `Gpu::new_cpu(PIPELINES)` -
codec decode is CPU-pinned unconditionally, no `--device` override - but
the Talker/MTP decode loop (`TalkerGen::load`/`MtpModel::load_inference`,
`crates/qwen3tts/src/{gen,mtp}.rs`) both call plain `Gpu::new(PIPELINES)`,
which reads the ambient `--device`/`BRAIN_DEVICE` selection like every
other model. On this box, with no `--device` flag given at all, that
ambient default resolves to the Arc iGPU (wgpu/Vulkan) whenever one is
present - so every run in the table below (all pre-dating an explicit
`--device` test) already had its Talker+MTP compute on the GPU, with only
the final codec-decode pass on CPU. Confirmed by directly comparing
`--device cpu` (forces everything, including Talker/MTP, onto
`brain-wgsl-cpu`) against `--device gpu` (Arc) on an identical request -
see the new table below.

All runs are the `synth`/`clone`/`design` path
(`pipeline::generate_codes`: KV-cached Talker, full-recompute MTP),
`temperature=0.9 top_k=50` (the reference's own defaults), wall-clock `time`
including process startup + checkpoint load (not steady-state decode alone).
RTF = wall-clock seconds / output-audio seconds (a from-scratch WGSL engine
timing itself, so this is real elapsed time on real hardware, not a vendor
benchmark).

### CPU vs GPU, controlled (same text, same seed, same `--max-frames 60`,
### run back-to-back on an otherwise-idle box - no concurrent build this
### time, unlike the table below)

| Device | Wall time | Audio | RTF | CPU utilization (`user`/`real`) |
|---|---|---|---|---|
| `--device cpu` (Talker+MTP+codec all CPU) | 1m40s | 3.36s | **29.7x** | 13.1x (heavy multi-core use) |
| `--device gpu` (Talker+MTP on Arc iGPU, codec still CPU-pinned) | 4m52s | 3.36s | **86.9x** | 3.1x |

**The Arc iGPU is ~2.9x SLOWER than the 22-core CPU path for this model**,
not faster - confirms the pre-existing "GPU (Vulkan) forward passes exist
for correctness checks but are not the path used for practical synthesis
speed" line in this document's own "Hardware and limits" section, now with
a real number behind it instead of just an assertion.

**That table is the BEFORE state, and its stated root cause was wrong.**
It attributed the loss to per-dispatch overhead being unavoidable for
batch-1 decode, and drew the conclusion that the CPU should be this
model's default device. Profiling it instead of reasoning about it found
three ordinary defects in this crate plus one in `mimi`, all fixable, and
the conclusion reverses: after them the GPU is **2.4-2.6x FASTER** than
the same CPU path, end to end, on the identical command. See "CPU vs GPU,
after the 2026-09-04 optimisation pass" below. The
`ambient_compute_set()` / default-CPU-override idea floated above is
therefore **dropped**, not deferred.

Also note: the CPU run above (RTF 29.7x, `user`/`real` ratio 13.1x - good
multi-core utilization) is dramatically faster than every CPU-driven number
in the table further below (RTF 93-129x, `user`/`real` ratio only ~2.9x -
poor utilization). Same engine, same hardware. The difference is system
load: the table below's runs were taken while a from-scratch workspace
`cargo build --release` (~60 crates) was running concurrently on the same
22 cores; this comparison table's runs were not. **The 93-129x figures
understate this engine's real CPU throughput by roughly 3-4x** - they are
real, honestly measured numbers for the conditions they were taken under,
just not this box's best case. 29.7x is closer to this box's real CPU
ceiling for this model.

### CPU vs GPU, after the 2026-09-04 optimisation pass

Same command, same text, same seed, same `--max-frames 60` (both stop at
EOS after 42 frames = 3.36 s of audio), run back-to-back twice each on an
otherwise-idle box with nothing compiling.

| Device | Wall (run 1 / run 2) | RTF | codec.decode | decode loop |
|---|---|---|---|---|
| `--device cpu` | 1m17.9s / 1m15.6s | **22.5x** | 49.4s / 48.1s | 22.1s / 22.2s |
| `--device gpu` (Arc iGPU) | **0m32.1s / 0m29.9s** | **8.9x** | 6.3s / 6.0s | 20.2s / 18.1s |

**The GPU is now 2.4-2.6x faster than the CPU**, where it was 2.9x
slower: a 7.4x swing on the identical command. Absolute GPU wall time is
**10.0x** better than the baseline row above (299-305s -> 30-32s); the
CPU path improved 1.34x over the same change set (101-102s -> 76-78s),
since two of the four fixes are device-independent.

The two runs' waveforms are the same waveform: cosine **0.9999999993**
with a worst-case difference of 1 LSB of 16-bit PCM, which is fp
reassociation inside the reductions, not a behavioural difference.

Reproduce (checkpoints already imported):

```
BRAIN_QWEN3TTS_WEIGHTS=out/tts-base06 \
BRAIN_QWEN3TTS_CKPT=<hf>/Qwen3-TTS-12Hz-0.6B-Base \
TTS_PROFILE=1 time ./target/release/brain --device {cpu,gpu} qwen3tts \
  synth --text "The quick brown fox jumps over the lazy dog." \
  --max-frames 60 --seed 5 --out /tmp/out.wav
```

#### What was actually wrong, and what each fix returned

`TTS_PROFILE=1` printed nothing at all for this path before (it only
instrumented `generate_codes_cached`, the CPU-only NPU-adjacent mirror);
instrumenting `generate_codes` is what found all of this. The baseline
split, on the GPU, over 42 frames:

```
prefix-stream(9 pos)=1465.3ms | talker-step total=9825.1ms (233.9ms/frame)
| mtp-residuals total=242406.8ms (5771.6ms/frame) | cb0-head total=100.8ms
```

so **95.6% of the GPU decode loop was the MTP residual fill**, not the
28-layer Talker and not overhead spread evenly across the model.

| # | Defect | Fix | Measured |
|---|---|---|---|
| 1 | `MtpModel::generate_residuals_with` called `logits()` (a full 16-position, 5-layer re-forward) once per residual codebook, 15x per audio frame | KV-cached incremental decode, one prebuilt tape per position (the MTP's sequence length is fixed, so no uniform is ever rewritten); mirrors the already-proven `gen_kv_mtp::CpuMtp` | with #2: mtp-residuals 5771.6 -> 300.5 ms/frame |
| 2 | Every linear dispatched the naive `matmul` (one thread per output, 4 KB-strided weight read per thread, zero coalescing at m=1) | `block::gemm_variant` over `matmul_gemv` (upgraded to `matmul_gemv_reg` by `gpu_core::upgrade`) / `matmul_reg3`, gated on `caps.workgroup_reductions` exactly as `qwen3::serve` gates its own | talker-step 233.9 -> 120.8 ms/frame |
| 3 | `logits()` evaluated all 15 `[2048,1024]` `lm_head` rows per residual step in a scalar loop; the caller read one. `codec_head_logits` was scalar too | `logits_at` (one row) + `hostmath::matvec` (AVX2+FMA, rayon) | cb0-head 2.4 -> 0.4 ms/frame |
| 4 | `mimi::Codec::from_weights` hardcoded `Gpu::new_cpu`, with no override - the codec was the one stage of a `--device gpu` run that never touched the GPU, and after 1-3 it was 64% of the wall clock | `Codec::{load_inference,from_weights}_on`; `qwen3tts::pipeline::decode_codes` builds on the ambient device | codec.decode 46.5s -> 6.3s |

Fixes 1 and 3 help the CPU path too (16x and 15x less arithmetic
respectively, on any device); 2 and 4 are GPU-only by construction -
`backend-cpu` reports `workgroup_reductions: false` and keeps the naive
reference, which it routes to the same AVX2 GEMM it always did.

#### Where the remaining time goes, and what would move it

At 30 s: ~6 s codec, ~18 s decode loop, ~6 s process start + 4.0 GiB of
checkpoint load. The decode loop is now the largest term and is within
~20% of the CPU's, which is the honest ceiling for this shape on this
box: batch-1 fp32 autoregressive decode is weight-bandwidth-bound, and
an integrated GPU reads those weights over the SAME DDR5 controller the
22-core CPU does, so there is no bandwidth advantage to win.

Per-kernel timing (`BRAIN_PROFILE=1`; note the Intel ANV timestamp
period is broken on this box, so the absolute ms are garbage and only
the shares and call counts are usable) shows every kernel's share
tracking its DISPATCH COUNT to within 0.5 percentage points - a
`matmul_gemv_reg` moving 8 MB and an `add2` moving 4 KB cost the same.
That puts the floor at roughly 0.15-0.2 ms per dispatch, ~588 dispatches
per Talker token and 1680 per MTP frame. **The only lever left is fewer
dispatches**, i.e. block fusion (fused QKV, fused gate/up, fused
RMSNorm+GEMV) - `kernel-performance.md` Phase 4's territory, a
kernel-shape change, and deliberately NOT attempted here.

One hypothesis was tested and **killed**: `TalkerGen::decode_cached`
issues 196 `Gpu::write` calls per token (7 position uniforms x 28
layers) and `WgpuBackend::write` flushes first, so each was an empty
`queue.submit(None)` - 197 submissions per token carrying no dispatches.
A batched `write_many` collapsed that to 546 submits per 13 steps from
3094 and changed the wall clock by **nothing** (talker-step 120.8 ->
122.5 ms/frame, inside run-to-run noise). Reverted; see
`kernel-performance.md`.

| Run | Checkpoint | Wall time | Audio | RTF |
|---|---|---|---|---|
| Speaker-free synth (English) | 0.6B-Base | 8m43s | 5.60s | 93.4x |
| Voice clone, ICL (ref transcript + in-tree codec encode) | 0.6B-Base | 6m46s | 4.16s | 97.6x |
| Voice clone, x-vector-only (no transcript) | 0.6B-Base | 6m02s | 3.52s | 102.9x |
| CustomVoice preset speaker ("vivian") | 0.6B-CustomVoice | 7m34s | 4.56s | 99.5x |
| CustomVoice preset + `--instruct` emotion ("ryan", excited) | 0.6B-CustomVoice | 6m22s | 2.96s | 129.2x |
| Speaker-free synth (German, `--lang german`) | 0.6B-Base | 4m36s | 2.56s | 107.9x |
| VoiceDesign, pure `--instruct` (no preset speaker) | 1.7B-VoiceDesign | 6m58s | 3.44s | 121.5x |

Mean RTF across the six 0.6B runs: **~105x realtime**; the one 1.7B run measured
above is slower still (121.5x), consistent with its larger Talker (2048-wide
vs 1024) and MTP. Confirms the existing "CPU codec decoding is sub-real-time"
note above understates it: the codec alone is not the bottleneck at this
scale, MTP + Talker CPU decode dominate, matching the "MTP and the codec are
now the dominant per-clip cost" line in the carried-over items below.

The 1.7B-VoiceDesign row above did not run cleanly on the first two
attempts - see "1.7B-family MtpModel crash" below, now fixed and reflected
in this row's real measured numbers.

Speaker-embedding cosine similarity (ECAPA x-vector, `brain qwen3tts sim`)
against the reference clip (`testdata/asr/audio/librispeech_mr_quilter.wav`,
LibriSpeech, ~5.9s):

| Output | spk-cosine vs reference |
|---|---|
| Voice clone, ICL | 0.9829 |
| Voice clone, x-vector-only | 0.9841 |
| Speaker-free synth (no cloning attempted) | 0.9079 |

The two clone modes land within 0.001 of each other and clearly above the
speaker-free baseline, which is itself surprisingly high (0.91) -- the
speaker-free default voice is not maximally distinct from an arbitrary
reference speaker in ECAPA embedding space, so 0.90-ish is this particular
metric's noise floor, not evidence of accidental cloning.

The first two attempts at the 1.7B-VoiceDesign row above both failed, for
two DIFFERENT reasons, before the real number above was measured:

1. **OOM-killed** (`Killed`, SIGKILL) partway through the first attempt,
   immediately after loading its 6.5 GiB `talker.safetensors`, while running
   concurrently with a from-scratch workspace `cargo build --release` (~60
   crates) on this 30 GiB-RAM box with swap already near full from prior
   sessions. Real operational finding, not a correctness bug: **the 1.7B
   variants need real memory headroom** (the 0.6B variants' ~4 GiB combined
   footprint has no such issue) -- worth a documented minimum-RAM note for
   this checkpoint size class.
2. **A real crash**, on the retry once memory pressure was gone:
   `assert_eq!(cb0_embed.len(), d)` panicked with `left=2048, right=1024`.
   This was a genuine, previously-unknown correctness bug -- `MtpModel`
   (`crates/qwen3tts/src/mtp.rs`) never applied `small_to_mtp_projection`,
   the real `Linear(2048->1024)` the 1.7B checkpoint ships to bridge its
   Talker's `hidden_size=2048` down to its MTP's own `hidden_size=1024` (the
   0.6B has no such tensor at all, both widths being 1024 there, which is
   exactly what let this go unnoticed until a real 1.7B checkpoint was
   actually run). Fixed same-session -- see the git log for the commit
   fixing `MtpModel`; `crates/qwen3tts/src/gen_kv_mtp.rs`'s `CpuMtp` and the
   NPU `MtpEngine` impls in `npu_gen.rs` already handled this correctly, so
   `MtpModel` (the full-recompute path `pipeline::generate_codes` -- what
   every default `synth`/`clone`/`design` command calls -- actually uses)
   was the one lagging implementation. The row above is the POST-fix
   measurement.

## Completion plan (2026-09-03 audit)

Verified against the real `Qwen/Qwen3-TTS-12Hz-{0.6B-Base,0.6B-CustomVoice,
1.7B-VoiceDesign}` checkpoints (imported and run end-to-end: synth, clone
(ICL + x-vector), CustomVoice preset speakers, VoiceDesign instruct -- see the
"Measured" section below for numbers, per `docs/performance/overview.md`'s
own convention that session-specific measurements live here, not in
`docs/`). Ordered cheapest/highest-leverage first; each phase is
independently shippable.

### Phase 0 - Bugs found by actually running it (do first, near-zero cost)

- [x] `crates/arch/src/lib.rs`'s `qwen3tts` entry declares
      `weights_env: &[("BRAIN_QWEN3TTS_WEIGHTS", "weights_dir"), ...]`, but
      `resolve::flag_twin` derives the CLI flag from the env var's suffix
      (`WEIGHTS` → `--weights`), not from the tuple's own second field. The
      real flag is `--weights-dir`. Fixed via a small, explicit
      variable->flag override list in `resolve.rs`, plus a resolver test
      asserting `--weights-dir X --ckpt Y` alone satisfies
      `weights_already_named("qwen3tts", ...)`.
- [x] `prompt.rs`'s module doc claimed the codec was decode-only with no
      in-tree encoder - false since `mimi::Codec::encode` landed and is
      called from `pipeline.rs`'s `clone()`/`clone_npu()`. Comment fixed.

### Phase 1 - Generation-control parity (spec exists in Qwen's own config)

Qwen's reference reads `do_sample/top_k/top_p/temperature/repetition_penalty`
for codebook-0 AND separate `subtalker_*` knobs for the MTP residual
codebooks from `generate_config.json`. Brain's `GenOpts` has exactly
`{max_frames, temperature, top_k, seed, min_new}` and the MTP residual
codebooks (1..15) are hardcoded greedy argmax (`gen_kv_mtp.rs`,
`npu_gen.rs::generate_residuals`) -- never sampled, no independent control.
Since the residual codebooks carry most of the acoustic detail, this is a
real perceptual-quality lever, not API decoration.

- [x] Write the spec test first: extend `GenOpts` with `top_p`,
      `repetition_penalty`, and an optional `residual: Option<ResidualOpts>`;
      tests asserting top_p/repetition_penalty actually filter (unit-level, on
      `sample_cb0` directly) and that residual sampling diverges from greedy
      across seeds (`pipeline::sampling_tests`, no checkpoint needed --
      `MtpModel::new_synthetic_on` weights).
- [x] Implement top-p (nucleus) and repetition-penalty on the existing
      codebook-0 `sample_cb0` path (`apply_top_p`/`apply_repetition_penalty` in
      `pipeline.rs`).
- [x] Implement optional sampling (temperature/top-k/top-p) on the MTP
      residual-codebook decode (`MtpModel::generate_residuals_with` +
      `sample_residual` in `mtp.rs`), defaulting to today's greedy behavior so
      nothing regresses silently. **Scope note**: this covers the
      full-recompute `MtpModel` path, which is what `pipeline::generate_codes`
      (the default `synth`/`clone`/`design` path) actually calls. The
      KV-cached `gen_kv_mtp::CpuMtp` mirror and the NPU `MtpEngine` impls in
      `npu_gen.rs` stay greedy-only for now (documented inline at each call
      site) -- giving them the same independent sampling is follow-up, not
      silently dropped.
- [x] Wire `--top-p --repetition-penalty --residual-temp --residual-top-k
      --residual-top-p` into `tts_cli.rs`'s `parse_common`. Verified live:
      `brain qwen3tts synth --top-p 0.9 --repetition-penalty 1.3
      --residual-temp 0.7 --residual-top-k 20 ...` runs end-to-end and writes
      finite audio.

### Phase 2 - Batched serving

- [x] `run_batch` (`qwen3tts::batch::run_batch`) for the Talker+MTP KV-cache
      decode path. **Scope, stated plainly**: this delivers interleaved
      (round-robin, one frame per request per round) scheduling over the CPU
      `CpuTalker`/`CpuMtp` engine, genuinely ragged (each request's own EOS/
      `max_frames` ends its rotation independently, proven by a 1-frame
      request finishing while a 6-frame one keeps going) - it is NOT a
      single batched GPU matmul across requests (`b>1` in every `Gqa`/`Step`
      this crate's GPU engine builds is a kernel-shape change, out of scope
      here) and does NOT reuse `crates/model`'s qwen3/qwen35 paged-KV
      scheduler (that scheduler is built around GPU-resident paged KV
      blocks; this is a CPU host-side round-robin over independent
      `CpuTalker`/`CpuMtp` instances - a smaller, different mechanism, not
      the "reuse, don't invent a second one" the earlier draft of this plan
      called for. Revisit if/when this needs to run on the GPU path). Each
      request also still reloads its own weights from disk rather than
      sharing one read-only set across the batch - a real, separate
      optimization not attempted here either.
      Tested (checkpoint-free, synthetic Talker+MTP): batched output for
      each request matches that SAME request run alone through
      `pipeline::generate_codes_cached`, bit-for-bit, at **sampling**
      temperature (0.8, not greedy) - a stronger bar than the originally
      planned "match at temperature=0" (greedy would not have caught
      cross-session RNG contamination; sampling does, and there wasn't any).
      A second test proves a 1-frame request drops out of rotation
      immediately while a 6-frame request in the same batch keeps running.
- [ ] Wire `run_batch` into an actual entry point. Not done this session:
      `tts_serve.rs`'s executor is built around the NPU `serve::TtsEngine` /
      `KvTalker` (OpenVINO), a DIFFERENT engine than the CPU
      `CpuTalker`/`CpuMtp` `run_batch` drives - and OpenVINO itself is
      absent on this box (see the Phase 3 streaming entry below), so
      wiring this in and testing it live aren't both possible here. The
      primitive is real and tested; production wiring (a CPU-side batched
      server entry point, or porting the interleaving idea to the NPU
      engine) is the next step.

### Phase 3 - Streaming (cheap half now, hard half later)

Two genuinely different gaps, don't conflate them:

- [ ] **Cheap, but NOT measurable on this box**: the `windowed`
      codec-during-generation path (`generate_codes_kv_streaming`,
      interleaves codec decode with Talker generation) already exists but is
      opt-in (`BRAIN_QWEN3TTS_CODEC=windowed`) -- the default
      `npu-stream`/`cpu-stream` paths generate the ENTIRE code sequence
      before streaming the codec decode. `brain qwen3tts serve` (the only
      entry point that exercises any of `npu-stream`/`cpu-stream`/`windowed`
      -- `synth`/`clone`/`design` don't run through `serve::TtsEngine` at
      all) needs the OpenVINO runtime regardless of target device
      (`NpuDevice::Cpu` still dlopens `libopenvino.so`, it just targets
      OpenVINO's CPU plugin instead of NPU silicon). Checked on this box:
      `libopenvino.so` itself is entirely absent (only two leftover
      NPU-compiler-loader shim libraries are registered in `ldconfig`, no
      Python `openvino` module either) -- `brain qwen3tts serve
      --design-... ` hangs past 40s with zero output even with
      `BRAIN_QWEN3TTS_NPU_DEVICE=cpu`. This isn't the NPU-firmware gap
      tracked elsewhere ([[brain-npu-container-blocked]]) -- it's one level
      more basic, the runtime library is missing outright. Measure
      `windowed`'s real time-to-first-audio vs default on a box with a real
      OpenVINO install, then make it the default if it wins (or expose it as
      a documented, tested `--stream-codec` flag either way).

      **Update (2026-09-04)**: installed the OpenVINO runtime on this box
      (`pip install openvino`, per `scripts/build/setup-npu-runtime.sh`,
      plus the `libze_intel_vpu.so.1` compat symlink
      [[brain-npu-container-blocked]] already root-caused). It now loads
      cleanly and reports `['CPU', 'GPU']` -- `libopenvino.so` is no longer
      the blocker. `NPU` is still absent from that list, which IS the
      already-tracked firmware gap: confirmed by installing `linux-firmware`
      in this container (does nothing - `request_firmware()` reads the
      HOST's own init mount namespace, never a container's, so a
      container-local firmware install cannot be seen by the already-running
      host kernel) and by the user reloading the `intel_vpu` kernel module
      **on the host itself** - `Core().available_devices` still reports only
      `['CPU', 'GPU']` after that reload. This means either the host's own
      `/lib/firmware/intel/vpu/` genuinely doesn't have the right file for
      this specific NPU (Meteor Lake VPU, PCI `8086:7d1d` per
      [[brain-npu-container-blocked]]), or the reload needs a full container
      restart to pick up a fresh `/dev/accel/accel0` binding (the device
      node's mtime did update at reload time, so the driver did rebind -
      inconclusive either way from inside the container). Real forward-motion
      on the underlying blocker, but the `windowed` TTFA measurement THIS
      item wants is still not runnable here.

      GPU numbers (a DIFFERENT, unrelated device) ARE now available for the
      Talker/MTP decode path, though NOT for `brain qwen3tts serve`'s
      NPU-only streaming engine this item is about - see the "Measured"
      section's CPU vs GPU table above.
- [ ] **Hard, lower priority**: true incremental TEXT streaming (extend the
      input token-by-token while acoustic decode is already running, the
      "dual-track" architecture Qwen's own README claims ~97ms TTFA for).
      Note: Qwen's own official Python wrapper does not fully expose this
      either (`non_streaming_mode=False` is documented as simulating
      streaming input, not true streaming) -- brain is behind the model's
      *architectural* ceiling here but not dramatically behind the normal
      upstream API. Sequence this after Phase 2 (batching benefits more
      requests than streaming latency does for most serving workloads).

### Phase 4 - Capability-surface completeness

`crates/qwen3tts/src/caps.rs`'s generic manifest (`brain caps qwen3tts`, what
D-Bus/HTTP/`brain do` reach) declares exactly one action, `synth`, and
`resident_tts.rs`'s D-Bus surface exposes exactly one, `speak` (env-configured
clone-or-not). Clone, VoiceDesign and CustomVoice are real and complete but
reachable ONLY from the dedicated `tts_cli.rs` -- invisible to anything that
discovers capabilities generically.

- [x] Add `clone`/`design` actions to `caps::manifest()` (params: text, ref
      audio path + optional ref-text for clone; text, instruct, optional
      speaker name for design) and wire them in `caps.rs`'s invoke dispatch.
      Verified two ways: unit tests (validation, missing-weights, and
      only-clone-needs-the-speaker-encoder) plus a real generation through
      `capability::Registry::run` end-to-end against an imported
      0.6B-CustomVoice checkpoint (46080 samples produced). Note:
      `brain qwen3tts <action>` still reaches the DEDICATED CLI
      (`tts_cli.rs`, ARCH_HANDLERS takes priority over the generic
      ARCH_TO_MODEL path for architectures with their own CLI module) - these
      new actions are reachable via D-Bus/HTTP/the `Registry` directly, not a
      new CLI spelling; that's the actual gap this phase closes.
- [x] Extend `resident_tts.rs` past the single `speak` action to match: added
      a `design` action (`qwen3tts::caps::design_spec`, per-call
      `instruct`/`speaker` params, unlike `speak`'s instance-fixed reference
      voice). Unit-tested (dispatch + empty-text rejection on both actions).
- [ ] Consolidate the private `tts serve` socket protocol into the standard
      D-Bus surface (carried over from the previous list); add an example
      D-Bus client under `examples/`. Not done this session - lower urgency
      than the generic-surface gap above (the socket server already serves
      its low-latency-streaming purpose; this is infra hygiene, not a
      functional gap).

### Phase 5 - Official-style full fine-tune

- [x] A full (non-LoRA) single-speaker SFT path -- `qwen3tts::sft::finetune_full`,
      alongside (not replacing) the existing LoRA path. Both share one
      `run_finetune` training loop (extracted from the old `finetune_lora`
      body verbatim) -- the two modes only differ in how `cfg`/`init` are
      built (LoRA-extended config with the base frozen, vs. the base config
      with every tensor trainable, `BRAIN_OFFLOAD_ADAM` set for the call
      matching `qwen3::finetune::Mode::FullOffload`'s convention). `brain
      qwen3tts finetune --full` selects it (default stays LoRA). Tested with
      a synthetic checkpoint + dataset (no real Qwen3-TTS weights needed):
      full fine-tuning measurably moves a base attention weight tensor,
      LoRA leaves the same tensor bit-for-bit identical -- the actual
      contract the two modes exist to provide.
- [ ] From-scratch codec/speaker-encoder training: `mimi::recon` ("Track C")
      exists but isn't wired to a CLI/spec path -- lowest priority, largest
      scope, only pursue if a concrete need shows up.

### Phase 6 - Parity/gradcheck completeness

- [x] `crates/qwen3tts/tests/parity.rs` (Talker golden-logit parity) and
      `tests/talker.rs` (Talker gradcheck) both exist and are real; the
      parity test used to be gated on a PyTorch golden dump
      (`testdata/tts/dumps/talker_ref/{tokens.u32,logits.f32}`) that had
      never been generated on this box, with no dump-generation script
      in-tree (unlike e.g. deepseek-ocr's `tools/goldens/*` convention) - so
      the gate only ever took its "not present" skip, and a skipped test is
      green. `tools/goldens/qwen3tts_dump_talker_reference.py` closes that:
      it drives the upstream `qwen-tts` 0.1.1 package's own
      `Qwen3TTSTalkerModel` (under the `transformers==4.57.3` pin
      `qwen3tts_ref.bootstrap` installs into a private directory; no
      `trust_remote_code`, the published checkpoint carries no remote
      modelling code) against the real `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
      weights and reproduces exactly the boundary the Rust side implements -
      `codec_head(model(inputs_embeds=codec_embedding(ids)))`, no text
      projection, no MTP, no codec, no prompt assembly (`Qwen3TTSTalkerModel.
      forward` has no `embed_tokens` - it's never set - so ids must be
      resolved through `codec_embedding` and passed as `inputs_embeds`, the
      same way the reference's own generation path does it). Weights are
      cast bf16 to fp32 and run fp32 on the CPU, matching what brain's
      importer does, so the expected agreement is fp32-tight, not a dtype
      allowance. Evidence, `BRAIN_DEVICE=cpu cargo test --release -p
      brain-qwen3tts --test parity`: `talker parity: max_abs=0.0005
      top1=64/64`, `test result: ok. 1 passed`. The dumper also re-derives,
      on every run, the M-RoPE claim `talker.rs` rests on - with all three
      mrope sections carrying the same position index,
      `apply_multimodal_rotary_pos_emb` equals the single-section
      half-split rotation brain applies, measured max abs 0.0 against the
      reference's own functions, so this is now proven per-run rather than
      just asserted in a comment. The token sequence is deterministic and
      synthetic (`codec_bos_id` then a seeded LCG over the acoustic range)
      rather than codec output: isolating `codec_embedding(cb0)` already
      puts the input rows off the production distribution (production sums
      sixteen per-codebook embeddings per frame), so what remains to
      measure is decoder and head numerics, and a seeded sequence
      reproduces byte-identically with no clip and no network. The dump
      itself is not committed - `testdata/` is gitignored, only the
      generator is. `tests/talker.rs`'s own gradcheck was already real and
      unaffected by this - the two tests check different things (golden
      logits vs. analytic-vs-finite-difference gradients) and both now
      genuinely run.
- [ ] The MTP code predictor has NO equivalent parity or gradcheck test at
      all (`tests/mtp.rs` is a forward smoke test only, checkpoint-gated, no
      golden comparison). **Checked this session and found the real reason
      it's missing, not just an oversight**: `MtpModel` has no backward pass
      at all (`mtp.rs`'s own module doc: "an inference forward (no
      backward) -- the Talker decoder carries the gradient-checked block
      coverage"). The Talker's gradcheck test (`tests/talker.rs`) works
      because `TalkerModel::new_trainable` exists with a full
      forward+backward+`gradcheck::CheckModel` implementation; mirroring it
      for MTP needs an equivalent `MtpModel::new_trainable` FIRST (a real,
      separate prerequisite - implementing MTP backward through
      `model::block`'s shared builders - not just writing the test file
      once that exists). Left open, correctly scoped as "add MTP backward,
      then the gradcheck test" rather than attempted as a quick mirror.
- [x] A cheap end-to-end quality signal that doesn't need a PyTorch reference:
      `crates/qwen3tts/tests/asr_roundtrip.rs` round-trips `pipeline::synth`
      output through `nemotronasr` (real checkpoint,
      `nvidia/nemotron-3.5-asr-streaming-0.6b`, ~2.4 GB, fetched for this
      audit) and asserts word error rate against the input text is below
      0.5. Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT`/
      `BRAIN_NEMOTRONASR` all being set and present, same convention as
      every other real-checkpoint test here.

      **It found a real bug on its first real run.** `synth("The quick
      brown fox jumps over the lazy dog.", seed=0, temperature=0.9,
      top_k=50, max_frames=200)` against the real `0.6B-Base` checkpoint
      produced 200 frames of genuinely near-total silence (confirmed by
      direct sample inspection at `--max-frames 30`: max |sample| ≈
      3.05e-5, RMS ≈ 1.27e-7, silent from frame 0, not a late decay or an
      early-EOS-then-padding artifact) - despite sampling being active,
      which the existing `GenOpts::default()` doc comment already
      identifies as the fix for the OTHER known collapse mode (greedy,
      `temperature=0`). This is a DIFFERENT, previously undocumented
      failure mode: sampled decode can still collapse to near-silence for
      some (text, seed) pairs. Reproduces reliably at `seed=0`; whether
      it's seed-specific or text-specific is NOT yet determined (a
      `seed=1` repro attempt was started but not completed before this
      audit ended - a real open item, not a claim either way). Not
      root-caused or fixed this session - flagged here as a genuine,
      reproducible gap the test now exists specifically to catch, exactly
      the kind of thing this test was written for. **Priority**: this
      likely deserves attention before the residual-codebook/repetition-
      penalty knobs added in Phase 1 above get real-world use, since a
      silent-collapse failure mode undermines confidence in ANY sampling
      configuration, not just the default one.

### Carried over, unchanged priority

- [ ] RMSNorm backward: coalesce to `rmsnorm_rows` (measured 11.2x forward;
      backward still per-element)
- [ ] Cancellation support for in-flight synth/clone requests
- [ ] A windowed attention mask in the codec for long-form decode beyond the
      current fixed window
- [ ] A fused single-inference MTP graph
- [ ] Wire a real TTS model into the runtime event/state-machine flow
      (currently only a stub model is wired there)
