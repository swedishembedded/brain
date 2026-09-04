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
- [x] **The MTP code predictor is gradient-checked**, and the prerequisite the
      previous version of this entry correctly identified - `MtpModel` had no
      backward pass at all - is what had to be built to get there. Golden-logit
      PARITY for the MTP is still open, blocked on the same missing PyTorch
      reference dump as the Talker's above; it is not split out as a new item
      because generating that dump answers both at once.

      **The prerequisite, re-confirmed rather than assumed.** No MTP training
      path existed anywhere in the workspace to reuse: `sft::finetune_lora` and
      `sft::finetune_full` both build a `qwen3::Qwen` from a TALKER checkpoint
      and never touch the code predictor, and `sft::MultiCodebookLabels`
      materialises the same-frame residual targets but had no model-side
      consumer. `TalkerModel::new_trainable` is a handful of lines only
      because the Talker decoder IS a `qwen3::Qwen`, which already carries a
      gradient-checked forward/backward; the MTP has no inner model, so this
      is the first backward over any of its wiring.

      **`MtpModel::new_trainable` / `new_trainable_on`** (`crates/qwen3tts/
      src/mtp.rs`) now provide forward + backward + the
      `param_names`/`read_weight`/`write_weight`/`read_grad`/`forward`/
      `zero_grads`/`backward` surface `gradcheck::CheckModel` wants, over all
      four parameter families: the decoder block set (device, `ParamStore`,
      `Role::Trainable`) plus the per-residual `codec_embedding` and `lm_head`
      tables and `small_to_mtp_projection` (host - that is where the served
      forward keeps them, and `write_weight` therefore takes `&mut self`;
      the `RefCell` the checker's `&self` surface wants lives in the test's
      adapter, not in the model every served run pays for).

      **What the backward composes.** The decoder half is the exact adjoint of
      `forward_steps` read bottom-up, out of the same shared `model::block`
      builders the forward uses: `rmsnorm_bwd` (trainable-gain arm, so all
      four per-block norms and the final one get gradients), `gqa_bwd`,
      `rope_bwd`, `swiglu_bwd`, plus a plain `matmul_dx`/`matmul_dw` pair for
      the seven per-layer linears. Nothing there was hand-rolled. The
      genuinely new code is the HOST half - exactly where the MTP differs from
      every decoder already gated here: the `num_code_groups - 1` separate
      per-position output heads (head `i-1` reads decoder position `i` and
      nothing else, so position 0's head gradient is structurally zero), the
      `small_to_mtp_projection` adjoint folded across every position, and the
      codec-embedding row scatter. The loss is `sft::ce_batch` over
      `MultiCodebookLabels`' unshifted same-frame residual targets - the
      aligned multi-codebook CE that already existed for a different purpose,
      now with its first model-side consumer.

      **It gates the PRODUCTION forward, not a training twin.** The trainable
      forward calls the very `assemble` (hence `project_to_hidden`), `hidden`
      and `head_row` a served run calls. An all-device training forward would
      have been easier to write and would have gated only itself - and the one
      bug this area has actually produced (the 2026-09-04 `embedding_dim !=
      d_model` width assertion) lived in `assemble`.

      **A second kernel table, not a longer first one.** `mtp::TRAIN_PIPELINES`
      is `PIPELINES` verbatim plus 12 backward kernels. `only_fwd_ids` keeps
      naming `block::UNREGISTERED` in every backward slot, so an
      inference-built handle still panics rather than dispatching a stand-in,
      and a served MTP does not compile kernels it never runs.
      `train_pipelines_extends_the_inference_table` pins the shared prefix,
      because every forward-slot const indexes both tables and appending to
      one only would silently shift the backward slots under the training tape.

      **`crates/qwen3tts/tests/mtp.rs`** gained three tests alongside the
      existing checkpoint-gated forward smoke:
      `mtp_analytic_grads_match_finite_differences` (the Talker's mirror, run
      at BOTH `MtpConfig::tiny` and `MtpConfig::tiny_projected`, so the 1.7B
      family's projection path is gradient-checked and not just the 0.6B
      identity one), `mtp_projection_grads_match_elementwise_finite_
      differences` (`small_to_mtp_projection` is the one MTP parameter a
      reverse pass folds across the whole sequence, which
      `directional_check`'s own doc records it is measurably blind to a
      *partial* error on), and `every_mtp_parameter_family_receives_gradient`
      (a `zero_grad_params` structural check, exempting only
      `codec_embedding.{i}` for `i >= num_code_groups - 2`, which at this
      sequence length is never fed as an input and is legitimately dead).
      Result: green at the workspace's own `(atol 4e-3, rtol 8e-2)` gate,
      worst relative error **3.42e-2** (`tiny`, `blocks.1.attn.k_norm.weight`)
      and **3.71e-2** (`tiny_projected`, `blocks.0.ln2.weight`).

      **It did not pass first time, and what it caught was a fixture defect,
      not a backward bug.** At the workspace's usual `eps = 5e-3` the
      `tiny_projected` run failed on five tensors, worst
      `small_to_mtp_projection.bias` at analytic -40.3 vs numeric -5.3. An
      eps sweep settled it: entry by entry, the central difference walks
      monotonically ONTO the analytic value (bias entry 14: -2.56 at 1e-2,
      -6.70 at 3e-3, -7.970 at 1e-3, -7.974 at 3e-4, -7.968 at 1e-4, against
      an analytic -7.9675) and stays there - a wrong gradient converges onto a
      different number, so the backward was right and the step was too coarse.
      The root cause was that `MtpConfig::tiny_projected`'s synthetic
      `small_to_mtp_projection` was initialised at the same flat 0.02 std as
      every other tensor. That is the correct scale at the REAL 1.7B shape
      (`1/sqrt(2048) = 0.022`) and 10x too small at a toy `embedding_dim` of
      24, so the toy projection ATTENUATED the residual stream to an rms of
      ~0.05 and left every downstream RMSNorm running at a ~20x gain - a
      miniature that does not behave like the model it stands in for. The
      synthetic builder now scales that one tensor by `1/sqrt(embedding_dim)`;
      the finite-difference step is 1e-3 rather than 5e-3, with the sweep
      table recorded in the test so the choice is evidence, not tuning.
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

- [x] The RMSNorm backward now selects the coalesced `rmsnorm_dx_rows`
      wherever the device runs workgroup reductions, through the SAME
      `model::block::rms_variant` seam the forward already used for
      `rmsnorm_rows` - see "Coalescing the RMSNorm backward" below for what
      was measured, where it wins, and where it does not
- [ ] The narrow-row half of that swap is still open: at the per-head
      QK-norm widths (`head_dim` 64/128) the cooperative backward kernel is
      a wash-to-loss, because its redundant 64-partial fold costs more than
      the reduction it folds once a row is narrower than the workgroup. The
      block norms carry the win today; a `d`-aware variant (fewer lanes per
      row, several rows per workgroup) would close the rest - tracked in
      `kernel-performance.md`
- [x] Cancellation support for in-flight synth/clone requests. The frame loop
      is the interruption point: `pipeline::generate_codes` (the path every
      default `synth`/`clone`/`design` runs), `pipeline::generate_codes_cached`
      (the KV-cached CPU-Talker mirror) and `batch::run_batch` now each poll a
      `capability::CancelToken` once per frame, between the Talker step and the
      next one. Per-frame, deliberately, not finer: a real 0.6B frame is
      hundreds of milliseconds, so a check inside a single Talker step or MTP
      residual fill would buy no measurable latency while threading a token
      through kernel dispatch. The token type is the workspace's existing one
      (`s3dit::finetune`, `s3dit::pipeline::generate`, `supir::pipeline::
      restore` all poll the same `CancelToken` between steps) rather than a new
      TTS-only flag.

      **Partial output is kept, not thrown away.** The two code-level loops
      return `Result<Vec<u32>, pipeline::Cancelled>`, where `Cancelled.partial`
      holds the frames produced before the check tripped, in the same
      `[frames*16]` layout a completed run returns - so a caller can hand it
      straight to `decode_codes` and keep the partial clip, or drop it. The
      wav-level entry points (`synth`/`clone`/`design`) do the latter and
      report `Err("cancelled")`, matching how every other cancellable action in
      the workspace reports an aborted run. `run_batch` takes a per-request
      token (`Vec<(Prompt, GenOpts, CancelToken)>`); a cancelled member leaves
      rotation exactly like one that hit EOS or its frame cap, returning its
      partial codes in its own slot while the rest of the batch runs to full
      length.

      Wired where it actually buys something: `caps.rs`'s three actions and
      `resident_tts.rs`'s `speak`/`design` now forward `inv.cancel`, so a
      D-Bus/HTTP/`brain do` caller that hangs up stops the loop instead of
      waiting out `max_frames`. `tts_cli.rs` passes an unarmed token on
      purpose - it is a foreground one-shot command where Ctrl-C already ends
      the process, so an in-process token would buy nothing there.

      Evidence, `pipeline::cancel_tests` (unit lane, synthetic checkpoints, no
      real weights needed): `a_mid_flight_cancel_stops_generation_early_with_
      partial_codes` calibrates the per-frame cost on the running machine,
      sizes `max_frames` so an uncancelled run would take seconds, starts the
      generation on a second thread, fires the cancel a couple hundred frames
      in, and asserts the run came back near the cancel point with far fewer
      than `max_frames` frames - and that the partial codes are bit-identical
      to the same request run uncancelled with `max_frames` set to exactly that
      many frames (so the partial is a genuine prefix, not a truncated buffer).
      Its own trace line on a heavily loaded box:

          [cancel-test] per_frame=2.514608ms max_frames=2000
                        cancelled at 500ms, returned after 501.9906ms
                        with 266 frames

      i.e. the cancel was observed ~2 ms after it was set (one frame), and 266
      of 2000 frames were produced instead of all 2000 - the loop stopped at
      the cancel, not at the frame cap. Sizing is calibrated rather than
      hardcoded, after a discarded warm-up run, because a cold-cache
      measurement over-estimates the per-frame cost and would size the run
      small enough to finish before the cancel lands.
      `an_already_cancelled_token_produces_no_frames` covers the pre-armed case
      (refused before the prefix is even streamed) and re-runs the same
      request with an unarmed token to prove the check did not become an
      unconditional early return.
      `batch::a_cancelled_batch_member_drops_out_without_truncating_the_others`
      covers the scheduler: the cancelled member yields zero frames, its
      neighbour still completes all 6 and its codes stay bit-identical to
      running it alone. Full `cargo test --release --offline -p brain-qwen3tts
      --lib`: 37 passed, 0 failed, 2 ignored.

      **Known gap, stated rather than papered over**: the NPU paths
      (`synth_npu`/`clone_npu`/`design_npu` -> `npu_gen::generate_codes_npu`/
      `generate_codes_kv`/`generate_codes_kv_streaming`) are NOT cancellable
      and deliberately take no token. Their frame loops drive compiled
      OpenVINO graphs that this box cannot build or exercise (see the NPU
      status above), and offering a token those loops silently ignore would be
      a lying API. The `Mode::Cpu` NPU variant routes through
      `generate_codes_cached` and passes an unarmed token for the same reason.

      Supporting cleanup, done rather than duplicated: the synthetic
      Talker+MTP checkpoint builders that `batch.rs`'s tests carried inline
      moved to `crates/qwen3tts/src/testsupport.rs` (a `#[cfg(test)]` module),
      so the new pipeline tests reuse them instead of adding a second copy of
      the tensor-name/shape knowledge. The temp directories those tests write
      are now owned by a `Scratch` guard that removes them on `Drop`, so a
      failing assertion no longer leaks synthetic checkpoints into the temp
      dir (the old code only cleaned up on the success path).
- [x] A windowed attention mask in the codec for long-form decode beyond the
      current fixed window **(2026-09-04)**. See "The windowed codec mask,
      and where it was actually missing" below for what the window is, which
      two of the codec's THREE copies of that transformer were ignoring it
      (including the one behind the default serve path), and the numbers.
- [ ] The SAME gap, still open, on the ENCODE side: the Mimi encoder
      transformer (`mimi::model::Codec::enc_transformer`) parses
      `encoder_config.sliding_window` (250, confirmed in the released
      `speech_tokenizer/config.json`) and dispatches plain `gqa_fwd`, so
      encoding more than 250 encoder frames - 10 s of audio, at the
      pre-downsample 25 Hz rate, well inside the length of a voice-cloning
      reference clip - over-attends exactly the way decode used to. Left
      unfixed deliberately rather than shipped alongside the decode fix:
      the encode path's only correctness witness is the reference
      code-match golden, whose clip is far shorter than 250 frames, so the
      one-line change cannot be shown red-then-green and would land as an
      unverified edit under an existing 100%-code-match parity claim.
      Needs a >250-frame reference dump first; that dump is the work item.
- [ ] A fused single-inference MTP graph
- [x] **A real TTS model in the runtime event/state-machine flow.**
      `runtime::tts::Qwen3TtsSynthModel` (`crates/runtime/src/tts.rs`)
      implements the controller's `SynthModel` seam for real: it owns a
      validated `TtsPaths` + `GenOpts` and dispatches per request to
      `pipeline::synth` (no `ref_audio`) or `pipeline::clone` (with one -
      x-vector, upgraded to the ICL path when a `ref_text` transcript comes
      too), returning the 24 kHz waveform `AudioStreamPump` slices into
      `audio_chunk` events. Only `FakeSynthModel` existed before, so
      `Synthesizing` had never streamed a real waveform.

      Gated behind an OPTIONAL `qwen3tts` feature on `brain-runtime`
      (matching `brain-cli`'s existing `vulkan-coopmat = ["dep:brain-vulkan"]`
      convention) because the dependency really is the whole stack:
      `brain-qwen3tts` -> `brain-npu` -> `brain-perf` -> `brain-apiserve`
      drags in the serving surface on top of codec/speaker/talker/MTP -
      measured, not assumed: the default `-p brain-runtime` tree is 176
      packages / 25 brain crates, byte-for-byte unchanged by this work; the
      feature adds 83 packages (57 brain crates). `cargo test --release -p
      brain-runtime --lib` (no feature) stays at 10 tests, 0.01s, untouched.

      Selectable from `brain serve --stdio` (a build carrying `brain-cli`'s
      `qwen3tts-synth` feature, forwarding to `brain-runtime/qwen3tts`),
      registered from the SAME env vars the D-Bus resident already reads
      (`BRAIN_QWEN3TTS_WEIGHTS`/`_CKPT`/`_LANG`) - one spelling across both
      serving surfaces, no new `brain serve` flag. The seam's
      `fn synth(&self, req) -> Vec<f32>` has no error channel; a failed
      generation logs to stderr and returns an empty waveform, which the
      controller already treats as "drained" (terminal `done:true`, back to
      `Idle`) - checkpoint validation is eager in `Qwen3TtsSynthModel::load`
      instead.

      Verified against a real 0.6B-Base checkpoint, not just compiled: two
      feature-gated tests in `crates/runtime/src/tts.rs` (24-frame cap,
      `min_new` raised to 16 so an early EOS can't make the assertions
      vacuous) - one asserts finite samples, the expected length, and
      speech-level RMS rather than the near-silent degenerate decode; the
      other drives it through `Controller::feed_event` and asserts the
      `audio_chunk` stream shape. Both produced the same clip - 46080
      samples, 1.92s, exactly the 24-frame budget - at `rms 0.0209 / peak
      0.1274` (voice, not the `rms ~0.004` greedy-collapse silence), as 3
      `audio_chunk` events (two 24000-sample chunks plus the terminal
      `done`).

## Coalescing the RMSNorm backward

**Where the change actually lives.** The Talker's decoder IS `qwen3::Qwen`
verbatim (`talker.rs` builds one; `sft.rs`'s LoRA fine-tune calls
`Qwen::backward` directly), so this crate has no RMSNorm backward of its
own to fix - registering `rmsnorm_dx_rows` in `crates/qwen3`'s pipeline list
and naming that slot in its `KernelIds` is the whole opt-in, and every
Talker gradient inherits it by construction. The other half of the original
bullet does not apply at all: `mtp.rs` builds NO backward graph (its
`only_fwd_ids` leaves every backward slot `UNREGISTERED` on purpose), so the
MTP's per-frame norms were only ever a forward measurement.

**Where the win is.** Measured with `bench_rmsnorm_dx` (release, real
device - an Intel Arc integrated GPU on Vulkan, min-of-8 per dispatch, four
back-to-back dispatches per sample so launch overhead is amortised),
`rmsnorm_dx_rows` against the per-element `rmsnorm_dx`:

| shape family | what dispatches it | speedup |
|---|---|---|
| `rows` 512-2048, `d` 896-5120 | the `d_model`-wide block norms (`ln1`, `ln2`, final) at training width | 2.5x - 6.7x |
| `rows` 1-8, `d` 1024-5120 | the same norms at decode width | 9.9x - 36.1x |
| `rows` 1k-16k, `d` 128 | `attn.q_norm` / `attn.k_norm` | 0.6x - 2.9x |
| `rows` 2k-8k, `d` 64 | a narrower `head_dim` | ~0.5x |

Agreement with the host reference (`model::hostmath::rmsnorm_dx_rows`) is
3e-7 to 2.7e-6 relative across the whole sweep, against the family's 2e-5
gate - the two kernels differ in reduction ORDER, not in math.

**Why the last two rows are not a win, and why that is a width question.**
`rmsnorm_dx_rows` folds its two 64-partial arrays redundantly in all 64
threads to stay inside the CPU JIT's single-top-level-barrier limit. That
fold is a fixed cost per row regardless of `d`; at `d` 5120 each thread has
already done 80 elements of real work and it disappears into them, at `d`
64 each thread did one element and the fold dominates. This is a property
of the row WIDTH, and must not be conflated with the row-COUNT threshold
that was once a real selector bug on this op.

**Whole-pass effect.** `crates/qwen3`'s own `bench_train_p40` train step
(0.6B-shaped, 4 layers, GQA 16/8, `head_dim` 128, b=2 t=256) moved from 483
ms to 457 ms best-of-runs with the registration flipped on and off in place
- a real improvement, but small enough to sit inside this integrated GPU's
run-to-run spread (which was wide here: the box was shared with another
workspace's compile during part of the sampling, and an integrated GPU
shares its memory bandwidth with exactly that). The pass is dominated by
its matmuls in any case. Per-kernel
DEVICE attribution, which would settle the size of the share exactly, is
not available on this box: `BRAIN_PROFILE=1`'s timestamp queries return
uncalibrated values on this Mesa/Intel driver, so only the dispatch COUNTS
from that run are usable - and those do confirm the new kernel is live and
the per-element one is never dispatched. Treat the kernel-level A/B as what
resolves this change and the whole-pass number as what shows it does not
regress.

**What gates it.** `crates/qwen3`'s `rmsnorm_dx_variant_agreement` module
pins the seam against the HOST reference at the exact `(rows, dim)` pairs
this decoder's backward tape dispatches, including the tiny gradcheck
fixture's own sub-workgroup widths. It was mutation-verified RED before
green: dropping one factor of `r` from the `coef` term (a plausible
backward-math slip) produced 1.2e-2 relative error against the 2e-5 gate.
The Talker's own `talker_analytic_grads_match_finite_differences` runs the
whole finite-difference check through the new kernel and is green, as are
the five `brain-gradcheck` checks over the same `qwen3::Qwen` decoder
(`check_qwen`, its LoRA/weighted/M-RoPE variants and `check_qwen2`). None
of these needs a checkpoint - they build synthetic weights from a `tiny()`
config, which is exactly what makes them usable as the gate for a kernel
swap.

## The windowed codec mask, and where it was actually missing (2026-09-04)

**The window.** The codec's `pre_transformer` is Mimi-derived, so its
attention is sliding-window causal, not plain causal: key `j` is masked out
of query `i` once `i - j >= sliding_window`, on every forward call. The
released checkpoint's `speech_tokenizer/config.json` sets
`decoder_config.sliding_window = 72` frames, which at 12.5 Hz is **5.76 s of
audio**. Nothing about this is a cache capacity or a buffer size: it is a
property of the mask, and it applies to a one-shot decode of a 10-minute clip
exactly as much as to a streamed one.

**Where it was missing.** The surprise of this item: the codec's
`pre_transformer` is written out THREE times in this repo, and only one of
the three had the window.

| Implementation | Called by | Mask before this change |
|---|---|---|
| `mimi::Codec::transformer` (WGSL, `block::gqa_fwd_win`) | `qwen3tts::pipeline::decode_codes` - the `synth`/`clone`/`design` CLI path | sliding-window, correct |
| `mimi::decode_stream::front` (pure host loop) | `qwen3tts::serve` with `BRAIN_QWEN3TTS_CODEC=cpu-stream` | **plain causal - `sliding_window` parsed, never applied** |
| `npu::codec_topology`'s `tf_mask` (ONNX initializer) | `qwen3tts::serve`'s **default** path, via `npu_gen::NpuStreamCodec` | **plain causal (`j > i` only)** |

So both *streaming server* paths were wrong, including the default one - and
those are precisely the paths that exist to emit long-form audio. Finding the
third one also changed the shape of the fix: repairing only the host decoder
would have made it disagree with the NPU front on any clip past 72 frames,
turning a shared defect into a backend-dependent waveform. Both had to move
together, which is what `npu_gen::stream_codec_tests::npu_stream_matches_cpu`
(NPU stream vs the CPU reference) would otherwise have started failing on.

Why it survived this long: `decode_stream`'s own real-weights parity test ran
at `t = 16` with the comment *"small T (< sliding_window) so attention is
plain causal"*. Below the window a sliding-window mask and a plain causal mask
are the same object, so the only test that ever compared two of these
implementations was pinned at a length where the defect is invisible by
construction. The lesson is not "add a test" but "a parity test whose input
sits inside the degenerate regime of the thing it is checking proves nothing
about that thing".

**What happened past the window.** Nothing loud. No panic, no assertion, no
wraparound, no truncation, no length or shape change: `decode_streaming`
returned exactly `T * 1920` samples as always. Frames past index 72 simply
attended over context the reference never exposed to them, and the streamed
waveform drifted further from the reference the longer the clip ran. A
silent-divergence failure mode, not a crash.

**Test that caught it** (`crates/mimi/tests/long_form_window.rs`, three
tests, no external checkpoint needed - a structurally complete tiny codec at
`sliding_window = 4`, decoded through both BRAIN implementations on the same
synthetic weights, so the witness is independent rather than a self-check):

| Case | Before | After |
|---|---|---|
| T=3 (inside one window), device vs host max-abs | 5.960e-6 | 5.960e-6 (unchanged) |
| T=40 (ten windows), device vs host max-abs | **2.565e-3** | **7.495e-6** |
| T=40, narrow vs wide window, device path | 2.566e-3 | 2.566e-3 |
| T=40, narrow vs wide window, host path | **0.000e0** | 2.564e-3 |

The last row is the direct statement of the bug: changing `sliding_window`
had *literally no effect* on the host decoder's output. The third row is the
guard against a future "fix" that drops the mask on both sides and passes
the parity rows vacuously. After the change the two paths agree to 7.5e-6,
the same fp noise floor as the in-window case (5.96e-6) - i.e. what is left
is WGSL-vs-host reassociation, not a mask difference.

The ONNX export has its own red-then-green test
(`crates/npu/tests/codec_onnx.rs::codec_onnx_attention_mask_is_sliding_window_causal`):
OpenVINO is absent in this environment so the graph cannot be RUN, but the
mask is a materialized initializer, so the test builds the graph, decodes the
serialized proto back, and asserts the real exported `tf_mask` bytes
element-by-element plus a closed-form live-key count
(`sum_i min(i+1, window)`), which a plain causal mask (`sum_i (i+1)`) cannot
hit. Before the fix it failed at `mask[3,0] must be masked out, got 0`.

**On the real checkpoint, and a measurement worth keeping.** The obvious
real-weights test - decode a clip longer than 72 frames through both brain
implementations and compare - was written first and is the wrong test.
`out/tts-base06/codec.safetensors` at T=144 gives a host-vs-device max-abs of
**3.9e-2** over 276480 samples, against ~2.4e-3 at T=16. That gap is not the
mask: it is ordinary fp divergence amplified by the SEANet stack, where
SnakeBeta is `x + (1/(exp(beta)+eps)) * sin(exp(alpha) * x)^2` and the `sin`
magnifies any input difference by `exp(alpha)` at each of twelve residual
units, then the next upsample stage spreads it. It grows with length whether
or not a mask is involved, so a bound on it measures the amplifier rather
than the thing under test - and it is worth recording because it sets a real
limit on how far a max-abs host-vs-device comparison can be pushed for THIS
architecture, independent of this work item.

`mimi::decode_stream::tests::windowed_parity_vs_codec_on_real_weights`
(opt-in, `--ignored`, needs `BRAIN_MIMI_WEIGHTS`) shrinks the WINDOW instead
of lengthening the clip: real weights and real architecture at
`sliding_window = 4`, T=20, which puts four fifths of the queries under a
truncated key set - a harder mask test than 72 would be at any clip length
worth decoding - while staying inside the length regime where host-vs-device
fp noise is already characterized. Measured:

| T=20, real weights, 38400 samples | max-abs |
|---|---|
| host vs device, `sliding_window = 4` (five windows deep) | 8.110e-3 |
| host vs device, unbounded window (plain causal) | 8.679e-3 |
| effect of narrowing the window, **host** path | **6.819e-1** |
| effect of narrowing the window, **device** path | **6.819e-1** |

Two results matter here. The windowed case agrees very slightly BETTER than
the plain-causal one, so whatever the ~8e-3 is, it is not the mask. And the
window itself moves the waveform by 0.68 on `[-1,1]` audio - a change roughly
80x larger than the two implementations' disagreement - by **the same amount
on both, to four significant figures**. That is the sharpest available
statement that they now implement the same mask: they do not merely agree,
they respond identically to the field that used to move only one of them (and
by 0.000e0 on the other).

That also forced the assertion to be relative rather than absolute. The first
version of this test pinned `< 5e-3`, borrowed from `e2e_parity_vs_codec`'s
~2.4e-3 at a different T and a different code draw, and went red on a correct
implementation - host-vs-device max-abs on this architecture is a property of
the SEANet amplifier and the particular codes, so an absolute ceiling here is
fitting to one draw. The test now asserts that windowing does not make the
two agree any worse than plain causal does, plus a loose order-of-magnitude
sanity cap, plus that a narrow window MOVES both waveforms - the last one
being what neither implementation could satisfy while ignoring the field.

**Short sequences are bit-identical, not merely close.** For `window > i`
the new loop iterates exactly the old key set in exactly the old order, so
the arithmetic is unchanged; the test asserts this as `assert_eq!` on the
whole waveform (decode at `sliding_window = 4` vs `sliding_window = 4096`
for a T=3 clip), on both implementations, rather than as a tolerance.

**Performance.** No regression, and a scaling improvement where the fix
landed. The device path is untouched apart from one host-side `if` per
layer (see below), so the 48 s CPU / 6.3 s GPU `codec.decode` figures in
"Measured" stand. The host path got strictly cheaper: its attention was
`O(T^2 * head_dim)` scored keys and is now `O(T * window * head_dim)`,
capped at 72 keys per query instead of growing with the clip - so the
longer the form, the more the fix saves rather than costs. No new
allocation, no new dispatch, no new kernel, and **no change to
`crates/model/src/block.rs`**: the shared windowed-attention builder
(`block::gqa_fwd_win` over `kernels::GQA_SCORES_WIN`, with its own
independent-oracle tests in `crates/model/tests/gqa_fwd_win.rs`) already
existed and was already what the codec's device path dispatched. The fix
was to make the other two implementations obey the same mask - so nothing
changed in an engine module ~20 other models depend on.

**One extra correctness fix, in all three.** `sliding_window == 0` (what a
hand-built `CodecConfig::default()` carries; `from_json` always yields 72)
was being passed straight to the kernel, where `i - j >= 0` masks *every*
key and `attn_softmax` turns the all-masked row into a uniform distribution
over the whole sequence, future positions included - a non-causal result out
of a config that merely left the field unset. All three implementations now
normalize `0` to "unbounded", so they agree on every config rather than only
on parsed ones.
