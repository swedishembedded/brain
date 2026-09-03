# qwen3tts - roadmap

Qwen3-TTS voice synthesis stack (Talker + MTP + Mimi-style codec + ECAPA
speaker encoder): speaker-free synthesis, voice cloning (x-vector and
in-context), instruct-style voice design, NPU/CPU/GPU backends, and LoRA
fine-tuning. Parity against the reference is verified.

## Not yet done

- [ ] The RMSNorm backward is still the per-element kernel; only the forward
      selects the coalesced `rmsnorm_rows` (measured 11.2x on both the Talker's
      per-token norms and the MTP's per-frame norms)
- [ ] Cancellation support for in-flight synth/clone requests
- [ ] Batched inference (`run_batch`) - only sequential single-request
      inference exists; autoregressive decode makes a genuine batched
      forward nontrivial
- [ ] Consolidate the private socket-based serving side-channel into the
      standard D-Bus serving surface
- [ ] An example D-Bus client for TTS
- [ ] A windowed attention mask in the codec for long-form decode beyond the
      current fixed window
- [ ] A fused single-inference MTP graph - MTP and the codec are now the
      dominant per-clip cost after the Talker path was optimized
- [ ] From-scratch training for the codec and speaker encoder (only Talker
      LoRA fine-tuning exists today)
- [ ] Wire a real TTS model into the runtime event/state-machine flow
      (currently only a stub model is wired there)

CPU codec decoding is computationally sub-real-time for this architecture;
the NPU backend is the only realtime synthesis path today.

## Measured (2026-09-03, CPU-only: Intel Arc iGPU box, no discrete GPU, no
## NPU firmware present, wgsl-cpu Cranelift JIT backend, 22 threads)

All runs are the default (non-NPU, non-`--device`) `synth`/`clone`/`design`
path (`pipeline::generate_codes`: KV-cached Talker, full-recompute MTP),
`temperature=0.9 top_k=50` (the reference's own defaults), wall-clock `time`
including process startup + checkpoint load (not steady-state decode alone).
RTF = wall-clock seconds / output-audio seconds (a from-scratch WGSL engine
timing itself, so this is real elapsed time on real hardware, not a vendor
benchmark).

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

- [ ] `crates/arch/src/lib.rs`'s `qwen3tts` entry declares
      `weights_env: &[("BRAIN_QWEN3TTS_WEIGHTS", "weights_dir"), ...]`, but
      `resolve::flag_twin` derives the CLI flag from the env var's suffix
      (`WEIGHTS` → `--weights`), not from the tuple's own second field. The
      real flag is `--weights-dir`. Result: `weights_already_named` never
      matches, so `brain qwen3tts synth/clone/design --weights-dir D --ckpt
      C ...` ALWAYS falls through to `supply::ensure_env_weights` and either
      hard-errors ("not pulled") or silently starts a multi-GB network fetch
      of the default ref, even though the user named both paths explicitly.
      Every invocation in this audit needed `BRAIN_QWEN3TTS_WEIGHTS`/
      `BRAIN_QWEN3TTS_CKPT` set as env vars instead of the documented flags
      to work around this. Fix `flag_twin` to use the tuple's own flag name
      (fixes this for every future architecture with a flag that doesn't
      match its env-var suffix, not just this one) or special-case
      `weights_dir`. Add a resolver test asserting `--weights-dir X --ckpt Y`
      alone satisfies `weights_already_named("qwen3tts", ...)`.
- [ ] `prompt.rs`'s module doc still says "brain's codec is decode-only... no
      encoder in-tree" -- false since the encoder landed (`mimi::Codec::encode`,
      called from `pipeline.rs`'s `clone()`/`clone_npu()`). Fix the comment;
      no behavior change.

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

- [ ] `run_batch` for the Talker+MTP KV-cache decode path. The hard part is
      genuine: autoregressive decode with per-request finish times means a
      ragged batch, not a fixed-shape one. Reuse the paged/continuous-batching
      pattern already proven for `qwen3`/`qwen35` (`crates/model`'s KV-cache
      scheduler) rather than inventing a second one -- see
      [[brain-evolve-core-for-models]] convention: hoist, don't copy.
      Write the batching invariance test FIRST (batched vs sequential output
      must match bit-for-bit at temperature=0, same seed) before touching
      `tts_serve.rs`'s executor.
- [ ] Replace `tts_serve.rs`'s single FIFO executor thread with the batched
      scheduler once `run_batch` exists.

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

- [ ] A full (non-LoRA) single-speaker SFT path -- unfreeze the base Talker,
      matching Qwen's own documented single-speaker fine-tuning workflow -- alongside (not replacing) the existing LoRA path, which stays the
      lighter-weight option. Reuse `qwen3`'s full-finetune plumbing rather
      than writing new training-loop code.
- [ ] From-scratch codec/speaker-encoder training: `mimi::recon` ("Track C")
      exists but isn't wired to a CLI/spec path -- lowest priority, largest
      scope, only pursue if a concrete need shows up.

### Phase 6 - Parity/gradcheck completeness

- [ ] `crates/qwen3tts/tests/parity.rs` (Talker golden-logit parity) and
      `tests/talker.rs` (Talker gradcheck) both exist and are real, but the
      parity test is gated on a PyTorch golden dump
      (`testdata/tts/dumps/talker_ref/{tokens.u32,logits.f32}`) that has never
      been generated on this box -- there is no dump-generation script in-tree
      today (unlike e.g. deepseek-ocr's `tools/goldens/*` convention). Write
      one (needs a `transformers`-based Qwen3-TTS reference run) so
      `make fetch/testdata` can produce it and the gate stops silently
      skipping.
- [ ] The MTP code predictor has NO equivalent parity or gradcheck test at
      all (`tests/mtp.rs` is a forward smoke test only, checkpoint-gated, no
      golden comparison). Add one, mirroring the Talker's.
- [ ] A cheap end-to-end quality signal that doesn't need a PyTorch reference:
      round-trip synth/clone output through brain's own ASR
      (`nemotronasr`/`qwen3asr`, already in-tree) and assert the transcribed
      text matches the input text above a WER threshold. Catches gross
      regressions (garbled audio, wrong language, silence) that a
      per-tensor logit diff can miss and a human isn't watching for on every
      CI run.

### Carried over, unchanged priority

- [ ] RMSNorm backward: coalesce to `rmsnorm_rows` (measured 11.2x forward;
      backward still per-element)
- [ ] Cancellation support for in-flight synth/clone requests
- [ ] A windowed attention mask in the codec for long-form decode beyond the
      current fixed window
- [ ] A fused single-inference MTP graph
- [ ] Wire a real TTS model into the runtime event/state-machine flow
      (currently only a stub model is wired there)
