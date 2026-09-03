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

- [ ] Write the spec test first: extend `GenOpts` with `top_p`,
      `repetition_penalty`, and an optional `residual: Option<GenOpts>` (or
      equivalent) for independent MTP-codebook sampling; a test asserting two
      generations with different `residual` settings diverge in their
      residual codes while codebook-0 stays identical (fixed seed).
- [ ] Implement top-p (nucleus) and repetition-penalty on the existing
      codebook-0 `sample_cb0` path.
- [ ] Implement optional sampling (temperature/top-k/top-p) on the MTP
      residual-codebook decode, defaulting to today's greedy behavior so
      nothing regresses silently.
- [ ] Wire `--top-p --repetition-penalty` (and the residual equivalents) into
      `tts_cli.rs`'s `parse_common`.

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

- [ ] **Cheap**: the `windowed` codec-during-generation path
      (`generate_codes_kv_streaming`, interleaves codec decode with Talker
      generation) already exists but is opt-in
      (`BRAIN_QWEN3TTS_CODEC=windowed`) -- the default `npu-stream`/`cpu-stream`
      paths generate the ENTIRE code sequence before streaming the codec
      decode. Measure `windowed`'s real time-to-first-audio vs default on
      this box's hardware, then make it the default if it wins (or expose it
      as a documented, tested `--stream-codec` flag either way).
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

- [ ] Add `clone`/`design` actions to `caps::manifest()` (params: text, ref
      audio path + optional ref-text for clone; text, instruct, optional
      speaker name for design) and wire them in `caps.rs`'s invoke dispatch.
- [ ] Extend `resident_tts.rs` past the single `speak` action to match.
- [ ] Consolidate the private `tts serve` socket protocol into the standard
      D-Bus surface (carried over from the previous list); add an example
      D-Bus client under `examples/`.

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
