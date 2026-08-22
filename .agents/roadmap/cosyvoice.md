# cosyvoice - roadmap

CosyVoice (`FunAudioLLM/CosyVoice`): LLM-based streaming zero-shot TTS. Two
released generations, both 0.5B, ported behind ONE architecture id
(`cosyvoice` names the family, not the release - see `crates/arch`'s naming
rule, and `wan`'s row for the precedent):

- **CosyVoice 2** (`FunAudioLLM/CosyVoice2-0.5B`) - `Qwen2LM` speech-token LM,
  `CausalMaskedDiffWithXvec` flow (an `UpsampleConformerEncoder` front end +
  `CausalConditionalDecoder` UNet CFM estimator), `HiFTGenerator` vocoder.
- **CosyVoice 3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) - `CosyVoice3LM`
  (same Qwen2.5-0.5B backbone, wider special-token tail), `CausalMaskedDiffWithDiT`
  flow (no encoder, a bare `PreLookaheadLayer` + a 22-layer adaLN-zero `DiT`
  CFM estimator), `CausalHiFTGenerator` vocoder (causal convs).

Two supporting architectures are separate crates/rows, not CosyVoice roles,
because they are independently useful (mirroring `qwen3tts`/`mimi`/`ecapatdnn`):
`s3tokenizer` (the FSQ supervised-semantic speech tokenizer, v2/v3) and
`campplus` (the CAM++ 192-d x-vector speaker encoder, byte-identical across
both generations).

**There is no public 1.5B checkpoint.** The CosyVoice 3 paper scales the LM
to 1.5B; only 0.5B was released. Since the LM config is a generic Qwen2.5
size (`qwen3::QwenConfig::qwen2(...)`), a 1.5B checkpoint would drop in as a
config change - recorded as an unreachable gap, never claimed as support.

Full architectural detail (exact YAML, layer-by-layer topology, streaming
constants, the FSQ/ISTFT/NSF math) was gathered against the real upstream
source before any Rust was written, per `.agents/rules/porting.md` §0-1; it
informs every phase below rather than being repeated here.

## Validation policy

Per-stage cosine ≥ 0.9999 **and** asserted `rel_l2` (lessons.md #2 - cosine
alone cannot see a dropped scale factor) against real-weight goldens; exact
integer token-id equality for the FSQ tokenizer (not cosine - these are
indices); every gradcheck run on both `backend-wgpu` and `BRAIN_DEVICE=cpu`
(lessons.md #5). This machine has 30 GB RAM and no discrete GPU - real-scale
residency gaps will be recorded honestly (as `minimaxmusic3` did), not
silently skipped.

## Phase 0: names reserved

`crates/arch`'s `ARCHS` table gained three rows (`cosyvoice`, `s3tokenizer`,
`campplus`) before any other code, per porting.md §0. Three new crates
(`brain-cosyvoice`, `brain-s3tokenizer`, `brain-campplus`) registered in the
workspace `Cargo.toml` (`members`, `default-members`, `workspace.dependencies`)
with module-doc-only `lib.rs` stubs stating scope and status. `docs/models/`
stub pages added for all three, in the `docs/models/instantid.md`
"not yet servable" style.

## Phase 1: golden dumper

`tools/goldens/cosyvoice_dump_reference.py`, real CosyVoice2-0.5B weights:
mel, CAM++ x-vector, S3Tokenizer FSQ token ids, LM prefill hidden/logits + a
reseeded real-sampler AR token sequence, flow CFM (conds/mu/embedding/
rand_noise/all 10 Euler steps/mel output), and HiFT (magnitude/phase/
waveform). Every component self-validated internally (an independent numpy
mel reimplementation; bit-exact reseed-and-rerun checks for the two
components with global-RNG dependencies). CosyVoice 3 goldens are a
deliberate, recorded follow-up, not covered.

## Phase 2: shared `crates/audio` infra

Mixed-radix FFT (Bluestein, for `n_fft=1920`), `center=False` STFT +
`MelConfig::cosyvoice_24k()`, `audio::istft` (overlap-add + NOLA
normalization - nothing in this workspace computed an inverse STFT before
this), `audio::resample::rational` (Kaiser-windowed-sinc, this workspace's
first from-scratch filter derivation of this kind), `elu`/`elu_bwd` kernels
+ `audio::act`, and hoisting `fold_weight_norm` out of
`minimaxmusic3::vocoder` into `audio::conv` (a previously-duplicated
function, now one implementation).

## Phase 3: `crates/campplus`

CAM++ speaker encoder: a 2D-conv `FCM` stem (verified from the real ONNX
graph node-by-node - which BatchNorms the exporter folds vs. leaves
standalone depends on consumer count, not module type, not assumed) feeding
a D-TDNN with `CAMLayer`'s context-aware masking. Two genuine gaps closed in
`crates/onnx`'s `walk` (`check_conv2d` for independent per-axis
kernel/stride/pad, `check_conv1d` adding dilation) - both pure additions,
`scrfd`/`arcface` (the existing consumers) re-verified unaffected. Forward
parity vs the real checkpoint: cosine 1.0000000000, `rel_l2` 2.3e-6.

## Phase 4: `crates/s3tokenizer`

S3Tokenizer v2 only (v3's 12-layer MinMo encoder is a distinct architecture,
not just a layer-count bump - deferred, recorded below). A strided Conv1d
stem (exact-erf GELU, matching torch's real default) feeding 6 RoPE +
FSMN-memory attention blocks, then the FSQ head as plain elementwise host
arithmetic (round-half-to-even, base-3 place values) - no GPU dispatch, since
the parity gate is exact integer equality. Weights bound positionally by
node name (the exporter kept module-hierarchy names on nodes but dropped
them from most initializers). Centralized `erf`/`gelu_exact` in
`model::hostmath` (previously duplicated independently in `mimi` and
`fastvlm`). Gate: all 87 tokens of the real reference clip match the ONNX
reference exactly.

## Phase 5: the LM (`Qwen2LM`, CosyVoice 2 only)

Hosted on `qwen3::QwenConfig::qwen2_0_5b()` (verified against the real
`CosyVoice-BlankEN/config.json`, matches exactly). Real finding, verified
empirically: `llm.pt` is self-contained, carrying its own independently
fine-tuned Qwen2.5-0.5B backbone that differs from
`CosyVoice-BlankEN/model.safetensors` by up to 0.265 max-abs on
`embed_tokens.weight` - `CosyVoice-BlankEN` supplies only the tokenizer/
config identity, never weights. Prompt assembly and AR decode reproduce
`Qwen2LM.inference()` verbatim via `qwen3::Qwen`'s incremental KV-cache
decode. `ras_sampling` ports the reference's repetition-aware nucleus
sampler algorithm-for-algorithm; the reference's `log_softmax`-then-`softmax`
call chain, checked rather than assumed load-bearing, turns out to be
mathematically inert (softmax is exactly shift-invariant).

Parity: prefill hidden state and `llm_decoder` logits both cosine
1.0000000000. Exact AR-token reproduction against the golden's 32 tokens is
an honest, documented gap, not a faked pass: the reference draws from
torch's own RNG, this port from `data::rng::Rng` - unrelated streams, so
only the sampling *algorithm* matches, not the exact draws. What IS
verified: every generated token is a valid id, no stop id leaks
mid-sequence, generation is deterministic given the same seed.

## Phase 6: flow decoder, CosyVoice 2 (`CausalMaskedDiffWithXvec`)

`UpsampleConformerEncoder` (ESPnet Transformer-XL-style relative-position
attention, ported with no existing brain precedent) feeding
`CausalConditionalDecoder` (a 56-transformer + 14-resnet-block UNet driven
by a 10-step classifier-free-guided Euler ODE solver). Host CPU throughout,
matching `crate::llm`'s own convention. Real finding, verified by reading
the reference line-for-line: the UNet never actually changes resolution
(`channels=[256]` makes every down/up stage `is_last`, so the "downsample"/
"upsample" convs are both stride-1 causal `Conv1d(256,256,3)`) - not assumed
from the class name.

The fixed CFM noise buffer (`torch.manual_seed(0); torch.randn([1,80,15000])`,
a plain attribute never stored in the checkpoint) is reproduced by a
bit-exact Rust port of PyTorch's CPU RNG (`flow::torch_rng`: MT19937 seeding/
tempering plus the AVX2 `normal_fill_16` Box-Muller kernel, including
replicating the compiler's own FMA instruction-fusion effects - found by
comparing GIMPLE dumps of a standalone-compiled `avx_mathfun.h` against a
first naive port that was off by 1 ULP on ~1% of values) - not a checked-in
data asset, which this repo's `no-large-or-binary-files` gate bans outright
for exactly this kind of regenerable buffer.

Parity vs the real `flow.pt`, all three rungs, real-weight, no
hand-assembled inputs: condition assembly (`conds`/`mu`/`embedding`) cosine
1.0000000000; the 10-step Euler loop replayed from the golden's own captured
entry state matches all 10 steps; a full independent from-scratch forward
(this port's own encoder feeding this port's own Euler loop, every fixture
input real and independent) matches the reference mel output. Streaming/
chunked attention is a documented, not-yet-implemented gap.

**Recorded performance gap, not a correctness gap**: this forward is
impractically slow in an unoptimized debug build (a real-weight test run
was still climbing steadily past 69 minutes before being switched to
`--release`, where the same test completes in about 5 minutes) - self-
attention in `flow.rs` is a raw scalar quadruple-nested loop, not yet
dispatched through `model::hostmath`'s AVX2+FMA rayon-parallel matmul path.
A device (WGSL Step-builder) or vectorized-host port is a natural follow-up
performance milestone once the forward is parity-proven, which it now is;
until then, real-weight tests against this component should be run with
`cargo test --release`, not the debug default.

## Phase 7a: CosyVoice 3 golden dumper

`tools/goldens/cosyvoice3_dump_reference.py`, the CosyVoice 3 sibling of
Phase 1's dumper: same self-validation discipline, extended to CV3's real
topology (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`), read line-by-line against
the live reference source rather than assumed from how this milestone was
originally scoped. `campplus.onnx` and `CosyVoice-BlankEN/` are hardlinked
from the CosyVoice 2 fetch rather than re-downloaded, verified byte-identical
by sha256 against the CV3 repo's own reported hashes first.

Real findings, several of which corrected the original scoping brief rather
than just confirming it:

- **CV3's mel front end is NOT unchanged from CV2**: `fmax` is `null`
  (librosa's `sr/2` default), not `8000` - caught by the dumper's own
  independent-mel self-validation failing at cosine 0.9734 with the CV2
  config copied over verbatim, fixed to 0.9999999 once corrected. Any Rust
  mel config for CV3 needs its own `fmax`, not `MelConfig::cosyvoice_24k()`
  as-is.
- `CosyVoice3LM`'s `sos`/`task_id`/`eos`/`fill` all read from
  `speech_embedding` (a `speech_token_size + 200`-wide table), never a
  separate `llm_embedding` table; `llm_decoder` projects to that same +200
  width (6761, not CosyVoice 2's 6564).
- `<|endofprompt|>` (151646) is empirically stable: `CosyVoice-BlankEN`'s
  base tokenizer has exactly 3 added tokens (151643-5), so
  `add_special_tokens` deterministically assigns 151646 next - matching the
  reference's own hardcoded assert on that id being present in the prompt.
- `CausalMaskedDiffWithDiT` has no encoder at all (not a config toggle) -
  condition assembly is bare `pre_lookahead_layer` + `repeat_interleave`
  (25 Hz tokens to 50 Hz, "the simple interpolation operation" the CosyVoice
  3 paper describes replacing the conformer encoder with).
- `CausalHiFTGenerator.inference()` has no `cache_source` parameter, a real
  signature difference from CosyVoice 2's `HiFTGenerator.inference`.
- **The HiFT RNG story genuinely differs from CosyVoice 2**: `SineGen2
  (causal=True)` in eval mode reads FIXED buffers drawn once at `__init__`
  (never checkpointed), not fresh per call - so two `inference()` calls on
  the same model instance are bit-exact without reseeding between them,
  proven by this dumper's own self-validation. Reproducibility hinges on the
  global RNG state at model-construction time, not at each inference call
  the way CosyVoice 2's HiFT needs.

34 files, 8.4 MB under `testdata/golden/cosyvoice3/` (gitignored): mel front
end, CAM++ x-vector, S3Tokenizer v3 FSQ tokens (87 tokens, exact-match
reseed check), CosyVoice3LM prefill hidden/logits + 32 reseeded AR tokens,
the DiT flow decoder (conds/mu/embedding/all 10 Euler steps/mel output, plus
DiT-internal `InputEmbedding`/`TimestepEmbedding` taps with independent-
recompute self-validation), and `CausalHiFTGenerator` magnitude/phase/
waveform. Every planned component dumped, every self-validation check
passing on a real run - no gaps.

## Phase 7b: CosyVoice 3 model port (LM, DiT flow, causal HiFT)

Three self-contained commits, each buildable and tested independently, using
Phase 7a's goldens as the parity oracle:

**`CosyVoice3LM`**: `CosyVoiceLm` (`crate::llm`) now hosts both generations
behind `CosyVoiceLmConfig::special_token_source` - `LlmEmbedding` (CosyVoice
2's dedicated `llm_embedding` table) or `SpeechEmbedding` (CosyVoice 3's
`sos`/`task_id` rows inside `speech_embedding` itself, no `llm_embedding`
table, bias-free `llm_decoder`). `stop_token_ids` widened from a fixed 3-id
array to a `Range<u32>` to cover CosyVoice 3's 200-entry special-token block.
Parity vs the real `llm.pt`: prefill hidden state and logits both cosine
**1.0000000000**.

**`CausalMaskedDiffWithDiT`**: no encoder at all - condition assembly is a
bare `PreLookaheadLayer(80->1024)` + `repeat_interleave` feeding a 22-layer
adaLN-zero `DiT` (`dim=1024, heads=16, dim_head=64, ff_mult=2`).
`crate::flow`'s condition assembly, cosine `t_scheduler`, causal
conv/mish/leaky_relu primitives, and the CFM noise buffer (`flow::torch_rng`)
are widened to `pub(crate)` and reused verbatim - the Euler loop and noise
are identical regardless of which estimator sits inside. **Real, non-obvious
finding**: `x_transformers`'s `RotaryEmbedding` rotates the query/key row
BEFORE the per-head reshape, so only the first `dim_head` channels ever get
rotated - heads 1-15 pass through unrotated. Caught by a full-forward
divergence (cosine ~0.993 with textbook per-head RoPE) despite every other
sub-stage tap matching exactly; reproducing the quirk verbatim closes it to
cosine 0.9999999997. Parity vs the real `flow.pt`: conds/mu/embedding and
DiT-internal taps at cosine **1.0000000000**; the full 10-step Euler loop and
an independent re-forward to mel at cosine **>= 0.9999999997**.

**`CausalHiFTGenerator`**: reuses `HiFTGenerator`'s exact topology, but every
conv is one-sided causal - `conv_pre` right-looking (kernel
`conv_pre_look_right+1=5`), `ups[i]` nearest-upsample + left-causal `Conv1d`
(`CausalConv1dUpsample`) rather than `ConvTranspose1d`, confirmed against the
real `hift.pt`'s own tensor shapes (`ups.0`'s weight is `(256,512,16)` =
`[Cout,Cin,K]`, a plain `Conv1d`'s layout, not `ConvTranspose1d`'s
`[Cin,Cout,K]`) - so `weight_norm`'s `dim=0` convention is `Cout` here for
every conv including `ups[i]`, the one case CosyVoice 2 needs `Cin` for.
**Real, empirically-caught finding**: `SineGen2`'s phase-upsample
interpolation mode is `"nearest"` under `causal=True`, not `"linear"` - a
first port assumed CosyVoice 2's linear mode for both generations, matching
CosyVoice 2's own tests while breaking CosyVoice 3 sharply (cosine ~0.28)
partway through the signal; `nsf_source_forward` is now a shared generic
selected by `nsf_source_forward`/`nsf_source_forward_causal`. CosyVoice 3's
RNG story is simpler than CosyVoice 2's: `SineGen2(causal=True)` reads a
buffer fixed once at construction rather than per call, modeled by
`Cv3HiftInstance` drawing its noise once and reusing it across every
`forward()`. Parity vs the real `hift.pt`: magnitude/phase/waveform at
cosine **>= 0.9999998**, the residual attributed to a known cause (the
reference upcasts `f0_predictor` to `float64` for causal inference; this
port stays `f32` throughout, reported rather than hidden by
`cv3_hift_parity.rs`).

All three commits' real-weight tests re-verified together in one run
alongside the unmodified CosyVoice 2 `flow_parity`/`hift_parity` suites
(confirming the shared-code widening introduced no regression): 13/13 tests
passed.

**Not done in this phase**: CosyVoice 3 pipeline reuse - `crate::pipeline`
still wires CosyVoice 2's five components only; composing the DiT flow
decoder, causal HiFT, and `CosyVoice3LM` into a CosyVoice-3
`pipeline::generate()` path is a recorded follow-up, not attempted here.

## Phase 8: HiFT vocoder (`HiFTGenerator`, CosyVoice 2 non-causal only)

`ConvRNNF0Predictor` (despite the name, no RNN) -> NSF harmonic source
excitation (`SourceModuleHnNSF`/`SineGen2`) -> BigVGAN-style conv trunk
(Snake `ResBlock`s, source-fused per upsample stage) -> ISTFT head -> 24 kHz
waveform. Every conv reuses `audio::conv`'s reference kernels,
`audio::snake`/`audio::act` for activations, `audio::istft` for the
STFT/ISTFT pair - no new kernels.

Real, load-bearing gotcha, verified directly against the reference (not
assumed): `SineGen2` draws fresh `torch.rand`/`torch.randn` from PyTorch's
global RNG on every call, so real HiFT output is not reproducible run-to-run
without reseeding. **One empirical finding narrows this further**: the
`rand_ini` draw (initial phase noise) is provably inert at HiFT's real
`upsample_scale=480` - the downsample-interpolation step's first sampled
input index is 239/240, never index 0, the one `rand_ini` perturbs (verified
by running the real reference twice with different seeds and confirming
bit-identical output). The only draw that genuinely reaches the output is
`torch.randn_like(sine_waves)`; the parity suite injects the exact values a
real reseeded run consumed (captured by an ad-hoc, uncommitted script - see
the honest-gap note below) to verify the conv-trunk+NSF-source+ISTFT math
bit-exactly without reimplementing that specific draw in Rust. Production
inference (`hift::forward_seeded`) draws its own noise from `data::rng::Rng`
- the same honest RNG-crossing gap `sampling`'s module doc already
documents for the LM.

Parity vs the real `hift.pt`: magnitude/phase/waveform match the reference
exactly given the same NSF noise; production `forward_seeded` is verified
deterministic and bounded given its own seed. `CausalHiFTGenerator`
(CosyVoice 3, causal convs, no `cache_source` state) was a deliberate
follow-up at the time this phase was written - see Phase 7b, where it was
implemented.

## Phase 9: pipeline (non-streaming), CosyVoice 2

`crates/cosyvoice/src/pipeline.rs` composes all five components - CAM++,
S3Tokenizer, the LM, the flow decoder, HiFT - into one
`pipeline::generate(paths, opts, text, ref_wav_path, ref_text)` call:
zero-shot voice cloning, text + a reference clip in, a real 24 kHz waveform
out, mirroring `CosyVoiceFrontEnd.frontend_zero_shot` +
`CosyVoiceModel.tts(finalize=True)` step for step (resample to 16/24 kHz,
CAM++ x-vector, S3Tokenizer prompt tokens, the 24 kHz prompt mel, the
reference's own `token_len = min(mel_frames/2, num_tokens)` truncation shared
by the LM and the flow decoder, `max_token_text_ratio`/`min_token_text_ratio`
sized off the target text only). Each stage's checkpoint is imported, used,
and dropped in its own scope before the next stage's import runs
(`minimaxmusic3::generate`'s own sequential-stage RAM discipline - `llm.pt`
alone is 2 GB on a 30 GB, no-discrete-GPU box).

Two genuine gaps closed to make this possible, both host-only math, no new
kernel: `audio::kaldi_fbank` (`torchaudio.compliance.kaldi.fbank`-compatible
mel features CAM++'s reference front end needs - ported line-for-line from
that library's own source; a GENUINELY different triangular-filter shape
from `audio::mel`'s existing Hz-domain-linear filters, since Kaldi's own
filter is linear in the MEL domain) and `pipeline::extract_prompt_mel` (the
24 kHz `matcha.utils.audio.mel_spectrogram`-equivalent glue: magnitude
spectrogram -> Slaney mel filter -> log-clamp, composed from
`audio::mel::power_spectrogram`/`mel_filterbank`, which already existed but
had never been driven end to end from raw audio before this milestone).
S3Tokenizer's own Whisper-style mel front end (`audio::asr_frontend::
qwen_logmel`) needed no new code at all - it already matches
`s3tokenizer.log_mel_spectrogram` exactly when called unpadded.

Verification, `crates/cosyvoice/tests/pipeline_e2e.rs`:

- `mel_frontend_matches_the_reference_mel`: `pipeline::extract_prompt_mel`
  against the real `mel_real_*` golden - cosine **0.9999999999**, `rel_l2`
  1.747e-5. This closes the one piece of new glue math that had no prior
  parity check (`audio::mel::power_spectrogram`/`mel_filterbank` were proven
  components; driving them end to end from raw 24 kHz audio into this exact
  magnitude/log-clamp formula had not been).
- `spliced_flow_and_hift_reproduce_the_reference_given_golden_tokens_and_xvec`:
  the composed-pipeline regression check porting.md's parity ladder calls
  "rung 4 with real weights" - the golden's own captured prompt/generated
  speech tokens, x-vector, and prompt mel through THIS crate's `flow::forward`,
  then flow's own mel output straight into `hift::forward` (the ad-hoc
  NSF-noise capture `hift_parity.rs` already documents) - proving the seam
  between two independently-proven components (flow's channel-major mel
  output needs no reshaping to become HiFT's input) still reproduces the
  reference end to end.
- `full_pipeline_produces_a_real_playable_wav_from_real_weights`: the actual
  milestone deliverable - `pipeline::generate()`, this port's OWN sampling,
  against the real reference clip
  (`resources/cosyvoice/source/asset/zero_shot_prompt.wav`), gated
  structurally (finite, bounded to `audio_limit`, non-silent RMS, a plausible
  duration, deterministic given the same seed, and the written WAV round-trips
  through `audio::wav::write`/`read`) rather than against a golden waveform -
  the LM/HiFT RNG-crossing gap (already documented before this milestone)
  makes byte-exact end-to-end parity the wrong gate, not a gap this milestone
  introduces.

A runnable example, `crates/cosyvoice/examples/synth.rs`
(`cargo run -p brain-cosyvoice --release --example synth -- <text> <ref.wav>
<ref transcript> [out.wav] [seed]`) - no `brain caps`/CLI-verb/D-Bus surface
yet, that is Phase 11's job.

`crates/arch/src/lib.rs` gained two new registrations this phase needed:
`campplus`'s row had NO `weights_env` at all despite its own parity test
already treating `BRAIN_CAMPPLUS_DIR` as canonical - now registered, not a
second ad-hoc convention; `cosyvoice`'s row gained a fourth `weights_env` role,
`BRAIN_COSYVOICE_TOKENIZER`, for the `CosyVoice-BlankEN` Qwen BPE identity the
LM's text side needs (tokenizer + config identity only, never weights - see
`crate::llm_import`'s module doc).

**Recorded gap, not silently skipped**: `audio::kaldi_fbank` has no bit-exact
golden to check against in this workspace - CAM++'s own real-weight parity
test reads its fbank input from a captured golden rather than computing it
in Rust, so nothing in this repo has ever run a real
`torchaudio.compliance.kaldi.fbank` and compared it against this port's
output. The pipeline's x-vector is therefore structurally, not numerically,
verified against the reference. A from-scratch capture + parity test
(mirroring `hift_parity.rs`'s own ad-hoc NSF-noise capture) is a recorded
follow-up, not attempted here.

Streaming (chunked `token2wav`, growing-prefix flow re-run, Hamming
cross-fade, `token_hop_len`/`token_overlap_len`/`mel_cache_len`/
`source_cache_len`) was evaluated as a stretch goal for this phase and
deliberately NOT attempted: the non-streaming path's own verification (above)
consumed the available time, and a streaming implementation is exactly the
kind of "gate that lies" risk this repo's culture warns about if rushed -
better a clearly-scoped follow-up than a half-verified streaming path.

## Phase 10 (partial): LM training - gradcheck + LoRA + overfit

Scoped to the speech-token LM only; the flow decoder (both CV2's UNet CFM
estimator and CV3's DiT CFM estimator) and the HiFT vocoder (both
generations) remain forward-only - see "Not yet done" below for why and what
a follow-up needs.

**Real finding: `crate::llm::CosyVoiceLm` cannot be trained through
`qwen3::Qwen`'s own training graph at all.** The plan going in was to reuse
`qwen3::Qwen`'s batched `set_batch`/`forward`/`backward` (as `qwen3tts`'s
Talker does) and wire `qwen3::lora` onto it directly. Reading both crates
line by line found this does not fit: `CosyVoiceLm` drives a **decode-only**
`qwen3::Qwen` build (`Qwen::from_tensors_decode`) one row at a time through
`step_embed`, which allocates no backward buffers at all
(`Qwen::run_backward` asserts `!self.decode_only`); and its three bolted-on
tables (`llm_embedding`/`speech_embedding`/`llm_decoder`) live entirely
outside `qwen3::Qwen`'s own parameter set with a different row count than the
backbone's tied `tok.weight`/`lm_head` (151936 real BPE ids vs. a separate
~6564/6761-row speech vocabulary) - `qwen3::Qwen`'s training path assumes ONE
table shared by embedding and head. The nearest seam,
`Qwen::enable_mm_splice`, replaces one CONTIGUOUS row range with externally
supplied embeddings while every other row still comes from the backbone's own
tied table; CosyVoice's `sos ++ text ++ task_id ++ speech` layout needs three
genuinely disjoint row sources feeding one sequence, which does not fit that
seam without changing its contract. `qwen3tts`'s Talker works around none of
this because it does not need to: its own trainable stream is literally a
private `qwen3::Qwen` instance with `vocab` sized to its own codec vocabulary
and `tie_embeddings = false` - no bolted-on tables outside `qwen3::Qwen`'s
own parameter set at all.

**Judgment call**: rather than retrofit `qwen3::Qwen` with new "drive the
transformer body from an externally-assembled embedding, read back a raw
hidden state" surface (invasive on a crate every other decoder-LM
architecture in this workspace depends on), `crates/cosyvoice/src/lmgrad.rs`
is a **fresh, self-contained, `Fp`-generic host reference** of the same
Qwen2-style decoder math (RMSNorm -> biased QKV -> half-split RoPE -> causal
GQA attention -> output projection -> residual -> RMSNorm -> SwiGLU MLP ->
residual, matched op-for-op against `qwen3::Qwen::forward_steps`) plus
CosyVoice's own embedding/head tables and its masked next-speech-token
cross-entropy objective - the same pattern `wan::grad`/`wan::modelgrad`,
`flux2::grad` and `ltxv::grad` already use for a model whose trainable graph
does not fit an existing device-model seam: `f64` is the finite-difference
gradcheck oracle, `f32` is the host trainer, one implementation for both.
LoRA (`crates/cosyvoice/src/lmlora.rs`) follows the same substitution:
`model::lora::Pair`'s host `W_eff = W + (α/r)·B·A` adapter (the SAME
machinery `wan::lora`/`flux2::lora`/`s3dit::lora` build on) targeting
`wq`/`wk`/`wv`/`wo` per layer - the same four projections
`qwen3::LoraCfg::attn` targets by default, `B = 0` at init, so a rank/alpha
choice means the same thing here as it does for any `qwen3`-hosted model in
this workspace, without `qwen3::lora`'s device-adapter machinery being
reachable from this crate's training path.

Gates (`crates/gradcheck/src/lib.rs`'s `check_cosyvoice_lm_block`/
`check_cosyvoice_lm`, `crates/cosyvoice/src/lmlora.rs`'s and
`crates/cosyvoice/tests/lm_overfit.rs`'s own tests), at tiny, deliberately
non-degenerate dims (`n_heads=6 ≠ head_dim=4`, `d_model=20 ≠
n_heads·head_dim=24`), covering BOTH `SpecialTokenSource` branches (CosyVoice
2's dedicated `llm_embedding` table and CosyVoice 3's
`sos`/`task_id`-as-speech_embedding-rows):

- Block-level FD (one decoder layer, every weight tensor plus the input
  adjoint `dx_in`): worst `rel_err` **1.09e-9** against the porting
  playbook's 1e-4 gate.
- Model-level FD (embedding assembly + N layers + final norm + `llm_decoder`
  head + masked cross-entropy): worst `rel_err` **1.92e-6** (CV2) /
  **7.61e-10** (CV3) against the 1e-3 gate.
- Confirmed the gate has teeth: a deliberately-flipped sign in the RoPE
  backward's adjoint failed both checks loudly (`rel_err` up to 1.74) before
  being reverted - the RED half of the TDD cycle, not assumed.
- LoRA exact no-op at init (`applied == base`, bit-for-bit) and measured
  descent (rank 4, 120 Adam steps, loss falls to <90% of its initial value
  with the base provably unmoved) - `crates/cosyvoice/src/lmlora.rs`'s tests.
- Single-example overfit: loss 2.196 -> 0.00054 over 400 Adam steps.
  Batch-of-4 overfit: mean loss 2.224 -> 0.00034 over 500 Adam steps
  (`crates/cosyvoice/tests/lm_overfit.rs`).
- **Both backends**: this module dispatches no `gpu_core` step, no WGSL, no
  `Backend` at all (confirmed by grep) - there is no kernel here for the
  documented `backend-cpu` workgroup-reduction bug class to hide in, so
  `BRAIN_DEVICE=cpu cargo test -p brain-gradcheck --lib cosyvoice` was run and
  produces the identical report, by construction, rather than being skipped.

## Phase 11: serving contract (caps, residency, CLI, D-Bus/HTTP)

Wired the already-working `pipeline::generate` into this repo's generic
model-serving layer: `crates/cosyvoice/src/caps.rs` (`MODEL`, `manifest`,
`resident_manifest`, one `synth` action), `crates/cli/src/
resident_cosyvoice.rs` (`from_env`, `InstanceKey`, `MemCost`, `activate`,
following `resident_minimaxmusic3.rs`'s literal shape), one `catalog.rs`
`ModelEntry` and one `ARCH_TO_MODEL` row in `crates/cli/src/resolve.rs`.
D-Bus/HTTP came for free through the existing `Provider`/`ResidentModel`
dispatch, confirmed by reading it rather than assumed - no D-Bus/HTTP code
needed touching.

**`variant` accepts both generation names, only one runs.** `synth`'s
schema takes `variant="cosyvoice2"` (the default, the only one
`pipeline::generate` actually wires) or `"cosyvoice3"` (accepted so a
client can discover the option, but rejected with a named, typed error
before any weight touches disk - CosyVoice 3 pipeline reuse is still Phase
9's own recorded follow-up, not attempted here).

**Reference audio is a blob, not a server-side path - a real design point,
not a mechanical wrapper.** `pipeline::generate` takes a filesystem path
(`ref_wav_path`) because its only caller until now was a local example
binary; a D-Bus/HTTP caller's reference clip lives on THEIR disk. `synth`
takes it as a `Media::Audio` input blob (matching `qwen3omnimoe`'s
speech-input convention) and bridges to the path-based signature with a
short-lived scratch WAV file, removed on drop whether `generate` succeeds
or fails. **Real, honestly-recorded finding**: this workspace's shared CLI
blob loader (`caps_cli.rs`'s `--in audio=file`, used by every model with an
audio input) downsamples any WAV to a fixed 16 kHz before an action ever
sees it - fine for ASR, but CosyVoice separately resamples its reference
clip to BOTH 16 kHz (CAM++/S3Tokenizer) AND 24 kHz (the prompt mel), so a
clip fed through the plain CLI path has its 24 kHz mel built from
already-band-limited audio (no content above 8 kHz) - a real fidelity
ceiling on `brain cosyvoice synth --in ref_audio=...`, not a correctness
bug. `decode_ref_audio` does not introduce this cap and is not limited by
it: given a raw `Blob` built directly (D-Bus/HTTP, not through that CLI
helper), it honours a WAV container's own sample rate or a raw-PCM blob's
own `meta.sample_rate` - full fidelity is available to any caller that
does not round-trip through the CLI's ASR-shaped blob loader.

**The `MemCost`/sequential-stage-drop tension, investigated and resolved
the same way `minimaxmusic3` already did, not with a new mechanism.**
`residency::ResidentModel::estimate` is meant to describe bytes reserved
for an instance's whole Hot lifetime; `pipeline::generate` holds no
checkpoint open across a whole call at all - each of its four stages
imports, uses, and drops its own weights in its own scope, and the
resident `Instance` itself holds nothing between calls but path strings.
Reporting the sum of all five checkpoints would over-reserve against a
peak that never occurs; reporting near-zero (the real idle footprint)
would under-reserve against the real, if brief, per-stage peak. `estimate`
reports the LARGEST single stage (`llm.pt`, ~2 GB, dwarfing `flow.pt`'s 451
MB and `hift.pt`'s 83 MB) - the same "budget the larger one, not the sum"
call `resident_minimaxmusic3.rs` already made for its own AR/denoise
stages, applied here for the identical reason, documented in
`resident_cosyvoice.rs`'s own module doc rather than re-litigated. One
related, real bug caught while writing the estimator: `CosyVoicePaths`'
six roles may all alias ONE directory (the released "one folder holds
everything" layout `pipeline.rs`'s own doc and `examples/synth.rs`'s
fallback both support) - summing whole-directory sizes per role would have
counted `llm.pt`/`flow.pt`/`hift.pt` five times over in that layout, so the
estimator stats one NAMED file per role instead
(`CosyVoiceResident::file_bytes`), pinned by a test that aliases every role
onto one directory and checks the result is still the single largest file,
not a multiple of it. No known fp32-promotion factor applies here (unlike
`minimaxmusic3`'s measured int8-to-fp32 doubling): `llm_import`/
`flow_import`/`hift_import` decode straight into `Vec<f32>` and
`CosyVoiceLm` builds a decode-only `qwen3::Qwen` over that same `f32`
backbone, so a checkpoint's own file size is the honest number.

**Validated end to end against real weights on this box, via the new CLI
verb** (`brain cosyvoice synth --text ... --ref_text ... --in
ref_audio=resources/cosyvoice/source/asset/zero_shot_prompt.wav --out
audio=out.wav`, release build): a real, playable WAV came out the other
end through the full served path (`resolve` -> `caps_cli::run_do` ->
`cosyvoice::caps::CosyVoiceProvider` -> `pipeline::generate`) - not just
structurally exercised, an actual run.

`examples/tts/cosyvoice_synth.py` follows `examples/musicgen/
generate_song.py`'s exact shape (a streaming `subscribe`, `meta.format ==
"wav"` written straight to disk) with one addition: it sends the
reference clip's raw WAV file bytes as the `ref_audio` input blob rather
than decoded PCM, deliberately demonstrating the full-fidelity path this
phase's own `decode_ref_audio` finding describes.

## Not yet done

- [x] Phase 7b: CosyVoice 3 model code (`CausalMaskedDiffWithDiT`'s DiT
      estimator, `CosyVoice3LM`, `CausalHiFTGenerator`)
- [ ] Phase 9 (CosyVoice 3): pipeline reuse now that Phase 7 has landed
- [ ] Phase 9 (streaming): chunked token2wav, growing-prefix flow re-run, cross-fade
- [ ] Phase 9 (kaldi fbank parity): a real, captured `torchaudio.compliance.kaldi.fbank` golden and a bit-exact gate for `audio::kaldi_fbank`
- [x] Phase 10 (LM): `lmgrad`/`lmlora` host reference, gradcheck (block + model, both `SpecialTokenSource` branches), LoRA no-op-at-init + descent, single/batch overfit
- [ ] Phase 10 (flow decoder, both generations): full fine-tune + LoRA gradcheck for CV2's `CausalConditionalDecoder` UNet CFM estimator and CV3's adaLN-zero DiT CFM estimator. Both are pure host scalar-loop `f32` forward code today (conformer relative-position attention, resnet+transformer UNet stages, partial-rotary DiT blocks) with zero float-type genericity and no backward of any kind - hand-deriving and gradient-checking a correct analytic backward for both (particularly the conformer's relative-position attention and the UNet's resnet/transformer mix) is real, substantial engineering distinct from the LM's, deliberately not attempted in the same pass as the LM so that the LM's own gates could be fully closed rather than three components left half-done.
- [ ] Phase 10 (HiFT vocoder, both generations): full fine-tune baseline (no LoRA precedent applies - it is a conv/ISTFT/NSF-source stack, not an attention model) - same blocker as the flow decoder: pure host `f32` forward only, no genericity, no backward. The NSF harmonic source generator and the ISTFT head are the two hardest pieces to differentiate correctly (neither has a precedent backward anywhere else in this workspace to check conventions against) and were the reason this component was scoped out rather than rushed. No discriminator (MPD/MSD/MRD) exists in this crate at all - GAN-style adversarial fine-tuning would be new architecture, not just a missing backward.
- [x] Phase 11: serving contract (caps, residency, CLI, D-Bus/HTTP)
- [ ] Phase 12: docs + README (not the quickstart, per instruction)
- [ ] Phase 13: NPU export + INT8 PTQ + optimization pass

## Recorded gaps (expected, not yet reached)

- No public CosyVoice 1.5B checkpoint exists to validate against (see above).
- No official llama.cpp/GGUF architecture entry for CosyVoice, `s3tokenizer`,
  or `campplus` - `gguf: None` on all three rows. The one community GGUF
  (`cstr/cosyvoice3-0.5b-2512-GGUF`) splits the model into five per-component
  files with `cosyvoice3.*`-prefixed tensors and is not a single-checkpoint
  import path; a GGUF importer is deferred.
- Exact AR-token / exact-noise-draw reproduction against the reference's
  PyTorch RNG (the LM's `ras_sampling` multinomial draws, HiFT's
  `randn_like(sine_waves)` NSF noise) is a deliberate, honestly-documented
  gap in both `crate::llm`/`crate::sampling` and `crate::hift`'s own module
  docs, for BOTH generations: production inference draws from
  `data::rng::Rng`, not a bit-exact port of torch's generator. The flow
  decoder's fixed CFM noise buffer is the one exception - THAT draw is
  bit-exactly reproduced (`flow::torch_rng`, see Phase 6), shared unchanged
  by CosyVoice 3's DiT estimator, because it is a single fixed constant
  computed once, not a per-request stream. CosyVoice 3's `Cv3HiftInstance`
  noise (fixed at construction, per Phase 7b) is likewise not a bit-exact
  torch-RNG port, only a faithful "drawn once, reused" structural match.
- HiFT's magnitude/phase/waveform parity rung depends on an ad-hoc,
  uncommitted noise capture for BOTH generations
  (`testdata/golden/cosyvoice/hift_real_nsf_noise.f32`,
  `testdata/golden/cosyvoice3/hift_real_nsf_noise.f32`, neither provisioned
  by `make fetch/testdata`) - a box with only the official goldens still
  runs each `hift`/`cv3_hift`'s own import-coverage and tiny-smoke tests,
  but not this specific rung. A proper capture script belongs in
  `tools/goldens/` as a follow-up if this needs to be routinely
  re-verifiable rather than a one-time proof.
- CosyVoice 3's `cv3_hift` runs `f0_predictor` in `f32`; the reference
  upcasts it to `float64` for causal inference specifically. The resulting
  residual (cosine still >= 0.9999998) is reported, not hidden, but closing
  it to CosyVoice 2's exact-match bar would need an `f64` compute path this
  crate does not have.
- The flow decoder's host-CPU forward is slow in a debug build (see Phase 6)
  - a real-weight test against it should use `cargo test --release`.
