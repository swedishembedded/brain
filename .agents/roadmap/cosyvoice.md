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
(CosyVoice 3, causal convs, no `cache_source` state) is a deliberate
follow-up, not implemented.

## Not yet done

- [ ] Phase 7: flow decoder, CosyVoice 3 (DiT CFM)
- [ ] Phase 9: pipeline + streaming (chunked token2wav, cross-fade)
- [ ] Phase 10: training (LM LoRA, flow + vocoder full/LoRA finetune, gradcheck)
- [ ] Phase 11: serving contract (caps, residency, CLI, D-Bus/HTTP)
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
  docs: production inference draws from `data::rng::Rng`, not a bit-exact
  port of torch's generator. The flow decoder's fixed CFM noise buffer is
  the one exception - THAT draw is bit-exactly reproduced (`flow::torch_rng`,
  see Phase 6), because it is a single fixed constant computed once, not a
  per-request stream.
- HiFT's magnitude/phase/waveform parity rung depends on an ad-hoc,
  uncommitted noise capture (`testdata/golden/cosyvoice/hift_real_nsf_noise.f32`,
  not provisioned by `make fetch/testdata`) - a box with only the official
  goldens still runs `hift`'s own import-coverage and tiny-smoke tests, but
  not this specific rung. A proper capture script belongs in
  `tools/goldens/` as a follow-up if this needs to be routinely
  re-verifiable rather than a one-time proof.
- The flow decoder's host-CPU forward is slow in a debug build (see Phase 6)
  - a real-weight test against it should use `cargo test --release`.
