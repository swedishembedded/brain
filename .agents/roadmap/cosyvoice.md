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

## Not yet done

- [ ] Phase 1: golden dumper (`tools/goldens/cosyvoice_dump_reference.py`)
- [ ] Phase 2: shared `crates/audio` infra (mixed-radix FFT for `n_fft=1920`,
      `center=False` STFT, `audio::istft`, a rational resampler, `elu`
      activation, hoisting `fold_weight_norm` out of `minimaxmusic3::vocoder`)
- [ ] Phase 3: `crates/campplus` (ONNX import + forward parity)
- [ ] Phase 4: `crates/s3tokenizer` (FSQ, exact token-id parity)
- [ ] Phase 5: the LM (Qwen2.5-0.5B hosted on `crates/qwen3`, prompt assembly,
      RAS sampling)
- [ ] Phase 6: flow decoder, CosyVoice 2 (UpsampleConformerEncoder + UNet CFM)
- [ ] Phase 7: flow decoder, CosyVoice 3 (DiT CFM)
- [ ] Phase 8: HiFT vocoder (NSF source-filter + ISTFT head)
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
