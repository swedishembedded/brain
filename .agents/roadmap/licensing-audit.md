# Licensing & copyright provenance audit - 2026-09-03

Full-repo forensic audit of code and weight/checkpoint provenance across
~90 crates, triggered by a second-hand external analysis alleging brain's
blanket `Apache-2.0` claim conflicted with several model ports being direct
translations of restrictively-licensed upstream source (SUPIR, CodeFormer,
VQGAN, WorldMirror-2, YOLOv8/Ultralytics, FastVLM). That analysis was
**not** independently verified against this repo before being written, and
its central claim on codeformer/vqgan was already visibly wrong (their own
doc-comments say "composition, not re-implementation" over brain's shared
`vae::blocks::Builder`). Nothing from that report was taken on faith; every
crate below was re-checked against real upstream source - local caches
under `resources/` where available, live `WebFetch` of the actual upstream
repo otherwise - before reaching a verdict.

Standing policy from this audit lives in `.agents/rules/licensing.md`.
Weight/checkpoint license inventory lives in `docs/compliance/
third-party-models.md`. Code attributions live in `/NOTICE.md`.

## Headline result

**The external report's six CRITICAL claims did not hold up.** Direct
comparison of brain's Rust against the real upstream source - SUPIR against
its actual cloned repo at `resources/supir/upstream-src/SUPIR`, the other
five via WebFetch of the real upstream files - found no evidence of source
translation in any of them: different variable names, different code
decomposition (a generic config-driven block interpreter rather than a
transcribed per-layer module list), hand-derived analytic backward passes
with no PyTorch-autograd equivalent to translate from, and in several cases
a documented algebraic optimization or bug fix absent from upstream
entirely. `.agents/rules/porting.md`'s stated methodology - facts from
upstream, implementation on brain's own kernel engine, numerical parity
gate - was actually followed for these six and, on sampling across ~50
more crates, for the overwhelming majority of the workspace.

The real findings split into three categories, none of which require
rewriting or removing a model:

1. **Real, permissively-licensed source transcriptions needing
   attribution** - fixed in `/NOTICE.md`. No license conflict: every one is
   MIT or Apache-2.0.
2. **Undocumented restrictive weight/checkpoint licenses** - the larger,
   previously-invisible risk, tracked in `docs/compliance/
   third-party-models.md`.
3. **One weights license that may restrict brain's own code, not just the
   checkpoint** - TimesFM-3's Non-Commercial contract, see below.

## Per-crate verdicts

Legend: **INDEPENDENT** = no evidence of source-text translation, verified
against real upstream. **ATTRIBUTION-NEEDED** = a specific data table or
algorithm recipe genuinely transcribed from a named upstream file, upstream
license permissive - fixed in `/NOTICE.md`.

### Verified CRITICAL leads from the external report - all INDEPENDENT

| Crate | Verdict | Alleged upstream | Real finding |
|---|---|---|---|
| `supir` | INDEPENDENT (high) | Fanghua-Yu/SUPIR (non-commercial) | Compared against the actual cloned repo. Same formulas (required for parity), completely different code shape - graph-recording over `vae::blocks::Builder`, no class hierarchy, a rewritten multiply-add to avoid a kernel upstream never needed. Weights license already documented in `docs/models/supir.md`. |
| `codeformer` | INDEPENDENT (med-high) | sczhou/CodeFormer (S-Lab, non-commercial) | Zero new kernels; reuses `vqgan::model::run_blocks` + shared Transformer step-builders. One 4-line doc-comment pseudocode block reuses upstream's own `tgt`/`tgt2` naming (standard pre-LN-transformer vocabulary, not CodeFormer-original) - reworded, not a code-copying risk. Weights license (S-Lab, non-commercial) not documented in `docs/models/codeformer.md`. |
| `vqgan` | INDEPENDENT (high) | same upstream as codeformer | Generic `Block` enum interpreter, config-driven, shared with `codeformer`. Same weights-license doc gap as codeformer. |
| `worldmirror2` | INDEPENDENT (high) | Tencent HY-World-2.0 (EU-excluded) | Compared two of the most cited classes (GLVControl-analogue trunk, camera head) against the real fetched upstream; same algorithm (required), explicit-buffer-lifetime GPU dispatch graph has no PyTorch analogue at all. Weights license (EU-excluded) already documented in `docs/models/worldmirror2.md`. |
| `yolov8` (`assign.rs`/`loss.rs`) | INDEPENDENT (high) | Ultralytics `TaskAlignedAssigner` (AGPL-3.0) | Fetched real `tal.py`: class-based, tensor-batched, `torch.topk`/`scatter_add_`, OOM-retry path. Brain: a free function, nested `Vec` loops, hand-sorted top-k, none of upstream's method/variable names. TAL is the published TOOD-paper algorithm - Ultralytics' own docstring cites a third-party reference for it. Weights (`yolov8n.pt`, AGPL/Enterprise) not documented in `docs/models/yolov8/readme.md`. |
| `fastvlm` | INDEPENDENT (high) | apple/ml-fastvlm + ml-mobileclip | The FastViTHD tower actually derives from Apple's separate **MIT-licensed** `ml-mobileclip`, not the more restrictive VLM-splice repo. Brain hand-writes analytic backward for every block (upstream relies on autograd) - real, substantial independent-engineering signature. Sharpest weights gap in the whole audit: `apple/FastVLM-0.5B` auto-fetches on a one-command opt-in with zero mention that its weights are Apple's Research-only `LICENSE_MODEL` (commercial use explicitly prohibited). |

### Real, permissive-license attribution gaps - fixed in `/NOTICE.md`

| Crate | Upstream | License | What was transcribed |
|---|---|---|---|
| `qwen3`/`qwen35`'s `gguf_import.rs`, `crates/gguf/src/leaf.rs` | llama.cpp/ggml | MIT | GGUF tensor-name vocabulary (self-admitted in `AGENTS.md`) |
| `crates/checkpoint/src/gguf.rs`, `crates/gguf/src/kquant.rs` | llama.cpp/ggml | MIT | GGUF quantization block layouts + dequant arithmetic (Q4_0...Q6_K, MXFP4, IQ4_NL/XS, TQ1/2_0) - self-admitted for the MXFP4/IQ/TQ family, pinned at llama.cpp commit `d7a2074...`. The earlier Q4_0...Q6_K/legacy K-quant dequant functions match ggml's `ggml-quants.c` line-for-line (confirmed against the cached source at `resources/ggml-ref/ggml-quants.c`) but carry no equivalent pinned-revision comment - only "M8's K-quant work," unpinned. |
| `crates/model/src/yarn.rs` | HuggingFace `transformers` | Apache-2.0 | YaRN RoPE-scaling derivation, self-admitted "ported line-for-line" |
| `crates/zipdepth` (`blocks.rs`, `config.rs`) | fabiotosi92/ZipDepth | MIT | `MODEL_CONFIGS` dimension table, `_pick_groups` (verbatim per brain's own comment), QARepBlock structure |
| `crates/rrdbnet/src/model.rs` | XPixelGroup/BasicSR | Apache-2.0 | RRDBNet/RRDB/ResidualDenseBlock structure. Checked BasicSR's own non-commercial sub-component list (StyleGAN2/DFDNet/etc.) - does not cover `rrdbnet_arch.py`, no contamination |
| `crates/cosyvoice` (`hift.rs`, `flow.rs`, `llm.rs`) | Alibaba CosyVoice + Matcha-TTS | Apache-2.0 + MIT | The one crate whose own doc-comments explicitly disclaim the facts-only methodology ("read directly, not from the paper," "reproduced verbatim," "read line-for-line") - real algorithm-for-algorithm ports, still not verbatim code (brain's own kernel dispatch throughout) |
| `crates/s3tokenizer/src/model.rs` | xingchensong/S3Tokenizer | Apache-2.0 | Softer case - one verbatim method name (`forward_fsmn`) beyond generic vocabulary; full `forward()` diff not completed (unverified depth) |
| `crates/kronos/src/decoder.rs` | shiyu-coder/Kronos | **MIT** (not Apache-2.0 - genuine license-family mismatch, now recorded) | `decode_s1`/`decode_s2` public API naming closer than checkpoint-loading strictly requires; `docs/models/kronos.md` was the only one of its four sibling forecasting-model pages missing a license section |
| `crates/gpt2` | karpathy/nanoGPT | MIT | Defensive/courtesy credit only - "nanogpt-parity" self-description, no copying found |

### Weights-license findings (code found clean; checkpoint license is the issue)

Full table in `docs/compliance/third-party-models.md`. Newly flagged there,
previously untracked: CodeFormer/VQGAN (S-Lab non-commercial), YOLOv8
(AGPL/Enterprise), FastVLM (Apple research-only, auto-fetched with no
warning), LFM2 ($10M revenue cap), Moondream 3 (BSL 1.1,
anti-competitive-hosting clause), SCRFD/ArcFace (InsightFace
non-commercial-research weights under an otherwise-MIT code license),
FLUX.1-dev/Kontext (BFL non-commercial, no opt-in gate unlike its `flux2`
sibling), Z-Image (license unverified - network-blocked during audit).
TimesFM-3 is qualitatively different, see below.

### TimesFM-3: a weights license that may reach brain's own code

`crates/timesfm3` - the **code** is INDEPENDENT (high confidence: a
documented, non-upstream algebraic fold of the query-scaling op, two
brain-invented kernels, no shared prose). The **weights license**
(`google/timesfm-3.0-pytorch`, TimesFM Non-Commercial License v1.0) is a
*contract*, not just a copyright grant, triggered by downloading/using/
creating a "Derivative" - and its own definition of "Derivative" is
explicitly broader than copyright's ("any work that incorporates, utilizes,
or is otherwise based on or derived from the TimesFM Model, its **logic**,
or parameters"). Brain's port was written by studying that exact logic and
validating against goldens dumped from the real checkpoint
(`.agents/rules/porting.md` step 1). Whether that contractual definition
reaches an independently-written Rust port is a real open legal question,
separate from and not resolved by the copyright analysis above.

## Housekeeping (not a third-party risk)

`crates/atif` is a same-company internal mirror of
`applications/sven/crates/atif` - same owner, same license, explicit
provenance note on both sides, confirmed. Its doc-comment's "byte-for-byte"
claim is stale: every file has cosmetically drifted (comment headers,
cross-crate references) since the last manual sync.

## What was not exhaustively covered

This audit read each crate's primary architecture file(s) plus its docs
page and any doc-comment naming a specific upstream construct - not every
line of every file in ~90 crates. Infra crates (`kernels`, `backend-*`,
`gpu-core`, `capability`, `residency`, `splat`, the `toy*` crates,
`apiserve`, `dbus`, `weightset`, `gradcheck`, `wgsl-cpu`) got a lighter
spot-check confirming brain's own "native" claim, which held except for the
two transcriptions above (`gguf.rs`, `yarn.rs`, both in `/NOTICE.md`).
Training/finetune/serving-surface code paths (as opposed to the forward
architecture) were generally not separately diffed against upstream. Treat
this as a strong first pass, not a file-by-file guarantee.
