# Third-party model weight / checkpoint licenses

This is an inventory of the license terms attached to the **released
checkpoints** brain's model crates load, fetch, or document a fetch path
for. It is a completely separate legal question from the **code** licenses
tracked in `/NOTICE.md` at the repo root: brain's own Rust implementation of an
architecture can be (and, per `.agents/roadmap/licensing-audit.md`, almost
always is) independently authored and Apache-2.0, while the checkpoint that
architecture was validated against, or that a user points brain at, carries
its own separate terms set by whoever trained and released it.

**Rule going forward:** any crate that auto-fetches a checkpoint under
anything other than a permissive (Apache-2.0/MIT/BSD) license must gate that
fetch behind an explicit opt-in (see `crates/flux2`'s `BRAIN_FLUX2_ALLOW_NC`
for the pattern), and `docs/models/<name>.md` must state the restriction
inline, not just here. See `.agents/rules/licensing.md`.

Last audited: 2026-09-03 (`.agents/roadmap/licensing-audit.md`). Re-check
before shipping anything that bundles or hosts inference on these weights.

## Restrictive / non-commercial / conditional

| Model (crate) | Checkpoint | License | Key restriction | Documented in brain's docs? |
|---|---|---|---|---|
| SUPIR (`supir`) | `SUPIR-v0Q` etc. | SUPIR Software License (SupPixel) | Non-commercial only; commercial use needs a separate license | Yes, `docs/models/supir.md`; no auto-fetch (by design) |
| WorldMirror-2 (`worldmirror2`) | HY-World-2.0 weights | Tencent Hunyuan community license | Excludes EU, UK, South Korea entirely | Yes, `docs/models/worldmirror2.md` |
| CodeFormer (`codeformer`) | `codeformer.pth` | S-Lab License 1.0 | Non-commercial; contact contributors for commercial use | No |
| VQGAN (`vqgan`) | `vqgan_code1024.pth` | S-Lab License 1.0 | Same as CodeFormer (shares upstream) | No |
| YOLOv8 (`yolov8`) | `yolov8n.pt` (auto-fetched) | AGPL-3.0, or Ultralytics Enterprise | Copyleft on code that uses it, or a paid Enterprise license | No - auto-fetch has no license notice |
| FastVLM (`fastvlm`) | `apple/FastVLM-0.5B` (opt-in `--autofetch`) | Apple ML Research Model License (`LICENSE_MODEL`) | Research purposes only; commercial use explicitly prohibited | No - sharpest gap found: one-command opt-in fetch, zero license warning |
| TimesFM-3 (`timesfm3`) | `google/timesfm-3.0-pytorch` | TimesFM Non-Commercial License v1.0 | Non-commercial/non-production use only; unconditional redistribution ban; broad contractual "Derivative" definition that plausibly reaches an independent from-scratch port built by studying its logic | Partially, `docs/models/timesfm3.md` - understates the contractual "Derivative" exposure, see ledger |
| LFM2 (`lfm2`) | `LiquidAI/LFM2-*` (auto-fetched) | LFM Open License v1.0 | Commercial use conditioned on < $10M annual revenue; above that threshold needs a separate LiquidAI commercial license | No |
| Moondream 3 (`moondream3`) | user-supplied (`BRAIN_MOONDREAM3_WEIGHTS`, no auto-fetch) | Business Source License 1.1 (M87 Labs) | Production use allowed except hosted/embedded offerings that compete with M87 Labs' paid product; converts to Apache-2.0 at a future Change Date | No |
| SCRFD (`scrfd`) | antelopev2 `scrfd_10g_bnkps.onnx` | InsightFace (code: MIT; weights: non-commercial research only) | Commercial use requires contacting InsightFace | No |
| ArcFace (`arcface`) | antelopev2 `glintr100.onnx` | Same as SCRFD | Same as SCRFD | No |
| FLUX.1-dev / Kontext-dev (`flux1`) | BFL `dev`/`Kontext` checkpoints | FLUX.1-dev Non-Commercial License (schnell is Apache-2.0) | Non-commercial only for `dev`/`Kontext` variants | No opt-in gate - `flux2` has `BRAIN_FLUX2_ALLOW_NC` for its analogous 9B weights |
| FLUX.2 Klein-9B / base-9B (`flux2`) | BFL 9B checkpoints (4B + VAE + Qwen3 encoder are Apache-2.0) | FLUX Non-Commercial License | Non-commercial only | Yes - gated behind `BRAIN_FLUX2_ALLOW_NC` and documented in `docs/models/flux2.md`, the reference pattern |
| LTX-2.x (`ltxv`) | LTX-2.x checkpoints | LTX-2.x Community License (Lightricks) | Commercial-use gating on a broadly-defined "Derivatives of LTX-2.x" | Not cross-checked against `docs/models/ltxv.md`'s current text |
| SDXL (`sdxlunet`, `controlnet`, `supir`'s frozen backbone) | Stability AI SDXL 1.0 | CreativeML Open RAIL++-M | Use-based restrictions (no illegal/harmful use), not a commercial-use ban, but distinct from Apache-2.0 | Not cross-checked against `docs/models/sdxlunet.md`'s current text |
| Z-Image (`s3dit`) | `Tongyi-MAI/Z-Image-Turbo` | Unverified - Tongyi-MAI's license text was unreachable from the audit environment | Unknown | No |

## Permissive (Apache-2.0 / MIT), no split from code license

Chronos-2, FinCast, Kronos (MIT), Gemma-4 (gated upstream by Google Terms of
Use once weights are added - forward-looking flag only, no real-weight path
yet), SAM1, SAM2 (checkpoints explicitly re-confirmed Apache-2.0 by
upstream, not just the code), CLIP/OpenCLIP/EVA-CLIP, PuLID (note: composes
FLUX.1-dev, see above), InstantID, DIAMOND, GenieRedux, Qwen3-TTS, Qwen3-ASR,
CosyVoice, Mimi (dual Apache-2.0/MIT), NeMo/nemotronasr, Wan2.1/2.2,
diffusers-derived crates (`diffusion`, `dit`, `vae`, `sdxlunet`'s *code*,
`controlnet`'s *code*), DeepSeek-V2/DeepSeek-OCR (code: MIT; a separate
DeepSeek `LICENSE-MODEL` governs the weights and was not diffed in detail),
LLaVA, T5, gpt2 (no real checkpoint fetch path).
