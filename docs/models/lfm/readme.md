# LFM2.5-Encoder (LiquidAI) in brain

`crates/lfm` implements the **LFM2.5-Encoder-230M** and **-350M** released
checkpoints: bidirectional hybrid short-conv/attention encoders with a tied
masked-LM head, MLM-pretrained (30% masking), usable context 8,192 tokens.
Reference: <https://huggingface.co/blog/LiquidAI/lfm2-5-encoders>.

## Architecture

Vocab 65,536 → hidden 1024; pre-LN RMSNorm (eps 1e-5) throughout; per layer
either mixer (from the checkpoint's `layer_types`) followed by a SwiGLU FFN:

- **Conv mixer**: `in_proj` H→3H row-thirds (B, C, X); `Bx = B⊙X`; depthwise
  conv1d k=3, groups=H, **symmetric padding** (the encoder's non-causal
  variant); `y = out_proj(C ⊙ conv)`. No biases.
- **Attention mixer**: separate q/k/v/out projections (no bias); **per-head
  QK-RMSNorm** (head_dim 64); RoPE rotate_half θ=1e6; **GQA 16Q/8KV**;
  **bidirectional** (non-causal) softmax.
- 230M: 14 layers (8 conv / 6 attn), FFN 2560. 350M: 16 layers (10 conv /
  6 attn), FFN 4608 (HF `block_auto_adjust_ff_dim` rule, resolved at import).
- Tied MLM head: `logits = hidden @ embeddingᵀ`; CE ignore −100, unshifted.

## How it is built (brain mapping)

Everything composes from shared builders — no LFM-private math:

| Piece | Shared implementation |
|---|---|
| bidirectional attention fwd/bwd | `model::block::{bidir_fwd,bidir_bwd}` (hoisted from seq2seq) |
| GQA→MHA head expansion | `kv_expand{,_bwd}.wgsl` + `model::block::kv_expand_{fwd,bwd}` (group 2; group 1 places q) |
| query-chunked long-context attention | `model::block::chunked_bidir_fwd` (hoisted from `model::vit`) |
| depthwise symmetric conv1d fwd/dx/dw | `conv1d{,_dx,_dw}.wgsl` via `audio::conv` (pad_low=1, high pad implicit) |
| NLC↔NCL layout | `nchw_nlc`/`nlc_nchw` (mutually adjoint) |
| eps-aware RMSNorm | `model::block::rmsnorm_eps_{fwd,bwd}` |
| RoPE / SwiGLU / CE / embedding | `rope_base`, `silu_mul`, `ce_*`, `embed_tile`/`emb_bwd` |
| GEMM selection | `model::block::pick_gemm` |
| tokenizer | `data::qwen_tokenizer::QwenBpe` — digit-run width (`\p{N}{1,3}`) auto-detected from tokenizer.json; `template_prefix()` = BOS; `special_id("<|mask|>")` = 16 |

Two attention regimes behind one layer loop (`crates/lfm/src/model.rs`):

- **Materialized** (`Lfm::new` / `load_inference`): full `[B,H,T,T]` scores,
  per-layer activation caches — parity gates and (later) training. T² memory:
  4 GiB at T=8192 > the ~2 GiB binding limit, so NOT for long context.
- **Chunked** (`Lfm::new_chunked` / `load_inference_chunked`): bounded
  `[H, chunk, T]` slab, shared layer scratch, ping-pong residuals, and the MLM
  head evaluated only at gathered probe rows (the `embed` kernel doubles as the
  row gather). This is the 8k path; chunked == materialized is gated bit-exact
  by `crates/lfm/tests/chunked_equiv.rs`.

**Padding warning:** bidirectional attention makes unmasked padding unsound —
pad tokens corrupt every real token's encoding (measured: garbage fill-mask).
The capability provider therefore builds at the exact request length; batched
padding needs zeroed pad states + an additive key mask (planned with batched
serving).

## Commands

```bash
brain lfm import    --hf <hf_dir> --out out/lfm-230m.safetensors
brain lfm fill-mask --weights F --tokenizer tokenizer.json --text "… <|mask|> …" [--topk K]
brain lfm embed     --weights F --tokenizer tokenizer.json (--text "…" | --input FILE) [--seq T]
brain do brain/lfm fill_mask --weights … --tokenizer … --text "…"      # capability surface
brain caps                                                        # discovery
```

The event API serves the same actions via generic `action_request` (no new
Event variant): `{"event":"action_request","model":"brain/lfm","action":"fill_mask",…}`.

## Parity (the gate)

`tools/goldens/lfm_dump_reference.py` bakes staged goldens from the released fp32
checkpoints through the repo's own `modeling_lfm2_bidirectional.py` (fixed
token ids — tokenizer parity is gated separately in `crates/data` with pinned
id vectors incl. adversarial digit runs). `crates/lfm/tests/parity.rs` checks
post-embedding, every layer output, final hidden, three MLM-logit probe rows,
and fill-mask top-1, for both models, both regimes:

- Measured (CPU + GPU backends): **cosine = 1.000000, rel_l2 ≤ 1e-5 at every
  stage**; fill-mask top-1 = " Paris" (id 5242) matching the reference.
- Run: `BRAIN_LFM25_230M=<dir> BRAIN_LFM25_350M=<dir> cargo test -p brain-lfm`.

## Status / measured numbers

See [status.md](status.md).
