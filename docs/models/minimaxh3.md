# MiniMax-H3 (joint video + audio generation)

A 33B rectified-flow diffusion transformer that denoises video and audio
latents together in **one packed self-attention stack** - unlike
[`ltxv`](ltxv.md)'s two-stream architecture, H3 has no separate audio blocks
and no A/V cross-attention; modality differences come entirely from
input/output projections and modality-tagged AdaLN conditioning. Text
conditioning is a Qwen3-VL encoder truncated to an intermediate hidden
layer. Released tasks: `t2va` (text -> video+audio), `fl2va`
(first/last-frame-conditioned), `ref2va` (up to 12 mixed image/video/audio
references).

**This port is in progress.** See `.agents/roadmap/minimaxh3.md` for the
phase ledger and which convention questions are settled vs. still open. No
capability below is claimed working until its own phase's real-weight parity
gate is green.

## Support

| Task | Status |
|---|:---:|
| `t2va` (text -> video+audio) | [ ] in progress |
| `fl2va` (first/last-frame -> video+audio) | [ ] in progress |
| `ref2va` (image/video/audio references -> video+audio) | [ ] in progress |
| Training (LoRA) | [ ] in progress |

## Getting the weights

**Weights are never auto-fetched.** `minimaxh3` has no `default_ref` in
`arch::ARCHS` - the same deliberate omission `supir`/`fincast` use, here for
a different reason (see Licensing below). Point brain at a local checkout
you obtained yourself:

```
BRAIN_MINIMAXH3_DIR=/path/to/MiniMax-H3   # contains FL2VA/ and Ref2VA/
```

The real repo (`MiniMaxAI/MiniMax-H3` on HuggingFace) is a two-level layout:
task-partition directories (`FL2VA/`, `Ref2VA/`), each a `model_index.json`-
rooted diffusers pipeline with its own `transformer/`, `text_encoder/`,
`audio_vae/`, `video_vae/`, `processor/` roles.

## Hardware and limits

2xTesla P40 (24GB each) + 184GB RAM is what this port targets. Pascal has no
bf16 compute and brain's loader demotes BF16->F32 on read, so the 66GB bf16
checkpoint is 132GB of fp32-equivalent bytes - too large to hold resident at
fp32. The real shard header shows `adaln_proj` alone is 260M params/block x
50 blocks = 13.0B of the 33B total; folding that into small precomputed
per-(step,modality) modulation tables (the `precompute-adaln` tool, once it
lands) turns the DiT into a ~19.3B backbone that fits fully resident at int8
(~19GB) across both cards. The Qwen3-VL text encoder is a separate
~33B-class model and cannot be co-resident with the DiT - generation runs in
sequential phases (encode -> free -> denoise -> free -> decode), the same
pattern `wan`'s pipeline already uses.

License: **MiniMax H3 Community License Agreement** - a territorial
carve-out (the ordinary grant excludes the EU, UK, South Korea and the US;
an organization in one of those regions needs MiniMax's separate
authorization), a >$20M/yr revenue registration clause, and a mandatory
"MiniMax H3" UI attribution requirement for any commercial product or
service. This port's Rust code is Apache-2.0, ported from the Apache-2.0
`diffusers`/`transformers` reference implementation - MiniMax's own
(community-licensed) repository is not translated into this crate. The
weights themselves are never vendored or auto-fetched; running against a
checkpoint you obtained yourself additionally requires
`BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1` (`minimaxh3::caps::check_license`),
confirming you have cleared the license's terms for your own use. See the
checkpoint's own `LICENSE` file for the authoritative text.
