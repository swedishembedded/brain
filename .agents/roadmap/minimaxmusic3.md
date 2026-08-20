# minimaxmusic3 - roadmap

MiniMax Music 3 (`MiniMaxAI/MiniMax-Music3`): lyrics + a structured music
description in, a full song out (up to 5 minutes, 44.1 kHz stereo). No
official inference code exists upstream - only an unmerged `diffusers` PR
(commit `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`) implements it; this port
reads that PR (and the checkpoint's own real tensor shapes, read directly
over HTTP range requests) as the reference, without vendoring any of its
files into this tree.

Five chained components, ~19B parameters total:

1. **Global LLM** - a real Qwen3-8B (`hidden=4096, layers=36, heads=32,
   kv_heads=8, head_dim=128, vocab=200000` - the checkpoint's own
   `qwen_7B/qwen_7B/config.json`, not the smaller published Qwen3-8B preset),
   reused verbatim from `crates/qwen3`. Autoregressive, CFG-guided: one
   semantic RVQ code per 25 Hz frame.
2. **RVQ depth decoder** (0.65B) - a 4-layer causal transformer that
   autoregressively predicts the 7 residual codebooks per frame from the
   Global LLM's hidden state, and owns the residual-code embedding table.
3. **Condition encoder** (25M) - softmax-mixes the 8 per-frame hidden states
   (LLM + 7 depth steps), projects, nearest-resamples from the 25 Hz frame
   rate to the Flow-VAE latent timeline.
4. **Flow-matching DiT** (2.4B) - 36-layer LayerNorm transformer (partial
   RoPE: 32-of-64 head dims rotate), denoises Flow-VAE latents in 200-frame
   chunks with a 100-frame hop, splicing consecutive chunks over a
   172-latent overlap.
5. **Vocoder** (123M) - a DAC-style decoder (SnakeBeta + weight-normalized
   conv/conv-transpose) folding 128 latent channels into 2 (stereo) at
   44.1 kHz.

## Validation policy

Real-weight tensor/layer parity for every component (streamed + int8 where
fp32 won't fit in this machine's ~21 GB usable RAM), plus one real, short
(a few seconds, a single denoise chunk, a short AR run) end-to-end
generation producing an actual WAV on this machine. A full 5-minute
generation (14 overlapping chunks, an AR loop over ~1500 frames on the 8B
LM) is not attempted here - recorded as a hardware-bound gap, not silently
skipped.

## Phase 1: golden dumper

`tools/goldens/minimaxmusic3_dump_reference.py` covers the four
`MiniMaxMusic3*`-prefixed diffusers classes (condition encoder, vocoder,
RVQ depth decoder, DiT) at both random-weight `--tiny` dims (matching
`crates/minimaxmusic3::config`'s `::tiny()`, no checkpoint needed) and
`--real` dims (real weights, `strict=True` state-dict load). The Global
LLM's own golden path is `transformers.Qwen3ForCausalLM` directly - no
`diffusers` PR dependency - and is deferred to the Global LLM milestone.

Reference source: an unmerged `diffusers` PR, installed into a scratch venv
this repo does not track (`pip install
"git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"`,
alongside `torch`/`transformers`/`safetensors`/`numpy`/`huggingface_hub`) -
see `requirements.txt`'s "NOT pip-installable" block. No file from that PR
is vendored into this tree.

Real weights for the three small components (condition encoder 97 MB,
vocoder 207 MB, RVQ depth decoder 1.3 GB - `resources/minimax-music3/`,
gitignored) were fetched and dumped; every `state_dict.load_state_dict(...,
strict=True)` succeeded on the first try, confirming the tensor names/shapes
recorded from the checkpoint's real safetensors headers were exactly right.
Real weights for the DiT (9.7 GB) and the Global LLM (17.2 GB via the
pre-split `language_model/`, or 18.5 GB via `qwen_7B/` - the pre-split dir
is simpler and is what those milestones will use, not the manual key-split
this roadmap's Phase 0 draft assumed) are deferred to their own milestones.

Measured output shapes (real dims, batch=1): condition encoder
`(1,5,32768) -> (1,17,2048)`; vocoder `(1,128,6) -> (1,2,3072)`; RVQ depth
decoder hidden `(1,8,4096)`.

## Not yet done

- [ ] Condition encoder: import + forward + parity
- [ ] Vocoder: import (incl. folding the checkpoint's `weight_g`/`weight_v`
      weight-norm pairs) + forward/backward + a multi-scale STFT/mel
      discriminator + adversarial training + gradcheck
- [ ] RVQ depth decoder: import + forward/backward + gradcheck + LoRA
- [ ] Flow-matching DiT: import + forward/backward + gradcheck + LoRA +
      int8 storage tier + pipeline sharding
- [ ] Global LLM: streamed import via `crates/qwen3` + an audio-code
      cross-entropy training objective
- [ ] Pipeline: prompt assembly, two-axis CFG (AR logits + DiT
      zero-conditioning), chunk windowing/overlap-splice, vocoder
      crop-and-stitch - one real short end-to-end WAV on this machine
- [ ] Serving contract: capability manifest, residency adapter, CLI verb,
      D-Bus, a runnable example
- [ ] `SNAKE_BETA` backward kernel (currently forward-only; needed for
      vocoder training)

## Recorded gaps (expected, not yet reached)

- No full 5-minute generation on this machine (no discrete GPU, ~21 GB
  usable RAM) - only a short single-chunk generation is exercised here.
- No NPU export path planned in the initial port.
- No multi-GPU shard parity unless a second GPU is available when that
  milestone lands.
