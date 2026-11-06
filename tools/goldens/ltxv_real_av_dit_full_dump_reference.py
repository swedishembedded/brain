#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for verifying ported generative
# media pipelines against their reference implementations for its clients. If
# your team needs expertise in video diffusion model inference then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump a full-depth LTX-2.5 audio-visual DiT forward on the real checkpoint.

`ltxv_real_dit_dump_reference.py` proves the port at real width on 2 of the
48 blocks, video only. This dumper runs the WHOLE model as a generation does:
the OFFICIAL `LTXModel` built by the reference's own configurator from the
real bf16 transformer file (`LTXModelConfigurator` +
`LTXV_MODEL_COMFY_RENAMING_MAP` and the connector-excluding selection, as
`DiffusionStage` builds it), all 48
audio-visual blocks, fp32 on the CPU, one velocity forward at one sigma.

The inputs are the reference's own: the latent states come from
`VideoLatentTools`/`AudioLatentTools.create_initial_state` (positions with
the VAE scale factors and the frame rate, the first-frame keyframes mask),
the modalities from `modality_from_latent_state`, and the text context is a
real prompt's connector outputs from `ltxv_duration_e2e_dump_reference.py`'s
golden (`--prompt-golden`), so the 1024-token cross-attention is a real one.

Dumped (`av_dit_full.safetensors`): both streams' latent, per-token
timesteps, positions and (video) keyframes mask; the sigma; the block
outputs after blocks 0, 11, 23, 35 and 47 (`video.block{N}`/`audio.block{N}`)
so a divergence is placed in depth; and both velocities (`video.velocity`,
`audio.velocity`).

Usage:
  python -I tools/goldens/ltxv_real_av_dit_full_dump_reference.py \\
      --transformer .../ltx-2.5-22b-distilled-transformer-bf16.safetensors \\
      --prompt-golden testdata/golden/ltxv/duration_e2e/duration_e2e.safetensors \\
      --out testdata/golden/ltxv/av_dit_full --seed 7
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# 17 frames at 24 fps, 128x128: a 3 x 4 x 4 video latent (48 tokens) and 18
# audio tokens - small enough for an fp32 CPU forward of the whole model,
# large enough that every axis of both streams' RoPE has more than one value.
FRAMES, HEIGHT, WIDTH, FPS = 17, 128, 128, 24.0
SIGMA = 1.0
TAPPED_BLOCKS = (0, 11, 23, 35, 47)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--prompt-golden", required=True, help="duration_e2e.safetensors (a real prompt's connector outputs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sigma", type=float, default=SIGMA, help="the step's sigma; the latent is pure noise, so 1.0 is the on-distribution value")
    ap.add_argument("--tap-every-block", action="store_true", help="tap all 48 block outputs, not five")
    ap.add_argument("--tap-block0", action="store_true", help="also tap block 0's attention, cross-attention and feed-forward outputs and both patchify projections")
    ap.add_argument("--weights-from-gguf", default=None,
                    help="replace every transformer weight with this Q8_0 GGUF's dequantized tensor, to separate the port from the weight format")
    ap.add_argument("--reference-root", default=str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv"),
                    help="the LTX-2 reference checkout (default: the repo's gitignored scratchpad copy)")
    args = ap.parse_args()
    for package in ("ltx-core", "ltx-pipelines"):
        sys.path.insert(0, str(Path(args.reference_root) / "packages" / package / "src"))
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
        transformer_checkpoint_selection_ops,
    )
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils.helpers import modality_from_latent_state

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    device, dtype = torch.device("cpu"), torch.float32

    pixel = VideoPixelShape(batch=1, frames=FRAMES, height=HEIGHT, width=WIDTH, fps=FPS)
    v_shape = VideoLatentShape.from_pixel_shape(pixel)
    a_shape = AudioLatentShape.from_video_pixel_shape(pixel)
    gen = torch.Generator().manual_seed(args.seed)
    v_noise = torch.randn(v_shape.to_torch_shape(), generator=gen)
    a_noise = torch.randn(a_shape.to_torch_shape(), generator=gen)
    v_state = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, FPS).create_initial_state(device, dtype, v_noise)
    a_state = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape).create_initial_state(device, dtype, a_noise)
    # Tabulated as the pipeline's `create_noised_state` leaves it.
    v_state = replace(v_state, denoise_mask=v_state.denoise_mask.compact(device))
    a_state = replace(a_state, denoise_mask=a_state.denoise_mask.compact(device))

    prompt = load_file(args.prompt_golden)
    v_context = prompt["prompt0_video_encoding"][None]
    a_context = prompt["prompt0_audio_encoding"][None]
    sigma = torch.tensor([args.sigma])
    video = modality_from_latent_state(v_state, v_context, sigma)
    audio = modality_from_latent_state(a_state, a_context, sigma)

    # `DiffusionStage._build_transformer`'s own build: the connectors belong to
    # the embeddings processor, so the transformer's selection excludes them.
    builder = SimpleBuilder(LTXModelConfigurator, (Path(args.transformer),), (LTXV_MODEL_COMFY_RENAMING_MAP,))
    selection = transformer_checkpoint_selection_ops(builder.model_config())
    if selection is not None:
        builder = builder.with_sd_ops((*builder.sd_ops, selection))
    model = builder.build(device=device, dtype=dtype).eval()
    print(f"built the full model: {len(model.transformer_blocks)} blocks", flush=True)
    if args.weights_from_gguf:
        from gguf import GGUFReader
        from gguf.quants import dequantize
        tensors = {t.name: t for t in GGUFReader(args.weights_from_gguf).tensors}
        replaced = 0
        for name, param in model.named_parameters():
            t = tensors.get(name)
            if t is None:
                raise KeyError(f"{name} is not in {args.weights_from_gguf}")
            param.data.copy_(torch.from_numpy(dequantize(t.data, t.tensor_type).reshape(param.shape).copy()))
            replaced += 1
        print(f"replaced {replaced} parameters with the GGUF's", flush=True)

    taps = {}

    def tap(i):
        def hook(_module, _inputs, output):
            v, a = output
            taps[f"video.block{i}"] = v.x[0].clone()
            taps[f"audio.block{i}"] = a.x[0].clone()
        return hook

    tapped = range(len(model.transformer_blocks)) if args.tap_every_block else TAPPED_BLOCKS
    handles = [model.transformer_blocks[i].register_forward_hook(tap(i)) for i in tapped]
    if args.tap_block0:
        def keep(name):
            def hook(_module, _inputs, output):
                taps[name] = output[0].clone()
            return hook
        b0 = model.transformer_blocks[0]
        for name, module in [("patchify_proj", model.patchify_proj), ("audio_patchify_proj", model.audio_patchify_proj),
                             ("b0.attn1", b0.attn1), ("b0.attn2", b0.attn2), ("b0.ff", b0.ff),
                             ("b0.audio_attn1", b0.audio_attn1), ("b0.audio_attn2", b0.audio_attn2), ("b0.audio_ff", b0.audio_ff),
                             ("b0.a2v", b0.audio_to_video_attn), ("b0.v2a", b0.video_to_audio_attn)]:
            handles.append(module.register_forward_hook(keep(name)))
    vx, ax = model(video, audio, None)
    for h in handles:
        h.remove()
    print(f"velocities: video {tuple(vx.shape)} audio {tuple(ax.shape)}", flush=True)

    tensors = {
        "video.latent": video.latent[0], "video.timesteps": video.timesteps.per_token()[0, :, 0],
        "video.positions": video.positions[0], "video.keyframes_mask": video.keyframes_mask[0].reshape(-1),
        "audio.latent": audio.latent[0], "audio.timesteps": audio.timesteps.per_token()[0, :, 0],
        "audio.positions": audio.positions[0],
        "sigma": sigma, "video.velocity": vx[0], "audio.velocity": ax[0], **taps,
    }
    tensors = {k: t.detach().to(torch.float32).contiguous() for k, t in tensors.items()}
    path = os.path.join(args.out, "av_dit_full.safetensors")
    save_file(tensors, path)
    manifest = {
        "run": {"frames": FRAMES, "height": HEIGHT, "width": WIDTH, "fps": FPS, "sigma": args.sigma, "seed": args.seed,
                "weights": "gguf" if args.weights_from_gguf else "bf16",
                "tapped_blocks": list(tapped), "prompt": "prompt0 of the prompt golden"},
        "av_dit_full.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                                    "tensors": {k: list(t.shape) for k, t in tensors.items()}},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "source": source_block(
            checkpoint="Lightricks/LTX-2.5",
            files=[args.transformer],
            hash_files=False,
            identity={"blocks": len(model.transformer_blocks), "video_tokens": int(tensors["video.latent"].shape[0]),
                      "audio_tokens": int(tensors["audio.latent"].shape[0]), "context_tokens": int(v_context.shape[1])},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
