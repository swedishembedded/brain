#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for verifying ported generative
# media pipelines against their reference implementations for its clients. If
# your team needs expertise in video diffusion model inference then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump LTX-2.5 end-to-end clip-length prediction goldens.

`ltxv_duration_head_dump_reference.py` proves the head's own math on
synthetic tokens. This dumper runs what a generation that names no length
runs: the OFFICIAL `PromptEncoder` (full-depth Gemma-4 text encoder, the
aggregate embedding, both embeddings connectors - loaded from the real bf16
transformer and text-encoder files exactly as `ti2vid_one_stage` loads them)
and the OFFICIAL `DurationPredictor` over its output, for a fixed prompt set.

Per prompt `i` it dumps

  prompt{i}_ids              [1024] the tokenizer's (left-padded) token ids
  prompt{i}_mask             [1024] which positions the connector reads as
                             the prompt's own - in the connector's layout,
                             RIGHT-padded (`create_embeddings` sorts the
                             valid tokens first, whatever the tokenizer did)
  prompt{i}_video_features   [1024, 4096] the aggregate embedding the video
                             connector reads, in that layout (padding rows
                             zeroed: the connector replaces them)
  prompt{i}_audio_features   [1024, 2048] the same for the audio connector
  prompt{i}_video_encoding   [1024, 4096] video connector output
  prompt{i}_audio_encoding   [1024, 2048] audio connector output
  prompt{i}_seconds_av       the head's raw seconds, both streams (a sound clip)
  prompt{i}_seconds_video    video stream alone (a silent clip)
  prompt{i}_seconds_audio    audio stream alone (text-to-audio)

so a port can be checked stage by stage: its tokens, its text encoder
against the features, its connectors against the encodings, then its
prediction against the seconds. Runs on the CPU in fp32
(`--dtype`), the precise reference rather than the bf16 GPU fast path.

Usage:
  python tools/goldens/ltxv_duration_e2e_dump_reference.py \\
      --transformer .../ltx-2.5-22b-distilled-transformer-bf16.safetensors \\
      --text-encoder .../gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \\
      --duration-head .../ltx-2.5-duration-head-bf16.safetensors \\
      --out testdata/golden/ltxv/duration_e2e
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# A single quick action and a long multi-beat scene: the two ends of the
# head's range a port must reproduce, not just its midpoint.
PROMPTS = [
    "a glass falls off a table and shatters",
    "a woman walks along a beach at sunset, stops to pick up a shell, turns it over in her hand, "
    "looks out at the waves, then slowly walks on as the light fades and the tide comes in",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--duration-head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--reference-root", default=str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv"),
                    help="the LTX-2 reference checkout (default: the repo's gitignored scratchpad copy)")
    args = ap.parse_args()
    for package in ("ltx-core", "ltx-pipelines"):
        sys.path.insert(0, str(Path(args.reference_root) / "packages" / package / "src"))
    from ltx_pipelines.utils.blocks.duration_predictor import DurationPredictor
    from ltx_pipelines.utils.blocks.prompt_encoder import PromptEncoder
    from ltx_pipelines.utils.model_paths import ModelPaths

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.dtype)
    device = torch.device("cpu")

    paths = ModelPaths.from_split(transformer_path=args.transformer, text_encoder_path=args.text_encoder,
                                  duration_head_path=args.duration_head)
    # `PromptEncoder.__call__`'s own two steps, run here so the tokens and the
    # features between them can be dumped too.
    encoder = PromptEncoder(paths, dtype=dtype, device=device)
    with encoder._text_encoder_ctx() as text_encoder:
        raw = text_encoder.encode(PROMPTS)
        ids = [torch.tensor([t for t, _ in text_encoder.tokenizer.tokenize_with_weights(p)["gemma"]]) for p in PROMPTS]
    processor = encoder._build_embeddings_processor()
    head = DurationPredictor.from_checkpoint(args.duration_head, dtype=dtype, device=device)._head

    tensors, run = {}, {"prompts": PROMPTS, "dtype": args.dtype}
    for i, (hidden_states, mask) in enumerate(raw):
        v_feats, a_feats = processor.feature_extractor(hidden_states, mask)
        out = processor.process_hidden_states(hidden_states, mask)
        v, a = out.video_encoding, out.audio_encoding
        assert out.attention_mask.all(), "the connector output is fully attendable by construction"
        # The connector's own view: valid tokens first, stably, exactly as
        # `create_embeddings` reorders them.
        order = torch.argsort(mask[0].to(torch.int32), descending=True, stable=True)
        keep = mask[0][order].bool().unsqueeze(-1)
        tensors[f"prompt{i}_ids"] = ids[i].to(torch.float32)
        tensors[f"prompt{i}_mask"] = mask[0][order].to(torch.float32)
        tensors[f"prompt{i}_video_features"] = torch.where(keep, v_feats[0][order], 0.0)
        tensors[f"prompt{i}_audio_features"] = torch.where(keep, a_feats[0][order], 0.0)
        seconds = {"av": head(v, a), "video": head(v, None), "audio": head(None, a)}
        tensors[f"prompt{i}_video_encoding"] = v[0]
        tensors[f"prompt{i}_audio_encoding"] = a[0]
        for k, s in seconds.items():
            tensors[f"prompt{i}_seconds_{k}"] = s
            run[f"prompt{i}_seconds_{k}"] = s.item()
        print(f"prompt {i}: " + ", ".join(f"{k} {s.item():.4f} s" for k, s in seconds.items()), flush=True)

    tensors = {k: t.detach().to(torch.float32).contiguous() for k, t in tensors.items()}
    path = os.path.join(args.out, "duration_e2e.safetensors")
    save_file(tensors, path)
    manifest = {
        "run": run,
        "duration_e2e.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                                     "tensors": {k: list(t.shape) for k, t in tensors.items()}},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "source": source_block(
            checkpoint="Lightricks/LTX-2.5",
            files=[args.transformer, args.text_encoder, args.duration_head],
            hash_files=False,
            identity={"video_dim": int(tensors["prompt0_video_encoding"].shape[-1]),
                      "audio_dim": int(tensors["prompt0_audio_encoding"].shape[-1]),
                      "tokens": int(tensors["prompt0_video_encoding"].shape[0])},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
