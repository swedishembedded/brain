#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving a ported vision-language
# model reproduces its reference numerically for its clients. If your team
# needs expertise in multimodal model porting then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Dump the reference goldens `crates/minicpmv` is gated against.

Runs the REAL `transformers.models.minicpmv4_6` processor and model (never a
vendored copy) on CPU in fp32 over the real `openbmb/MiniCPM-V-4.6`
checkpoint, and records:

  inputs/*.png            every input image, so the Rust side decodes the same
                          bytes (tracked repo images plus deterministic
                          synthetic ones at aspect ratios that exercise the
                          slicing policy's branches)
  processor/<case>.safetensors
                          per processor case: `input_ids`, `target_sizes`,
                          `grids`, `num_patches` and every resized view the
                          image processor produced (`resized.<i>`, uint8
                          values as f32, [3,H,W]); `pixel_values` for a few
                          cases (it is a pure function of the resized views)
  video/sample_frames.json
                          `sample_frames` index lists over several clip
                          metadata combinations
  model/<case>.safetensors
                          per model case: the processor outputs, vision taps
                          (embeddings, encoder layers 0/6/26, the window
                          merger, post-layernorm), the merger output, the LM
                          input embeddings, LM layer outputs, final-norm
                          output, the logits of the last rows, the greedy
                          continuation and (one case) the label loss
  manifest.json           cases, options, prompts, library versions and the
                          `source` block (golden_source.py)

Every quantity is saved as f32 (ints cast exactly).

Usage:
  python tools/goldens/minicpmv_dump_reference.py --model <dir> [--out DIR]
"""

import argparse
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
LOGIT_ROWS = 8
GREEDY_TOKENS = 24


def f32(x):
    return torch.as_tensor(np.asarray(x)).to(torch.float32).contiguous()


def synthetic(name, w, h, seed):
    """A deterministic test card: smooth gradients plus seeded noise plus a few
    solid shapes, so resampling, slicing and patch order all leave a mark."""
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1), (xx + yy) / max(w + h - 2, 1)], -1) * 200.0
    img += rs.rand(h, w, 3).astype(np.float32) * 55.0
    for _ in range(6):
        x0, y0 = rs.randint(0, w), rs.randint(0, h)
        x1, y1 = min(w, x0 + rs.randint(1, max(2, w // 3))), min(h, y0 + rs.randint(1, max(2, h // 3)))
        img[y0:y1, x0:x1] = rs.randint(0, 256, 3)
    return name, Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def load_inputs(out_dir):
    images = {}
    for name, rel in [("room", "docs/reference/models/flux2/staging-source.jpg"), ("doc", "docs/getting-started/img/doc.png")]:
        images[name] = Image.open(os.path.join(REPO, rel)).convert("RGB")
    for name, img in [
        synthetic("tall", 300, 1400, 1),
        synthetic("wide", 1600, 200, 2),
        synthetic("tiny", 40, 30, 3),
        synthetic("large", 2600, 1900, 4),
        synthetic("odd", 517, 389, 5),
    ]:
        images[name] = img
    os.makedirs(os.path.join(out_dir, "inputs"), exist_ok=True)
    for name, img in images.items():
        img.save(os.path.join(out_dir, "inputs", f"{name}.png"))
    # Re-read the PNGs so the reference consumes exactly the bytes Rust will.
    return {n: Image.open(os.path.join(out_dir, "inputs", f"{n}.png")).convert("RGB") for n in images}


def synthetic_clip(t, h, w):
    """A clip with known content: a white square moving left to right over a
    background whose colour changes once per quarter of the clip."""
    frames = np.zeros((t, h, w, 3), np.uint8)
    colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    side = h // 4
    for i in range(t):
        frames[i] = colours[min(3, i * 4 // t)]
        x0 = int((w - side) * i / max(t - 1, 1))
        frames[i, h // 2 - side // 2 : h // 2 + side // 2, x0 : x0 + side] = 255
    return frames


class ResizeRecorder:
    """Wraps the image processor's `resize` to record every view it produces."""

    def __init__(self, processor):
        self.proc = processor
        self.orig = processor.resize
        self.views = []
        processor.resize = self

    def __call__(self, image, *a, **k):
        out = self.orig(image, *a, **k)
        self.views.append(out.detach().clone())
        return out

    def take(self):
        v, self.views = self.views, []
        return v


def prompt_for(proc, parts):
    msgs = [{"role": "user", "content": parts}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def processor_cases(proc, images, out_dir, manifest):
    rec = ResizeRecorder(proc.image_processor)
    cases = []
    for name in images:
        cases.append((f"{name}_default", [name], {}))
    cases += [
        ("room_4x", ["room"], {"downsample_mode": "4x"}),
        ("room_slices36", ["room"], {"max_slice_nums": 36}),
        ("large_slices36", ["large"], {"max_slice_nums": 36}),
        ("room_noslice", ["room"], {"slice_mode": False}),
        ("room_slices1", ["room"], {"max_slice_nums": 1}),
        ("room_noid", ["room"], {"use_image_id": False}),
        ("room_wide_tall", ["room", "wide", "tall"], {}),
    ]
    with_pixels = {"room_default", "tiny_default", "room_4x", "wide_default"}
    os.makedirs(os.path.join(out_dir, "processor"), exist_ok=True)
    for case, names, opts in cases:
        parts = [{"type": "image"} for _ in names] + [{"type": "text", "text": "Describe."}]
        text = prompt_for(proc, parts)
        rec.take()
        out = proc(images=[images[n] for n in names], text=[text], images_kwargs=dict(opts), return_tensors="pt")
        views = rec.take()
        bf = proc.image_processor([images[n] for n in names], **{k: v for k, v in opts.items() if k != "use_image_id"})
        t = {
            "input_ids": f32(out["input_ids"][0]),
            "target_sizes": f32(out["target_sizes"]),
            "grids": f32(bf["grids"]),
            "num_patches": f32(bf["num_patches_per_image"]),
        }
        for i, v in enumerate(views):
            t[f"resized.{i}"] = f32(v.squeeze(0) if v.dim() == 4 else v)
        if case in with_pixels:
            t["pixel_values"] = f32(out["pixel_values"][0])
        save_file(t, os.path.join(out_dir, "processor", f"{case}.safetensors"))
        manifest["processor"][case] = {"images": names, "options": opts, "prompt": text, "n_views": len(views)}
        print(f"processor {case}: ids {tuple(out['input_ids'].shape)} views {len(views)}")
    rec.proc.resize = rec.orig


def sample_frame_cases(proc, out_dir, manifest):
    rows = []
    for total, fps, duration, max_frames, stack in [
        (60, 10.0, 6.0, 128, 1),
        (60, 10.0, 6.0, 128, 3),
        (73, 29.97, 2.4357, 128, 1),
        (9000, 30.0, 300.0, 128, 1),
        (9000, 30.0, 300.0, 32, 1),
        (1500, 25.0, 60.0, 64, 1),
        (150, 25.0, 6.0, 128, 5),
        (24, 24.0, 1.0, 128, 1),
    ]:
        md = VideoMetadata(total_num_frames=total, fps=fps, duration=duration, width=64, height=64)
        idx = proc.video_processor.sample_frames(md, max_num_frames=max_frames, stack_frames=stack)
        rows.append({"total_num_frames": total, "fps": fps, "duration": duration, "max_num_frames": max_frames, "stack_frames": stack, "indices": [int(i) for i in idx]})
    os.makedirs(os.path.join(out_dir, "video"), exist_ok=True)
    with open(os.path.join(out_dir, "video", "sample_frames.json"), "w") as f:
        json.dump(rows, f, indent=1)
    manifest["video_sample_frames"] = len(rows)


class Taps:
    def __init__(self, model, vision_taps):
        self.t = {}
        self.handles = []
        m = model.model
        lm = m.language_model
        if vision_taps:
            vt = m.vision_tower
            self._out(vt.embeddings, "vision.embeddings")
            for i in (0, 6, len(vt.encoder.layers) - 1):
                self._out(vt.encoder.layers[i], f"vision.layer.{i}")
            self._out(vt.vit_merger, "vision.vit_merger")
            self._out(vt.post_layernorm, "vision.post_layernorm")

        def pre(mod, args, kwargs):
            if kwargs.get("inputs_embeds") is not None and "lm.inputs_embeds" not in self.t:
                self.t["lm.inputs_embeds"] = kwargs["inputs_embeds"][0].detach().clone()

        self.handles.append(lm.register_forward_pre_hook(pre, with_kwargs=True))
        for i in (0, 3, len(lm.layers) - 1):
            self._out(lm.layers[i], f"lm.layer.{i}", first_only=True)
        self._out(lm.norm, "lm.norm", first_only=True)

    def _out(self, mod, name, first_only=False):
        def hook(_m, _i, o):
            o = o[0] if isinstance(o, (tuple, list)) else o
            if first_only and name in self.t:
                return
            self.t[name] = (o[0] if o.dim() == 3 else o).detach().clone()

        self.handles.append(mod.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()


def model_cases(model, proc, images, out_dir, manifest):
    clip = synthetic_clip(60, 240, 320)
    clip_md = VideoMetadata(total_num_frames=60, fps=10.0, duration=6.0, width=320, height=240)
    cases = [
        ("text", [], None, "What is the capital of France? Answer in one word.", {}, False),
        ("room_16x", ["room"], None, "Describe this image in one sentence.", {}, True),
        ("room_4x", ["room"], None, "Describe this image in one sentence.", {"downsample_mode": "4x"}, True),
        ("doc_16x", ["doc"], None, "What does the text in this image say?", {}, False),
        ("two_images", ["room", "wide"], None, "How do these two images differ?", {}, True),
        ("video_16x", [], "clip", "What happens in this video?", {}, False),
    ]
    os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
    for case, names, video, question, opts, vision_taps in cases:
        parts = [{"type": "image"} for _ in names]
        if video:
            parts.append({"type": "video"})
        parts.append({"type": "text", "text": question})
        text = prompt_for(proc, parts)
        kw = {}
        if names:
            kw["images"] = [images[n] for n in names]
            kw["images_kwargs"] = dict(opts)
        if video:
            kw["videos"] = [clip]
            kw["videos_kwargs"] = dict(video_metadata=[clip_md], do_sample_frames=True, **opts)
        inp = proc(text=[text], return_tensors="pt", **kw)
        fwd_kw = {"downsample_mode": opts["downsample_mode"]} if "downsample_mode" in opts else {}
        taps = Taps(model, vision_taps)
        with torch.no_grad():
            out = model(**inp, **fwd_kw)
        taps.close()
        t = dict(taps.t)
        if names or video:
            feats = model.get_image_features if names else model.get_video_features
            with torch.no_grad():
                if names:
                    vo = feats(inp["pixel_values"], inp["target_sizes"], **fwd_kw)
                else:
                    vo = feats(inp["pixel_values_videos"], inp["target_sizes_videos"], **fwd_kw)
            t["merger"] = torch.cat(vo.pooler_output, 0)
        ids = inp["input_ids"][0]
        t["input_ids"] = f32(ids)
        for k in ("pixel_values", "target_sizes", "pixel_values_videos", "target_sizes_videos"):
            if k in inp:
                t[k] = f32(inp[k][0] if k.startswith("pixel") else inp[k])
        t["logits_last"] = out.logits[0, -LOGIT_ROWS:].float()
        with torch.no_grad():
            gen = model.generate(**inp, **fwd_kw, max_new_tokens=GREEDY_TOKENS, do_sample=False)
        cont = gen[0, ids.shape[0]:]
        t["greedy"] = f32(cont)
        entry = {"images": names, "video": video, "options": opts, "prompt": text, "question": question,
                 "greedy_text": proc.tokenizer.decode(cont, skip_special_tokens=False)}
        if case == "room_16x":
            # Label loss over the greedy answer appended to the prompt: the
            # quantity a fine-tuner minimises, prompt rows masked with -100.
            full = torch.cat([ids, cont]).unsqueeze(0)
            labels = full.clone()
            labels[0, : ids.shape[0]] = -100
            lin = dict(inp)
            lin["input_ids"] = full
            lin["attention_mask"] = torch.ones_like(full)
            with torch.no_grad():
                lo = model(**lin, labels=labels, **fwd_kw)
            t["loss_ids"] = f32(full[0])
            t["loss"] = f32([lo.loss.item()])
            entry["loss"] = lo.loss.item()
        if video:
            entry["video_meta"] = {"total_num_frames": 60, "fps": 10.0, "duration": 6.0, "height": 240, "width": 320}
            save_file({"frames": f32(clip)}, os.path.join(out_dir, "video", "clip.safetensors"))
        save_file({k: v.float().contiguous() for k, v in t.items()}, os.path.join(out_dir, "model", f"{case}.safetensors"))
        manifest["model"][case] = entry
        print(f"model {case}: T={ids.shape[0]} greedy={entry['greedy_text']!r}")


def stack_frames_defect(proc, model, manifest):
    """Record whether `stack_frames>1` survives the full reference path. The
    reference builds per-second composites but then processes the raw frames,
    so the placeholder count and the feature count can disagree."""
    clip = synthetic_clip(60, 240, 320)
    md = VideoMetadata(total_num_frames=60, fps=10.0, duration=6.0, width=320, height=240)
    text = prompt_for(proc, [{"type": "video"}, {"type": "text", "text": "What happens?"}])
    inp = proc(text=[text], videos=[clip], videos_kwargs=dict(video_metadata=[md], do_sample_frames=True, stack_frames=3), return_tensors="pt")
    n_tokens = int((inp["input_ids"][0] == VIDEO_TOKEN_ID).sum())
    try:
        with torch.no_grad():
            model(**inp)
        verdict = "ran"
    except Exception as e:  # the defect is the finding, recorded verbatim
        verdict = f"{type(e).__name__}: {e}"
    manifest["stack_frames_3"] = {"video_tokens": n_tokens, "units": int(inp["target_sizes_videos"].shape[0]), "forward": verdict[:400]}
    print("stack_frames=3:", manifest["stack_frames_3"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local openbmb/MiniCPM-V-4.6 snapshot directory")
    ap.add_argument("--out", default=os.path.join(REPO, "testdata", "minicpmv"))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval()
    cfg = model.config
    manifest = {
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "image_token_id": IMAGE_TOKEN_ID,
        "video_token_id": VIDEO_TOKEN_ID,
        "logit_rows": LOGIT_ROWS,
        "processor": {},
        "model": {},
    }
    images = load_inputs(args.out)
    processor_cases(proc, images, args.out, manifest)
    sample_frame_cases(proc, args.out, manifest)
    model_cases(model, proc, images, args.out, manifest)
    stack_frames_defect(proc, model, manifest)

    weights = sorted(os.path.join(args.model, f) for f in os.listdir(args.model) if f.endswith(".safetensors"))
    tc, vc = cfg.text_config, cfg.vision_config
    manifest["source"] = source_block(
        checkpoint="openbmb/MiniCPM-V-4.6",
        files=weights,
        identity={
            "text_n_layers": tc.num_hidden_layers, "text_d_model": tc.hidden_size, "vocab_size": tc.vocab_size,
            "vision_n_layers": vc.num_hidden_layers, "vision_d_model": vc.hidden_size, "insert_layer_id": cfg.insert_layer_id,
        },
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
