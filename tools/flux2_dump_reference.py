#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump FLUX.2 Klein reference goldens for brain parity tests.

Runs the diffusers Flux2KleinPipeline (CPU, fp32) and dumps stage goldens:
  schedule.safetensors  sigma schedules + mu for several (H, W, steps) combos
  text.safetensors      chat-templated ids + per-layer hidden taps + concat ctx
  vae.safetensors       deterministic image -> moments/mean/packed+bn latent -> decode
  dit.safetensors       transformer I/O captured by forward hook (t2i step 0)
  dit_edit.safetensors  transformer I/O with one reference image (edit step 0)
  e2e.safetensors       per-step latents + final latent + decoded image (t2i)
  e2e_edit.safetensors  same for the edit run
  manifest.json         shapes, sha256 per file, run parameters, versions

Usage:
  python tools/flux2_dump_reference.py --weights <FLUX.2-klein-4B dir> \
      --out testdata/flux2/klein-4b [--height 512 --width 512 --steps 4 --seed 42]
"""

import argparse, hashlib, json, os, sys

import torch
from safetensors.torch import save_file

PROMPT = "a red fox sitting on a mossy rock in a misty forest, morning light"
TAP_LAYERS = [9, 18, 27]


def det_image(h, w):
    """Deterministic RGB test pattern in [-1, 1], shape (3, h, w)."""
    ys = torch.linspace(0, 3.14159, h).unsqueeze(1).expand(h, w)
    xs = torch.linspace(0, 6.28318, w).unsqueeze(0).expand(h, w)
    r = torch.sin(xs + ys)
    g = torch.cos(2.0 * xs) * torch.sin(0.5 * ys)
    b = 2.0 * (ys / 3.14159) - 1.0
    return torch.stack([r, g, b], 0).contiguous()


def save(out, name, tensors, manifest):
    # everything as f32 — brain's safetensors reader is F32/F16/BF16-only, and
    # token ids / byte values are exactly representable
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    torch.manual_seed(args.seed)

    from diffusers import Flux2KleinPipeline
    import diffusers.pipelines.flux2.pipeline_flux2_klein as P

    print("loading pipeline fp32 ...", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(args.weights, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)

    # ---- schedule ------------------------------------------------------------
    sched = {}
    for (hh, ww, st) in [(args.height, args.width, args.steps),
                         (1024, 1024, 4), (1024, 1024, 50), (768, 1360, 4)]:
        seq = (hh // 16) * (ww // 16)
        mu = P.compute_empirical_mu(seq, st)
        import numpy as np
        sig_in = np.linspace(1.0, 1.0 / st, st)
        pipe.scheduler.set_timesteps(sigmas=sig_in, mu=mu)
        sched[f"sigmas_{hh}x{ww}_s{st}"] = pipe.scheduler.sigmas.clone()
        sched[f"mu_{hh}x{ww}_s{st}"] = torch.tensor([mu])
    save(args.out, "schedule.safetensors", sched, manifest)

    # ---- text ----------------------------------------------------------------
    tok, te = pipe.tokenizer, pipe.text_encoder
    templated = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    enc = tok(templated, return_tensors="pt", padding="max_length",
              truncation=True, max_length=512)
    with torch.no_grad():
        out = te(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                 output_hidden_states=True, use_cache=False)
    taps = {f"hidden_{k}": out.hidden_states[k][0] for k in TAP_LAYERS}
    ctx_manual = torch.cat([out.hidden_states[k][0] for k in TAP_LAYERS], dim=-1)
    with torch.no_grad():
        ctx_pipe = pipe.encode_prompt(prompt=PROMPT, device="cpu",
                                      num_images_per_prompt=1)
        ctx_pipe = ctx_pipe[0] if isinstance(ctx_pipe, tuple) else ctx_pipe
    diff = (ctx_manual - ctx_pipe[0]).abs().max().item()
    print(f"manual vs encode_prompt max abs diff: {diff:.3e}", flush=True)
    assert diff < 1e-4, "manual text path diverges from pipeline"
    save(args.out, "text.safetensors",
         {"input_ids": enc.input_ids[0].to(torch.int32),
          "template_bytes": torch.tensor(list(templated.encode()), dtype=torch.uint8),
          **taps, "ctx": ctx_manual}, manifest)

    # ---- vae -----------------------------------------------------------------
    vae = pipe.vae
    img = det_image(args.height, args.width).unsqueeze(0)
    with torch.no_grad():
        moments = vae.encode(img).latent_dist.parameters  # (1, 64, H/8, W/8)
        mean = moments.chunk(2, dim=1)[0]                 # (1, 32, H/8, W/8)
    b, c, hh, ww = mean.shape
    packed = mean.view(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4)
    packed = packed.reshape(b, c * 4, hh // 2, ww // 2)   # (1, 128, H/16, W/16)
    rm = vae.bn.running_mean.view(1, -1, 1, 1)
    rv = vae.bn.running_var.view(1, -1, 1, 1)
    eps = float(vae.config.batch_norm_eps)
    normed = (packed - rm) / torch.sqrt(rv + eps)
    denorm = normed * torch.sqrt(rv + eps) + rm
    unpacked = denorm.view(b, c, 2, 2, hh // 2, ww // 2).permute(0, 1, 4, 2, 5, 3)
    unpacked = unpacked.reshape(b, c, hh, ww)
    with torch.no_grad():
        dec = vae.decode(unpacked).sample                 # (1, 3, H, W)
    save(args.out, "vae.safetensors",
         {"image": img[0], "moments": moments[0], "latent_mean": mean[0],
          "latent_packed_norm": normed[0], "decoded": dec[0],
          "bn_running_mean": vae.bn.running_mean, "bn_running_var": vae.bn.running_var,
          "bn_eps": torch.tensor([eps])}, manifest)

    # ---- e2e runs with transformer-I/O hook + per-step latents ---------------
    def run(tag, ref_image):
        cap, step_latents = {}, []

        def pre_hook(mod, a, kw):
            if "hs" not in cap:
                cap["hs"] = kw["hidden_states"].clone()
                cap["ctx"] = kw["encoder_hidden_states"].clone()
                cap["timestep"] = kw["timestep"].clone().float()
                cap["img_ids"] = kw["img_ids"].clone().float()
                cap["txt_ids"] = kw["txt_ids"].clone().float()
            return None

        def post_hook(mod, a, kw, out):
            if "out" not in cap:
                cap["out"] = (out[0] if isinstance(out, tuple) else out.sample).clone()
            return None

        h1 = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        h2 = pipe.transformer.register_forward_hook(post_hook, with_kwargs=True)

        def cb(p, i, t, kwargs):
            step_latents.append(kwargs["latents"].clone())
            return kwargs

        gen = torch.Generator("cpu").manual_seed(args.seed)
        kw = dict(prompt=PROMPT, height=args.height, width=args.width,
                  num_inference_steps=args.steps, generator=gen,
                  output_type="np", callback_on_step_end=cb,
                  callback_on_step_end_tensor_inputs=["latents"])
        if ref_image is not None:
            kw["image"] = ref_image
        res = pipe(**kw)
        h1.remove(); h2.remove()
        image = torch.from_numpy(res.images[0])           # (H, W, 3) in [0,1]
        save(args.out, f"dit{tag}.safetensors", cap, manifest)
        lat = {f"latents_step{i}": l[0] for i, l in enumerate(step_latents)}
        save(args.out, f"e2e{tag}.safetensors", {**lat, "image": image}, manifest)

    run("", None)
    from PIL import Image
    import numpy as np
    ref_np = ((det_image(args.height, args.width).permute(1, 2, 0) + 1) * 127.5)
    ref_pil = Image.fromarray(ref_np.numpy().astype(np.uint8))
    run("_edit", [ref_pil])

    manifest["params"] = {"prompt": PROMPT, "seed": args.seed,
                          "height": args.height, "width": args.width,
                          "steps": args.steps, "tap_layers": TAP_LAYERS,
                          "weights": os.path.abspath(args.weights),
                          "torch": torch.__version__,
                          "diffusers": __import__("diffusers").__version__}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
