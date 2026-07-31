#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump CLIP-family reference goldens for brain's `crates/clip` parity ladder.

Three encoders, one crate (docs/imaging/plan.md phase 3b):

  clip_l/text.safetensors        SDXL `text_encoder`   — HF CLIPTextModel, 12L x 768,
                                 quick_gelu. Consumed by SDXL (penultimate) and by
                                 FLUX.1 (pooled EOS output).
  openclip_bigg/text.safetensors SDXL `text_encoder_2` — HF CLIPTextModelWithProjection
                                 (OpenCLIP ViT-bigG/14 text tower), 32L x 1280, gelu.
                                 Consumed by SDXL (penultimate + PROJECTED pooled).
  sdxl/cond.safetensors          the exact SDXL conditioning pair
                                 (concat penultimates -> 2048, pooled -> 1280),
                                 cross-checked against diffusers'
                                 StableDiffusionXLPipeline.encode_prompt.
  eva02_l336/image.safetensors   EVA02-CLIP-L/14-336 IMAGE tower (PuLID's `eva_clip`,
                                 the authoritative reference), 24L x 1024, 2D RoPE,
                                 SwiGLU + subln. Consumed by PuLID.
  tokenizer/ids.safetensors      CLIP BPE ids for a fixed string set (both SDXL
                                 tokenizers; they differ ONLY in the pad token).
  manifest.json                  every tensor's shape + dtype, sha256 per file,
                                 the reference config, and the run parameters.

Everything is CPU + fp32 with fixed seeds and fixed synthetic inputs, and every
tensor is stored as f32 (brain's safetensors reader is F32/F16/BF16-only; token
ids are exactly representable).

Per-STAGE taps are captured with forward hooks on the real modules, so the Rust
parity test is a pure replay and no convention is re-derived by hand.

Usage:
  python3 tools/clip_dump_reference.py \
      --sdxl   /path/to/sdxl-base-1.0 \
      --eva-ckpt /path/to/EVA02_CLIP_L_336_psz14_s6B.pt \
      --eva-code /path/to/PuLID          # dir containing the `eva_clip` package
      --out    testdata/clip
Any of --sdxl / --eva-ckpt may be omitted to dump only the other family.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

# Fixed synthetic inputs. Two prompts of different length so the EOS-pooling
# index and the causal mask are both exercised (EOS lands at row 0 col 17,
# row 1 col 6 — see `eos_index` in the goldens).
PROMPTS = [
    "a red fox sitting on a mossy rock in a misty forest, morning light",
    "a photo of a cat",
]
# Strings the tokenizer golden covers: casing, punctuation, digits, unicode,
# a >77-token string (truncation), and the empty prompt.
TOKENIZER_STRINGS = [
    "a red fox sitting on a mossy rock in a misty forest, morning light",
    "a photo of a cat",
    "",
    "Hello, World!  MiXeD Case   and   collapsed\twhitespace",
    "digits 0123456789 and symbols #@$%^&*()_+-=[]{}|;':\",./<>?",
    "café naïve 你好 \U0001f98a emoji",
    "a " * 120 + "truncated tail",
]
SEED = 0


# ---------------------------------------------------------------- io helpers


def save(out_dir, rel, tensors, manifest, extra=None):
    """Save `tensors` as f32 safetensors under `out_dir/rel` and manifest it."""
    tensors = {
        k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()
    }
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    entry = {
        "sha256": sha,
        "bytes": os.path.getsize(path),
        # dtype is f32 for every tensor by construction; shapes are the contract.
        "dtype": "F32",
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    if extra:
        entry.update(extra)
    manifest["files"][rel] = entry
    print(f"wrote {rel}: {len(tensors)} tensors, {os.path.getsize(path) / 1e6:.1f} MB",
          flush=True)


class Taps:
    """Collect module inputs/outputs by name; every tap is a forward hook."""

    def __init__(self):
        self.t = {}
        self.h = []

    def out(self, name, module, index=0):
        def hook(_m, _a, o):
            o = o[index] if isinstance(o, tuple) else o
            self.t[name] = o.detach().clone()

        self.h.append(module.register_forward_hook(hook))

    def inp(self, name, module, index=0):
        def hook(_m, a, _o):
            self.t[name] = a[index].detach().clone()

        self.h.append(module.register_forward_hook(hook))

    def seq_out(self, name, module):
        """Append every call's output — for modules invoked more than once."""
        bucket = self.t.setdefault(name, [])

        def hook(_m, _a, o):
            bucket.append((o[0] if isinstance(o, tuple) else o).detach().clone())

        self.h.append(module.register_forward_hook(hook))

    def seq_inp(self, name, module):
        bucket = self.t.setdefault(name, [])

        def hook(_m, a, _o):
            bucket.append(a[0].detach().clone())

        self.h.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.h:
            h.remove()
        self.h = []


# ------------------------------------------------------------- text towers


def dump_text_tower(tag, enc_dir, tok_dir, out_dir, manifest):
    """Dump one CLIP text tower with per-stage taps. Returns the pieces SDXL needs."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    with open(os.path.join(enc_dir, "config.json")) as f:
        cfg_json = json.load(f)
    arch = cfg_json["architectures"][0]
    cls = {
        "CLIPTextModel": CLIPTextModel,
        "CLIPTextModelWithProjection": CLIPTextModelWithProjection,
    }[arch]
    variant = "fp16" if os.path.exists(
        os.path.join(enc_dir, "model.fp16.safetensors")) else None
    te = cls.from_pretrained(enc_dir, variant=variant, dtype=torch.float32).eval()
    tok = CLIPTokenizer.from_pretrained(tok_dir)
    cfg = te.config
    n_layers = cfg.num_hidden_layers

    ti = tok(PROMPTS, padding="max_length", max_length=tok.model_max_length,
             truncation=True, return_tensors="pt")
    ids = ti.input_ids
    # SDXL passes NO attention mask: CLIP's mask is causal-only, and the pad
    # rows therefore cannot influence any content row. Recorded, not assumed.
    assert "attention_mask" in ti, "tokenizer must expose the mask we then drop"

    # transformers 5.x flattens CLIPTextModel onto the model itself, while
    # CLIPTextModelWithProjection still wraps one as `.text_model`.
    core = te.text_model if hasattr(te, "text_model") else te
    layers = core.encoder.layers
    taps = Taps()
    taps.out("embed", core.embeddings)
    taps.out("tok_embed", core.embeddings.token_embedding)
    taps.out("pos_embed", core.embeddings.position_embedding)
    l0 = layers[0]
    taps.out("l0_ln1", l0.layer_norm1)
    taps.out("l0_q", l0.self_attn.q_proj)
    taps.out("l0_k", l0.self_attn.k_proj)
    taps.out("l0_v", l0.self_attn.v_proj)
    taps.out("l0_attn_out", l0.self_attn.out_proj)
    taps.out("l0_ln2", l0.layer_norm2)
    taps.out("l0_mlp_fc1", l0.mlp.fc1)
    taps.out("l0_mlp_out", l0.mlp.fc2)
    taps.out("l0_out", l0)
    taps.inp("final_ln_in", core.final_layer_norm)
    for li in range(n_layers):
        taps.out(f"layer{li}_out", layers[li])

    with torch.no_grad():
        out = te(ids, output_hidden_states=True)
    taps.remove()

    hs = out.hidden_states
    assert len(hs) == n_layers + 1, f"{tag}: hidden_states {len(hs)} != {n_layers + 1}"
    # hidden_states[0] is the embedding output and hidden_states[k] is the output
    # of encoder.layers[k-1] -- proven here rather than assumed.
    assert torch.equal(hs[0], taps.t["embed"]), f"{tag}: hs[0] is not the embedding out"
    for li in range(n_layers):
        assert torch.equal(hs[li + 1], taps.t[f"layer{li}_out"]), \
            f"{tag}: hs[{li + 1}] is not layers[{li}] out"
    # last_hidden_state = final_layer_norm(hidden_states[-1]); the hidden-state
    # tuple is PRE-final-LN, which is what makes "penultimate" un-normalized.
    assert torch.equal(taps.t["final_ln_in"], hs[-1]), f"{tag}: final LN input != hs[-1]"
    lhs = out.last_hidden_state
    assert torch.allclose(core.final_layer_norm(hs[-1]), lhs, atol=0, rtol=0), \
        f"{tag}: last_hidden_state != final_layer_norm(hs[-1])"

    # EOS pooling. CLIP's eot id (49407) is the largest id in the sequence, so
    # transformers' eos_token_id==2 branch uses argmax(ids); assert that this
    # coincides with "first occurrence of the real eot id" for these inputs.
    eot_id = int(tok.eos_token_id)
    argmax_idx = ids.argmax(dim=-1)
    first_eot = (ids == eot_id).int().argmax(dim=-1)
    assert torch.equal(argmax_idx, first_eot), f"{tag}: argmax pooling index != first eot"
    pooled = lhs[torch.arange(lhs.shape[0]), argmax_idx]
    # CLIPTextModel exposes it as `pooler_output`; CLIPTextModelWithProjection
    # returns only the PROJECTED `text_embeds`, so the pre-projection pooled
    # vector is reconstructed here and validated through the projection below.
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        assert torch.equal(out.pooler_output, pooled), \
            f"{tag}: pooler_output != lhs[eos]"

    penult = hs[-2]  # == hs[n_layers - 1] == output of encoder.layers[n_layers - 2]
    assert torch.equal(penult, taps.t[f"layer{n_layers - 2}_out"]), \
        f"{tag}: penultimate index mismatch"

    tensors = {
        "input_ids": ids.to(torch.int32),
        "attention_mask": ti.attention_mask.to(torch.int32),
        "eos_index": argmax_idx.to(torch.int32),
        "tok_embed": taps.t["tok_embed"],
        "pos_embed": taps.t["pos_embed"],
        "embed": taps.t["embed"],
        "l0_ln1": taps.t["l0_ln1"],
        "l0_q": taps.t["l0_q"],
        "l0_k": taps.t["l0_k"],
        "l0_v": taps.t["l0_v"],
        "l0_attn_out": taps.t["l0_attn_out"],
        "l0_ln2": taps.t["l0_ln2"],
        "l0_mlp_fc1": taps.t["l0_mlp_fc1"],
        "l0_mlp_out": taps.t["l0_mlp_out"],
        "l0_out": taps.t["l0_out"],
        "last_hidden_state": lhs,
        "penultimate": penult,
        "pooled": pooled,
    }
    for li in range(n_layers):
        tensors[f"layer{li}_out"] = taps.t[f"layer{li}_out"]

    proj = None
    if arch == "CLIPTextModelWithProjection":
        proj = out.text_embeds
        assert torch.equal(proj, te.text_projection(pooled)), \
            f"{tag}: text_embeds != text_projection(pooled)"
        tensors["text_embeds"] = proj

    save(out_dir, f"{tag}/text.safetensors", tensors, manifest, extra={
        "reference": f"transformers {arch}",
        "weights": os.path.basename(enc_dir),
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "max_position_embeddings": cfg.max_position_embeddings,
            "vocab_size": cfg.vocab_size,
            "hidden_act": cfg.hidden_act,
            "layer_norm_eps": cfg.layer_norm_eps,
            "projection_dim": cfg.projection_dim,
            "eos_token_id": cfg.eos_token_id,
            "pad_token_id": cfg.pad_token_id,
        },
        "notes": {
            "hidden_states": "hs[0]=embeddings out, hs[k]=encoder.layers[k-1] out, "
                             "all PRE final_layer_norm",
            "penultimate_index": f"hidden_states[-2] == hidden_states[{n_layers - 1}] "
                                 f"== output of encoder.layers[{n_layers - 2}] "
                                 f"(0-based) of {n_layers}; NOT layer-normed",
            "pooled": "last_hidden_state[argmax(input_ids)] after final_layer_norm",
            "attention": "causal mask only; SDXL passes attention_mask=None",
            "tokenizer_pad_token": tok.pad_token,
            "tokenizer_pad_id": int(tok.pad_token_id),
        },
    })
    return {"penultimate": penult, "pooled": pooled, "text_embeds": proj,
            "ids": ids, "model": te, "tokenizer": tok, "arch": arch}


def dump_sdxl_cond(sdxl_dir, out_dir, manifest, one, two):
    """The exact SDXL conditioning pair, cross-checked against diffusers."""
    cond = torch.cat([one["penultimate"], two["penultimate"]], dim=-1)
    pooled = two["text_embeds"]

    checked = False
    try:
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

        sched = EulerDiscreteScheduler.from_pretrained(sdxl_dir, subfolder="scheduler")
        pipe = StableDiffusionXLPipeline(
            vae=None, text_encoder=one["model"], text_encoder_2=two["model"],
            tokenizer=one["tokenizer"], tokenizer_2=two["tokenizer"],
            unet=None, scheduler=sched, force_zeros_for_empty_prompt=True)
        pipe.set_progress_bar_config(disable=True)
        with torch.no_grad():
            pe, _, ppe, _ = pipe.encode_prompt(
                prompt=PROMPTS, device="cpu", num_images_per_prompt=1,
                do_classifier_free_guidance=False)
        d_cond = (pe - cond).abs().max().item()
        d_pool = (ppe - pooled).abs().max().item()
        print(f"encode_prompt cross-check: cond {d_cond:.3e}, pooled {d_pool:.3e}",
              flush=True)
        assert d_cond == 0.0 and d_pool == 0.0, "manual SDXL path diverges from diffusers"
        checked = True
    except ImportError as e:  # diffusers absent -> record it, never fake it
        print(f"WARNING: diffusers unavailable ({e}); SDXL cond NOT cross-checked",
              flush=True)

    save(out_dir, "sdxl/cond.safetensors",
         {"prompt_embeds": cond, "pooled_prompt_embeds": pooled,
          "input_ids_1": one["ids"].to(torch.int32),
          "input_ids_2": two["ids"].to(torch.int32)},
         manifest, extra={
             "reference": "diffusers StableDiffusionXLPipeline.encode_prompt",
             "cross_checked_against_diffusers": checked,
             "notes": {
                 "prompt_embeds": "concat([clip_l penultimate (768), "
                                  "bigG penultimate (1280)], dim=-1) -> 2048",
                 "pooled_prompt_embeds": "bigG text_projection(pooled EOS) -> 1280; "
                                         "CLIP-L's pooled output is NOT used by SDXL",
             },
         })


def dump_tokenizer(sdxl_dir, out_dir, manifest):
    from transformers import CLIPTokenizer

    tensors, notes = {}, {}
    for name, sub in (("tok1", "tokenizer"), ("tok2", "tokenizer_2")):
        tok = CLIPTokenizer.from_pretrained(os.path.join(sdxl_dir, sub))
        enc = tok(TOKENIZER_STRINGS, padding="max_length",
                  max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt")
        tensors[f"{name}_ids_padded"] = enc.input_ids.to(torch.int32)
        tensors[f"{name}_mask"] = enc.attention_mask.to(torch.int32)
        # unpadded lengths, so a Rust tokenizer can be gated before padding rules
        lens = [len(tok(s, truncation=True,
                        max_length=tok.model_max_length).input_ids)
                for s in TOKENIZER_STRINGS]
        tensors[f"{name}_len"] = torch.tensor(lens, dtype=torch.int32)
        notes[name] = {
            "vocab_size": tok.vocab_size,
            "model_max_length": tok.model_max_length,
            "bos": tok.bos_token, "bos_id": int(tok.bos_token_id),
            "eos": tok.eos_token, "eos_id": int(tok.eos_token_id),
            "pad": tok.pad_token, "pad_id": int(tok.pad_token_id),
        }
    save(out_dir, "tokenizer/ids.safetensors", tensors, manifest, extra={
        "reference": "transformers CLIPTokenizer (slow, byte-level BPE)",
        "strings": TOKENIZER_STRINGS,
        "tokenizers": notes,
        "notes": {
            "difference": "tokenizer and tokenizer_2 share vocab.json/merges.txt "
                          "and differ ONLY in the pad token (<|endoftext|>=49407 "
                          "vs '!'=0)",
        },
    })


# ------------------------------------------------------------- EVA02 image


def det_image(h, w):
    """Deterministic RGB test pattern in [0, 1], shape (3, h, w)."""
    ys = torch.linspace(0.0, 3.14159265, h).unsqueeze(1).expand(h, w)
    xs = torch.linspace(0.0, 6.28318531, w).unsqueeze(0).expand(h, w)
    r = 0.5 + 0.5 * torch.sin(xs + ys)
    g = 0.5 + 0.5 * torch.cos(2.0 * xs) * torch.sin(0.5 * ys)
    b = ys / 3.14159265
    return torch.stack([r, g, b], 0).contiguous()


def dump_eva(ckpt, code_dir, out_dir, manifest):
    sys.path.insert(0, code_dir)
    from eva_clip import create_model
    from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    name = "EVA02-CLIP-L-14-336"
    model = create_model(name, ckpt, force_custom_clip=True, precision="fp32",
                         device="cpu")
    v = model.visual.eval()
    with open(os.path.join(code_dir, "eva_clip", "model_configs",
                           f"{name}.json")) as f:
        eva_cfg = json.load(f)

    # The reference loads with strict=False and drops the checkpoint's RoPE
    # frequency buffers (they are recomputed). Prove nothing ELSE was skipped.
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck_visual = {k[len("visual."):] for k in raw if k.startswith("visual.")}
    have = set(v.state_dict().keys())
    missing = sorted(have - ck_visual)
    unused = sorted(ck_visual - have)
    assert all("freqs_" in m for m in missing), f"unexpected missing visual keys: {missing[:8]}"
    assert all("freqs_" in u for u in unused), f"unexpected unused visual keys: {unused[:8]}"
    del raw

    mean = torch.tensor(getattr(v, "image_mean", OPENAI_DATASET_MEAN)).view(3, 1, 1)
    std = torch.tensor(getattr(v, "image_std", OPENAI_DATASET_STD)).view(3, 1, 1)
    size = v.image_size if isinstance(v.image_size, int) else v.image_size[0]
    raw_img = det_image(size, size)
    px = ((raw_img - mean) / std).unsqueeze(0)

    n_blocks = len(v.blocks)
    taps = Taps()
    taps.out("patch_embed", v.patch_embed)
    for bi in range(n_blocks):
        taps.out(f"block{bi}_out", v.blocks[bi])
    b0 = v.blocks[0]
    taps.inp("b0_in", b0)
    taps.out("b0_norm1", b0.norm1)
    # q/k/v are NOT module calls in the reference (`F.linear(x, w, bias)` with an
    # asymmetric bias set), so no hook fires on q_proj/k_proj/v_proj; they are
    # recomputed below from the tapped attention input using the same expression.
    taps.inp("b0_attn_in", b0.attn)
    taps.inp("b0_inner_ln_in", b0.attn.inner_attn_ln)
    taps.out("b0_inner_ln", b0.attn.inner_attn_ln)
    taps.out("b0_attn_proj", b0.attn.proj)
    taps.out("b0_norm2", b0.norm2)
    taps.out("b0_mlp_w1", b0.mlp.w1)
    taps.out("b0_mlp_w2", b0.mlp.w2)
    taps.out("b0_mlp_ffn_ln", b0.mlp.ffn_ln)
    taps.out("b0_mlp_out", b0.mlp)
    # `rope` is one shared module invoked twice per block (q then k); record the
    # first two calls, which are block 0's q and k.
    taps.seq_inp("rope_in", v.rope)
    taps.seq_out("rope_out", v.rope)
    taps.out("norm_out", v.norm)
    taps.inp("head_in", v.head)
    taps.out("head_out", v.head)

    with torch.no_grad():
        cls_out, hidden = v(px, return_all_features=False, return_hidden=True,
                            shuffle=False)
    taps.remove()

    # Block-0 q/k/v, recomputed with the reference's own bias asymmetry:
    #   q = x @ q_proj.W^T + q_bias ; k = x @ k_proj.W^T (NO bias) ; v = ... + v_bias
    a0, x0 = b0.attn, taps.t["b0_attn_in"]
    assert torch.equal(x0, taps.t["b0_norm1"]), "attn input is not norm1 out"
    assert a0.k_proj.bias is None and a0.q_proj.bias is None, "unexpected fused qkv bias"
    with torch.no_grad():
        b0_q = torch.nn.functional.linear(x0, a0.q_proj.weight, a0.q_bias)
        b0_k = torch.nn.functional.linear(x0, a0.k_proj.weight, None)
        b0_v = torch.nn.functional.linear(x0, a0.v_proj.weight, a0.v_bias)
    # ...and prove the recomputation is the real thing: replaying the tail of the
    # reference attention from these q/k/v must reproduce the tapped inner_attn_ln
    # input exactly.
    B, N, C = x0.shape
    nh = a0.num_heads
    with torch.no_grad():
        qh = b0_q.reshape(B, N, nh, -1).permute(0, 2, 1, 3)
        kh = b0_k.reshape(B, N, nh, -1).permute(0, 2, 1, 3)
        vh = b0_v.reshape(B, N, nh, -1).permute(0, 2, 1, 3)
        qh = torch.cat((qh[:, :, :1], a0.rope(qh[:, :, 1:])), -2).type_as(vh)
        kh = torch.cat((kh[:, :, :1], a0.rope(kh[:, :, 1:])), -2).type_as(vh)
        probs = ((qh * a0.scale) @ kh.transpose(-2, -1)).softmax(dim=-1)
        ctx = (probs @ vh).transpose(1, 2).reshape(B, N, -1)
    d = (ctx - taps.t["b0_inner_ln_in"]).abs().max().item()
    print(f"eva block0 attention replay max abs diff: {d:.3e}", flush=True)
    assert d == 0.0, "recomputed block-0 q/k/v do not reproduce the reference"

    # PuLID taps the INPUT of blocks 4/8/12/16/20 (`0 < idx <= 20 and idx % 4 == 0`),
    # i.e. the output of blocks 3/7/11/15/19.
    pulid_idx = [i for i in range(n_blocks) if 0 < i <= 20 and i % 4 == 0]
    assert len(hidden) == len(pulid_idx) == 5, f"PuLID hidden count {len(hidden)}"
    for j, bi in enumerate(pulid_idx):
        assert torch.equal(hidden[j], taps.t[f"block{bi - 1}_out"]), \
            f"PuLID hidden[{j}] is not block{bi - 1} out"

    tensors = {
        "image_raw": raw_img,
        "image_mean": mean.view(3),
        "image_std": std.view(3),
        "pixel_values": px[0],
        "cls_token": v.cls_token[0, 0],
        "pos_embed": v.pos_embed[0],
        "rope_freqs_cos": v.rope.freqs_cos,
        "rope_freqs_sin": v.rope.freqs_sin,
        "patch_embed": taps.t["patch_embed"][0],
        "block_in": taps.t["b0_in"][0],
        "b0_norm1": taps.t["b0_norm1"][0],
        "b0_q": b0_q[0],
        "b0_k": b0_k[0],
        "b0_v": b0_v[0],
        "b0_rope_q_in": taps.t["rope_in"][0][0],
        "b0_rope_q_out": taps.t["rope_out"][0][0],
        "b0_rope_k_in": taps.t["rope_in"][1][0],
        "b0_rope_k_out": taps.t["rope_out"][1][0],
        "b0_attn_ctx": taps.t["b0_inner_ln_in"][0],
        "b0_inner_ln": taps.t["b0_inner_ln"][0],
        "b0_attn_proj": taps.t["b0_attn_proj"][0],
        "b0_norm2": taps.t["b0_norm2"][0],
        "b0_mlp_w1": taps.t["b0_mlp_w1"][0],
        "b0_mlp_w2": taps.t["b0_mlp_w2"][0],
        "b0_mlp_ffn_ln": taps.t["b0_mlp_ffn_ln"][0],
        "b0_mlp_out": taps.t["b0_mlp_out"][0],
        "norm_out": taps.t["norm_out"][0],
        "head_in": taps.t["head_in"][0],
        "head_out": taps.t["head_out"][0],
        "cls_embed": cls_out[0],
        "cls_embed_l2norm": (cls_out / cls_out.norm(2, 1, True))[0],
    }
    for bi in range(n_blocks):
        tensors[f"block{bi}_out"] = taps.t[f"block{bi}_out"][0]
    for j, bi in enumerate(pulid_idx):
        tensors[f"pulid_hidden{j}"] = hidden[j][0]

    save(out_dir, "eva02_l336/image.safetensors", tensors, manifest, extra={
        "reference": f"PuLID eva_clip create_model('{name}') .visual",
        "weights": os.path.basename(ckpt),
        "config": eva_cfg["vision_cfg"],
        "derived": {
            "num_heads": eva_cfg["vision_cfg"]["width"]
            // eva_cfg["vision_cfg"]["head_width"],
            "head_dim": eva_cfg["vision_cfg"]["head_width"],
            "rope_dim_half": eva_cfg["vision_cfg"]["head_width"] // 2,
            "mlp_hidden": int(eva_cfg["vision_cfg"]["width"]
                              * eva_cfg["vision_cfg"]["mlp_ratio"]),
            "num_patches": v.patch_embed.num_patches,
            "seq_len": v.patch_embed.num_patches + 1,
            "embed_dim": eva_cfg["embed_dim"],
        },
        "notes": {
            "architecture": "NOT vanilla ViT: qkv is three bias-asymmetric linears "
                            "(q has q_bias, k has NO bias, v has v_bias), attention "
                            "output passes inner_attn_ln (subln) before proj, the "
                            "MLP is naive SwiGLU (w1 silu-gated by w2, ffn_ln, w3), "
                            "and there is no layer_scale (gamma_1/gamma_2 absent)",
            "rope": "2D axial RoPE applied to q and k EXCLUDING the cls token "
                    "(tokens 1..576 only); freqs are recomputed by the reference, "
                    "not loaded from the checkpoint (pt_seq_len=16 -> ft_seq_len=24 "
                    "interpolation)",
            "act": "SwiGLU act is nn.SiLU",
            "pooling": "use_mean_pooling=False -> norm(x)[:, 0] then head (1024->768)",
            "pulid_hidden": f"outputs of blocks {[i - 1 for i in pulid_idx]} "
                            f"(= inputs of blocks {pulid_idx})",
            "pulid_id_cond_vit": "cls_embed L2-normalized (cls_embed_l2norm)",
        },
    })


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdxl", help="SDXL base dir (text_encoder{,_2}, tokenizer{,_2})")
    ap.add_argument("--eva-ckpt", help="EVA02_CLIP_L_336_psz14_s6B.pt")
    ap.add_argument("--eva-code", help="dir containing the `eva_clip` package (PuLID)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.sdxl and not args.eva_ckpt:
        ap.error("nothing to do: pass --sdxl and/or --eva-ckpt")

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"files": {}, "params": {
        "prompts": PROMPTS, "seed": SEED,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }}

    if args.sdxl:
        one = dump_text_tower("clip_l", os.path.join(args.sdxl, "text_encoder"),
                              os.path.join(args.sdxl, "tokenizer"), args.out, manifest)
        two = dump_text_tower("openclip_bigg",
                              os.path.join(args.sdxl, "text_encoder_2"),
                              os.path.join(args.sdxl, "tokenizer_2"), args.out, manifest)
        dump_sdxl_cond(args.sdxl, args.out, manifest, one, two)
        dump_tokenizer(args.sdxl, args.out, manifest)
        manifest["params"]["sdxl_weights"] = os.path.abspath(args.sdxl)
        try:
            manifest["params"]["diffusers"] = __import__("diffusers").__version__
        except ImportError:
            manifest["params"]["diffusers"] = None
        del one, two

    if args.eva_ckpt:
        if not args.eva_code:
            ap.error("--eva-ckpt needs --eva-code (the dir holding `eva_clip`)")
        dump_eva(args.eva_ckpt, args.eva_code, args.out, manifest)
        manifest["params"]["eva_weights"] = os.path.abspath(args.eva_ckpt)
        manifest["params"]["eva_code"] = os.path.abspath(args.eva_code)
        manifest["params"]["timm"] = __import__("timm").__version__

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
