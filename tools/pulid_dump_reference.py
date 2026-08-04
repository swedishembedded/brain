#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump PuLID-FLUX v0.9.1 reference goldens for brain's `crates/pulid` parity tests.

Three files, each gating a different rung of the ladder:

  idformer.safetensors   the ID-embedding pipeline: `IDFormer(id_cond, id_vit_hidden)`
                         -> the 32 projected ID tokens, plus every internal stage.
                         **The inputs are read from the fixtures brain already
                         gates**: the ArcFace 512-d embedding from
                         `face/antelopev2/e2e.safetensors` (`photo0_embedding`,
                         insightface antelopev2 `glintr100` — the SAME model
                         `PuLIDPipeline.get_id_embedding` calls) and the EVA-CLIP
                         L2-normalized cls embedding + 5 tapped hidden states from
                         `clip/eva02_l336/image.safetensors`. So a brain run that
                         reproduces this golden has composed its OWN parity-gated
                         ArcFace and EVA-CLIP towers into the ID embedding.

  ca.safetensors         `PerceiverAttentionCA(id_tokens, img_tokens)` for the
                         first and last of the 20 cross-attention modules, with
                         module 0's internals tapped, on a deterministic
                         `img` slab.

  flux_cond.safetensors  ONE conditioned transformer evaluation: the reduced-depth
                         FLUX.1 transformer (the same truncation
                         `tools/flux1_dump_reference.py` dumps as `dit_small`)
                         replayed on the SAME inputs, once WITHOUT the ID
                         (self-validation: must reproduce `dit_small`'s `out`
                         bit-for-bit) and once WITH it, following the reference
                         injection rule transcribed from PuLID's `flux/model.py`.

Injection rule (PuLID `flux/model.py::Flux.forward`, v0.9.1), transcribed here and
asserted by `crates/pulid`'s schedule test:

    ca_idx = 0
    for i, block in enumerate(double_blocks):
        img, txt = block(...)
        if i % 2 == 0 and id is not None:
            img = img + id_weight * pulid_ca[ca_idx](id, img); ca_idx += 1
    ...
    for i, block in enumerate(single_blocks):
        ... real_img, txt = split(x)
        if i % 4 == 0 and id is not None:
            real_img = real_img + id_weight * pulid_ca[ca_idx](id, real_img); ca_idx += 1

i.e. the ID contribution is ADDED TO THE IMAGE RESIDUAL STREAM (never concatenated
as tokens), the image rows are the QUERIES and the ID tokens the KEYS/VALUES, and
`ca_idx` is a single sequential counter shared by both loops.

Usage:
  python tools/pulid_dump_reference.py \
      --pulid   /path/to/pulid_flux_v0.9.1.safetensors \
      --code    /path/to/PuLID \
      --testdata testdata \
      [--transformer /path/to/FLUX.1-Kontext-dev/transformer] \
      [--small-double 2 --small-single 2 --id-weight 1.0 --seed 42]

`--transformer` is optional; without it the `flux_cond` golden is skipped.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

DOUBLE_INTERVAL = 2
SINGLE_INTERVAL = 4


def save(out_dir, name, tensors, manifest, extra=None):
    tensors = {
        k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()
    }
    path = os.path.join(out_dir, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {
        "sha256": h,
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    if extra:
        manifest[name].update(extra)
    keys = list(tensors)
    head = ", ".join(f"{k}{list(tensors[k].shape)}" for k in keys[:5])
    print(f"wrote {name}: {len(keys)} tensors [{head}{' ...' if len(keys) > 5 else ''}]",
          flush=True)


def split_state_dict(sd):
    """`pulid_encoder.*` / `pulid_ca.*` -> per-module state dicts."""
    out = {}
    for k, v in sd.items():
        mod = k.split(".")[0]
        out.setdefault(mod, {})[k[len(mod) + 1:]] = v.to(torch.float32)
    return out


# --------------------------------------------------------------------------
# 1. the ID-embedding pipeline
# --------------------------------------------------------------------------


def dump_idformer(enc, testdata, out_dir, manifest):
    face = os.path.join(testdata, "face/antelopev2/e2e.safetensors")
    eva = os.path.join(testdata, "clip/eva02_l336/image.safetensors")
    for p in (face, eva):
        if not os.path.exists(p):
            sys.exit(f"missing fixture {p} (run tools/arcface_dump_reference.py / "
                     f"tools/clip_dump_reference.py first)")
    f = load_file(face)
    e = load_file(eva)

    # `PuLIDPipeline.get_id_embedding`:
    #   id_ante_embedding = face_info['embedding']          (raw, NOT normalized)
    #   id_cond_vit       = clip_vision(...)  / ||.||_2     (L2-normalized cls)
    #   id_cond           = cat([id_ante_embedding, id_cond_vit], dim=-1)
    arc = f["photo0_embedding"].to(torch.float32).view(1, -1)
    vit = e["cls_embed_l2norm"].to(torch.float32).view(1, -1)
    assert arc.shape[1] == 512 and vit.shape[1] == 768, (arc.shape, vit.shape)
    id_cond = torch.cat([arc, vit], dim=-1)
    hidden = [e[f"pulid_hidden{j}"].to(torch.float32).unsqueeze(0) for j in range(5)]
    assert all(h.shape[1:] == (577, 1024) for h in hidden), [h.shape for h in hidden]

    caps = {}
    with torch.no_grad():
        # Re-derive the forward stage by stage so every intermediate is tapped;
        # asserted against the reference module's own output at the end.
        x = enc.id_embedding_mapping(id_cond)
        x = x.reshape(-1, enc.num_id_token, enc.dim)          # [1, 5, 1024]
        caps["id_tokens"] = x[0].clone()
        latents = enc.latents.repeat(id_cond.size(0), 1, 1)   # [1, 32, 1024]
        latents = torch.cat((latents, x), dim=1)              # [1, 37, 1024]
        caps["latents_in"] = latents[0].clone()
        li = 0
        for i in range(5):
            vf = getattr(enc, f"mapping_{i}")(hidden[i])
            caps[f"map{i}_out"] = vf[0].clone()
            ctx = torch.cat((x, vf), dim=1)                   # [1, 582, 1024]
            if i == 0:
                caps["ctx0"] = ctx[0].clone()
            for attn, ff in enc.layers[i * enc.depth: (i + 1) * enc.depth]:
                latents = attn(ctx, latents) + latents
                caps[f"layer{li}_attn"] = latents[0].clone()
                latents = ff(latents) + latents
                caps[f"layer{li}_ff"] = latents[0].clone()
                li += 1
        assert li == 10, li
        out = latents[:, :enc.num_queries] @ enc.proj_out     # [1, 32, 2048]
        ref = enc(id_cond, hidden)
    d = (out - ref).abs().max().item()
    print(f"idformer staged vs module forward: max abs {d:.3e}", flush=True)
    assert d == 0.0, "staged IDFormer replay does not reproduce the module"

    caps["id_cond"] = id_cond[0]
    caps["arcface_embedding"] = arc[0]
    caps["eva_cls_l2norm"] = vit[0]
    for j, h in enumerate(hidden):
        caps[f"vit_hidden{j}"] = h[0]
    caps["id_embedding"] = out[0]
    save(out_dir, "idformer.safetensors", caps, manifest, extra={
        "reference": "PuLID pulid.encoders_transformer.IDFormer, pulid_flux_v0.9.1",
        "inputs_from": {
            "arcface_embedding": "testdata/face/antelopev2/e2e.safetensors:photo0_embedding",
            "eva_cls_l2norm": "testdata/clip/eva02_l336/image.safetensors:cls_embed_l2norm",
            "vit_hidden{0..4}": "testdata/clip/eva02_l336/image.safetensors:pulid_hidden{0..4}",
        },
        "notes": {
            "id_cond": "cat([arcface_512 (raw, unnormalized), eva_cls_768 (L2-normalized)])",
            "structure": "5 groups of depth//5 = 2 layers; group i sees "
                         "ctx = cat(id_tokens(5), mapping_i(vit_hidden[i])(577))",
            "latents": "cat(learned latents(32), id_tokens(5)) = 37 rows; "
                       "only the first 32 survive proj_out",
            "attn": "PerceiverAttention: q from latents, kv from "
                    "cat(norm1(ctx), norm2(latents)) = 619 rows, 16 heads x 64",
        },
    })
    return out


# --------------------------------------------------------------------------
# 2. the injected cross-attention module
# --------------------------------------------------------------------------


def dump_ca(cas, id_embedding, n_img, seed, out_dir, manifest):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, n_img, cas[0].norm2.weight.shape[0], generator=g) * 0.5
    caps = {"img": img[0], "id": id_embedding[0]}
    with torch.no_grad():
        m = cas[0]
        # taps of module 0, re-derived (asserted against the module below)
        xn = m.norm1(id_embedding)
        ln = m.norm2(img)
        q = m.to_q(ln)
        kv = m.to_kv(xn)
        caps["ca0_norm1_id"] = xn[0]
        caps["ca0_norm2_img"] = ln[0]
        caps["ca0_q"] = q[0]
        caps["ca0_kv"] = kv[0]
        for k in (0, len(cas) - 1):
            o = cas[k](id_embedding, img)
            caps[f"ca{k}_out"] = o[0]
        # ctx (pre `to_out`) of module 0
        from pulid.encoders_transformer import reshape_tensor
        import math
        kk, vv = kv.chunk(2, dim=-1)
        qh = reshape_tensor(q, m.heads)
        kh = reshape_tensor(kk, m.heads)
        vh = reshape_tensor(vv, m.heads)
        sc = 1 / math.sqrt(math.sqrt(m.dim_head))
        w = (qh * sc) @ (kh * sc).transpose(-2, -1)
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        ctx = (w @ vh).permute(0, 2, 1, 3).reshape(1, img.shape[1], -1)
        caps["ca0_ctx"] = ctx[0]
        d = (m.to_out(ctx) - caps["ca0_out"].unsqueeze(0)).abs().max().item()
    print(f"ca0 staged vs module forward: max abs {d:.3e}", flush=True)
    assert d == 0.0, "staged PerceiverAttentionCA replay does not reproduce the module"
    save(out_dir, "ca.safetensors", caps, manifest, extra={
        "reference": "PuLID pulid.encoders_transformer.PerceiverAttentionCA",
        "notes": {
            "call": "pulid_ca[k](x=id, latents=img) -> the IMAGE rows are the "
                    "QUERIES (dim 3072) and the ID tokens the KEYS/VALUES (kv_dim 2048)",
            "scale": "q and k are each scaled by dim_head**-0.25, i.e. the "
                     "product is the usual 1/sqrt(dim_head) = 1/sqrt(128)",
            "kv": "to_kv emits [n_id, 2*2048]; chunk(2, -1) => k at column 0, "
                  "v at column 2048 — exactly brain's fused-KV cross layout",
            "modules": f"{len(cas)} modules; only 0 and {len(cas) - 1} are dumped",
        },
    })
    return img


# --------------------------------------------------------------------------
# 3. one conditioned transformer evaluation
# --------------------------------------------------------------------------


def build_small_transformer(tr_dir, keep_d, keep_s):
    """A reduced-depth FluxTransformer2DModel holding the REAL first blocks."""
    import json as _json

    from diffusers import FluxTransformer2DModel

    with open(os.path.join(tr_dir, "config.json")) as f:
        cfg = _json.load(f)
    full_d, full_s = cfg["num_layers"], cfg["num_single_layers"]
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["num_layers"], cfg["num_single_layers"] = keep_d, keep_s
    model = FluxTransformer2DModel(**cfg)

    def keep(name):
        for pfx, lim in (("transformer_blocks.", keep_d),
                         ("single_transformer_blocks.", keep_s)):
            if name.startswith(pfx):
                return int(name[len(pfx):].split(".")[0]) < lim
        return True

    sd, seen = {}, 0
    for fn in sorted(os.listdir(tr_dir)):
        if not fn.endswith(".safetensors"):
            continue
        shard = load_file(os.path.join(tr_dir, fn))
        seen += len(shard)
        for k, v in shard.items():
            if keep(k):
                sd[k] = v.to(torch.float32)
        del shard
    print(f"transformer: {seen} tensors on disk, {len(sd)} kept at depth "
          f"{keep_d}+{keep_s} (full {full_d}+{full_s})", flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    return model.eval(), full_d, full_s


def dump_flux_cond(tr_dir, cas, id_embedding, testdata, keep_d, keep_s, id_weight,
                   out_dir, manifest):
    small = os.path.join(testdata, "flux1/kontext-dev/dit_small.safetensors")
    if not os.path.exists(small):
        sys.exit(f"missing fixture {small} (run tools/flux1_dump_reference.py)")
    fx = load_file(small)
    model, full_d, full_s = build_small_transformer(tr_dir, keep_d, keep_s)

    kw = dict(
        hidden_states=fx["hs"].unsqueeze(0).float(),
        encoder_hidden_states=fx["ctx"].unsqueeze(0).float(),
        pooled_projections=fx["pooled"].unsqueeze(0).float(),
        timestep=fx["timestep"].float(),
        guidance=fx["guidance"].float(),
        img_ids=fx["img_ids"].float(),
        txt_ids=fx["txt_ids"].float(),
        return_dict=False,
    )

    def run(active_id):
        """`active_id=None` -> plain transformer; otherwise the PuLID injection."""
        state = {"ca_idx": 0}
        taps = {}
        handles = []

        def mk(kind, i, interval):
            def hook(mod, args, kwargs, out):
                ehs, hs = out
                if active_id is not None and i % interval == 0:
                    ca = cas[state["ca_idx"]]
                    state["ca_idx"] += 1
                    hs = hs + id_weight * ca(active_id, hs)
                taps[f"{kind}{i}_img"] = hs[0].detach().clone()
                taps[f"{kind}{i}_txt"] = ehs[0].detach().clone()
                return (ehs, hs)
            return hook

        for i, b in enumerate(model.transformer_blocks):
            handles.append(b.register_forward_hook(mk("db", i, DOUBLE_INTERVAL),
                                                   with_kwargs=True))
        for i, b in enumerate(model.single_transformer_blocks):
            handles.append(b.register_forward_hook(mk("sg", i, SINGLE_INTERVAL),
                                                   with_kwargs=True))
        with torch.no_grad():
            out = model(**kw)[0][0]
        for h in handles:
            h.remove()
        return out, taps, state["ca_idx"]

    base, base_taps, n0 = run(None)
    assert n0 == 0
    d = (base - fx["out"]).abs().max().item()
    print(f"self-validation: reduced-depth replay vs dit_small `out` max abs {d:.3e}",
          flush=True)
    assert d < 1e-4, "the harness does not reproduce the flux1 golden — inputs differ"

    cond, cond_taps, n_used = run(id_embedding)
    n_expect = len(range(0, keep_d, DOUBLE_INTERVAL)) + len(range(0, keep_s, SINGLE_INTERVAL))
    assert n_used == n_expect, (n_used, n_expect)
    delta = (cond - base).abs().max().item()
    print(f"conditioned vs unconditioned prediction: max abs {delta:.4f} "
          f"({n_used} CA modules fired)", flush=True)
    assert delta > 1e-3, "the ID injection changed nothing — the golden would be vacuous"

    caps = {"id": id_embedding[0], "out_uncond": base, "out_cond": cond,
            "id_weight": torch.tensor([id_weight])}
    caps.update({f"uncond_{k}": v for k, v in base_taps.items()})
    caps.update({f"cond_{k}": v for k, v in cond_taps.items()})
    save(out_dir, "flux_cond.safetensors", caps, manifest, extra={
        "reference": "diffusers FluxTransformer2DModel truncated to "
                     f"{keep_d} double + {keep_s} single blocks (full {full_d}+{full_s}), "
                     "PuLID injection per flux/model.py",
        "inputs_from": "testdata/flux1/kontext-dev/dit_small.safetensors",
        "ca_indices": {
            "double": list(range(0, keep_d, DOUBLE_INTERVAL)),
            "single": list(range(0, keep_s, SINGLE_INTERVAL)),
            "note": "ca_idx is ONE sequential counter over both loops; at full "
                    f"depth that is doubles {list(range(0, full_d, DOUBLE_INTERVAL))} "
                    f"-> ca 0..{full_d // DOUBLE_INTERVAL + (full_d % DOUBLE_INTERVAL > 0) - 1} "
                    "then singles -> the rest",
        },
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulid", required=True, help="pulid_flux_v0.9.1.safetensors")
    ap.add_argument("--code", required=True, help="the PuLID repo (holds `pulid/`)")
    ap.add_argument("--testdata", default="testdata")
    ap.add_argument("--transformer", help="FLUX.1[-Kontext]-dev/transformer dir")
    ap.add_argument("--small-double", type=int, default=2)
    ap.add_argument("--small-single", type=int, default=2)
    ap.add_argument("--id-weight", type=float, default=1.0)
    ap.add_argument("--n-img", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, args.code)
    from pulid.encoders_transformer import IDFormer, PerceiverAttentionCA

    out_dir = os.path.join(args.testdata, "pulid")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    torch.manual_seed(args.seed)

    sd = split_state_dict(load_file(args.pulid))
    assert set(sd) == {"pulid_encoder", "pulid_ca"}, sorted(sd)

    enc = IDFormer().eval()
    enc.load_state_dict(sd["pulid_encoder"], strict=True)
    n_ca = 1 + max(int(k.split(".")[0]) for k in sd["pulid_ca"])
    cas = []
    for i in range(n_ca):
        m = PerceiverAttentionCA().eval()
        pfx = f"{i}."
        m.load_state_dict({k[len(pfx):]: v for k, v in sd["pulid_ca"].items()
                           if k.startswith(pfx)}, strict=True)
        cas.append(m)
    print(f"loaded IDFormer + {n_ca} PerceiverAttentionCA modules", flush=True)

    id_embedding = dump_idformer(enc, args.testdata, out_dir, manifest)
    dump_ca(cas, id_embedding, args.n_img, args.seed, out_dir, manifest)
    if args.transformer:
        dump_flux_cond(args.transformer, cas, id_embedding, args.testdata,
                       args.small_double, args.small_single, args.id_weight,
                       out_dir, manifest)
    else:
        print("--transformer not given: skipping flux_cond.safetensors", flush=True)

    manifest["params"] = {
        "weights": os.path.abspath(args.pulid),
        "code": os.path.abspath(args.code),
        "seed": args.seed, "n_img": args.n_img, "id_weight": args.id_weight,
        "small_double": args.small_double, "small_single": args.small_single,
        "num_ca": n_ca,
        "double_interval": DOUBLE_INTERVAL, "single_interval": SINGLE_INTERVAL,
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
