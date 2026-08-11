#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Component-scoped parity goldens for Qwen3-Omni-30B-A3B-Instruct.

The full checkpoint is 70.5 GB bf16 across 28 010 tensors — too large to load
wholesale on this box even once, let alone twice (brain's own device copy).
Per the plan of record (`.agents/roadmap/omni.md`), parity is validated
**component by component**: each dump below streams ONLY the tensors that
component needs straight out of the sharded safetensors files (via
`model.safetensors.index.json`), builds the real upstream HF module for that
component, and runs one fixed, synthetic, fp32, CPU forward. This is the same
selective-load pattern `qwenvl_decoder_dump_reference.py` uses for the (also
oversized) Qwen3-VL-4B decoder.

No random inputs — every input below is deterministic (arange/fixed lists),
so a re-run byte-matches. Nothing here is a full end-to-end parity claim; each
golden isolates one component so a mismatch localizes to one crate.

Components dumped:
  audio    thinker.audio_tower  (AuT, 32L)      -> omni_audio.safetensors
  vision   thinker.visual        (ViT, 27L)      -> omni_vision.safetensors
  layer0   thinker.model layer 0 (MoE decoder)   -> omni_layer0.safetensors
  rope     get_rope_index, mixed text/audio/image/video ids (no weights)
                                                  -> omni_rope.safetensors
  talkcp   talker.code_predictor (5L)            -> omni_code_predictor.safetensors
  code2wav code2wav (RVQ decode -> waveform)      -> omni_code2wav.safetensors

usage: omni_dump_reference.py <hf_checkpoint_dir> [out_dir] [component ...]
  (no component args => dump all of them)

env:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 are set unconditionally below —
  goldens must come from the checkpoint already on disk, never a live fetch.
"""
import glob
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if len(sys.argv) < 2:
    sys.exit(__doc__)
HF_DIR = sys.argv[1]
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.join(
    os.path.dirname(__file__), "..", "..", "testdata", "golden", "omni")
os.makedirs(OUT_DIR, exist_ok=True)
REQUESTED = [a for a in sys.argv[3:] if not a.startswith("--")] or None

torch.manual_seed(0)  # only matters for module init of any *unloaded* params (should be none)


def load_config():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    with open(os.path.join(HF_DIR, "config.json")) as f:
        raw = json.load(f)
    return Qwen3OmniMoeConfig.from_dict(raw), raw


def shard_index():
    """weight name -> shard filename, from model.safetensors.index.json."""
    idx_path = os.path.join(HF_DIR, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)["weight_map"]
    # single-file checkpoint fallback
    shards = sorted(glob.glob(os.path.join(HF_DIR, "*.safetensors")))
    assert len(shards) == 1, f"no index.json and not exactly one shard in {HF_DIR}"
    with safe_open(shards[0], "pt") as f:
        return {k: os.path.basename(shards[0]) for k in f.keys()}


WEIGHT_MAP = None  # lazily populated only if a dump needs real weights


def load_prefixed(prefix, rename=lambda s: s, layer_filter=None):
    """Stream every tensor under `prefix.` from its shard, renamed by `rename`.

    `layer_filter(int) -> bool` restricts to specific `...layers.N...` indices
    when set, so e.g. only layer 0 of a 48-layer stack is ever read.
    """
    global WEIGHT_MAP
    if WEIGHT_MAP is None:
        WEIGHT_MAP = shard_index()
    wanted = {k: v for k, v in WEIGHT_MAP.items() if k.startswith(prefix + ".")}
    if layer_filter is not None:
        wanted = {k: v for k, v in wanted.items() if _layer_ok(k, layer_filter)}
    by_shard = {}
    for k, shard in wanted.items():
        by_shard.setdefault(shard, []).append(k)
    sd = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(HF_DIR, shard), "pt") as f:
            for k in keys:
                sd[rename(k[len(prefix) + 1:])] = f.get_tensor(k).float()
    assert sd, f"no tensors found under prefix '{prefix}' (check config/shard layout)"
    return sd


def _layer_ok(key, layer_filter):
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return layer_filter(int(parts[i + 1]))
    return True  # not a per-layer tensor (embed/norm/etc.) -> always keep


def save(name, tensors, **meta):
    out = os.path.join(OUT_DIR, f"omni_{name}.safetensors")
    save_file(tensors, out, metadata={k: str(v) for k, v in meta.items()})
    print(f"[{name}] -> {out} ({len(tensors)} tensors)")


# --------------------------------------------------------------------------- audio
def dump_audio(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    acfg = cfg.thinker_config.audio_config
    model = Qwen3OmniMoeAudioEncoder(acfg)
    sd = load_prefixed("thinker.audio_tower")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"audio tower missing: {missing[:8]}"
    assert not unexpected, f"audio tower unexpected: {unexpected[:8]}"
    model.eval()

    # A fixed 4s@16kHz-equivalent mel: chunk_len is 2*n_window frames; use 2 chunks.
    chunk = 2 * acfg.n_window  # 100 mel frames/chunk
    n_frames = 2 * chunk
    mel = torch.arange(acfg.num_mel_bins * n_frames, dtype=torch.float32)
    mel = (mel % 23 - 11.0) / 11.0  # deterministic, bounded, non-trivial pattern
    mel = mel.reshape(acfg.num_mel_bins, n_frames)
    with torch.no_grad():
        out = model(input_features=mel, feature_lens=torch.tensor([n_frames]))
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    save("audio", {
        "mel": mel, "feature_lens": torch.tensor([n_frames], dtype=torch.int32),
        "hidden": hidden.squeeze(0).contiguous(),
    }, src="thinker.audio_tower", n_layers=acfg.encoder_layers, d_model=acfg.d_model)


# -------------------------------------------------------------------------- vision
def dump_vision(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    vcfg = cfg.thinker_config.vision_config
    model = Qwen3OmniMoeVisionEncoder(vcfg)
    sd = load_prefixed("thinker.visual")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"vision tower missing: {missing[:8]}"
    assert not unexpected, f"vision tower unexpected: {unexpected[:8]}"
    model.eval()

    # One 2x2-merge-block image: grid (t=1, h=4, w=4) patches -> patch_vec_dim wide rows.
    t, h, w = 1, 4, 4
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
    n_patches = t * h * w
    patches = torch.arange(n_patches * patch_dim, dtype=torch.float32)
    patches = (patches % 17 - 8.0) / 8.0
    patches = patches.reshape(n_patches, patch_dim)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)
    with torch.no_grad():
        out = model(hidden_states=patches, grid_thw=grid_thw)
    save("vision", {
        "patches": patches, "grid_thw": grid_thw.to(torch.int32),
        "hidden": out.last_hidden_state.contiguous(),
        **{f"deepstack{i}": t.contiguous() for i, t in enumerate(out.deepstack_features or [])},
    }, src="thinker.visual", depth=vcfg.depth, hidden=vcfg.hidden_size,
       deepstack_indexes=str(vcfg.deepstack_visual_indexes))


# --------------------------------------------------------------------------- layer0
def fuse_experts(sd, layer, n_experts):
    """The checkpoint stores one gate_proj/up_proj/down_proj per expert
    (matching brain's own model::moe, which reads them individually -- no
    Rust-side change needed here). The transformers module class this dumper
    runs the reference through instead wants ONE stacked parameter per
    projection (`Qwen3OmniMoeThinkerTextExperts`/`...TalkerTextExperts`:
    `gate_up_proj [E, 2*ff, d]` gate and up concatenated along dim 1,
    `down_proj [E, d, ff]`) -- a transformers-internal loading convention,
    replicated here by hand since this is a raw state_dict load, not
    `from_pretrained` (which would apply it automatically via the model's own
    checkpoint-conversion hook)."""
    prefix = f"layers.{layer}.mlp.experts."
    gate_up, down = [], []
    for e in range(n_experts):
        gate = sd.pop(f"{prefix}{e}.gate_proj.weight")
        up = sd.pop(f"{prefix}{e}.up_proj.weight")
        down.append(sd.pop(f"{prefix}{e}.down_proj.weight"))
        gate_up.append(torch.cat([gate, up], dim=0))
    sd[f"{prefix}gate_up_proj"] = torch.stack(gate_up, dim=0)
    sd[f"{prefix}down_proj"] = torch.stack(down, dim=0)


def dump_layer0(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

    tcfg = cfg.thinker_config.text_config
    # Truncate to 1 layer (qwenvl_decoder_dump_reference.py's pattern) so only
    # layer 0's 128 experts (~605 M params) are ever materialized, not all 48.
    tcfg.num_hidden_layers = 1
    model = Qwen3OmniMoeThinkerTextModel(tcfg)
    sd = load_prefixed("thinker.model", layer_filter=lambda i: i == 0)
    fuse_experts(sd, 0, tcfg.num_experts)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"layer0 missing: {missing[:8]}"
    assert not unexpected, f"layer0 unexpected: {unexpected[:8]}"
    model.eval()

    tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]  # fixed, valid ids
    ids = torch.tensor([tokens], dtype=torch.long)
    router_out = {}

    def hook(_mod, _inp, out):
        # Qwen3OmniMoeThinkerTextTopKRouter.forward -> (router_logits,
        # router_scores [top-k values], router_indices [top-k ids]) -- the
        # SparseMoeBlock wrapping it returns only the combined hidden state,
        # so the router itself (`mlp.gate`) is the hook target, not `mlp`.
        router_out["logits"], router_out["scores"], router_out["indices"] = (t.detach() for t in out)

    handle = None
    gate = getattr(getattr(model.layers[0], "mlp", None), "gate", None)
    if gate is not None:
        handle = gate.register_forward_hook(hook)
    # DIAGNOSTIC: post_attention_layernorm's INPUT is xmid (post-attention,
    # pre-MoE residual) -- exactly the stage omni::thinker::layer_fwd's xmid
    # buffer should match, isolating attention-stage correctness from the
    # MoE FFN stage.
    xmid_out = {}

    def xmid_hook(_mod, inp, _out):
        xmid_out["xmid"] = inp[0].detach()

    xmid_handle = model.layers[0].post_attention_layernorm.register_forward_hook(xmid_hook)
    # `Qwen3OmniMoeThinkerTextModel.forward` always applies its top-level
    # `self.norm` (the final decoder-stack RMSNorm) after the layer loop, even
    # truncated to 1 layer -- `out.last_hidden_state` is therefore
    # `model.norm(layer0_output)`, NOT layer 0's own raw output. Hook
    # `model.norm`'s INPUT to get the raw one-layer output that
    # `omni::thinker::layer_fwd` (a single decoder layer, no final norm)
    # actually produces, so the comparison is apples-to-apples.
    prenorm_out = {}

    def prenorm_hook(_mod, inp, _out):
        prenorm_out["prenorm"] = inp[0].detach()

    prenorm_handle = model.norm.register_forward_hook(prenorm_hook)
    with torch.no_grad():
        out = model(input_ids=ids)
    if handle:
        handle.remove()
    xmid_handle.remove()
    prenorm_handle.remove()

    tensors = {
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "hidden": out.last_hidden_state.squeeze(0).contiguous(),
        "xmid": xmid_out["xmid"].squeeze(0).contiguous(),
        "layer_out": prenorm_out["prenorm"].squeeze(0).contiguous(),
    }
    if "logits" in router_out:
        tensors["router_logits"] = router_out["logits"].contiguous()
        tensors["router_topk_ids"] = router_out["indices"].to(torch.int32).contiguous()
        tensors["router_topk_weights"] = router_out["scores"].contiguous()
    save("layer0", tensors, src="thinker.model.layers.0", n_experts=tcfg.num_experts,
         top_k=tcfg.num_experts_per_tok, hidden_size=tcfg.hidden_size)


# ------------------------------------------------------------------- talker layer0
def dump_talker_layer0(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerModel

    tcfg = cfg.talker_config.text_config
    tcfg.num_hidden_layers = 1
    model = Qwen3OmniMoeTalkerModel(tcfg)
    sd = load_prefixed("talker.model", layer_filter=lambda i: i == 0)
    fuse_experts(sd, 0, tcfg.num_experts)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"talker layer0 missing: {missing[:8]}"
    assert not unexpected, f"talker layer0 unexpected: {unexpected[:8]}"
    model.eval()

    # Fixed, valid codec ids (< vocab_size=3072) -- Qwen3OmniMoeTalkerModel has
    # no self.embed_tokens (real usage always builds inputs_embeds itself: text
    # projection + codec_embedding + thinker-hidden splice), so this dumper
    # embeds via the model's own codec_embedding table and calls with
    # inputs_embeds, matching how a real caller would never pass input_ids here.
    codec_ids = [17, 401, 2050, 9, 3071, 88, 512, 6, 1200]
    ids = torch.tensor([codec_ids], dtype=torch.long)
    with torch.no_grad():
        embeds = model.codec_embedding(ids)

    router_out = {}

    def hook(_mod, _inp, out):
        router_out["logits"], router_out["scores"], router_out["indices"] = (t.detach() for t in out)

    gate = getattr(getattr(model.layers[0], "mlp", None), "gate", None)
    handle = gate.register_forward_hook(hook) if gate is not None else None

    xmid_out = {}

    def xmid_hook(_mod, inp, _out):
        xmid_out["xmid"] = inp[0].detach()

    xmid_handle = model.layers[0].post_attention_layernorm.register_forward_hook(xmid_hook)

    # Same "hidden is model.norm(layer0_output), not layer0's own raw output"
    # distinction as dump_layer0 -- see that function's comment.
    prenorm_out = {}

    def prenorm_hook(_mod, inp, _out):
        prenorm_out["prenorm"] = inp[0].detach()

    prenorm_handle = model.norm.register_forward_hook(prenorm_hook)
    with torch.no_grad():
        out = model(inputs_embeds=embeds)
    if handle:
        handle.remove()
    xmid_handle.remove()
    prenorm_handle.remove()

    tensors = {
        "codec_ids": torch.tensor(codec_ids, dtype=torch.int32),
        "hidden": out.last_hidden_state.squeeze(0).contiguous(),
        "xmid": xmid_out["xmid"].squeeze(0).contiguous(),
        "layer_out": prenorm_out["prenorm"].squeeze(0).contiguous(),
    }
    if "logits" in router_out:
        tensors["router_logits"] = router_out["logits"].contiguous()
        tensors["router_topk_ids"] = router_out["indices"].to(torch.int32).contiguous()
        tensors["router_topk_weights"] = router_out["scores"].contiguous()
    save("talker_layer0", tensors, src="talker.model.layers.0", n_experts=tcfg.num_experts,
         top_k=tcfg.num_experts_per_tok, hidden_size=tcfg.hidden_size,
         shared_expert_intermediate_size=tcfg.shared_expert_intermediate_size)


# ------------------------------------------------------------------------------ rope
def dump_rope(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    tc = cfg.thinker_config
    # No weights needed -- get_rope_index is pure config + id arithmetic. Build
    # the object via __new__ to skip allocating the full (huge) module tree.
    thinker = Qwen3OmniMoeThinkerForConditionalGeneration.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    thinker.config = tc
    thinker.spatial_merge_size = tc.vision_config.spatial_merge_size

    # A synthetic mixed prompt: system/user text, one image (grid 1x4x4 -> 4
    # merged tokens at merge_size 2), one audio span (2 frames -> 2 tokens after
    # the 2x downsample used elsewhere in this dumper), assistant text.
    vs, ve = tc.vision_start_token_id, tc.vision_end_token_id
    ims, ime = tc.image_token_id, tc.image_token_id
    aus, aue, au = tc.audio_start_token_id, tc.audio_end_token_id, tc.audio_token_id
    n_img_tok = 4  # (4*4)/(2*2)
    n_aud_tok = 2
    ids = (
        [tc.user_token_id] + [1, 2, 3]
        + [vs] + [ims] * n_img_tok + [ve]
        + [aus] + [au] * n_aud_tok + [aue]
        + [4, 5]
    )
    input_ids = torch.tensor([ids], dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        position_ids, mrope_deltas = thinker.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attn,
            audio_seqlens=torch.tensor([n_aud_tok]),
        )
    save("rope", {
        "input_ids": input_ids.squeeze(0).to(torch.int32),
        "image_grid_thw": image_grid_thw.to(torch.int32),
        "position_ids": position_ids.squeeze(1).to(torch.int32),  # [3, seq]
        "mrope_deltas": mrope_deltas.to(torch.int32),
    }, src="Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index",
       mrope_section=str(tc.text_config.rope_scaling["mrope_section"]))


# ------------------------------------------------------------------------- talkcp
def dump_code_predictor(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
    )

    ccfg = cfg.talker_config.code_predictor_config
    model = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(ccfg)
    sd = load_prefixed("talker.code_predictor")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if "lm_head" not in m], f"code_predictor missing: {missing[:8]}"
    model.eval()

    # The "prefill" branch (the one that actually consumes inputs_embeds
    # directly, rather than looking `input_ids` up in a per-step embedding
    # table) triggers only when inputs_embeds.shape[1] > 1 -- the smallest
    # real prefill is [1, 2, h]: position 0 is the Talker hidden-state
    # conditioning, position 1 is codebook-0's embedded token; this call
    # predicts codebook-1 (generation_steps = shape[1] - 2 = 0).
    h = ccfg.hidden_size
    embeds = ((torch.arange(2 * h, dtype=torch.float32) % 13) - 6.0) / 6.0
    with torch.no_grad():
        out = model(inputs_embeds=embeds.view(1, 2, h))
    save("code_predictor", {
        "in_embed": embeds.view(2, h).contiguous(), "logits": out.logits.squeeze(0).contiguous(),
    }, src="talker.code_predictor", n_layers=ccfg.num_hidden_layers, n_code_groups=ccfg.num_code_groups)


# ----------------------------------------------------------------------- code2wav
def dump_code2wav(cfg):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

    c2w = cfg.code2wav_config
    model = Qwen3OmniMoeCode2Wav(c2w)
    sd = load_prefixed("code2wav")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"code2wav missing: {missing[:8]}"
    assert not unexpected, f"code2wav unexpected: {unexpected[:8]}"
    model.eval()

    # A short deterministic code sequence: [1, num_quantizers, T].
    T = 8
    codes = torch.zeros(1, c2w.num_quantizers, T, dtype=torch.long)
    for q in range(c2w.num_quantizers):
        size = c2w.semantic_codebook_size if q < c2w.num_semantic_quantizers else c2w.codebook_size
        codes[0, q] = torch.arange(T, dtype=torch.long) % size
    with torch.no_grad():
        wav = model(codes)
    save("code2wav", {
        "codes": codes.squeeze(0).to(torch.int32), "wav": wav.squeeze(0).contiguous(),
    }, src="code2wav", num_quantizers=c2w.num_quantizers, total_upsample=int(model.total_upsample))


COMPONENTS = {
    "audio": dump_audio, "vision": dump_vision, "layer0": dump_layer0,
    "talker_layer0": dump_talker_layer0,
    "rope": dump_rope, "talkcp": dump_code_predictor, "code2wav": dump_code2wav,
}


def main():
    cfg, _raw = load_config()
    names = REQUESTED or list(COMPONENTS)
    unknown = [n for n in names if n not in COMPONENTS]
    if unknown:
        sys.exit(f"unknown component(s) {unknown}; choose from {list(COMPONENTS)}")
    for name in names:
        print(f"--- {name} ---")
        COMPONENTS[name](cfg)


if __name__ == "__main__":
    main()
