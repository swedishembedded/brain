#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference-parity gates for looped
# mixture-of-experts decoders for its clients. If your team needs expertise in
# proving a depth-attention, multi-pass decoder matches its published reference
# then you can procure our services by sending an email to
# info@swedishembedded.com.
"""Bake Inkling 2 (looped) reference goldens for brain's parity ladder.

Two dumps, both driven through the released transformers reference
(`Inkling2LoopForCausalLM` / `Inkling2LoopForConditionalGeneration`):

  tiny   inkling2_dump_reference.py tiny [out_dir]
         A random 6-layer config (block size 2, prelude 0-1, core 2-3 run 3
         passes, coda 4-5, sliding and full layers in every segment, 8 experts
         top-2, small vocabulary, no vision tower) with seeded weights written
         as a checkpoint IN THE PUBLISHED NAMING (`model.llm.*`, interleaved
         `w13`, `controllers.*`, `middle_cycle_cross_pass_*`, `attn_res.*` with
         the +1 layer offset; every tensor f32). The dumper proves that naming by
         loading the file back through the reference's own conversion mapping and
         comparing every parameter. It records, per logical visit, every tensor a
         port needs to localise a mismatch, plus a cached prefill + token-by-token
         decode that pins the cache semantics.

  real   inkling2_dump_reference.py real <checkpoint dir> [out_dir]
         The published checkpoint on CPU (experts dequantized to bf16 by the
         reference, activations unquantized), three short chat prompts rendered
         with the checkpoint's own template. Records per-visit hidden states,
         router choices and final logits (top-k for every position, full rows
         for a few).

  real-f32  inkling2_dump_reference.py real-f32 <checkpoint dir> [out_dir]
         The same dump with the reference loaded in f32 (experts dequantized to
         f32, every activation f32; about 110 GB of host memory). Against
         `real` it is the reference's own bf16-versus-f32 spread, which sets the
         thresholds a port running f32 activations is held to.

Every tensor is stored as f32 (integers cast exactly). The topology (logical
visit -> physical layer, pass, attention kind, reference cache slot) is written
to `manifest.json`.
"""
import gc
import json
import os
import re
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

SEED = 20261121
HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "..", "testdata", "golden", "inkling2")

# The tiny text config, in the published `config.json` vocabulary. `layer_types`
# is not a published key: the reference ignores `local_layer_ids` and derives the
# layer kinds from `(i + 1) % 6`, which at six layers would leave the core with
# no full-attention layer. Both keys are written and agree.
TINY_TEXT = {
    "model_type": "inkling2_loop_text",
    "model_max_length": 4096,
    "torch_dtype": "float32",
    "hidden_size": 64,
    "num_hidden_layers": 6,
    "vocab_size": 80,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "d_rel": 4,
    "rel_extent": 6,
    "q_bias": False,
    "o_bias": False,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 3.0,
    "rel_l2_clip_threshold": 3.0,
    "local_layer_ids": [0, 2, 4],
    "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
    "sconv_kernel_size": 4,
    "unpadded_vocab_size": 72,
    "final_logit_softcapping": 1.5,
    "swa_head_dim": 16,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "sliding_window_size": 4,
    "n_layers_per_attn_res_block": 2,
    "attn_res_normalize_values": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 32,
    "mlp_gate_softcap": 1.5,
    "mlp_up_softcap": 3.0,
    "route_scale": 2.5,
    "use_gate_bias": True,
    "eos_token_id": 70,
    "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
}
TINY_TOKENS = [3, 17, 5, 71, 42, 8, 0, 63, 29, 11]
TINY_DECODE = [7, 55, 23, 4]

REAL_PROMPTS = [
    [{"role": "user", "content": "What is the capital of Sweden?"}],
    [{"role": "user", "content": "Write a haiku about rust on an old bicycle."}],
    [{"role": "user", "content": "Compute 17 * 23 and explain the steps briefly."}],
]
REAL_TOPK = 64
# Positions (counted from the end) whose full logit rows are stored.
REAL_FULL_ROWS = (1, 2, 4)


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------
def visit_plan(tc):
    """Logical visits in execution order, with the reference's cache slot.

    The reference keeps one cache layer per physical layer (used by the
    prelude, the first core pass and the coda) and appends one copy of the
    core's layers per extra pass (`cache_offset` in `Inkling2LoopTextModel`).
    """
    n, s, e, p = tc.num_hidden_layers, tc.loop_layer_index_start, tc.loop_layer_index_end, tc.loop_num_passes
    core = e - s
    out = []

    def add(layer, pas, seg, slot):
        kind = "sliding" if tc.layer_types[layer] == "hybrid_sliding" else "full"
        out.append({
            "visit": len(out),
            "layer": layer,
            "pass": pas,
            "segment": seg,
            "kind": kind,
            "window": tc.sliding_window_size if kind == "sliding" else None,
            "block_boundary": layer % tc.n_layers_per_attn_res_block == 0,
            "cache_slot": slot,
        })

    for layer in range(s):
        add(layer, 0, "prelude", layer)
    for pas in range(p):
        for layer in range(s, e):
            slot = layer if pas == 0 else n + (pas - 1) * core + (layer - s)
            add(layer, pas, "core", slot)
    for layer in range(e, n):
        add(layer, 0, "coda", layer)
    assert len(out) == tc.total_num_hidden_layers
    return out


# --------------------------------------------------------------------------
# published naming (inverse of the reference's conversion mapping)
# --------------------------------------------------------------------------
_RENAMES = [
    (r"^lm_head\.weight$", "model.llm.unembed.weight"),
    (r"^model\.embed_tokens\.embed_norm\.", "model.llm.embed_norm."),
    (r"^model\.embed_tokens\.weight$", "model.llm.embed.weight"),
    (r"^model\.final_input\.query$", "model.llm.attn_res.output_query"),
    (r"^model\.layers\.(\d+)\.mlp_input\.query$", r"model.llm.attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.linear\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.linear\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.linear\.", ".attn.wv_dv."),
    (r"\.self_attn\.q_proj\.norm\.", ".attn.q_l2_clip."),
    (r"\.self_attn\.relative_position_bias\.linear\.", ".attn.wr_du."),
    (r"\.self_attn\.relative_position_bias\.norm\.", ".attn.rel_l2_clip."),
    (r"\.self_attn\.relative_position_bias\.proj$", ".attn.rel_logits_proj.proj"),
    (r"\.self_attn\.o_proj\.linear\.", ".attn.wo_ud."),
    (r"\.self_attn\.o_proj\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.(k|v)_proj\.sconv\.conv1d\.weight$", r".attn.\1_sconv.weight"),
    (r"\.self_attn\.(k|v)_proj\.sconv\.static_weight$", r".attn.\1_sconv.static_w"),
    (r"\.self_attn\.(k|v)_proj\.sconv\.weight_proj\.weight$", r".attn.\1_sconv.weight_proj"),
    (r"\.post_attention_convolution\.conv1d\.weight$", ".attn_sconv.weight"),
    (r"\.post_mlp_convolution\.conv1d\.weight$", ".mlp_sconv.weight"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
    (r"\.mlp\.experts\.down_proj$", ".mlp.experts.w2_weight"),
    (r"\.q_message_gate\.", ".controllers.attn."),
    (r"\.k_message_gate\.", ".controllers.k."),
    (r"\.v_message_gate\.", ".controllers.v."),
    (r"\.mlp_message_gate\.", ".controllers.mlp."),
    (r"\.controllers\.(attn|k|v|mlp)\.down_proj\.weight$", r".controllers.\1.down_weight"),
    (r"\.controllers\.(attn|k|v|mlp)\.message_gate_proj\.weight$", r".controllers.\1.gain_weight"),
    (r"\.controllers\.(attn|k|v|mlp)\.forget_gate_proj\.weight$", r".controllers.\1.forget_weight"),
    (r"^model\.layers\.", "model.llm.layers."),
    (r"^model\.norm\.", "model.llm.norm."),
]


def to_published(sd, tc):
    """The reference's text-model state dict in the published naming."""
    out = {}
    for key, value in sd.items():
        value = value.detach().float().contiguous()
        if key == "model.message_input.query":
            # (num_loop_layers, 4 = q,k,v,mlp, hidden) -> (q, mlp) pairs and (k, v) pairs.
            out["model.llm.middle_cycle_cross_pass_queries"] = value[:, [0, 3]].flatten(0, 1).contiguous()
            out["model.llm.middle_cycle_cross_pass_kv_queries"] = value[:, 1:3].flatten(0, 1).contiguous()
            continue
        m = re.match(r"^model\.layers\.(\d+)\.attention_input\.query$", key)
        if m:
            # `attn_queries.{i}` belongs to layer i + 1: layer 0 has none.
            out[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = value
            continue
        m = re.match(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$", key)
        if m:
            # concatenated [gate; up] rows -> row-interleaved g0, u0, g1, u1, ...
            e, rows, cols = value.shape
            inter = value.view(e, 2, rows // 2, cols).transpose(1, 2).reshape(e, rows, cols).contiguous()
            out[f"model.llm.layers.{m.group(1)}.mlp.experts.w13_weight"] = inter
            continue
        if key == "lm_head.weight":
            pad = torch.zeros(tc.vocab_size - value.shape[0], value.shape[1])
            value = torch.cat([value, pad], dim=0).contiguous()
        new = key
        for pat, rep in _RENAMES:
            new = re.sub(pat, rep, new)
        assert new.startswith("model.llm."), f"no published name for {key}"
        assert new not in out, f"two parameters map to {new}"
        out[new] = value
    return out


def randomize(model, gen):
    """Seeded weights at scales that keep every branch of the reference active
    (clips that clip, softcaps that bend, routers that disagree)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            shape = p.shape
            if name.endswith(("layernorm.weight", "embed_norm.weight", "model.norm.weight")):
                v = 1.0 + 0.2 * torch.randn(shape, generator=gen)
            elif name.endswith(".scale"):
                v = 1.0 + 0.3 * torch.randn(shape, generator=gen).abs()
            elif name.endswith(".query"):
                v = torch.randn(shape, generator=gen)
            elif name.endswith("conv1d.weight"):
                v = 0.5 * torch.randn(shape, generator=gen)
            elif name.endswith("static_weight"):
                v = 0.3 * torch.randn(shape, generator=gen)
            elif name.endswith("relative_position_bias.proj"):
                v = torch.randn(shape, generator=gen)
            elif name.endswith("embed_tokens.weight"):
                v = torch.randn(shape, generator=gen)
            elif name.endswith("experts.gate_up_proj"):
                v = torch.randn(shape, generator=gen) / shape[-1] ** 0.5 * 2.0
            elif name.endswith("experts.down_proj"):
                v = torch.randn(shape, generator=gen) / shape[-1] ** 0.5
            elif len(shape) == 2:
                v = torch.randn(shape, generator=gen) / shape[-1] ** 0.5
            else:
                raise SystemExit(f"randomize: no rule for {name} {tuple(shape)}")
            p.copy_(v)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.5 * torch.randn(b.shape, generator=gen))


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------
class Recorder:
    """Forward hooks that tag every captured tensor with its logical visit.

    Decoder layers run in logical-visit order, so a counter advanced by each
    decoder-layer call is the visit index. `message_input` runs just BEFORE the
    layer it feeds, so it records under the next visit.
    """

    def __init__(self, model, plan, full=True):
        inner = model.model
        self.tm = inner if hasattr(inner, "layers") else inner.language_model
        self.plan = plan
        self.full = full
        self.next_visit = 0
        self.cur = None
        self.out = {}
        self.handles = []
        for layer in self.tm.layers:
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self._layer_post))
            self._sub(layer.mlp.gate, self._router)
            if full:
                if layer.attention_input is not None:
                    self._sub(layer.attention_input, self._tag("attn_in", stack=True))
                self._sub(layer.mlp_input, self._tag("mlp_in", stack=True))
                self._sub(layer.self_attn, self._tag("attn", first=True))
                self._sub(layer.post_attention_convolution, self._tag("attn_conv"))
                self._sub(layer.mlp, self._tag("mlp"))
                self._sub(layer.post_mlp_convolution, self._tag("mlp_conv"))
                self._sub(layer.self_attn.relative_position_bias, self._relbias)
                self._sub(layer.self_attn.relative_position_bias.norm, self._tag("rel_states"))
                for g in ("q", "k", "v", "mlp"):
                    gate = getattr(layer, f"{g}_message_gate", None)
                    if gate is not None:
                        self._sub(gate, self._tag(f"gated_{g}"))
        if full:
            self._sub(self.tm.message_input, self._messages)
            self._sub(self.tm.final_input, self._final_in)
            self._sub(self.tm.norm, self._final_norm)

    def _sub(self, mod, fn):
        self.handles.append(mod.register_forward_hook(fn))

    def close(self):
        for h in self.handles:
            h.remove()

    def put(self, key, t):
        assert key not in self.out, f"recorded twice: {key}"
        self.out[key] = t.detach().float().contiguous().clone()

    def _layer_pre(self, mod, args, kwargs):
        self.cur = self.next_visit
        self.next_visit += 1
        v = self.cur
        assert mod is self.tm.layers[self.plan[v]["layer"]], f"visit {v} ran an unexpected layer"
        hidden, block_states = args[0], args[1]
        self.put(f"visit{v}.in", hidden[0])
        if self.full:
            # [n_blocks, T, H]
            self.put(f"visit{v}.block_states", block_states[0].permute(1, 0, 2))

    def _layer_post(self, mod, args, out):
        self.put(f"visit{self.cur}.out", out[0])

    def _tag(self, name, stack=False, first=False):
        def hook(mod, args, out):
            if stack:
                t = torch.stack(list(out), dim=0)[:, 0]  # [nq, T, H]
            else:
                t = (out[0] if first else out)[0]
            self.put(f"visit{self.cur}.{name}", t)
        return hook

    def _router(self, mod, args, out):
        logits, weights, indices = out
        # Sort each token's picks by expert id: the reference's top-k is unsorted.
        order = indices.argsort(dim=-1)
        self.put(f"visit{self.cur}.router_logits", logits)
        self.put(f"visit{self.cur}.router_indices", indices.gather(-1, order))
        self.put(f"visit{self.cur}.router_weights", weights.gather(-1, order))

    def _relbias(self, mod, args, out):
        self.put(f"visit{self.cur}.position_bias", out[0])

    def _messages(self, mod, args, out):
        # [T, 4 (q,k,v,mlp), H] -> [4, T, H]
        self.put(f"visit{self.next_visit}.messages", out[0].permute(1, 0, 2))

    def _final_in(self, mod, args, out):
        self.put("final_in", out[0][0])

    def _final_norm(self, mod, args, out):
        self.put("final_norm", out[0])


def conv_states(cache, plan):
    out = {}
    for v in plan:
        layer = cache.layers[v["cache_slot"]]
        for j in range(4):
            st = layer.conv_states.get(j)
            assert st is not None, f"visit {v['visit']}: conv state {j} missing"
            out[f"visit{v['visit']}.conv{j}"] = st[0].detach().float().contiguous().clone()
    return out


# --------------------------------------------------------------------------
# tiny
# --------------------------------------------------------------------------
def dump_tiny(out):
    from transformers import AutoConfig, Inkling2LoopForCausalLM
    from transformers.models.inkling2.modeling_inkling2 import get_relative_distance

    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)

    tc = AutoConfig.for_model(**TINY_TEXT)
    tc._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = Inkling2LoopForCausalLM(tc).float().eval()
    randomize(model, torch.Generator().manual_seed(SEED))
    plan = visit_plan(tc)

    # --- the checkpoint, in the published naming, proven by a reference round trip
    published = to_published(model.state_dict(), tc)
    save_file(published, os.path.join(ckpt, "model.safetensors"))
    cfg_json = {
        "dtype": "float32",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "eos_token_id": 70,
        "text_config": TINY_TEXT,
    }
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(out))) as tmp:
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(dict(TINY_TEXT, architectures=["Inkling2LoopForCausalLM"]), f)
        os.link(os.path.join(ckpt, "model.safetensors"), os.path.join(tmp, "model.safetensors"))
        back = Inkling2LoopForCausalLM.from_pretrained(tmp, dtype=torch.float32, attn_implementation="eager")
        a, b = model.state_dict(), back.state_dict()
        assert a.keys() == b.keys(), f"round trip keys differ: {set(a) ^ set(b)}"
        for k in a:
            assert torch.equal(a[k].float(), b[k].float()), f"round trip changed {k}"
        del back
    print(f"published naming round-trips: {len(published)} tensors")

    ids = torch.tensor([TINY_TOKENS])
    T = ids.shape[1]
    g = {"tokens": ids[0].float(), "positions": torch.arange(T).float()}

    # --- uncached forward, every intermediate recorded
    rec = Recorder(model, plan)
    with torch.no_grad():
        embed = model.model.embed_tokens(ids)
        res = model(input_ids=ids, use_cache=False)
    rec.close()
    assert rec.next_visit == len(plan)
    g.update(rec.out)
    g["embed"] = embed[0].float()
    g["logits"] = res.logits[0].float()
    cap = tc.final_logit_softcapping
    head = model.lm_head(rec.out["final_norm"]).float()
    assert torch.allclose(cap * torch.tanh(head / cap), g["logits"], atol=1e-5), "logits are not softcap(head(norm))"
    g["relative_distance.full"] = get_relative_distance(embed, None, is_sliding=False).float()
    g["relative_distance.sliding"] = get_relative_distance(embed, None, is_sliding=True).float()

    # Self-check the recorded depth-attention semantics on one visit: a
    # softmax over the RMS-normed [block_states; partial] with the learned query,
    # mixing the NORMED states.
    v = next(p for p in plan if p["segment"] == "core" and p["pass"] == 1 and p["kind"] == "full")["visit"]
    layer = model.model.layers[plan[v]["layer"]]
    stack = torch.cat([g[f"visit{v}.block_states"], g[f"visit{v}.in"][None]], dim=0)  # [D, T, H]
    normed = stack * torch.rsqrt(stack.square().mean(-1, keepdim=True) + 1e-6)
    q = layer.attention_input.query.float().view(-1, tc.hidden_size)
    w = torch.softmax(torch.einsum("qh,dth->tqd", q, normed) * tc.hidden_size ** -0.5, dim=-1)
    mix = torch.einsum("tqd,dth->qth", w, normed)
    assert torch.allclose(mix, g[f"visit{v}.attn_in"], atol=1e-5), "depth attention self-check failed"

    # --- cached prefill + token-by-token decode
    full_ids = torch.tensor([TINY_TOKENS + TINY_DECODE])
    with torch.no_grad():
        ref_full = model(input_ids=full_ids, use_cache=False).logits[0].float()
        rec = Recorder(model, plan, full=False)
        pre = model(input_ids=ids, use_cache=True)
        rec.close()
        cache = pre.cache
        g.update({f"prefill.{k}": t for k, t in conv_states(cache, plan).items()})
        g["prefill.logits"] = pre.logits[0].float()
        for s, tok in enumerate(TINY_DECODE):
            rec = Recorder(model, plan, full=False)
            step = model(input_ids=torch.tensor([[tok]]), cache=cache, use_cache=True)
            rec.close()
            cache = step.cache
            g[f"decode{s}.logits"] = step.logits[0, -1].float()
            for k, t in rec.out.items():
                g[f"decode{s}.{k}"] = t
            g.update({f"decode{s}.{k}": t for k, t in conv_states(cache, plan).items()})
    g["decode.tokens"] = torch.tensor(TINY_DECODE).float()
    g["decode.full_logits"] = ref_full
    pre_err = (g["prefill.logits"] - g["logits"]).abs().max().item()
    dec_err = [(g[f"decode{s}.logits"] - ref_full[T + s]).abs().max().item() for s in range(len(TINY_DECODE))]
    print(f"cached prefill vs uncached max|d| {pre_err:.3e}; decode step vs full forward max|d| {dec_err}")
    assert pre_err < 1e-4, "cached prefill disagrees with the uncached forward"

    stats = branch_stats(g, plan, tc)
    save_file(g, os.path.join(out, "golden.safetensors"))
    manifest = {
        "model": "inkling2",
        "kind": "tiny",
        "seed": SEED,
        "tokens": TINY_TOKENS,
        "decode_tokens": TINY_DECODE,
        "dtype": "float32",
        "attn_implementation": "eager",
        "decode_vs_full_max_abs": dec_err,
        "prefill_vs_uncached_max_abs": pre_err,
        "branch_stats": stats,
        "visits": plan,
        "keys": {
            "visit{v}.in/out": "[T,H] carried partial sum entering / leaving the visit",
            "visit{v}.block_states": "[n_blocks,T,H] completed block states seen by the visit",
            "visit{v}.attn_in": "[nq,T,H] depth-attention outputs feeding attention (nq=1 sliding, 3 full q/k/v); absent at layer 0",
            "visit{v}.mlp_in": "[1,T,H] depth-attention output feeding the MoE (before the message gate)",
            "visit{v}.messages": "[4,T,H] q,k,v,mlp messages from the previous pass (core passes >= 1)",
            "visit{v}.gated_{q,k,v,mlp}": "[T,H] message-gate outputs",
            "visit{v}.attn": "[T,H] attention output (o_proj)",
            "visit{v}.attn_conv / mlp / mlp_conv": "[T,H] after the post-attention conv / MoE / post-MoE conv",
            "visit{v}.rel_states": "[T,heads,d_rel] relative states after scale, clip (and log scale on full layers)",
            "visit{v}.position_bias": "[heads,T,Tk] relative-position bias added to the scores",
            "visit{v}.router_*": "[T,E] logits; [T,k] indices and weights sorted by expert id",
            "embed / final_in / final_norm / logits": "[T,H] normed embedding, final depth mix, final norm; [T,V] softcapped logits",
            "relative_distance.{full,sliding}": "[T,Tk] query minus key position",
            "prefill.visit{v}.conv{j}": "[C,K] conv state j after the cached prefill (0 k, 1 v, 2 post-attn, 3 post-mlp)",
            "decode{s}.*": "token-by-token step s after the prefill: logits [V], visit{v}.in/out [1,H], conv states",
            "decode.full_logits": "[T+steps,V] uncached forward over prompt + decode tokens",
        },
        "source": source_block(
            {
                "hidden": tc.hidden_size,
                "layers": tc.num_hidden_layers,
                "visits": len(plan),
                "heads": tc.num_attention_heads,
                "kv_heads": tc.num_key_value_heads,
                "head_dim": tc.head_dim,
                "experts": tc.n_routed_experts,
                "top_k": tc.num_experts_per_tok,
                "vocab": tc.vocab_size,
                "unpadded_vocab": tc.unpadded_vocab_size,
                "n_tokens": T,
            },
            checkpoint="tiny random config of the inkling2_loop reference",
            files=[os.path.join(ckpt, "model.safetensors")],
        ),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}: {len(g)} tensors, {len(plan)} visits; branches {stats}")


def branch_stats(g, plan, tc):
    """Count how often the clipped / capped branches fired, so a golden that
    never exercises a branch is visible in its manifest."""
    rel = torch.cat([g[f"visit{p['visit']}.rel_states"].flatten(0, -2) for p in plan])
    picks = torch.cat([g[f"visit{p['visit']}.router_indices"].flatten() for p in plan])
    return {
        "rel_states_at_clip": int((rel.norm(dim=-1) > tc.rel_l2_clip_threshold * 0.999).sum()),
        "rel_states_total": int(rel.shape[0]),
        "logits_beyond_half_cap": int((g["logits"].abs() > 0.5 * tc.final_logit_softcapping).sum()),
        "distinct_experts_picked": int(picks.unique().numel()),
    }


# --------------------------------------------------------------------------
# real
# --------------------------------------------------------------------------
def dump_real(ckpt_dir, out, dtype=torch.bfloat16):
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        ckpt_dir, dtype=dtype, attn_implementation="eager"
    ).eval()
    tc = model.config.text_config
    plan = visit_plan(tc)
    g = {}
    prompts = []
    for i, msgs in enumerate(REAL_PROMPTS):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        prompts.append({"messages": msgs, "rendered": text, "n_tokens": int(ids.shape[1])})
        rec = Recorder(model, plan, full=False)
        with torch.no_grad():
            res = model(input_ids=ids, use_cache=False)
        rec.close()
        assert rec.next_visit == len(plan)
        logits = res.logits[0].float()
        g[f"p{i}.tokens"] = ids[0].float()
        for k, t in rec.out.items():
            g[f"p{i}.{k}"] = t
        top = logits.topk(REAL_TOPK, dim=-1)
        g[f"p{i}.logits_topk_values"] = top.values.contiguous()
        g[f"p{i}.logits_topk_indices"] = top.indices.float().contiguous()
        for back in REAL_FULL_ROWS:
            g[f"p{i}.logits_row_m{back}"] = logits[-back].contiguous()
        print(f"prompt {i}: {ids.shape[1]} tokens, next token {tok.decode([int(logits[-1].argmax())])!r}", flush=True)
        gc.collect()
    save_file(g, os.path.join(out, "golden.safetensors"))
    manifest = {
        "model": "inkling2",
        "kind": "real",
        "dtype": f"{str(dtype).removeprefix('torch.')} (experts dequantized by the reference, activations unquantized)",
        "attn_implementation": "eager",
        "prompts": prompts,
        "topk": REAL_TOPK,
        "full_rows_from_end": list(REAL_FULL_ROWS),
        "visits": plan,
        "keys": {
            "p{i}.tokens": "[T] prompt ids (chat template, generation prompt appended)",
            "p{i}.visit{v}.in/out": "[T,H] carried partial sum entering / leaving the visit",
            "p{i}.visit{v}.router_*": "[T,E] logits; [T,k] indices and weights sorted by expert id",
            "p{i}.logits_topk_*": "[T,k] top-k logits (after the softcap) and their ids",
            "p{i}.logits_row_m{n}": "[V] full logits row n positions from the end",
        },
        "source": source_block(
            {
                "hidden": tc.hidden_size,
                "layers": tc.num_hidden_layers,
                "visits": len(plan),
                "experts": tc.n_routed_experts,
                "top_k": tc.num_experts_per_tok,
                "vocab": tc.vocab_size,
                "unpadded_vocab": tc.unpadded_vocab_size,
            },
            checkpoint="thinkingmachines/Inkling-2-Mini",
            files=[os.path.join(ckpt_dir, "model-00001-of-00001.safetensors")],
            hash_files=False,
        ),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    size = os.path.getsize(os.path.join(out, "golden.safetensors"))
    print(f"wrote {out}: {len(g)} tensors, {size / 1e6:.1f} MB")


def main():
    args = sys.argv[1:]
    if args[:1] == ["tiny"] and len(args) <= 2:
        dump_tiny(args[1] if len(args) == 2 else os.path.join(TESTDATA, "tiny"))
    elif args[:1] == ["real"] and len(args) in (2, 3):
        dump_real(args[1], args[2] if len(args) == 3 else os.path.join(TESTDATA, "real"))
    elif args[:1] == ["real-f32"] and len(args) in (2, 3):
        dump_real(args[1], args[2] if len(args) == 3 else os.path.join(TESTDATA, "real-f32"), dtype=torch.float32)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
