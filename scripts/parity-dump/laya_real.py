#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a REAL-WEIGHT forward pass of `convaiinnovations/laya`'s own
`DecisionModel` (`rl_common.py`, Apache-2.0, vendored inline - the same class
this script's own `build_model` would construct, kept in sync by hand rather
than importing the checkpoint's own copy so this script has no path
dependency on where `rl_common.py` happens to live) on a real
`(state, question)` pair, for `crates/modernbert/tests/laya_real_parity.rs`
to compare against `modernbert::import_dir` + `ModernBert` + `LayaHead`.

Runs the RAW forward (option logits, act logits) - not `rl_agent_api.py`'s
`RLAgent.system_one`, which additionally divides by a per-qtype/cardinality
TEMPERATURE before its own softmax. Dividing every logit by the same positive
constant does not change which index is the max, so comparing raw logits'
argmax is exactly equivalent to comparing `system_one`'s calibrated argmax,
without this script needing to reproduce `temp_bucket`'s own bucketing logic.

Real weights are F16-derived bf16-trained (per the model card) - see the
Laya plan's own "parity tolerance is two different bars" note: this is a
BEHAVIORAL check (argmax/decision agreement), never a tight numeric
tolerance, because bf16-to-fp32 error compounds across 28 pre-LN layers.

This script is resource-prep tooling, never part of brain's build/test path.
Fixtures are NOT committed (`crates/modernbert/tests/fixtures/` is entirely
gitignored).

Usage:
  python3 scripts/parity-dump/laya_real.py --dir ~/.local/share/brain/models/convaiinnovations/laya \\
      --out <scratch>/fixtures-laya-real
  cp <scratch>/fixtures-laya-real/manifest.json crates/modernbert/tests/fixtures/laya_real/

Swedish Embedded AB implements from-scratch GPU kernel ports of released
transformer checkpoints for clients who need an architecture their existing
inference stack does not cover, validated against the real reference
implementation rather than assumed. If your team needs a model brought up
this way, you can procure our services by emailing info@swedishembedded.com.
"""
import argparse
import json
import pathlib

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def serialize_state(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q):
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q, max_len, head_max_len, option_order=None, truncate_left=False):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id] + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


class DecisionModel(nn.Module):
    """Vendored, byte-for-byte, from the real `rl_common.py` (Apache-2.0)."""

    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


# A handful of real (state, question) pairs, the same shapes
# `crates/modernbert/tests/laya_real_parity.rs` builds independently in Rust
# via `modernbert::build_sequence` - kept small since this runs the real
# 28-layer encoder on CPU.
CASES = [
    {
        "name": "triage_choice",
        "state": {"channel": "chat", "turns": 4, "last_message": "My card was charged twice, please refund one of them.",
                   "tags": ["billing", "duplicate-charge"]},
        "question": {"t": "choice", "ins": "What is the customer's primary intent?",
                     "crit": {"refund": "wants money back", "complaint": "expressing dissatisfaction",
                              "question": "asking for information", "other": ""}},
    },
    {
        "name": "escalation_noul",
        "state": "Customer: This is the third time I've contacted support about this. I want to speak to a manager immediately.",
        "question": {"t": "noul", "ins": "Is the customer asking to escalate to a human or manager?"},
    },
    {
        "name": "satisfaction_score",
        "state": {"summary": "Agent resolved the billing dispute within one message and offered a discount."},
        "question": {"t": "score", "ins": "Rate how satisfied the customer likely is, from 0 (very unhappy) to 3 (delighted).",
                     "crit": ["very unhappy", "neutral", "satisfied", "delighted"]},
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(d / "rl_agent_config.json") as f:
        rl_cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(str(d / "tokenizer"))
    ecfg = AutoConfig.from_pretrained(str(d / "encoder"))
    enc = AutoModel.from_config(ecfg, attn_implementation="eager")
    model = DecisionModel(enc, rl_cfg["head_layers"], len(rl_cfg["act_costs"]) + 1)
    sd = load_file(str(d / "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model.eval()

    manifest = {"max_len": rl_cfg["max_len"], "head_max_len": rl_cfg["head_max_len"], "cases": []}
    for case in CASES:
        q = case["question"]
        ids, markers = build_sequence(tok, case["state"], q, rl_cfg["max_len"], rl_cfg["head_max_len"])
        assert len(markers) == len(render_options(q)), f"{case['name']}: options did not fit"
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        marker_pos = torch.tensor([markers], dtype=torch.long)
        marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
        qtype = torch.tensor([QTYPES[q["t"]]], dtype=torch.long)
        with torch.no_grad():
            logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        logits = logits[0].tolist()
        act_logits = act_logits[0].tolist()
        manifest["cases"].append({
            "name": case["name"], "state": case["state"], "question": q,
            "ids": ids, "markers": markers, "qtype": QTYPES[q["t"]],
            "logits": logits, "act_logits": act_logits,
            "argmax_option": int(max(range(len(logits)), key=lambda i: logits[i])),
            "argmax_act": int(max(range(len(act_logits)), key=lambda i: act_logits[i])),
        })
        print(f"{case['name']}: logits={logits} argmax={manifest['cases'][-1]['argmax_option']} "
              f"act_logits={act_logits} argmax_act={manifest['cases'][-1]['argmax_act']}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
