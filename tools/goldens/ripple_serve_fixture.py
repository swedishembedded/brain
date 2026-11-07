#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that a served model
# answers over its HTTP API exactly as its published reference format
# prescribes, for its clients. If your team needs expertise in end-to-end
# serving tests of large language models without their weights then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Model directories for serving Ripple end to end.

Both modes write a Hugging Face directory a pull would leave (`config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`), laid out as `OUT/<vendor>/<repo>` for `brain serve --models-dir OUT`.

  scripted  A tiny random Ripple (3 layers: two sliding, one global; hidden 64) whose `lm_head` is designed so greedy
            decoding replays one fixed reply, whatever the prompt: reasoning, a blank line, a JSON tool call, end of
            turn. The embedding of each reply token is a random unit vector and the `lm_head` row of its successor is
            that vector scaled up, so the successor wins the argmax by a wide margin while the attention and MLP
            weights are small random values that the engine still computes. The tokenizer is byte level with the
            reply's pieces as extra tokens. It tests the serving path (template, tokenizer, scheduler, scanner,
            OpenAI surface); the transformer math is gated against the reference in `crates/ricursive/tests`.
            The same directory also holds `Ripple-scripted-DFlash2` beside it, a tiny RANDOM DFlash2 draft model for it
            (width 64, three layers reading the target's three, a window of 6, blocks of five rows): it proposes at
            random, so a speculative run exercises the draft, the tap retention and the verify, and the output must
            still be the fixed reply.
  partial   Layers 0 and 4 of the real checkpoint (a sliding and a global layer, with the embedding, final norm and
            `lm_head`) as a two-layer model, from a directory of `layers-0-4.safetensors` (or the NVFP4 file) beside the
            release's `config.json`, tokenizer and template: the weights file with layer 4 renamed layer 1, `config.json`
            cut to [sliding, global], the rest linked in place.

    python3 -I tools/goldens/ripple_serve_fixture.py scripted OUT TEMPLATE.jinja
    python3 -I tools/goldens/ripple_serve_fixture.py partial OUT SRC_DIR [--nvfp4]

The exact reply of `scripted` is `REPLY` below; `tests/e2e/ripple.bats` asserts on it.
"""

import array
import json
import math
import os
import random
import struct
import sys

VENDOR = "ricursiveAI"

# The pieces of the scripted reply, each one token: reasoning the scanner splits out, then one tool call.
REPLY = ["<think>", "\nThe user wants the weather, so I call the tool.\n", "</think>", "\n\n", "<tool_call>", '\n{"name": "get_weather", "arguments": {"city": "Zürich", "unit": "celsius"}}\n', "</tool_call>", "<|im_end|>"]

# Tokens the release defines above its byte-level vocabulary, in id order from 100256 (the tiny model starts at 256).
SPECIALS = [("<|extra_id_0|>", False), ("<|endoftext|>", True), ("<|fim_prefix|>", True), ("<|fim_middle|>", True), ("<|fim_suffix|>", True), ("|||PHONE_NUMBER|||", False), ("|||EMAIL_ADDRESS|||", False), ("|||IP_ADDRESS|||", False), ("<|im_start|>", True), ("<|im_end|>", True)]

HIDDEN, FFN, HEADS, KV_HEADS, HEAD_DIM, WINDOW = 64, 96, 4, 2, 16, 4
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"]


def bytes_to_unicode():
    """GPT-2's reversible byte to printable character map."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


def write_safetensors(path, tensors):
    """`tensors`: name -> (shape, list of floats), written as F32."""
    header, offset, blobs = {}, 0, []
    for name, (shape, values) in tensors.items():
        raw = array.array("f", values).tobytes()
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for raw in blobs:
            f.write(raw)


def unit_vectors(n, dim, rng):
    out = []
    for _ in range(n):
        v = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        out.append([x / norm for x in v])
    return out


def scripted(out_dir, template):
    rng = random.Random(7)
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    added = []  # (id, content, special)
    next_id = 256
    for content, special in SPECIALS:
        added.append((next_id, content, special))
        next_id += 1
    ids = {content: i for i, content, _ in added}
    # Reply pieces that are not already tokens become plain added tokens; the template writes `<think>` as text.
    for piece in REPLY:
        if piece not in ids:
            ids[piece] = next_id
            added.append((next_id, piece, False))
            next_id += 1
    vocab_size = next_id + 8
    newline = vocab[b2u[ord("\n")]]

    chain = [newline] + [ids[p] for p in REPLY]  # the prompt ends in a newline; each token is followed by the next
    assert len(set(chain)) == len(chain), "the scripted chain needs distinct tokens"
    codes = unit_vectors(vocab_size, HIDDEN, rng)
    embed_scale = 8.0
    embed = [x for v in codes for x in v]
    # logits[u] = lm_head[u] . norm(h); h is the scaled embedding of the current token, so the row of the successor
    # is the current token's code scaled to dominate the cross terms of every other row.
    head = [[0.0] * HIDDEN for _ in range(vocab_size)]
    for cur, nxt in zip(chain, chain[1:]):
        head[nxt] = [x * 40.0 for x in codes[cur]]
    # After the end of turn nothing follows: the engine stops there.
    tensors = {
        "model.embed_tokens.weight": ((vocab_size, HIDDEN), embed),
        "model.norm.weight": ((HIDDEN,), [1.0] * HIDDEN),
        "lm_head.weight": ((vocab_size, HIDDEN), [x for row in head for x in row]),
    }
    q_dim, kv_dim = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

    def small(rows, cols):
        return ((rows, cols), [rng.uniform(-1, 1) * 0.5 / math.sqrt(cols) for _ in range(rows * cols)])

    for layer in range(len(LAYER_TYPES)):
        p = f"model.layers.{layer}."
        # The norms after each branch scale what it adds to the residual stream: small, so the token's own code stays
        # the dominant direction at the head while the branches still run.
        for norm, gain in (("input_layernorm", 1.0), ("post_attention_layernorm", 0.05), ("pre_feedforward_layernorm", 1.0), ("post_feedforward_layernorm", 0.05)):
            tensors[p + norm + ".weight"] = ((HIDDEN,), [gain] * HIDDEN)
        tensors[p + "self_attn.q_proj.weight"] = small(q_dim, HIDDEN)
        tensors[p + "self_attn.k_proj.weight"] = small(kv_dim, HIDDEN)
        tensors[p + "self_attn.v_proj.weight"] = small(kv_dim, HIDDEN)
        tensors[p + "self_attn.o_proj.weight"] = small(HIDDEN, q_dim)
        tensors[p + "self_attn.g_proj.weight"] = small(q_dim, HIDDEN)
        tensors[p + "self_attn.q_norm.weight"] = ((HEAD_DIM,), [1.0] * HEAD_DIM)
        tensors[p + "self_attn.k_norm.weight"] = ((HEAD_DIM,), [1.0] * HEAD_DIM)
        tensors[p + "self_attn.value_res_lambda"] = ((1,), [0.0])
        tensors[p + "mlp.gate_proj.weight"] = small(FFN, HIDDEN)
        tensors[p + "mlp.up_proj.weight"] = small(FFN, HIDDEN)
        tensors[p + "mlp.down_proj.weight"] = small(HIDDEN, FFN)
    repo = os.path.join(out_dir, VENDOR, "Ripple-scripted")
    os.makedirs(repo, exist_ok=True)
    write_safetensors(os.path.join(repo, "model.safetensors"), tensors)
    eos = [ids["<|im_end|>"], ids["<|endoftext|>"]]
    config = {
        "architectures": ["RicursiveForCausalLM"], "model_type": "ricursive", "vocab_size": vocab_size, "hidden_size": HIDDEN, "intermediate_size": FFN,
        "num_hidden_layers": len(LAYER_TYPES), "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM, "sliding_window": WINDOW,
        "layer_types": LAYER_TYPES, "use_gated_attention": True, "gate_full_precision": True, "use_valres": True, "ffn_clamp_threshold": 20.0,
        "embed_scale": embed_scale, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "tie_word_embeddings": False,
        "max_position_embeddings": 2048, "bos_token_id": ids["<|endoftext|>"], "eos_token_id": eos, "pad_token_id": ids["<|endoftext|>"],
    }
    json.dump(config, open(os.path.join(repo, "config.json"), "w"), indent=2)
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": s} for i, c, s in added],
        "normalizer": None, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}, "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }
    json.dump(tokenizer, open(os.path.join(repo, "tokenizer.json"), "w"), ensure_ascii=False)
    tcfg = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|endoftext|>", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "model_max_length": 2048,
            "chat_template": "{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{%- endfor %}"}
    json.dump(tcfg, open(os.path.join(repo, "tokenizer_config.json"), "w"))
    with open(template) as src, open(os.path.join(repo, "chat_template.jinja"), "w") as dst:
        dst.write(src.read())
    print(repo)
    scripted_draft(out_dir, vocab_size)


def scripted_draft(out_dir, vocab_size):
    """A tiny random DFlash2 draft model for the scripted Ripple, laid out as a pulled `<vendor>/<repo>-DFlash2`
    (what `--speculative dflash` finds beside the target). Weights are small gaussians, the convolutions start as the
    identity on the current token (tap 0 = 1), as the published draft's do."""
    rng = random.Random(11)
    d, ff, heads, kv_heads, head_dim, layers, group, taps, rank = HIDDEN, FFN, HEADS, KV_HEADS, HEAD_DIM, 3, 8, 4, 8
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim

    def gauss(rows, cols, scale=0.08):
        return ((rows, cols), [rng.gauss(0, scale) for _ in range(rows * cols)])

    def gain(n):
        return ((n,), [1.0 + rng.gauss(0, 0.2) for _ in range(n)])

    def base():
        values = [rng.gauss(0, 0.1) + (1.0 if t == 0 else 0.0) for _ in range(2) for t in range(taps) for _ in range(d)]
        return ((2, taps, d), values)

    t = {
        "fc.weight": gauss(d, 3 * d, 0.06),
        "hidden_norm.weight": gain(d),
        "norm.weight": gain(d),
        "candidate_selector.hidden_projection.weight": gauss(rank, d, 0.2),
        "candidate_selector.predecessor_codebook": gauss(vocab_size, rank, 1.0),
        "candidate_selector.successor_codebook": gauss(vocab_size, rank, 1.0),
    }
    for i in range(layers):
        p = f"layers.{i}."
        t.update({
            p + "input_layernorm.weight": gain(d), p + "post_attention_layernorm.weight": gain(d),
            p + "self_attn.q_proj.weight": gauss(q_dim, d), p + "self_attn.k_proj.weight": gauss(kv_dim, d), p + "self_attn.v_proj.weight": gauss(kv_dim, d),
            p + "self_attn.o_proj.weight": gauss(d, q_dim), p + "self_attn.q_norm.weight": gain(head_dim), p + "self_attn.k_norm.weight": gain(head_dim),
            p + "mlp.gate_proj.weight": gauss(ff, d), p + "mlp.up_proj.weight": gauss(ff, d), p + "mlp.down_proj.weight": gauss(d, ff),
            p + "attention_conv.base_kernel": base(), p + "attention_conv.kernel_projection.weight": gauss(2 * taps * (d // group), d, 0.1),
            p + "mlp_conv.base_kernel": base(), p + "mlp_conv.kernel_projection.weight": gauss(2 * taps * (d // group), d, 0.1),
        })
    repo = os.path.join(out_dir, VENDOR, "Ripple-scripted-DFlash2")
    os.makedirs(repo, exist_ok=True)
    write_safetensors(os.path.join(repo, "model.safetensors"), t)
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": d, "intermediate_size": ff, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "rope_theta": 10000.0, "sliding_window": 6, "use_sliding_window": True,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"], "no_rope_layers": [2], "is_causal": False,
        "vocab_size": vocab_size, "draft_vocab_size": vocab_size, "num_target_layers": len(LAYER_TYPES), "max_position_embeddings": 2048,
        "dflash_config": {"block_size": 5, "conv_group_size": group, "conv_kernel_size": taps, "mask_token_id": vocab_size - 1, "selector_rank": rank, "selector_top_k": 4, "target_layer_ids": [0, 1, 2]},
    }
    json.dump(config, open(os.path.join(repo, "config.json"), "w"), indent=2)
    print(repo)


def renumbered_copy(src, dst, mapping):
    """`src` (safetensors) as `dst` with the tensors of layer `old` renamed to layer `new` for each (old, new) in `mapping`.
    The data region is copied unchanged: only the header names move."""
    with open(src, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        renamed = {}
        for name, entry in header.items():
            for old, new in mapping.items():
                prefix = f"model.layers.{old}."
                if name.startswith(prefix):
                    name = f"model.layers.{new}." + name[len(prefix):]
                    break
            renamed[name] = entry
        kept = {n: e for n, e in renamed.items() if n == "__metadata__" or not n.startswith("model.layers.") or int(n.split(".")[2]) in set(mapping.values())}
        text = json.dumps(kept, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 8)
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(text)))
            out.write(text)
            f.seek(data_start)
            while True:
                block = f.read(1 << 24)
                if not block:
                    break
                out.write(block)


def partial(out_dir, src_dir, nvfp4):
    """The real checkpoint's layers 0 (sliding) and 4 (global) as a two-layer model: the config cut to
    [sliding, global] and the weights file with layer 4 renamed layer 1 (the files hold exactly those two layers, with
    the embedding, the final norm and `lm_head`). Everything else is linked in place."""
    repo = os.path.join(out_dir, VENDOR, "Ripple-65B-layers-0-4" + ("-NVFP4" if nvfp4 else ""))
    os.makedirs(repo, exist_ok=True)
    config = json.load(open(os.path.join(src_dir, "config.json")))
    types = config.get("layer_types") or ["full_attention" if p == 0 else "sliding_attention" for p in config["hybrid_layer_pattern"]]
    config["num_hidden_layers"] = 2
    config["layer_types"] = [types[0], types[4]]
    config.pop("hybrid_layer_pattern", None)
    json.dump(config, open(os.path.join(repo, "config.json"), "w"), indent=2)
    renumbered_copy(os.path.join(src_dir, "layers-0-4-nvfp4.safetensors" if nvfp4 else "layers-0-4.safetensors"), os.path.join(repo, "model.safetensors"), {0: 0, 4: 1})
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        link = os.path.join(repo, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join(os.path.abspath(src_dir), name), link)
    print(repo)


def main(argv):
    if len(argv) >= 4 and argv[1] == "scripted":
        scripted(argv[2], argv[3])
    elif len(argv) >= 4 and argv[1] == "partial":
        partial(argv[2], argv[3], "--nvfp4" in argv)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
