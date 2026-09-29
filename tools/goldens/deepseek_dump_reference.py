#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump reference goldens for the DeepSeek dense text decoders brain serves
through its `llama` / `qwen2` config variants of the qwen3 decoder.

Covers every `deepseek-ai/*` text checkpoint up to 8B (R1-Distill-Qwen-1.5B/7B,
R1-Distill-Llama-8B, deepseek-coder 1.3b/6.7b/7b-v1.5, deepseek-math-7b,
deepseek-llm-7b) - they differ only in config (bias, GQA, RoPE scaling, eps),
tokenizer shape and chat template, which is exactly what this dump pins.

Written under `--out` (a `testdata/deepseek/<repo>/` directory):

  tokenizer.json     a hostile corpus and, per string, the ids of BOTH HF
                     encode paths - `tokenizers.Tokenizer.from_file` (the file's
                     own post-processor) and `AutoTokenizer` (which re-derives
                     BOS from `tokenizer_config.json`'s `add_bos_token`; the two
                     disagree for several of these checkpoints) - plus decodes.
  chat.json          `apply_chat_template` renders of fixed conversations, with
                     and without the generation prompt (errors recorded, not
                     skipped: a template that refuses a system turn is a fact).
  rope.safetensors   the model's own `inv_freq` + `attention_scaling`, and this
                     script's independent recomputation of them (asserted equal,
                     so the golden cannot be a copy of a wrong table).
  forward.safetensors  a fixed prompt through the first `--layers` decoder
                     layers (every layer with `--full`), fp32 on CPU: embeddings,
                     layer-0 input norm, post-bias q/k/v, post-RoPE q/k, every
                     hidden state, and the logits.
  generate.json      greedy continuation ids for the prompt (and, for a
                     checkpoint with FIM specials, for an infill prompt).
  manifest.json      shapes/sha256 of everything above plus the `source` block
                     (`golden_source.py`) and the library versions used.

Runs in the reference venv (torch CPU + transformers + tokenizers):

    resources/deepseek/.venv/bin/python tools/goldens/deepseek_dump_reference.py \\
        --model <hf_dir> --out testdata/deepseek/<repo> [--layers 2 | --full]

Swedish Embedded AB builds bit-level reference parity for model ports on edge
inference engines for its clients. If your team needs expertise in bringing a
model family onto its own engine without silent numeric drift, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import argparse
import json
import math
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import sha256_of, source_block  # noqa: E402

PROMPT = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
FIM_PREFIX = "def add(a, b):\n"
FIM_SUFFIX = "\n    return c\n"
GREEDY_TOKENS = 32

# Deliberately hostile: every branch of the five tokenizer shapes these
# checkpoints ship (Qwen2 / Llama-3 / DeepSeek-LLM Split-sequence / coder v1)
# plus the normalizer (NFC) and added-token matching.
CORPUS = [
    "Hello, world!",
    "  leading and trailing spaces  ",
    "line one\nline two\r\nline three\r\n\n\n",
    "tabs\tand\tmixed \t whitespace",
    "trailing whitespace at the end   ",
    "numbers 1 12 123 1234 12345 3.14159 -42 1e10 ٣٤٥ ①②",
    "don't won't I'm you're they've she'll he'd IT'S",
    "中文分词测试，包括标点符号。日本語のテキスト。한국어 텍스트",
    "mixed English中文English 123中文456",
    "Ωμέγα Кириллица Ελληνικά über café naïve",
    "é café Å Å",  # NFC-decomposable: é, café, Å (two forms), Ångström sign
    "emoji 😀👍🏽 family 👨‍👩‍👧 flags 🇸🇪",
    "punctuation!!! ??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& ((())) [[]] {{}}",
    "fullwidth！＂＃（）ｗｏｒｌｄ　ideographic space",
    "code: fn main() { let x = vec![1, 2, 3]; println!(\"{:?}\", x); }",
    "<think>\nreasoning\n</think>\n\nanswer",
    "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜>hello<｜end▁of▁sentence｜>",
    "<｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>",
    "<|im_start|>user\nhi<|im_end|>",
    "x" * 300,
    "",
]

CONVERSATIONS = {
    "single": [{"role": "user", "content": "What is 2+2?"}],
    "system": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name a prime."},
    ],
    "reasoning_history": [
        {"role": "user", "content": "What is 3*3?"},
        {"role": "assistant", "content": "<think>\n3 times 3 is 9.\n</think>\n\nThe answer is 9."},
        {"role": "user", "content": "And 4*4?"},
    ],
    "whitespace_cjk": [{"role": "user", "content": "  翻译：hello world  \n"}],
}


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def dump_tokenizer(model_dir, out):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    raw = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    auto = AutoTokenizer.from_pretrained(model_dir)
    rows = []
    for text in CORPUS:
        file_plain = raw.encode(text, add_special_tokens=False).ids
        rows.append({
            "text": text,
            "file_plain": file_plain,
            "file_special": raw.encode(text, add_special_tokens=True).ids,
            "auto_plain": auto(text, add_special_tokens=False)["input_ids"],
            "auto_special": auto(text)["input_ids"],
            "decode_plain": raw.decode(file_plain, skip_special_tokens=False),
        })
    specials = {t.content: i for i, t in auto.added_tokens_decoder.items()}
    write_json(os.path.join(out, "tokenizer.json"), {
        "vocab_size": raw.get_vocab_size(with_added_tokens=True),
        "bos_token": auto.bos_token,
        "eos_token": auto.eos_token,
        "added_tokens": dict(sorted(specials.items(), key=lambda kv: kv[1])),
        "rows": rows,
    })
    return auto


def dump_chat(auto, out):
    renders = {}
    for name, conv in CONVERSATIONS.items():
        entry = {"messages": conv}
        for gen in (False, True):
            key = "with_generation_prompt" if gen else "without_generation_prompt"
            try:
                entry[key] = auto.apply_chat_template(conv, tokenize=False, add_generation_prompt=gen)
            except Exception as e:  # a template that refuses a turn shape is a recorded fact
                entry[key + "_error"] = f"{type(e).__name__}: {e}"
        renders[name] = entry
    write_json(os.path.join(out, "chat.json"), {"has_template": auto.chat_template is not None, "renders": renders})


def independent_inv_freq(cfg, head_dim):
    """Recompute the scaled RoPE table from the config alone - the oracle the
    model's own buffer is checked against, so a golden cannot silently record a
    table the reference itself got wrong for this config."""
    theta = float(cfg.rope_theta if getattr(cfg, "rope_theta", None) is not None else cfg.rope_parameters["rope_theta"])
    base = [1.0 / (theta ** (2 * j / head_dim)) for j in range(head_dim // 2)]
    rs = getattr(cfg, "rope_scaling", None) or {}
    if not rs and getattr(cfg, "rope_parameters", None):
        rs = {k: v for k, v in cfg.rope_parameters.items() if k != "rope_theta"}
    kind = rs.get("rope_type", rs.get("type", "default"))
    if kind in (None, "default"):
        return base, 1.0, "none"
    if kind == "linear":
        return [f / rs["factor"] for f in base], 1.0, "linear"
    if kind == "llama3":
        factor, low, high = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
        orig = rs["original_max_position_embeddings"]
        low_wl, high_wl = orig / low, orig / high
        out = []
        for f in base:
            wl = 2 * math.pi / f
            if wl < high_wl:
                out.append(f)
            elif wl > low_wl:
                out.append(f / factor)
            else:
                smooth = (orig / wl - low) / (high - low)
                out.append((1 - smooth) * f / factor + smooth * f)
        return out, 1.0, "llama3"
    raise SystemExit(f"rope scaling type {kind!r} has no independent recomputation here")


def dump_forward(model_dir, out, layers, full):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_dir)
    total_layers = cfg.num_hidden_layers
    n = total_layers if full else min(layers, total_layers)
    cfg.num_hidden_layers = n
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    rot = model.model.rotary_emb
    inv = rot.inv_freq.detach().float()
    want, want_scale, kind = independent_inv_freq(cfg, head_dim)
    want_t = torch.tensor(want, dtype=torch.float32)
    err = (inv - want_t).abs().max().item() / want_t.abs().max().item()
    assert err < 1e-6, f"model inv_freq disagrees with the independent recomputation ({kind}): rel {err}"
    assert abs(float(rot.attention_scaling) - want_scale) < 1e-7, f"attention_scaling {rot.attention_scaling} != {want_scale}"
    save_file({"inv_freq": inv.contiguous(), "attention_scaling": torch.tensor([float(rot.attention_scaling)])},
              os.path.join(out, "rope.safetensors"), metadata={"scaling": kind})

    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    taps = {}
    layer0 = model.model.layers[0]
    hooks = [
        layer0.input_layernorm.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_input_norm", o)),
        layer0.self_attn.q_proj.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_q", o)),
        layer0.self_attn.k_proj.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_k", o)),
        layer0.self_attn.v_proj.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_v", o)),
    ]
    with torch.no_grad():
        res = model(ids, output_hidden_states=True)
    for h in hooks:
        h.remove()

    # Post-RoPE q/k, recomputed with the reference's own rotary helpers from the
    # hooked projections - the rung that separates "wrong table" from "wrong
    # rotation layout" when the layer-0 comparison fails.
    t = ids.shape[1]
    pos = torch.arange(t).unsqueeze(0)
    cos, sin = rot(taps["l0_q"], pos)
    q = taps["l0_q"].view(1, t, cfg.num_attention_heads, head_dim).transpose(1, 2)
    k = taps["l0_k"].view(1, t, cfg.num_key_value_heads, head_dim).transpose(1, 2)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    tensors = {
        "input_ids": ids[0].to(torch.int32),
        "l0_input_norm": taps["l0_input_norm"][0],
        "l0_q": taps["l0_q"][0],
        "l0_k": taps["l0_k"][0],
        "l0_v": taps["l0_v"][0],
        "l0_q_rope": q_rot[0].transpose(0, 1).reshape(t, -1),
        "l0_k_rope": k_rot[0].transpose(0, 1).reshape(t, -1),
        "logits": res.logits[0],
    }
    for i, h in enumerate(res.hidden_states):
        tensors[f"hidden_{i:02d}"] = h[0]
    save_file({k: v.detach().contiguous() for k, v in tensors.items()}, os.path.join(out, "forward.safetensors"),
              metadata={"layers": str(n), "total_layers": str(total_layers), "prompt": PROMPT})

    gen = {"layers": n, "prompt": PROMPT, "prompt_ids": ids[0].tolist()}
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=GREEDY_TOKENS, do_sample=False)
    gen["greedy_ids"] = g[0, t:].tolist()
    specials = {tk.content for tk in tok.added_tokens_decoder.values()}
    fim = ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>")
    if all(s in specials for s in fim):
        text = f"{fim[0]}{FIM_PREFIX}{fim[1]}{FIM_SUFFIX}{fim[2]}"
        fids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            fg = model.generate(fids, max_new_tokens=GREEDY_TOKENS, do_sample=False)
        gen["fim"] = {"text": text, "ids": fids[0].tolist(), "greedy_ids": fg[0, fids.shape[1]:].tolist()}
    write_json(os.path.join(out, "generate.json"), gen)
    return cfg, n, total_layers, kind


def weight_files(model_dir):
    names = sorted(os.listdir(model_dir))
    st = [n for n in names if n.endswith(".safetensors")]
    return [os.path.join(model_dir, n) for n in (st or [n for n in names if n.endswith(".bin")])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="HF checkpoint directory")
    ap.add_argument("--out", required=True, help="output directory (testdata/deepseek/<repo>)")
    ap.add_argument("--checkpoint", help="<vendor>/<repo> for the manifest (default: deepseek-ai/<dirname>)")
    ap.add_argument("--layers", type=int, default=2, help="decoder layers to run (default 2)")
    ap.add_argument("--full", action="store_true", help="run every decoder layer")
    ap.add_argument("--hash", action="store_true", help="sha256 the weight files (slow for 7B)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    auto = dump_tokenizer(args.model, args.out)
    dump_chat(auto, args.out)
    cfg, n, total, kind = dump_forward(args.model, args.out, args.layers, args.full)

    import tokenizers
    import transformers

    files = {}
    for name in sorted(os.listdir(args.out)):
        if name != "manifest.json":
            files[name] = sha256_of(os.path.join(args.out, name))
    checkpoint = args.checkpoint or "deepseek-ai/" + os.path.basename(os.path.normpath(args.model))
    manifest = {
        "files": files,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
        "source": source_block(
            checkpoint=checkpoint,
            files=weight_files(args.model),
            hash_files=args.hash,
            identity={
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": total,
                "dumped_layers": n,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "intermediate_size": cfg.intermediate_size,
                "vocab_size": cfg.vocab_size,
            },
        ),
        "config": {"architecture": cfg.architectures[0], "rope_scaling": kind},
    }
    write_json(os.path.join(args.out, "manifest.json"), manifest)
    print(f"wrote {args.out}: {n}/{total} layers, rope={kind}")


if __name__ == "__main__":
    main()
