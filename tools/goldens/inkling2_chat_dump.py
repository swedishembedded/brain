#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump `apply_chat_template` renders for Inkling 2 Mini, the reference for
`data::chat_template` over the checkpoint's own `chat_template.jinja`.

Writes `<out>/chat.json`: fixed conversations (single turn, system, multi-turn
with assistant thinking history, tool declarations with a tool call and its
result, image parts, a few malformed shapes), each rendered with and without the
generation prompt and under named, numeric and invalid `reasoning_effort`
values. A render that raises is recorded as an error, not skipped: a template
that refuses a shape is a fact the Rust side must reproduce.

Run in the reference venv:

    ~/project/shared/venvs/inkling-ref/bin/python tools/goldens/inkling2_chat_dump.py \\
        --model <hf_dir> --out testdata/inkling2

Swedish Embedded AB implements chat-template parity for model ports on edge
inference engines for its clients. If your team needs expertise in rendering
prompts exactly as a model's reference stack does, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import argparse
import json
import os

from transformers import AutoTokenizer

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"units": {"type": "string", "enum": ["c", "f"]}, "city": {"type": "string"}}, "required": ["city"]},
    }},
    {"name": "no_args", "description": "", "parameters": {}},
]

CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"units": "c", "city": "Malmö", "days": [1, 2.5, None, True], "nested": {"b": 1, "a": "å"}}}}

CONVERSATIONS = {
    "single": [{"role": "user", "content": "What is 2+2?"}],
    "system": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name a prime."},
    ],
    "system_after_user": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late system"},
        {"role": "user", "content": "again"},
    ],
    "thinking_history": [
        {"role": "user", "content": "What is 3*3?"},
        {"role": "assistant", "reasoning_content": "3 times 3 is 9.", "content": "The answer is 9."},
        {"role": "user", "content": "And 4*4?"},
    ],
    "thinking_only_assistant": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "hmm", "content": ""},
    ],
    "named_assistant": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "name": "analyst", "content": "a"},
    ],
    "tool_round_trip": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in Malmö?"},
        {"role": "assistant", "reasoning_content": "Need the tool.", "content": "Let me check.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 12}"},
        {"role": "assistant", "content": "It is 12 degrees."},
    ],
    "tool_result_named": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "no_args"}}]},
        {"role": "tool", "name": "no_args", "content": "done"},
    ],
    "parts": [
        {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}}, "plain part"]},
    ],
    "whitespace_cjk": [{"role": "user", "content": "  翻译：hello world  \n"}],
    "empty": [],
    "bad_role": [{"role": "wizard", "content": "x"}],
    "string_arguments": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
    ],
    "audio_part": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}],
    "assistant_image": [{"role": "assistant", "content": [{"type": "image", "image": "x"}]}],
}

# conversation -> tools passed
TOOLS_FOR = {"tool_round_trip": TOOLS, "tool_result_named": TOOLS, "single": TOOLS}

EFFORTS = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max", " high ", 0, 0.0, 0.5, 0.25, 0.7, 0.9, 0.99, 1, "0.5", 1.0, -0.1, 0.995, "ultra"]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def render(auto, conv, tools, gen, effort):
    kwargs = {}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    try:
        return {"text": auto.apply_chat_template(conv, tools=tools, tokenize=False, add_generation_prompt=gen, **kwargs)}
    except Exception as e:  # a refused shape is a recorded fact
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    auto = AutoTokenizer.from_pretrained(args.model)
    cases = []
    for name, conv in CONVERSATIONS.items():
        tools = TOOLS_FOR.get(name)
        for gen in (False, True):
            cases.append({"name": name, "messages": conv, "tools": tools, "add_generation_prompt": gen, "reasoning_effort": None, **render(auto, conv, tools, gen, None)})
    # every effort value on a plain conversation and on one with a system turn first
    for name in ("single", "system"):
        for effort in EFFORTS[1:]:
            conv = CONVERSATIONS[name]
            cases.append({"name": name, "messages": conv, "tools": None, "add_generation_prompt": True, "reasoning_effort": effort, **render(auto, conv, None, True, effort)})
    write_json(os.path.join(args.out, "chat.json"), {"cases": cases})
    print(f"{len(cases)} cases, {sum('error' in c for c in cases)} errors")


if __name__ == "__main__":
    main()
