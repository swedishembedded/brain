# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that a served model
# tokenizes and prompts exactly like its published reference for its clients.
# If your team needs expertise in tokenizer and chat-template parity then you
# can procure our services by sending an email to info@swedishembedded.com.
"""Golden for the Ripple-65B tokenizer and chat template.

Encodes and decodes a varied text sample with the `tokenizers` library and renders conversations through
`transformers`' `apply_chat_template` on the published `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja`. The Rust gate (`crates/catalog/tests/it/ricursive_chat.rs`) reproduces both byte for byte.

Messages are written in the OpenAI wire form a client sends (a tool call's `arguments` is a JSON string); the
reference renders the form `transformers` expects (`arguments` parsed), as a serving runtime hands it.

    python -I tools/goldens/ripple_chat_dump.py MODEL_DIR OUT.json
"""

import copy
import json
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer

TEXTS = [
    "Hello, world!",
    "x = 1234567890 + 3.14159; y=-42",
    "I'M here, you'RE there, it's 4:05pm. DON'T",
    "a  b   c\n\n\n  d\t\te \r\n f",
    "    def f(self):\n        return {'k': [1, 2]}\n",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "café naïve 日本語テキスト 한국어 🙂👍 Ελληνικά العربية עברית",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nok\n</think>\nHello<|im_end|>",
    "<|endoftext|><|fim_prefix|>def f():<|fim_suffix|>return 1<|fim_middle|>",
    "|||PHONE_NUMBER||| and <|extra_id_0|> are ordinary text",
    "trailing spaces   \n\n   \t",
    "https://example.com/a/b?c=d&e=f#g",
    "The quick brown fox jumps over the lazy dog. " * 12,
    "",
    " ",
    "\n\n\n",
    "ÿ\u0000\u0001 control and   nbsp ​ zwsp",
    "1 22 333 4444 55555 666666 7777777",
]

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "execute_bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Zürich", "unit": "celsius"}, ensure_ascii=False)}}
CALL2 = {"id": "call_2", "type": "function", "function": {"name": "execute_bash", "arguments": json.dumps({"command": "echo 'héllo' && pwd"})}}

CASES = [
    ("user only", [{"role": "user", "content": "Hi"}], None, False, {}),
    ("user, generation prompt", [{"role": "user", "content": "Hi"}], None, True, {}),
    ("system and user", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], None, True, {}),
    ("multi turn", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3+3?"}], None, True, {}),
    ("assistant with reasoning", [{"role": "user", "content": "2+2?"}, {"role": "assistant", "reasoning_content": "Add them.", "content": "4"}, {"role": "user", "content": "again"}], None, True, {}),
    ("reasoning cleared", [{"role": "user", "content": "2+2?"}, {"role": "assistant", "reasoning_content": "Add them.", "content": "4"}, {"role": "user", "content": "again"}], None, True, {"clear_thinking": True}),
    ("content already thinks", [{"role": "user", "content": "q"}, {"role": "assistant", "reasoning_content": "dup", "content": "<think>\nown\n</think>\nanswer"}], None, False, {}),
    ("tools", [{"role": "user", "content": "Weather in Zürich?"}], TOOLS, True, {}),
    ("tools and system", [{"role": "system", "content": "Use tools when needed."}, {"role": "user", "content": "Weather in Zürich?"}], TOOLS, True, {}),
    ("tool call and result", [{"role": "user", "content": "Weather in Zürich?"}, {"role": "assistant", "content": None, "tool_calls": [CALL]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"}], TOOLS, True, {}),
    ("tool call with text and reasoning", [{"role": "user", "content": "Weather?"}, {"role": "assistant", "reasoning_content": "Need a tool.", "content": "Checking.", "tool_calls": [CALL]}, {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}, {"role": "assistant", "content": "It is sunny."}, {"role": "user", "content": "thanks"}], TOOLS, True, {}),
    ("parallel calls and results", [{"role": "user", "content": "Do both."}, {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2]}, {"role": "tool", "tool_call_id": "call_1", "content": "21C"}, {"role": "tool", "tool_call_id": "call_2", "content": "three files"}], TOOLS, True, {}),
    ("system mid conversation", [{"role": "user", "content": "a"}, {"role": "system", "content": "note"}, {"role": "user", "content": "b"}], None, True, {}),
    ("unicode content", [{"role": "user", "content": "日本語 🙂 café\n\n  indented"}], None, True, {}),
]


def reference_form(messages):
    out = copy.deepcopy(messages)
    for m in out:
        for call in m.get("tool_calls") or []:
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return out


def main(model_dir, out_path):
    raw = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    hf = AutoTokenizer.from_pretrained(model_dir)
    tokenizer = []
    for text in TEXTS:
        ids = raw.encode(text, add_special_tokens=False).ids
        tokenizer.append({"text": text, "ids": ids, "decoded": raw.decode(ids, skip_special_tokens=False), "hf_ids": hf.encode(text, add_special_tokens=False)})
        assert tokenizer[-1]["hf_ids"] == ids, f"transformers and tokenizers disagree on {text!r}"
    templates = []
    for name, messages, tools, gen, kwargs in CASES:
        rendered = hf.apply_chat_template(reference_form(messages), tools=tools, tokenize=False, add_generation_prompt=gen, **kwargs)
        # The request as client text: JSON written with the keys in the order the client chose, which a tool
        # schema's rendering depends on (a parsed-and-reserialized copy would sort or reorder them).
        templates.append({"name": name, "messages_json": json.dumps(messages, ensure_ascii=False), "tools_json": json.dumps(tools, ensure_ascii=False) if tools else None, "add_generation_prompt": gen, "kwargs_json": json.dumps(kwargs), "rendered": rendered})
    json.dump({"tokenizer": tokenizer, "templates": templates, "eos_token_id": hf.eos_token_id, "bos_token_id": hf.bos_token_id, "pad_token_id": hf.pad_token_id}, open(out_path, "w"), ensure_ascii=False)
    print(f"{len(tokenizer)} texts, {len(templates)} conversations -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
