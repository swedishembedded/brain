#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the tokenizer reference golden for Inkling 2 Mini.

Writes `<out>/tokenizer.json`: a hostile corpus and, per string, the ids the
`tokenizers` library produces from the checkpoint's own `tokenizer.json`
(`file_plain`, no post-processor) plus its decode. The tokenizer is byte-level
BPE with no normalizer, an o200k-style `Split` regex, `ByteLevel(use_regex:
false)` and `ignore_merges`, so the corpus targets the places such a pipeline
disagrees between implementations: digit runs, CJK beside Latin, `\\r\\n`,
trailing spaces, emoji and ZWJ sequences, contractions, long runs, and every
added special token spelled inline.

Run in the reference venv:

    ~/project/shared/venvs/inkling-ref/bin/python tools/goldens/inkling2_tokenizer_dump.py \\
        --model <hf_dir> --out testdata/inkling2

Swedish Embedded AB implements tokenizer parity for model ports on edge
inference engines for its clients. If your team needs expertise in making an
engine encode text exactly as the model's reference tokenizer does, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import argparse
import json
import os

from tokenizers import Tokenizer

TOOL_CALL = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Malmö","units":"c"}}<|end_message|>'
THINKING = "<|message_model|><|content_thinking|>The user wants the weather.<|end_message|><|message_model|><|content_text|>Sunny.<|end_message|><|content_model_end_sampling|>"

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
    "é café Å Å",
    "emoji \U0001f600\U0001f44d\U0001f3fd family \U0001f468‍\U0001f469‍\U0001f467 flags \U0001f1f8\U0001f1ea",
    "punctuation!!! ??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& ((())) [[]] {{}}",
    "fullwidth！＂＃（）ｗｏｒｌｄ　ideographic space",
    "code: fn main() { let x = vec![1, 2, 3]; println!(\"{:?}\", x); }",
    "def f(x):\n    return x  # comment\n\n\n    pass",
    "CamelCaseIdentifier snake_case_identifier SCREAMING_CASE HTTPServerError",
    "https://example.com/path?q=1&r=%20#frag user@example.com /usr/local/bin",
    "<think>\nreasoning\n</think>\n\nanswer",
    "<|im_start|>user\nhi<|im_end|>",
    TOOL_CALL,
    THINKING,
    '{"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}}}]}',
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>hi<|end_message|>",
    "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
    "<|content_tool_error|>boom<|end_message|><|content_invoke_tool_text|>x<|end_message|>",
    "<|unused|><|endoftext|><|begin_of_text|><|content_xml|><|message_tool|>",
    "<|message_model|>",
    "<|not_a_token|> <|message_model| > <|message_model",
    "x" * 300,
    "ab" * 150,
    " " * 70,
    "\n" * 40,
    "1" * 50,
    "",
]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    specials = {}
    corpus = list(CORPUS)
    for tid in range(199998, raw.get_vocab_size(with_added_tokens=True)):
        content = raw.id_to_token(tid)
        specials[content] = tid
        corpus.append(f"a{content}b")
    corpus.append("".join(specials))
    rows = []
    for text in corpus:
        ids = raw.encode(text, add_special_tokens=False).ids
        rows.append({"text": text, "file_plain": ids, "decode_plain": raw.decode(ids, skip_special_tokens=False)})
    write_json(os.path.join(args.out, "tokenizer.json"), {
        "vocab_size": raw.get_vocab_size(with_added_tokens=True),
        "added_tokens": specials,
        "rows": rows,
    })
    print(f"{len(rows)} rows, vocab {raw.get_vocab_size(with_added_tokens=True)}")


if __name__ == "__main__":
    main()
