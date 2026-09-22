#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump two ground-truth fixtures for Laya M4 against the REAL
`convaiinnovations/laya` release (Apache-2.0), read directly from a
`brain pull`ed checkpoint directory:

1. A tokenizer DIFFERENTIAL battery: real strings encoded by the real
   `tokenizers.Tokenizer` loaded straight from `<dir>/tokenizer/tokenizer.json`
   (`add_special_tokens=False`, matching `data::qwen_tokenizer::QwenBpe::encode`'s
   own template-free contract) - not `QwenBpe`'s reimplementation. This is the
   empirical check the Laya M4 plan calls for: Laya's `tokenizer.json` declares
   a bare `ByteLevel` pre-tokenizer with `use_regex: true` and NO explicit
   `pattern`, which the real `tokenizers` library resolves to the ORIGINAL
   GPT-2 regex (case-SENSITIVE contractions, uncapped digit runs) - genuinely
   different from `QwenBpe`'s hardcoded cl100k-style default (case-insensitive,
   capped). The battery below is chosen to probe exactly those two axes, plus
   whitespace/punctuation/unicode cases that should NOT diverge.

2. `build_sequence` GOLDEN cases: the exact Rust reproduction target, run
   through a small vendored copy of `rl_common.py`'s own `build_sequence`/
   `render_options`/`serialize_state` (Apache-2.0, inlined rather than a new
   pip dependency, mirroring `scripts/parity-dump/modernbert.py`'s own
   `DecisionHead` inline copy) against the SAME real tokenizer.

This script is resource-prep tooling, never part of brain's build/test path.
Fixtures are NOT committed (same convention as every other `scripts/parity-dump/*.py`
- `crates/modernbert/tests/fixtures` is gitignored); copy the output where the
Rust test expects it.

Usage:
  python3 scripts/parity-dump/laya_seq.py --dir ~/.local/share/brain/models/convaiinnovations/laya \\
      --out <scratch>/fixtures-laya-seq
  cp <scratch>/fixtures-laya-seq/manifest.json crates/modernbert/tests/fixtures/laya_seq/

Swedish Embedded AB implements from-scratch GPU kernel ports of released
transformer checkpoints for clients who need an architecture their existing
inference stack does not cover, validated against the real reference
implementation rather than assumed. If your team needs a model brought up
this way, you can procure our services by emailing info@swedishembedded.com.
"""
import argparse
import json
import pathlib

from tokenizers import Tokenizer
from transformers import AutoTokenizer

# ---------------------------------------------------------------- vendored rl_common.py (Apache-2.0)
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


# ---------------------------------------------------------------- tokenizer differential battery
BATTERY = [
    "don't", "DON'T", "Don'T", "can't", "CAN'T", "I'm", "I'M", "it's", "IT'S",
    "we'll", "WE'LL", "you're", "YOU'RE", "should've", "SHOULD'VE",
    "1234567890", "12345", "0", "42", "3.14159", "price $1299.99", "year 2026",
    "hello world", "Hello, World!", "multiple   spaces", "tab\ttab",
    "newline\ntest", 'quote"quote', "emoji test", "cafe naive uber",
    "  leading spaces", "trailing spaces  ", "MiXeD CaSe TeXt",
    "a.b.c.d", "192.168.1.1", "user@example.com", "https://example.com/path?q=1",
    '{"state": {"turns": 3, "last": "hi"}}',
    '{"a": 1, "b": [1, 2, 3], "c": {"d": true, "e": null}}',
    "The quick brown fox jumps over the lazy dog.",
    "Testing punctuation: ,.;:!?()[]{}",
    "under_score and-dash and.dot",
    "Numbers 1 22 333 4444 55555 666666",
    "It's a test. Don't fail! CAN'T you see?",
    "single'quote", "multi''quotes",
    "Naive cafe with accents",
    "japanese text sample",
    "no options fit here" * 3,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="brain pull convaiinnovations/laya checkpoint dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw_tok = Tokenizer.from_file(str(d / "tokenizer" / "tokenizer.json"))
    hf_tok = AutoTokenizer.from_pretrained(str(d / "tokenizer"))

    manifest = {"tokenizer_cases": [], "special_tokens": {}, "build_sequence_cases": []}

    added = json.loads((d / "tokenizer" / "tokenizer.json").read_text())["added_tokens"]
    for name, content in [("unk", "[UNK]"), ("cls", "[CLS]"), ("sep", "[SEP]"), ("pad", "[PAD]"), ("mask", "[MASK]")]:
        hit = next(t for t in added if t["content"] == content)
        manifest["special_tokens"][name] = hit["id"]

    for text in BATTERY:
        ids = raw_tok.encode(text, add_special_tokens=False).ids
        manifest["tokenizer_cases"].append({"text": text, "ids": ids})

    def to_ordered(v):
        """Tag every dict as {"__obj__": [[k, to_ordered(v)], ...]} so the
        Rust side (whose serde_json does NOT enable `preserve_order` -
        verified this session, see crates/modernbert/src/sequence.rs's own
        module doc) can reconstruct INSERTION order from a JSON ARRAY
        (always order-preserving) instead of a JSON OBJECT (not, in this
        workspace). The single-key wrapper object itself has no ordering
        ambiguity (exactly one key)."""
        if isinstance(v, dict):
            return {"__obj__": [[k, to_ordered(vv)] for k, vv in v.items()]}
        if isinstance(v, list):
            return [to_ordered(x) for x in v]
        return v

    def case(name, state, q, max_len=512, head_max_len=192, option_order=None):
        ids, markers = build_sequence(hf_tok, state, q, max_len, head_max_len, option_order=option_order)
        manifest["build_sequence_cases"].append({
            "name": name, "state": to_ordered(state),
            "question": {"t": q["t"], "ins": q["ins"], "crit": to_ordered(q.get("crit"))},
            "max_len": max_len, "head_max_len": head_max_len,
            "option_order": option_order,
            "ids": ids, "markers": markers,
        })

    # 1. multi-key JSON state, choice question, several options with descriptions.
    case(
        "choice_multikey_state",
        {"user": "alice", "turns": 3, "last_message": "I need a refund", "tags": ["billing", "urgent"]},
        {"t": "choice", "ins": "What is the customer's primary intent?",
         "crit": {"refund": "wants money back", "complaint": "", "question": "asking for info"}},
    )

    # 2. score question.
    case(
        "score_question",
        {"summary": "Agent resolved the billing dispute quickly."},
        {"t": "score", "ins": "Rate the agent's helpfulness from 0 to 3.",
         "crit": ["not helpful", "somewhat helpful", "helpful", "very helpful"]},
    )

    # 3. noul question, default criteria text (no crit).
    case(
        "noul_default_criteria",
        "Customer said: this is unacceptable, I want a manager.",
        {"t": "noul", "ins": "Is the customer escalating?"},
    )

    # 3b. noul question, explicit criteria text.
    case(
        "noul_explicit_criteria",
        "Customer said: thanks, that solved it!",
        {"t": "noul", "ins": "Is the customer satisfied?",
         "crit": {"true": "customer expressed satisfaction", "false": "customer did not"}},
    )

    # 4. shrink path: many, long options so opt_budget < 16.
    many_opts = {("option_%d" % i): ("a fairly long description of option number %d for shrink testing" % i) for i in range(12)}
    case(
        "shrink_path_many_long_options",
        {"context": "a state blob with enough content to matter " * 4},
        {"t": "choice", "ins": "Pick the single best matching category from the following long list of options",
         "crit": many_opts},
    )

    # 5. plain string state (serialize_state's str branch).
    case(
        "string_state",
        "just a plain string state, not JSON",
        {"t": "choice", "ins": "Does this look like spam?", "crit": {"yes": "", "no": ""}},
    )

    # 6. option_order: the TRAINING-only path (`encode_record` shuffles a
    # non-score question's options every epoch so the model cannot answer
    # from a position). Untested until brain gained a training loop, and the
    # one place the packing and the target can silently disagree: the packed
    # option texts move but the marker list must move with them.
    case(
        "option_order_permuted_choice",
        {"user": "bob", "last_message": "my card was swallowed by the atm"},
        {"t": "choice", "ins": "What is the customer's primary intent?",
         "crit": {"refund": "wants money back", "card issue": "a problem with a card",
                  "complaint": "", "question": "asking for info"}},
        option_order=[2, 0, 3, 1],
    )

    # 6b. the same permutation on the SHRINK path, where every option is
    # truncated to `per` tokens - a shrink that ran before the permutation
    # would truncate the wrong options.
    case(
        "option_order_permuted_shrink",
        {"context": "a state blob with enough content to matter " * 4},
        {"t": "choice", "ins": "Pick the single best matching category from the following long list of options",
         "crit": many_opts},
        option_order=[11, 3, 7, 0, 5, 9, 1, 10, 2, 8, 4, 6],
    )

    # 6c. the IDENTITY permutation must be byte-identical to passing none -
    # otherwise `option_order` is doing something beyond reordering.
    case(
        "option_order_identity_choice",
        {"user": "bob", "last_message": "my card was swallowed by the atm"},
        {"t": "choice", "ins": "What is the customer's primary intent?",
         "crit": {"refund": "wants money back", "card issue": "a problem with a card",
                  "complaint": "", "question": "asking for info"}},
        option_order=[0, 1, 2, 3],
    )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"wrote {len(list(out.iterdir()))} files, {total / 1024:.0f} KiB -> {out}")


if __name__ == "__main__":
    main()
