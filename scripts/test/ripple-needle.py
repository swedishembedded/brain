#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that a long-context
# language model still finds what it was told after quantization and sparse
# attention, at the context lengths it is sold for, for its clients. If your
# team needs expertise in long-context retrieval evaluation then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Needle-in-a-haystack probe of a served model over the OpenAI chat API.

A random six-digit passkey is hidden at a chosen depth of a filler text of a
chosen length and the model is asked for it; the score is the share of trials
that answer the exact digits. Run it against the same `brain serve` once per
configuration under test (dense, `--sparse-attention`, each `--kv-cache`) and
compare the tables: a configuration that loses the needle at a depth the dense
one finds has lost retrieval there, which the retained-attention share of
`brain lsa eval` can only suggest.

The filler is shuffled words, seeded and never repeated, so neither a prefix
cache nor prompt-lookup speculation can answer from the text. A length is a
target in words (`--words-per-token`, default 0.6); the table reports the
prompt tokens the server counted.

  python3 -I scripts/test/ripple-needle.py --url http://127.0.0.1:8798 \\
      --model ricursiveAI/Ripple-65B-Preview-NVFP4 --keys keys.json \\
      --lengths 8192,32768,131072 --depths 0.05,0.5,0.95 --trials 3

`--keys` is the file `brain serve --api-keys-out` writes (its `openai` key).
Exit status 1 when any trial failed to get an answer from the server (a
timeout or an error is not a missed needle); 0 otherwise, whatever the score.
"""

import argparse
import json
import random
import secrets
import sys
import urllib.error
import urllib.request

WORDS = (
    "amber anchor basalt beacon birch cobalt copper coral dune ember fjord flint glacier granite harbor heron iris "
    "juniper kelp lagoon lantern larch maple marble meadow mist moss nectar oak onyx orchid otter pebble pine plume "
    "quartz reef ridge river sable sage silt slate spruce summit tarn thistle tide timber tundra umber valley willow"
).split()


def filler(rng, words):
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(words))


def prompt(rng, words, depth, passkey):
    before = int(words * depth)
    return (
        f"{filler(rng, before)}\n\nThe passkey is {passkey}. Remember it.\n\n{filler(rng, words - before)}\n\n"
        "What is the passkey? Answer with the six digits only."
    )


def ask(url, key, model, text, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 24, "temperature": 0}).encode()
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keys", required=True, help="the --api-keys-out file of the server")
    ap.add_argument("--lengths", default="8192,32768", help="target prompt lengths in tokens")
    ap.add_argument("--depths", default="0.05,0.5,0.95", help="where the needle sits, 0 (start) to 1 (end)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--words-per-token", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=None, help="default: random, printed")
    args = ap.parse_args()
    seed = secrets.randbits(31) if args.seed is None else args.seed
    print(f"seed {seed}", file=sys.stderr)
    rng = random.Random(seed)
    with open(args.keys) as f:
        key = json.load(f)["openai"]
    depths = [float(v) for v in args.depths.split(",")]
    failures = 0
    print(f"{'length':>9} {'depth':>6} {'found':>7} {'prompt tokens':>14}")
    for length in (int(v) for v in args.lengths.split(",")):
        for depth in depths:
            found, tokens = 0, 0
            for _ in range(args.trials):
                passkey = f"{rng.randrange(10**6):06d}"
                try:
                    reply = ask(args.url, key, args.model, prompt(rng, int(length * args.words_per_token), depth, passkey), args.timeout)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    why = e.read().decode(errors="replace")[:300] if isinstance(e, urllib.error.HTTPError) else e
                    print(f"{length:>9} {depth:>6.2f} request failed: {e}: {why}", file=sys.stderr)
                    failures += 1
                    continue
                answer = reply["choices"][0]["message"].get("content") or ""
                found += passkey in answer
                tokens = reply["usage"]["prompt_tokens"]
            print(f"{length:>9} {depth:>6.2f} {found:>3}/{args.trials:<3} {tokens:>14}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
