#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Ask a served decision model typed questions over HTTP, linking nothing of brain.

A decision model returns a calibrated distribution over options the CALLER
supplies in the request. This client reaches one the way any service would:
plain HTTP, stdlib only, against a `brain serve` that is already running. It
needs no weights of its own, and says what to pull when the server has no
decision model.

    brain pull convaiinnovations/laya
    brain serve --openai 8788 --api-keys-out /tmp/keys.json &
    python3 samples/python/decision/served-decide/decide_client.py \\
        --base-url http://127.0.0.1:8788 --api-key "$KEY"

Three calls, each showing one part of the contract:

  * `models`   `POST /v1/run` on `brain/decision`: which decision models the
               server has, ready or not, validated or not. Loads nothing.
  * `systemone` `POST /v1/systemone`: the shared wire shape - questions keyed
               by name, probabilities keyed by option.
  * `decide`   `POST /v1/run` on the chosen model: the capability's own,
               richer answer - the full ranked distribution, the model's
               backend, latency and whether the state was truncated.

Exit codes follow the rest of this tree: 0 on success, 77 when the environment
cannot run it (no server, or no decision model to ask), which prints what to do.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# The exit code every example in this tree uses to mean "this environment can't
# run me" (see samples/python/api/openai-client/openai_client.py).
EXIT_SKIP = 77

STATE = "Help! My payments have been failing for three days and nobody answers support."

QUESTIONS = {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}


def skip(reason: str) -> "None":
    print(f"SKIP: {reason}", file=sys.stderr)
    sys.exit(EXIT_SKIP)


class Client:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.key = api_key

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except ValueError:
                return e.code, {}

    def run(self, model: str, action: str, params: dict) -> tuple[int, dict]:
        return self.post("/v1/run", {"model": model, "action": action, "params": params})


def error_line(status: int, body: dict) -> str:
    err = body.get("error", {})
    return f"HTTP {status} [{err.get('code', '?')}] {err.get('message', '')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8788", help="a running `brain serve --openai PORT`")
    ap.add_argument("--api-key", default="", help="the key `brain serve` printed (`APIKEY openai <key>`)")
    ap.add_argument("--model", default="auto", help='a decision model id, or "auto" (default)')
    args = ap.parse_args()

    client = Client(args.base_url, args.api_key)

    try:
        status, listing = client.run("brain/decision", "models", {})
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        skip(f"no brain server at {args.base_url} ({e}); start one with `brain serve --openai 8788`")
    if status == 401:
        skip("the server refused the API key; pass the one `brain serve` printed with --api-key")
    if status == 404:
        skip("this server does not serve decision models; it needs a brain build with the decision family")
    if status != 200:
        print(error_line(status, listing), file=sys.stderr)
        return 1

    models = listing["outputs"]["models"]
    print("decision models on this server:")
    for m in models:
        verdict = "validated" if m["validated"] else "unvalidated"
        print(f"  {m['id']:<40} {m['state']:<14} {m['backend'] or '-':<7} {verdict}")
    ready = [m for m in models if m["state"] == "ready"]
    if not ready:
        skip("no decision model is installed; run `brain pull convaiinnovations/laya` and restart `brain serve`")

    print(f"\nPOST /v1/systemone  (model: {args.model})")
    status, answer = client.post("/v1/systemone", {"model": args.model, "state": STATE, "questions": QUESTIONS})
    if status == 404 and args.model == "auto":
        # `auto` serves only a model with a recorded validation; an unvalidated
        # one has to be named, which is a decision for the operator.
        chosen = ready[0]["id"]
        print(f"  auto selected nothing ({error_line(status, answer)})\n  asking {chosen} by name instead")
        status, answer = client.post("/v1/systemone", {"model": chosen, "state": STATE, "questions": QUESTIONS})
    if status != 200:
        print(error_line(status, answer), file=sys.stderr)
        return 1
    print(json.dumps(answer["answers"], indent=2))
    print(f"  answered by {answer['model']} on the {answer['brain']['backend']} backend, "
          f"usage.output_tokens={answer['usage']['output_tokens']}")

    print(f"\nPOST /v1/run  {answer['model']} decide")
    status, full = client.run(
        answer["model"],
        "decide",
        {
            "state": STATE,
            "questions": json.dumps(
                [
                    {
                        "id": "department",
                        "type": "choice",
                        "instruction": "Which team should handle this?",
                        "options": [{"id": k, "description": v} for k, v in QUESTIONS["department"]["criteria"].items()],
                    }
                ]
            ),
        },
    )
    if status != 200:
        print(error_line(status, full), file=sys.stderr)
        return 1
    out = full["outputs"]
    first = out["answers"][0]
    print(f"  selected {first['selected']!r}, confidence {first['confidence']:.3f}, "
          f"{out['latency_ms']:.1f} ms, state truncated: {out['flags']['state_truncated']}")
    for r in first["ranked"]:
        print(f"    {r['id']:<12} {r['probability']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
