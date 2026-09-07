#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Talk to GLM-5.2's MLA + sigmoid noaux_tc MoE decoder over D-Bus.

**Scope, honestly**: `brain/glm`'s `generate` action
(`crates/glmdsa/src/caps.rs` + `crates/cli/src/resident_llm.rs::GlmResident`)
is a raw completion, not a chat surface - GLM is **char-level** (the
checkpoint carries its own vocabulary), so there is no `messages`, no chat
template, no per-token streaming deltas: just `prompt`/`max_new`/`temp`/
`top_k`/`seed` in, `text` out. Decoding is
`glmdsa::sample::generate_kv` (the KV-cached fast path) end to end - the
served path and `brain glmdsa infer` sample identically.

**This example offers `--dbus` only.** GLM's `generate` action is not
`.streaming()` - `crates/apiserve/src/catalog.rs::api_caps` gates
`/v1/chat/completions` and `/v1/messages` on that flag, and GLM does not set
it yet (a streaming `generate` is future work). So `--openai`/
`--anthropic` are refused outright, with that explanation, rather than
silently hanging or degrading against an HTTP endpoint that will never see
this model. Reach GLM over D-Bus (this script) or `brain do glm generate ...`.

Examples:
  # D-Bus (needs BRAIN_GLMDSA_WEIGHTS=... brain serve --dbus running):
  python3 examples/llm/glmdsa.py --dbus --prompt "Once upon a time" --max_new 64

  # Quick, deps-free wire-contract check against the mock resident (no GLM
  # checkpoint needed -- exercises the exact same generate action shape):
  BRAIN_MOCK=1 dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/llm/glmdsa.py --dbus --model brain/mock --prompt hi
  '
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainBase, skip  # noqa: E402

_HTTP_REFUSAL = (
    "glmdsa.py: --{flag} is not supported -- brain/glm's `generate` action is not "
    "`.streaming()` (crates/apiserve/src/catalog.rs::api_caps gates /v1/chat/completions "
    "and /v1/messages on that flag), so it is not reachable "
    "over HTTP. Use --dbus (or `brain do`) instead."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dbus", action="store_true", help="use the D-Bus transport (the only transport this example serves)")
    ap.add_argument("--openai", metavar="URL", help="refused -- see the module docstring")
    ap.add_argument("--anthropic", metavar="URL", help="refused -- see the module docstring")
    ap.add_argument("--prompt", required=True, help="the prompt to continue")
    ap.add_argument("--max_new", type=int, default=128, help="number of new tokens to generate (default 128)")
    ap.add_argument("--temp", type=float, default=0.8, help="sampling temperature; <= 0 = greedy (default 0.8)")
    ap.add_argument("--top_k", type=int, default=40, help="top-k filter; 0 or negative disables it (default 40)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    ap.add_argument("--model", default="brain/glm", help="served model name (default brain/glm; brain/mock for a deps-free wire-contract check)")
    args = ap.parse_args()

    if args.openai:
        print(_HTTP_REFUSAL.format(flag="openai"), file=sys.stderr)
        sys.exit(2)
    if args.anthropic:
        print(_HTTP_REFUSAL.format(flag="anthropic"), file=sys.stderr)
        sys.exit(2)
    if not args.dbus:
        skip("exactly --dbus is required (this example serves no other transport for brain/glm)")

    from brain_py.dbus import BrainDBus

    brain: BrainBase = BrainDBus()

    served = brain.models()
    if args.model not in served:
        skip(f"model {args.model!r} is not served (served: {served}); for real GLM: "
             "BRAIN_GLMDSA_WEIGHTS=<checkpoint.safetensors> brain serve --dbus")

    text = brain.generate(
        prompt=args.prompt,
        model=args.model,
        max_new=args.max_new,
        temp=args.temp,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
