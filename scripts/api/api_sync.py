#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Refresh the vendored upstream OpenAPI specs and report drift.

Single-source-of-truth policy (AGENTS.md): brain has at most TWO sources of truth —
its code, and the *vendored upstream* provider specs under
`crates/apiserve/tests/specs/` (a cached copy of what the providers publish). The
jsonschema conformance tests in `crates/apiserve/tests/api.rs` validate brain's
emitted/accepted bodies against those vendored specs. There is NO separately
hand-maintained "brain spec".

This script pulls each provider's current upstream OpenAPI document, canonicalizes it
to JSON, semantically diffs it against the vendored copy (added/removed/changed
schemas + paths), and — unless `--check` — updates the vendored file. After updating,
run `cargo test -p brain-apiserve` to see which conformance tests now fail; adapt the
handlers, re-green, and (per the security invariant) run the API security audit.

Usage:
    python3 scripts/api/api_sync.py [--check] [--provider openai|anthropic|openrouter]

Exit code is non-zero when drift is found (so CI/`--check` can gate).
"""
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

try:
    import yaml  # for the OpenAI spec, published as YAML
except ImportError:
    yaml = None

REPO = Path(__file__).resolve().parent.parent.parent
VENDOR = REPO / "crates" / "apiserve" / "tests" / "specs"

# provider -> (vendored filename, upstream URL, format)
SOURCES = {
    "openai": ("openai.json",
               "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml",
               "yaml"),
    "anthropic": ("anthropic.json",
                  "https://raw.githubusercontent.com/laszukdawid/anthropic-openapi-spec/main/hosted_spec.json",
                  "json"),
    "openrouter": ("openrouter.json",
                   "https://openrouter.ai/openapi.json",
                   "json"),
}


def fetch(url: str, fmt: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "brain-api-sync"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    if fmt == "yaml":
        if yaml is None:
            sys.exit("api-sync: PyYAML required for the OpenAI spec (pip install pyyaml)")
        return yaml.safe_load(io.BytesIO(raw))
    return json.loads(raw)


def canon(obj: dict) -> str:
    """Stable JSON serialization so diffs are content-only, not whitespace. `default=str`
    coerces YAML-parsed dates (OpenAI's spec) to strings, matching the vendored JSON."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def keyset(doc: dict, section: str) -> set:
    return set((doc.get(section) or {}).keys())


def drift(old: dict, new: dict) -> list:
    """A concise semantic diff over the parts the conformance tests care about."""
    out = []
    for section in ("paths", "components/schemas"):
        if "/" in section:
            a = (old.get("components") or {}).get("schemas") or {}
            b = (new.get("components") or {}).get("schemas") or {}
        else:
            a, b = old.get(section) or {}, new.get(section) or {}
        added = sorted(set(b) - set(a))
        removed = sorted(set(a) - set(b))
        changed = sorted(k for k in set(a) & set(b) if canon(a[k]) != canon(b[k]))
        for k in added:
            out.append(f"  + {section}: {k}")
        for k in removed:
            out.append(f"  - {section}: {k}")
        for k in changed:
            out.append(f"  ~ {section}: {k}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, do not write")
    ap.add_argument("--provider", choices=list(SOURCES), help="only this provider")
    args = ap.parse_args()

    providers = [args.provider] if args.provider else list(SOURCES)
    any_drift = False
    for p in providers:
        fname, url, fmt = SOURCES[p]
        vendored_path = VENDOR / fname
        print(f"== {p}: {url}")
        try:
            upstream = fetch(url, fmt)
        except Exception as e:  # noqa: BLE001 — report and continue other providers
            print(f"  ! fetch failed: {e}")
            any_drift = True
            continue
        vendored = json.loads(vendored_path.read_text()) if vendored_path.exists() else {}
        d = drift(vendored, upstream)
        if not d:
            print("  up to date")
            continue
        any_drift = True
        print("\n".join(d[:200]))
        if len(d) > 200:
            print(f"  … {len(d) - 200} more changes")
        if not args.check:
            vendored_path.write_text(canon(upstream))
            print(f"  updated {vendored_path.relative_to(REPO)}")
        else:
            print("  (--check: not written)")

    if any_drift and not args.check:
        print("\nVendored specs updated. Next: `cargo test -p brain-apiserve` to see which")
        print("conformance tests now fail, adapt the handlers, then run the API security audit")
        print("(.agents/rules/api-security.md) before shipping.")
    return 1 if any_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
