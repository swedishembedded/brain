#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Print `-p <package>` for every ENGINE package (the workspace default members
# plus `crates/vulkan`), for workspace-wide gates to splice into a cargo
# invocation instead of using `--workspace`.
#
# Why this is not `--workspace`: that selects samples/*/* too. A sample NAMES
# the SDK surfaces it uses (samples/README.md), and cargo's v2 resolver unifies
# features across everything an invocation selects - so `--workspace` unions
# every sample's feature set onto `brain`, producing a configuration that
# matches neither the engine build nor any individual sample, and paying a cold
# build for it. Samples are linted separately, one `cargo clippy -p <sample>`
# each, inside scripts/gates/check-samples.sh. That mirrors the default-members
# split exactly: engine gates cover the engine, the samples gate covers samples.
#
# `crates/vulkan` is added back explicitly because it is deliberately excluded
# from `default-members` (the optional cooperative-matrix path) and dropping
# `--workspace` would otherwise silently stop linting it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python3 - <<'PY'
import tomllib, pathlib

root = tomllib.load(open("Cargo.toml", "rb"))
members = root["workspace"]["default-members"]
out = []
for m in members:
    manifest = pathlib.Path(m) / "Cargo.toml"
    if not manifest.is_file():
        raise SystemExit(f"default-member {m} has no Cargo.toml")
    name = tomllib.load(open(manifest, "rb"))["package"]["name"]
    out.append(f"-p {name}")
# Excluded from default-members on purpose; still engine code, still linted.
out.append("-p brain-vulkan")
print(" ".join(out))
PY
