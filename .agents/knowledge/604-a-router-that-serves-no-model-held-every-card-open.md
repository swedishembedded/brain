<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 604. A router that serves no model held every card open: device selection ran before the command was known

**Lesson:** a process-wide step that runs "before any subcommand" runs for the subcommands that need no device too.
Decide whether a command claims devices from its arguments first, and make the claim enforceable: a process that
declared it uses no device must fail loudly if anything probes one, not open it quietly.

## What happened

On 2026-11-26 the router of a 16-GH200 fleet (`brain serve --openai 127.0.0.1:8990 --upstreams-file F`, brain
b726c045b) held 35 `/dev/nvidia*` file descriptors (`/proc/<pid>/fd`), on three of its node's four cards, for its
whole lifetime, beside servers that needed every byte and every context of those cards.

## Why

`main` resolves `--device` for every command before dispatching it (`select_backend`), and with no `--device` the
default set is resolved by probing the machine (`devices::resolve_cli_default_compute_set` -> `Inventory::probe`):
the device registry enumerates every card through native Vulkan, wgpu and the CUDA driver, and the hardware probe
loads libcuda. Only after that does `serve` parse `--upstream` and learn that it routes and loads nothing. The
documentation already said a router "loads no model and uses no device"; nothing made that true.

## What to do

- `select_backend` recognises a routing serve from its arguments (`routes_only`), resolves nothing, installs no
  placer, and calls `gpu_core::devices::claim_no_device`: from then on the device registry and the hardware probe
  panic with the reason instead of opening a GPU API. A given `--device` or `--backend` is reported as not read.
- Proof without a GPU: `crates/cli/tests/it/serve_connection.rs`, `a_router_claims_no_device`, starts a router with
  no GPU driver visible (CUDA devices hidden, no Vulkan or EGL driver file) and `--trace-gpu 5`, and asserts the trace
  shows no wgpu instance and no device registry. Before the fix the trace showed both, so the same test also catches
  a regression on a machine whose drivers are hidden.
- A trace goes to stdout unless `--trace-output FILE` names a file: a test asserting on a device trace must read the
  file, or it reads stderr, finds nothing and passes against the bug (the first version of this test did).
- After: the router holds 0 `/dev/nvidia*` descriptors and 14 descriptors in all.
