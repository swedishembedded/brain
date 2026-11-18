<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 512. A directory of adapters was bound to the one base the adapter flags chose

**Lesson:** a source of many adapters has to bind each one to the base its own card names. Choosing one host for
the source, as a single-adapter flag does, puts every adapter on that host and refuses the rest as "trained against
another base".

## What was reported

A model store holding Qwen3-4B and Qwen3.8-27B, served without `--checkpoint`, with `--serve-adapters DIR`: the
4B's adapters were refused, verified against the 27B.

## Cause

`continuous_train::Host::pick` chooses the one resident `--adapter`, `--adapter-manifest` and `--watch-adapters` put
their adapter on (the flag-selected Qwen3, else the one released Qwen3.8, else the one Ripple). `--serve-adapters`
went through the same choice: `BankDirectory` held one `ServedBase` and one `LiveAdapters`. The store's own Qwen3
bases were not even handed to the CLI (`Serving` carried only the flag-selected one).

## Fix

`BankDirectory` holds every served base (`BankHost`: base and its live registry; `Serving::qwen_scanned` carries the
scan's Qwen3 bases). `llmserve::adapter_release::verify_adapter_among` reads the file once and binds it to the base
its card names: the base id (or a quantisation of it), else the base digest it records; then it is verified against
that base exactly as `verify_adapter` does. A base that is not served is `BindError::BaseNotServed`, naming the base
and the served ones. A file whose card moves to another base leaves the old base first.

Test: `adapter_directory` `each_adapter_rides_on_the_base_its_card_names` (two toy bases; it fails when every adapter
is put on the first base).
