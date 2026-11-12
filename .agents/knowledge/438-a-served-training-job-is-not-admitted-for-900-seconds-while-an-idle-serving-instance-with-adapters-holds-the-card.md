<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 438. A served training job is not admitted for 900 seconds while an idle serving instance with adapters holds the card

**Lesson:** `brain serve --training` is documented to take a card by evicting idle serving instances. Twice
on a single Qwen3.8-27B-FP8 server (gpu3, `--watch-adapters`, `--adapter-targets all`) a `brain/train` job
was refused with "not admitted within the deadline (queue depth 1, deadline 900000 ms)" although nothing was
running on the card. A server that had not yet answered any request admitted the same job at once.

## What happened

- Run 1: a 2-step job trained and published an adapter (3m36s). The adapter was dropped into the watched
  directory and a bench run sent 7 requests through it. The next 40-step job waited 900.2 s and was refused;
  `nvidia-smi` showed 86 GB resident and 0% utilisation, and the only connection to the port was the client.
  Restarting the server and resubmitting trained all 40 steps (3859 s).
- Run 2 (reproduction, same server, A2 adapter served and 160 requests answered earlier, no request in
  flight): a 2-step job of the same dataset waited 900.09 s and was refused; 86 GB resident, 0% utilisation.

## What this establishes, and what it does not

Reproduced twice: a loaded, idle, adapter-serving instance blocks admission for the full deadline. Not
isolated: whether the adapter bank, the watched directory, the resident KV pool, or the instance being
loaded at all is what refuses eviction (the one success had none of them). A test needs a fake instance
that is loaded, idle and holds an adapter, and asserts that `check_job` evicts it. Workaround: restart the
server before a training job, or train on a card that serves no adapters.
