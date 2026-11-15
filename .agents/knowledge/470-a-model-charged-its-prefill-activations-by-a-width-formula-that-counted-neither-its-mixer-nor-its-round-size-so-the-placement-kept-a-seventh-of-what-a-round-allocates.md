<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 470. A model charged its prefill activations by a width formula that counted neither its mixer nor its round size, so the placement kept a seventh of what a round allocates

**Lesson:** An activation charge is a claim about memory, so it needs a measurement that holds it: compare the
device's peak live bytes over the widest round against the charge in a test on the real checkpoint. A generic
width formula undercounts every block it does not model, and a round size that a constant decided can change
without the charge moving.

## What happened

Raising the 35B-A3B (`qwen35moe`) prefill round from 256 to the dense engine's 4096 rows on tensor cores lifted
the 16K prefill from 4.25K to 6.26K tok/s. Before doing so, the round's real footprint was measured against what
`catalog::resident_qwen35moe` charges as `MemCost::activation` (gpu3, GH200, int8 weights, bf16 KV, peak live
bytes of the engine's handles above the resident weights and caches):

| round | peak above resident | charged before | charged now |
|---|---|---|---|
| 256 rows | 0.31 GiB | 0.03 GiB | 0.36 GiB |
| 4096 rows | 3.42 GiB | 0.46 GiB | 4.39 GiB |

## Why

- The charge was `DecoderWidths::serve_peak_bytes`: the residual, the attention projections and the
  feed-forward widths of one forward. The Gated-DeltaNet mixer is not in it (its projections are 3x the
  attention's here, and its f32 chunk form allocates scratch of its own), nor is the private recurrent slot a
  prompt runs over, nor the layers the host queues ahead of the device.
- Every buffer a round takes is per call from its row count, so nothing failed: the card had the room. The
  charge mattered twice anyway: a placement on a fuller card would have died on an allocation after loading, and
  the engine's handles keep freed blocks only up to that charge (knowledge 431).

## What to do

- `qwen35moe::serve::activation_bytes` is the one charge (catalog, engine retention, profile binary):
  `prefill_round_bytes` counts the slot plus a layer's words (residual pair, the larger mixer, the expert slots)
  times a live-layer ratio calibrated on the measurement, the shape `qwen35::int8_gguf_resident::
  prefill_workspace_bytes` takes for the dense model.
- A round left to the device is charged at the widest the policy picks (`model::serve::prefill_round_rows_for`,
  now shared by both engines), since the placement asks before the device is known.
- Gate: `qwen35moe` `prefill_round_census::the_activations_of_a_prefill_round_of_the_real_checkpoint_fit_their_charge`.
  Re-run it whenever the round's kernels change: the ratio belongs to the current GDN form.
