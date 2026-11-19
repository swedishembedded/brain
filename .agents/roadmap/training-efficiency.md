# training efficiency - roadmap

Scope: Where a served LoRA training step spends its time on Qwen3-class models (diagnosis of 2026-11-19), the parity-preserving fixes, and the deferred items that change numerics or draw order.

**Rule.** Every change must pass the tier-1 parity harness described in `generic-training-capabilities.md` ("Release checks"):
on fixed real records, loss, gradients and post-step weights stay within the old path's run-to-run band. An item that changes
numerics or the draw order is a new arm, checked deliberately, never shipped as an optimisation.

Measured before the work (one GH200, Qwen3-4B bf16 LoRA rank 16, all 7 projections): packed SFT reached about 2% MFU and 1.3k
useful tokens/s (3.0k computed; rows padded to the 16k block, 56% of the work); a policy-gradient step with behaviour and reference
scoring took about 150 s (27 s of it scoring); a KTO run started 55 minutes late (eager reference scoring of the dataset) and then
ran 170 s per step.

## Diagnosis (certain = read in code; hypothesis = needs a profile)

| # | Finding | Status |
|---|---|---|
| 1 | Weighted-loss objectives (policy gradient, KTO, DPO) never trim rows: every pass computes the full block (`Qwen::needed_len`) | certain |
| 2 | The 2% SFT figure may predate SFT row trimming (0db97e677); check `step_work.padded_tokens` | hypothesis |
| 3 | KTO does about 2x the work: the 2-row batch is forwarded twice, row 1 backwarded at weight 0 | certain |
| 4 | KTO's 55-minute start is eager reference scoring of the whole dataset (`check_data`) | certain |
| 5 | Recompute runs every layer forward twice, about 25-30% of FLOPs; `SavedPolicy::Keep` would fit | certain |
| 6 | Batch hard-wired to 1, sequential accumulation with a blocking readback per micro-batch | certain |
| 7 | Policy-gradient scoring is serial; disabling adapters still runs their GEMMs and round-trips B through the host; `gradient_scan` reads every gradient to the host | certain |
| 8 | The LM head runs on unsupervised rows, about 7% | certain |
| 9 | Qwen3 held-out scoring reloads the base twice per run, at b=1 | certain |
| 10 | Attention is about half the FLOPs at 16k | hypothesis |
| 11 | fp32 activation traffic, separate casts and elementwise passes | hypothesis |

## Order

1. Parity-preserving, CPU-testable (lane E1): A1 trim under weighted loss (about 2.3x for policy gradient, KTO, DPO); A2 KTO
   single-row passes (about 2x); A3 lazy cached KTO reference (removes the 55 min); A4 in-process Qwen3 scoring with a cached base
   score (about 10-15 min per run); A7 policy-gradient base-only pass and a device-side gradient norm.
2. Profile on the next lease (nsys and ncu; profile through brain's generic perf path).
3. Then choose by measured cost among the deferred items below.

## Deferred items (each its own change, validated by the tiers)

| ID | Item | Expected gain | Why deferred |
|---|---|---|---|
| R1 | A5: length-bucketed micro-batches (b>1) sized by memplan; bucket-shuffled order | 1.2-2x on rows of 4k or less, about 1.1x at 16k | changes the draw order, so not parity-preserving |
| R2 | `SavedPolicy::Keep` for qwen3 when memplan says activations fit (no recompute) | about 25% fewer FLOPs | needs the trainer's keep path, a footprint term and an on-card check |
| R3 | LoRA fused into the base GEMM epilogue (one product instead of base plus two TF32 LoRA products) | profile decides | kernel work; rounding changes, tier 2 |
| R4 | bf16 activations end to end (no fp32-to-bf16 cast per product) | profile decides | numerics change, tier 2 |
| R5 | Fused rmsnorm/rope/swiglu/add epilogues | profile decides | kernel work |
| R6 | Attention efficiency at 16k (native kernel efficiency via ncu) | possibly large: about half the FLOPs at 16k | needs the profile first |
| R7 | Multi-GPU data parallelism over the existing `GradSync` (`fit_controlled_sync`) | up to about Nx per run | larger change, needs multi-card tests |
| R8 | Reference and behaviour scoring on a second card, overlapped with training | removes the scoring wall-clock | depends on R7's plumbing |
| R9 | Per-record attention isolation, to pack several records into one row | removes padding at b=1 | masks and positions per segment, a numerics risk |

## Status

- Open: A1-A4 and A7 (lane E1 in progress, parity harness on real records); the GPU profile (nsys/ncu) of the unfixed path; R1-R9.
- Done before this plan (kept for the record): rows trimmed to the last supervised token and held-out scoring on the device
  (0db97e677); LoRA reservation from the trainer's buffers; shared per-shape cast scratch (fwd +5.2 GiB and bwd +12.2 GiB of tape
  copies removed; 33.06 to 17.13 GiB at 4k); the helper that plans Qwen3 `--max-batch` from the card (a fixed 16 gave 1.53k tok/s at
  32 concurrent against 4.45k at 64); policy trace taken from decode (781 to 2579 tok/s at 32 concurrent).
