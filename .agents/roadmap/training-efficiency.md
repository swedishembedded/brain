# training efficiency - roadmap

Scope: Where a served LoRA training step spends its time on Qwen3-class models (diagnosis of 2026-11-19), the parity-preserving fixes, and the deferred items that change numerics or draw order.

**Rule.** Every change must pass the tier-1 parity harness described in `generic-training-capabilities.md` ("Release checks"),
implemented as `brain_testutil::step_parity` (eight real SFT records, pinned by digest; the old path run twice gives the band,
floored at the binary32 agreement bound of the step's longest reduction):
on fixed real records, loss, gradients and post-step weights stay within the old path's run-to-run band. An item that changes
numerics or the draw order is a new arm, checked deliberately, never shipped as an optimisation.

Measured before the work (one GH200, Qwen3-4B bf16 LoRA rank 16, all 7 projections): packed SFT reached about 2% MFU (a whole-run
estimate the step profile refuted, knowledge 533) and 1.3k useful tokens/s (3.0k computed; rows padded to the 16k block, 56% of
the work); a policy-gradient step with behaviour and reference
scoring took about 150 s (27 s of it scoring); a KTO run started 55 minutes late (eager reference scoring of the dataset) and then
ran 170 s per step.

## Diagnosis (certain = read in code; hypothesis = needs a profile)

| # | Finding | Status |
|---|---|---|
| 1 | Weighted-loss objectives (policy gradient, KTO, DPO) never trim rows: every pass computes the full block (`Qwen::needed_len`) | fixed (A1) |
| 2 | The 2% SFT figure may predate SFT row trimming (0db97e677); check `step_work.padded_tokens` | refuted by the lane PF profile: a warm SFT step is 14% of bf16 peak, padding 2% |
| 3 | KTO does about 2x the work: the 2-row batch is forwarded twice, row 1 backwarded at weight 0 | fixed (A2) |
| 4 | KTO's 55-minute start is eager reference scoring of the whole dataset (`check_data`) | fixed (A3) |
| 5 | Recompute runs every layer forward twice, about 25-30% of FLOPs; `SavedPolicy::Keep` would fit | certain |
| 6 | Batch hard-wired to 1, sequential accumulation with a blocking readback per micro-batch | certain |
| 7 | Policy-gradient scoring is serial; disabling adapters still runs their GEMMs and round-trips B through the host; `gradient_scan` reads every gradient to the host | certain |
| 8 | The LM head runs on unsupervised rows, about 7% | certain |
| 9 | Qwen3 held-out scoring reloads the base twice per run, at b=1 | certain |
| 10 | Attention is about half the FLOPs at 16k | measured (lane PF): attention is 74% of a warm SFT step at about 8% of peak |
| 11 | fp32 activation traffic, separate casts and elementwise passes | hypothesis |

## Order

1. Parity-preserving, CPU-testable (lane E1): A1 trim under weighted loss (about 2.3x for policy gradient, KTO, DPO); A2 KTO
   single-row passes (about 2x); A3 lazy cached KTO reference (removes the 55 min); A4 in-process Qwen3 scoring with a cached base
   score (about 10-15 min per run); A7 policy-gradient base-only pass and a device-side gradient norm.
2. Profiled (lane PF, 2026-11-19, nsys and ncu): attention is 74% of a warm SFT step at about 8% of peak, base GEMMs at the
   roof, host gaps under 2%; so R6 comes first among the deferred items below.
3. Then choose by measured cost among the rest.

## Deferred items (each its own change, validated by the tiers)

| ID | Item | Expected gain | Why deferred |
|---|---|---|---|
| R6 | **First.** Hopper-native causal flash attention, forward and backward (wgmma/TMA; the current kernels use Ampere-style 64x32 tiles) | attention is 74% of an SFT step at about 8% of peak; estimated step 4.0 s to about 1.5 s at 13.5k (about 2.7x) | large kernel work; a new candidate gated against WGSL, tier 2 numerics |
| R10 | LoRA GEMMs in fp32/TF32 plus the fp32-to-bf16 casts around them | about 7% of step time | after R6; rounding changes, tier 2 |
| R1 | A5: length-bucketed micro-batches (b>1) sized by memplan; bucket-shuffled order | 1.2-2x on rows of 4k or less, about 1.1x at 16k | changes the draw order, so a run is no longer the same run: a new arm, not an optimisation |
| R2 | `SavedPolicy::Keep` for qwen3 when memplan says activations fit (no recompute) | about 25% fewer FLOPs | needs the trainer's keep path, a footprint term and an on-card check |
| R3 | LoRA fused into the base GEMM epilogue (one product instead of base plus two TF32 LoRA products) | profile decides (see R10) | kernel work; rounding changes, tier 2 |
| R4 | bf16 activations end to end (no fp32-to-bf16 cast per product) | profile decides | numerics change, tier 2 |
| R5 | Fused rmsnorm/rope/swiglu/add epilogues | profile decides | kernel work |
| R7 | Multi-GPU data parallelism over the existing `GradSync` (`fit_controlled_sync`) | up to about Nx per run | larger change, needs multi-card tests |
| R8 | Reference and behaviour scoring on a second card, overlapped with training | removes the scoring wall-clock | depends on R7's plumbing |
| R9 | Per-record attention isolation, to pack several records into one row | removes padding at b=1 | masks and positions per segment, a numerics risk |

## Parity-preserving items (lane E1)

Each landed with its own step-parity test on the real records (CPU, `Fp32Reference` GEMMs, a small Qwen3-shaped model on the
records' compact vocabulary; an ignored on-card variant runs the device's own GEMM selection). Numbers: run-to-run band, then the
new path's largest difference from the old (loss relative, log-probabilities absolute, tensors relative L2).

| ID | Item | Parity (band / new) | Gain | Status |
|---|---|---|---|---|
| A1 | Weighted-loss rows trimmed to the last supervised position (`Qwen::needed_len`, `write_weights` per-row prefix); test `qwen3` `step_parity` | run-to-run 0, band 2.4e-4; new: loss 0, log-probs 0, gradients 3.1e-7, weights 2.5e-8 | computed tokens 8192 to 5376 a step on the fixture (rows of about 600 in a 1024 block); at a 16k block and records of 4-6k, the 2-3x the plan estimated for policy gradient, KTO and DPO passes | done |
| A2 | KTO at b=1: k mismatched forwards for z, then own row forward and backward, denominator both rows' count; test `rl` `kto_step_parity` | run-to-run 0, band 2.4e-4; new 0 in every field | per drawn example two forwards and a backward of one row instead of two rows: about 2x | done |
| A3 | KTO reference scored on first draw, under the start-of-run adapter once weights moved (`AdapterState::with_applied`); kept in memory beside the example (the policy-gradient disk cache is for its larger per-token scores); a model without adapters is still scored whole | eager vs lazy over two steps: run-to-run 0, band 1.2e-4; new 0 (scoring under the live adapter fails with log-probs off by 0.083) | the run starts at once (the 55 min before step 1 is gone); examples never drawn are never scored | done; the outcome's mean log-ratios are over the examples the run drew |
| A4 | Qwen3 held-out scoring in process with a cached base score | not attempted | about 10-15 min per run | open: the in-process `model::score_base`/`score_tuned` path reports no token accuracy (the reported `token_accuracy` would be lost) and scores the trained model rather than the exported adapter that is served; needs a generic per-position argmax readout on the trainer's model, and a base-score cache needs a location contract (a host-resolved store path), so it is its own change |
| A7a | Policy-gradient base pass through `AdapterOverride::BaseOnly` | not attempted | removes the zero-B adapter GEMMs of base-scoring passes and the host round trip of every B (about 66 MB read, 130 MB written per scoring at rank 16, 4B) | open: the override needs `&mut` and re-records the forward tape; the objective holds the model by `&`, so it needs an interior-mutable override on the `Model` surface |
| A7b | Device-side gradient norm instead of `gradient_scan` | not attempted | about 66 MB of gradients read to the host per step (tens of ms against a step of minutes, well under 1%) | deferred for its size; the optimiser's clip already runs on the device, and the guard's f64 host norm would change to an f32 device reduction (tier 2) |
| A8 | Deferred loss readback in sequential accumulation | not attempted | lane PF measured host gaps under 2% of a step | deferred: below the measured gap; the weighted objectives read the log-probabilities each micro-batch anyway |
| A6 | Head product skipped for rows with no supervised token (`HeadCe`) | not attempted | about 7% of an SFT step (prompt rows are most of a record) | open: the recorded tape fixes every dispatch's row count, so skipping rows per batch needs a gather/scatter of supervised rows with device-side counts or per-batch recording; kernel work |
| D1 | The policy-gradient objective fills `step_work` (forward, backward, tokens) as SFT does | test `rl` `a_policy_gradient_step_reports_its_work` | the step's time is no longer all put down to the optimizer | done |
| D2 | `--activations` help lists exactly `SavedPolicy::ACTIVATIONS_CHOICES` | test `cli` `the_activations_help_names_what_the_parser_accepts` | | done |

## Status

- Done (lane E1, 2026-11-19): the tier-1 harness, A1, A2, A3, D1, D2 (table above).
- Open: A4, A6, A7a (table above); R6 first, then R10 and R1-R9. A7b and A8 deferred for their measured size.
- Known flaky, unrelated: `rl` `mixture_anchor_regression` draws random seeds and failed once in six runs on the CPU backend
  (primary-only training did not raise the anchor loss by 0.05); it trains full, unweighted rows, which trimming never touches.
- Done before this plan (kept for the record): rows trimmed to the last supervised token and held-out scoring on the device
  (0db97e677); LoRA reservation from the trainer's buffers; shared per-shape cast scratch (fwd +5.2 GiB and bwd +12.2 GiB of tape
  copies removed; 33.06 to 17.13 GiB at 4k); the helper that plans Qwen3 `--max-batch` from the card (a fixed 16 gave 1.53k tok/s at
  32 concurrent against 4.45k at 64); policy trace taken from decode (781 to 2579 tok/s at 32 concurrent).
