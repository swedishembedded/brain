<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 440. Policy-gradient steps on the 27B show isolated KL and gradient-norm spikes at lr 1e-5 and a blow-up at 3e-5

**Lesson:** Served policy-gradient training of Qwen3.8-27B-FP8 (fp8 base, rank 16, alpha 32, 8 records per step,
standardised advantage, KL beta 0.02, clip 0.2, reference = start) has typical per-step KL of 0.001-0.004 and
gradient norm 0.2-0.6, but single steps jump by orders of magnitude, and a higher lr turns the jump into divergence.

## Measured

- lr 3e-5, compiled-material rollouts: step 5 KL 0.0016, grad norm 0.75; step 6 KL 26.6, grad norm 89.5. The run
  was stopped at step 6.
- lr 1.2e-5, same recipe and task family, 40 steps: step 26 KL 0.50, grad norm 21.1 (steps 25 and 27 at 0.002-0.004,
  0.2-0.4); the run finished with last-step KL 0.0025, mean KL 0.0159 (the mean is the spikes).
- lr 2e-5 (positive-only advantage, accounting rollouts), step 27: KL 0.108 against a neighbours' 0.003-0.006.

## What this does and does not establish

Reproduced on two task families and three learning rates: a rare step carries a few tokens with an extreme
log-ratio against the reference, and its size grows with lr. Not isolated: which records or tokens (the
per-record log-ratios were not dumped), whether a per-token ratio or advantage bound (`--advantage-bound`,
`--sequence-advantage-cap`) removes it, and whether the spike does lasting damage (the evaluated adapters
showed none: paired deltas against the base replicate are within noise). A run sized to KL 0.02-0.05 per round
therefore needs a per-step guard (skip or clip a step whose KL or gradient norm exceeds a bound) before lr is raised.
