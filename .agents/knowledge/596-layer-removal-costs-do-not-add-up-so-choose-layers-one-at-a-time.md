<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 596. Layer removal costs do not add up: choose removed layers one at a time, and prefer narrower over shallower

**Lesson:** Ranking layers by the KL of removing each alone and removing the cheapest k is a poor joint choice once
several layers go, especially after another edit has changed the model. Re-measure after every choice (greedy
removal on the model as edited so far), and when a parameter budget can be met by narrowing MLPs or by removing
layers, measure both: on Sten-TTS-1.7B narrowing won.

## What was measured
Sten-TTS-1.7B (Qwen3-TTS Talker 28 layers, MLP 6144), n460 GH200, held-out KL to the unedited model (Talker +
0.3 code predictor) on 128 validation records, calibration on 128 training records, before any recovery. Every
plan first trims the text vocabulary to the training texts plus the full corpus inventory (37,569 of 151,676 ids),
then keeps `keep` Fisher-ranked channels in every Talker MLP, then removes `count` Talker layers. All land at
947-957 M parameters in model.safetensors (the rung's ceiling is 958 M).

| MLP keep | layers removed | one-shot by single KL | greedy |
|---:|---:|---:|---:|
| 3008 | 6 | 1.085 | 0.906 |
| 2752 | 5 | 1.023 | 0.889 |
| 2560 | 4 | 0.929 | 0.862 |
| 2368 | 3 | 0.868 | 0.868 |
| 2176 | 2 | 0.892 | - |
| 1920 | 0 | 0.963 | - |

The single-layer ranking on the base (layers 8, 12, 6, 9, 13, 7 cheapest) and the greedy choice on the narrowed
model (8, 12, 6, 25, 20, 7) part after three layers; with three removals they agree and the KLs are identical.

## What to do
`remove_layers` with `greedy: true` (by `kl` or `loss`) chooses on the calibration records, building the model once
and masking each candidate set (`modelopt::surgery::greedy_removal`, `with_layers_masked`); the held-out records stay
a measurement. Rung 1 of the Sten-TTS ladder uses MLP 2560 and 4 greedy layers (8, 12, 6, 25): 952.6 M
parameters, held-out loss 3.435 and KL 0.864 before recovery (the unedited loss 2.609 on that run's 128 records; it
keeps the held-out shard's ids too, knowledge 598, which changes which records are measured).
