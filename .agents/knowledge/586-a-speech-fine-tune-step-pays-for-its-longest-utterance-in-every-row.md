<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 586. A speech fine-tune step pays for its longest utterance in every row

**Lesson:** in `speech_sft` every Talker row of a micro-batch is computed to the batch's longest record, so a batch of
utterances of random length (1.5-30 s) computes about 2.9 times the Talker rows it needs, and data-parallel ranks wait
for whichever drew the longest record. Batching records of about one length (`length_bucket_steps`) made one GH200
32 % faster and two 34 % faster. Balancing the ranks' frame counts did nothing: the cost follows the longest row, not
the frames.

## What happened
Data parallel over NCCL scaled at about 79 % a rank warm. nsys showed a reduction of 2.9 GB taking 19 ms when the
ranks arrived together, and 40-200 ms when one waited; per-step compute differed between the ranks by up to 300 ms.
Dealing each step's records longest first in snake order (each rank within one record's frames of the others) was
measured paired on two cards and changed nothing (7,461 / 6,995 against 7,177 / 7,420 frames a second). Logging each
step's time against its useful frames and its computed rows (on one card, 28 warm steps) gave correlations of 0.51 and
0.72: the step pays for the rows it computes, and those are `batch x longest`.

## What worked, measured (paired, alternated, 48 warm steps, batch 32, GH200 cards 1 and 2)
- `length_bucket_steps: 16`: the records of 16 micro-batches sorted by length and cut into micro-batches, taken in a
  seeded order, each dealt to the ranks every `world`-th record. One card 6,145 / 6,159 frames a second by the wall
  clock against 4,681 / 4,621; two ranks 10,396 / 10,750 against 7,915 (about 86 % of a lone card a rank).

## Traps met on the way
- A 16-step run on a synthetic shard of random text projects new text rows on the host every step (30-38 ms, card
  idle); measure past one epoch, or the cold text cost hides what the steady state does.
- The memory-peak sampler ran `nvidia-smi` back to back, slowing the run it measured and every card of the node; it
  now samples every 50 ms.
- `speech_sft_bench run --card N` trained on card 0 and only read card N's memory; brain's card indices ignore
  `CUDA_VISIBLE_DEVICES`, so pick cards with `--card` / `--cards`.
