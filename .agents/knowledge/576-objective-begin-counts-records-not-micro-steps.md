<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 576. `Objective::begin` is told how many records were drawn, not how many micro-steps

**Lesson:** the shared fit loop positions a resumed objective with
`obj.begin(seed, steps_done * grad_accum * batch_size)` (`model::train`): a count of
RECORDS. The trait's own documentation said "`draws_done` micro-steps into its data",
and an objective that believed it (the Qwen3-TTS speech objective, which draws its
own batches of utterances) resumed `batch_size` times further into its stream than
the run it continued. Nothing failed: the resumed run trained on other records,
and only an exact-resume test (N steps straight against k steps, a restart and
N - k) showed it, as different weights after the restart and a different frame
count at the first step that drew past the cut.

The causal-LM objective counts records too (`drawn_records = draws_done * world`),
so the loop's count is the contract; the trait documentation now says so. An
objective whose micro-batch is not the loop's `batch_size` either divides by it or
the run sets `batch_size` to the objective's batch (what `qwen3tts::speech_sft`
does).

## Evidence (2026-11-25)

`qwen3tts::speech_sft::tests::a_resumed_run_equals_the_straight_run`, tiny random
checkpoint, 4 steps of 2 utterances, resumed after 2: with the micro-step reading,
step 4 trained on 9 frames resumed against 8 straight and the decoder weights
differed; with the record reading every trained tensor is bit-identical.
