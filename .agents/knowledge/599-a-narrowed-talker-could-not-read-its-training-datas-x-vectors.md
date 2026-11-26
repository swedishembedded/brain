<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 599. A narrowed Talker could not read its training data's x-vectors

**Lesson:** An edit that rotates a model's residual stream changes every input that enters that stream, including
inputs that live outside the checkpoint (here the speaker x-vectors stored in the training shards). The edit must leave
the map for those inputs beside the checkpoint, or the cut cannot be trained on the data it came from.

## What happened
Rung 2 of the Sten-TTS ladder rotates the Talker stream 2048 -> 1536 (`rotate_stream`, gain tolerance 0.25). Weight
surgery measured it correctly, because its evaluator projected the x-vectors through the rotation in memory. The
exported cut kept no record of that rotation, and `speech_sft` refused the v2 shards: "dataset's x-vectors are 2048 wide
and the base's speaker encoder writes 1536". About 99 % of the 194,769 records use the x-vector slot (only the 12 named
voices use codec rows). Knowledge 591 had recorded the same limit for distillation across widths.

## What to do
The surgery writes `speaker_projection.safetensors` (`projection`, `[to, from]`) into a cut with a rotated Talker
stream. For a cut of a cut, the new map is composed with the base's own, so it always maps from the width the data was
made at. Export carries the file. `speech_sft`, the served job's checks and the surgery's calibration read a record's
x-vector through it, and a teacher reads the stored x-vector (`Example::source_xvector`). Upstream code ignores the
file, because a release with named voices serves those voices from codec rows.
