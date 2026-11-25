<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 598. A shard of the teacher's tokenizer was read as the trimmed student's ids

**Lesson:** Token ids are only meaningful with the tokenizer that wrote them. A shard records that tokenizer's digest
(`producers["tokenizer"]`); every reader of its ids must check it against the model's, and a model whose vocabulary was
trimmed and renumbered has a different tokenizer even though its tokens are a subset.

## What happened
Weight surgery trims the text vocabulary of Sten-TTS-1.7B to 37,570 of 151,676 ids and renumbers them densely.
`speech_sft` read the v2 shards' text ids (written with the full tokenizer) as the student's ids: for the distillation
of the cut it refused `<|im_start|>` (151644, past the student's table); ids below 37,570 would silently have named
other tokens, and a plain fine-tune of a trimmed checkpoint on such shards would have trained on scrambled text.
Separately, the trim kept the ids the calibration shard stores but not the held-out shard's, and one validation record
stores an id its text's reference encoding does not produce (knowledge 594), so the recovery could not read the
held-out shard.

## What to do
`speech_sft` compares the shards' tokenizer digest with the base's: equal (or none recorded) reads the ids unchanged;
the teacher's reads them through the teacher's tokens (`TextVocabMap::to_student`); any other is refused, naming both.
The trim keeps the ids both shards store. Re-encoding shards per student is never needed for a student cut from the
teacher that wrote them.
