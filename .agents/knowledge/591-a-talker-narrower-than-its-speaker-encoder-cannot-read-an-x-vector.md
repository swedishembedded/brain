<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 591. A Talker narrower than its speaker encoder cannot read an x-vector

**Lesson:** in Qwen3-TTS the speaker x-vector is summed into the Talker's input rows as one more row of the Talker's
width (`enc_dim == talker hidden_size`: 2048 for the 1.7B, 1024 for the 0.6B). A student whose residual width was
reduced can therefore not read the x-vectors a shard carries (they are the teacher's speaker encoder's width), and a
teacher of another width cannot read the student's. A distillation across widths trains named voices only (their
codec rows stand in the speaker slot); `speech_sft` refuses a record in the x-vector slot by name in that case, and
accepts a shard of named voices whose x-vectors are of another width, since nothing reads them.

## Consequences for a compressed release
- A narrowed `custom_voice` checkpoint is fine: upstream builds no speaker encoder for it and speaks only its named
  voices.
- A narrowed `base` checkpoint (voice cloning) needs its speaker encoder's output rotated into the student's basis
  with the rest of the residual stream (the same rotation the residual-width edit applies to the codec tables), and
  shards re-encoded with it; until then cloning cannot be distilled across widths.
- A student that keeps the teacher's width (layers, FFN channels or heads removed) reads the same x-vectors and
  distils x-vector records too.

Where: `crates/qwen3tts/src/speech_sft/mod.rs` (`check_teacher`), `objective.rs` (`check_data`).
