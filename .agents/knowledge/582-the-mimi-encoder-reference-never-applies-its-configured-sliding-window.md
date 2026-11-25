<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 582. The Qwen3-TTS codec encoder's reference never applies its configured sliding window

**Lesson:** the released `speech_tokenizer/config.json` carries `encoder_config.sliding_window: 250`, and
`mimi::Codec`'s encoder transformer applied it (`gqa_fwd_win`, window 250). The reference encode does not: `qwen-tts`
0.1.1 runs transformers 4.57.3's `MimiModel` encoder, whose transformer builds its mask with `create_causal_mask`, so a
single-shot encode attends every earlier frame of the clip. A config field the reference parses is not a field the
reference uses; only a golden longer than the field's threshold can tell them apart. The encode golden in the suite
was 2 s (50 encoder frames), far under 250, so the windowed encode passed it at 100 %.

The cost was silent: every clip longer than 10 s (250 encoder frames at 25 Hz) was encoded to codes that diverge from
the reference's from frame 125 on - training data and long clone references alike. The field is gone from
`EncoderConfig`; the encoder attention is plain causal.

## Evidence (2026-11-25)

Upstream golden (`tools/goldens/qwen3tts_codec_dump_reference.py --seconds 24`) on 24 s of real Swedish speech (six
corpus clips concatenated, resampled to 24 kHz), 300 frames x 16 codes, `mimi/tests/it/encode.rs`
`encode_matches_reference_codes` on the CPU backend:

| encoder attention | code match | cb0 | frames 0-124 | frames 125-299 |
|---|---|---|---|---|
| window 250 (before) | 62.02 % | 68.3 % | 100 % | 22-68 % per 25-frame block |
| plain causal (after) | 100.00 % | 100 % | 100 % | 100 % |

Reproduce: dump the golden with `--seconds 24` (a clip over 10 s is what matters), then
`BRAIN_DEVICE=cpu BRAIN_BACKEND=cpu make test/one P=brain-mimi T=it ARGS=encode_matches`.
