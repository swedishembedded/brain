<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 585. A sampled reply whose text is the answer's was refused under a template brain does not read

**Lesson:** a rollout record's `token_ids` are admitted when they are the tokenizer's encoding of the answer's text,
or when they read back as the same reply (`ChatTemplate::reads_as_same_reply`). Bytes a sampler draws need not be
UTF-8: every decode renders a stray byte as U+FFFD, and U+FFFD re-encodes to three other bytes, so the encoding test
fails on them by construction. The read-back test then answered `false` for every template whose replies brain does
not parse (no reply flavor), even when the decoded ids and the record's text were the same string. Two texts that
are equal once the end-of-turn marker is taken off are the same reply under any template; that comparison now comes
first. Spec: `data` `chat_template::tests::a_reply_whose_text_is_the_answers_reads_as_the_same_reply_under_any_template`.

## Evidence (2026-11-25, lane agrpo-e2e2)

The toy qwen3 checkpoint (byte-level tokenizer, bare template, no reply flavor) sampled at temperature 1 on the CPU
stack: the streaming job refused the first group sent, `messages[2].token_ids` of 163 ids against 289 rendered
tokens, first differing at position 1 (sampled byte 185, rendered 239, the first byte of U+FFFD's encoding). The
record's text equalled the decoded ids with `<|im_end|>` removed for all four rollouts of the group. With the change
the CPU stack ran to completion with every offered group taken.
