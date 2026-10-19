<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 330. A repeated tool call id gives every call every result, and no gate noticed

`brain serve` named every tool call `call_0` (each response numbered its own calls from 0), so a recorded agent
run held many calls with one id. A producer that pairs results to calls by id then gave every call every result of
the whole run: 147 of 231 examples of a training set had duplicated, reordered and FUTURE tool results in their
context. brain's record validation only asked that a tool result name a call made earlier, which `call_0` satisfied
trivially, so the corrupt set trained (loss 2.5 to 0.04, memorised) and the adapter looped on the one memorised opening.

The rule is now one function, `data::chat::check_tool_pairing`: every assistant tool call id is unique in the
conversation; a tool message naming an id answers exactly one earlier call that no earlier tool message answered; a
tool message with no id answers the earliest unanswered call (the order the template renders consecutive results in)
and must have one to answer. It runs in `messages_from_wire` (the `generic-messages-v2` reader, the preference reader
and `qwen3 lora_train` all go through it; the wire layer additionally requires an id on every tool message) and in
`ChatSample::encode_with` (SDK fine-tune, full fine-tune, preference and eval encoding), so neither a file nor an
in-memory sample can reach the trainer unchecked. Errors name the line (from the reader) and the id.

Consequence for preference pairs: a candidate that repeats a tool call must carry a fresh id; re-using the prompt's id
is the same corruption. The producer side is fixed too (`brain serve` issues ids unique per response and conversation), but the check, not the producer, is the guarantee: an id scheme can regress, a refused record cannot
be trained on.
