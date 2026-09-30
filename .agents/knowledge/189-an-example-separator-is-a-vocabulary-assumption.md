<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 189. An in-band example separator is a vocabulary assumption

Chat datasets closed every example with Qwen's `<|endoftext|>`, id 151643,
and the trainer found example boundaries by scanning for it. That only works
for a vocabulary that has that id, and means the same thing there.

- **For any other vocabulary it is an out-of-range id.** DeepSeek-Coder has
  32256 ids and DeepSeek-LLM 102400. A dataset written for them carried an
  id past their embedding tables, which is why three callers had to insist
  on a vocabulary above 151643, including test models built with 151644
  embedding rows just to hold the separator.
- **A real end-of-sequence id would not do either.** It occurs inside
  multi-turn samples, after every assistant turn, so splitting on it cuts one
  conversation into several examples.

Datasets now index their examples out of band: `<split>.ex.bin` holds each
example's `u64` start offset, and the stream holds only the conversations'
own tokens. `data::chat::write_split` is the one writer. A row pads with id
0, since a pad position is never a target and nothing before it attends to
it. The trainer still reads a legacy dataset by its separator, when the
stream holds 151643 and the vocabulary covers it.

Two further findings about rendering SFT data through a checkpoint's own
template:

- **The template writes the BOS, once.** `QwenBpe::encode` adds no BOS of
  its own, so encoding each message's range of one render gives exactly the
  template's `{{ bos_token }}`.
- **A reasoning model's template can drop the reasoning you mean to train.**
  DeepSeek-R1's template keeps only what follows the last `</think>` in every
  assistant turn. That is right when rendering history at inference and
  wrong for reasoning SFT. `RenderOpts::keep_reasoning` hides a trained
  turn's `</think>` from the template behind a private-use sentinel and
  restores it in the rendered text. It is opt-in, `--keep-reasoning` or the
  `lora_train` action's `keep_reasoning`, so no existing dataset changes
  silently.
