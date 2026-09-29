<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 176. A config default is a guess about which model wrote it

`QwenConfig::from_json` falls back to a default for every key it does not
find. That is fine for a key the writer always writes. It is a silent guess
for a key the writer leaves out. Two such gaps turned working checkpoints
into different models on reload.

**The writer left the architecture switches out.** `to_json` never wrote
`qk_norm` or `attention_bias`, and `from_json` defaults them to Qwen3's
values: QK-norm on, no bias. Every Llama or Qwen2 decoder saved as a brain
checkpoint therefore reloaded as a Qwen3:

- A Llama gained a QK-norm it never had.
- A Qwen2 lost its q/k/v biases.

`json_roundtrip_is_identity` now pins `from_json_checked(to_json(c)) == c`
over every preset, including the Llama and Qwen2 ones and a scaled and a
LoRA config.

**The reader was handed a GGUF.** Every loader did
`from_json(&reader.config())`. For a GGUF, `config()` is the raw KV map
(`qwen3.block_count`, ...), so not one brain key matched. Every field took
its default, which is `tiny()`: vocab 23, two layers, d_model 16. The
resident that serves a `.gguf` then built that shape. It read tensors by
brain names that a GGUF does not use.

`QwenConfig::from_reader` is now the one resolver:

- A GGUF's shape comes from its KV metadata (`config_from_gguf`).
- A brain header goes through `from_json_checked`, which refuses a missing
  shape key.
- A bare Hugging Face directory is refused, with a pointer to
  `hf::decoder_config`.

`qwen3::open_checkpoint` pairs that config with a source that reads under
brain's names, for either format. `a_gguf_decodes_exactly_as_its_import`
proves the GGUF build decodes bit-identically to its imported checkpoint.

Only a hand-built or test config should reach the unchecked `from_json`.
