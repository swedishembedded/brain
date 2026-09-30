<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 190. A tokenizer hash identifies a library version, not a tokenizer

llama.cpp's converter picks `tokenizer.ggml.pre` from a table of
`chkhsh` values. Each is the SHA-256 of the ids
`transformers.AutoTokenizer.encode(CHK_TXT)` returns. Those ids include
whatever BOS that `transformers` version adds, so the hash moves with the
library, not with the tokenizer.

- **In the pinned reference environment none of the DeepSeek hashes match
  the table.** deepseek-llm-7b-base hashes to `93105512…`, while llama.cpp
  lists `049ecf76…` for the same repo. The two coder-v1 checkpoints match
  no entry at all, which is why llama.cpp's own converter refused
  deepseek-coder-1.3b.
- **The pre-tokenizer's own structure is a stable key.**
  `qwen3::export::gguf_pre_tokenizer` hashes the `tokenizer.json`
  `pre_tokenizer` object instead (compact JSON, keys sorted). That hash
  sorts the store's eight tokenizers into exactly four families:
  - Qwen3 and R1-Qwen → `qwen2`
  - R1-Llama → `llama-bpe`
  - coder-v1 → `deepseek-coder`
  - llm, math and coder-v1.5 → `deepseek-llm`
- An unknown structure is refused. A GGUF whose `pre` names the wrong
  regex tokenizes differently from the training and still loads.

Two more of the converter's rules had to be matched before brain's export
equalled llama.cpp's conversion:

- **An added token can be a control token by its look.** DeepSeek-Coder's
  `<pad>` is not flagged special, but llama.cpp's `does_token_look_special`
  types it CONTROL.
- **BOS/EOS addition is read from the post-processor.** A `ByteLevel`
  post-processor adds neither, whatever `tokenizer_config.json`'s
  `add_bos_token` says. The chat template writes the BOS itself.

With both matched, deepseek-coder-1.3b-instruct exported by brain has the
same tensors, byte for byte (the llama q/k permute included), the same
non-descriptive metadata and the same `llama-tokenize` output as the file
llama.cpp's converter writes.
