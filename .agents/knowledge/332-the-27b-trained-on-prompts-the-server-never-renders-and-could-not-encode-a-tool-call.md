<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 332. The 27B trainer rendered prompts the server never renders, and could not encode a tool call

Question: does a chat fine-tune of the Qwen3.8-27B (`brain::ChatFineTune`, the path a caller's `train` step
takes) train on the tokens `brain serve` prompts the same model with? Three separate defects said no.
Found 2026-10-19 by rendering real agent requests through both.

* **Architecture.** `ChatFineTune` defaulted to `qwen3`; a Hugging Face checkpoint of the 27B
  (`Qwen3_5ForConditionalGeneration`) was only trainable if the caller named `qwen35`, which
  no caller did. An unnamed architecture now comes from the base's `config.json`
  (`architectures[0]`, else `model_type`) through the `brain_arch` registry, falling back to `qwen3`.
* **Tool calls.** Qwen3.8's template iterates `tool_call.arguments|items`, so it needs an object;
  Qwen3's passes a string through. Records carry the JSON text, so every record with a tool call
  failed to encode on the 27B ("cannot convert value into pairs"). `ChatTemplate::reads_argument_objects`
  now says which the template wants and the sample hands it the parsed arguments, key order kept.
* **Thinking switch.** With thinking on (the default) the 27B template adds "Reasoning effort is
  set to xhigh..." to the system turn. `no_think_block` compared the asked prompt with a plain
  render made with thinking ON, so the two differed before the assistant turn, no block was
  found, and the trainer concluded the model never reasons: it rendered every training prompt
  with thinking on (extra system text, an open `<think>`), while the server renders with
  `enable_thinking` false (no extra text, a closed empty block). The probe and the supervised
  suffix now render with the switches off.

Check (VERIFIED-BY-RUN): trainer context token counts equal the served `prompt_tokens` of the
27B GGUF server on 11 of 11 real requests (2157 to 2341 tokens); the cross-check test
`a_qwen38_training_prompt_is_the_prompt_the_server_renders` compares the training prompt of a
two-call tool conversation with `qwen_chat::render` (Qwen3.8 flavor) and needs the Qwen3.8
`tokenizer_config.json` (point `QWEN3_DIR` at the FP8 directory), else it skips loudly.

The check was made on requests with one tool turn at most; a conversation with many results
should be compared the same way before a long run.
