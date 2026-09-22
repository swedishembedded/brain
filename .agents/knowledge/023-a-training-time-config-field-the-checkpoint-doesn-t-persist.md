<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 23. A training-time config field the checkpoint doesn't persist makes the whole feature a no-op

`QwenConfig::to_json` never emitted the `lora` field; `from_json` hardcoded
`lora: None`. The LoRA forward/backward were correct and gradient-checked
(`gradcheck::check_qwen_lora`) - `Qwen::save` wrote the trained `*.lora_a`/
`*.lora_b` tensors to disk, exactly as designed. But the very next
`load_inference` rebuilt the param list from the reloaded config, which had
no `lora` entry, so it never allocated slots for those tensors: `lora_for()`
returned `None`, `lora_fwd` never dispatched, and the reloaded model was
bit-for-bit the untrained base. Every fine-tune "worked" (the training loop
ran, loss went down) and then silently did nothing the moment the process
that trained it exited.

Nobody noticed because the only test of it asserted `exact_match_after >=
exact_match_before` - "did not regress below baseline", which an unchanged
base model also satisfies exactly. This is #16 again, with a twist: the
statistic didn't just fail to distinguish success from failure, it couldn't
have distinguished them even in principle, because the two are
byte-identical once you skip the reload. **A test of a save/load feature
that never actually closes the save→load round trip is not testing the
feature.**

The fix is a two-line `Option` round-trip
(`crates/qwen3/src/config.rs`), but the gate that would have caught it needed
three things together, none optional: (1) take a few real optimizer steps so
`B ≠ 0` (a fresh LoRA init is `B = 0`, making the delta zero regardless of
whether it loads - see #4 on degenerate test setups hiding a whole bug
class), (2) **save, then load in a fresh process/struct** rather than
comparing against the live trained model still sitting in memory, (3) assert
the reloaded logits *differ from the base* by a real margin, not merely
"didn't get worse". `crates/qwen3/tests/lora_roundtrip.rs` does all three;
`crates/qwen3/tests/lora_learning_gate.rs` (Gate A) goes one further and
reintroduces this exact defect on purpose as its own verification that the
gate has teeth, rather than trusting the story that it would have caught it.
