# Chat-template-driven SFT training — status and known gaps

This is the workstream ledger for named LoRA adapters trained from bench's
exported SFT data (`.todo/bench-training.md`), in the house `status.md` style.

## Done

- **Generic chat-template rendering** (`crates/data/src/chat_template.rs`):
  executes a checkpoint's OWN `chat_template` Jinja string from its
  `tokenizer_config.json` via `minijinja` + `minijinja_contrib`'s pycompat
  layer, instead of hand-porting each model family's template control flow
  into Rust. Verified byte-identical, on the real Qwen3-0.6B template, against
  `crates/data/src/qwen_chat.rs` (a hand-transcribed, already-scrutinized
  Qwen3-specific port) across three cases including tool calls/results, the
  tools-schema preamble, and `enable_thinking=false`
  (`crates/data/tests/chat_template_cross_check.rs`). This is what makes the
  approach generic: a new model family (GLM, a future import) needs no new
  Rust template code, only its own `chat_template` string.
- **`ChatSample`/`ChatMessage`** (`crates/data/src/chat.rs`): a packed
  multi-turn training sample with per-message `train: bool`, rendered through
  the checkpoint's real template rather than an approximation. A `role` can be
  `"tool"` and is passed through as-is — the template merges consecutive tool
  turns and places think-blocks on its own; nothing here pre-folds or
  pre-renders that.
- **A strict wire schema on both sides of the bench/brain boundary.** bench's
  `generic-messages-v2` export is validated against a `pydantic` schema
  (`benchlib/datasets/formats/schema.py`) before the build writes it; brain's
  reader deserializes into `deny_unknown_fields` serde structs
  (`crates/data/src/chat.rs`'s `Wire*` types), never permissive
  `serde_json::Value` indexing with silent `.unwrap_or(default)` fallbacks. A
  missing/mistyped/unexpected field is a hard error naming the exact field,
  not a silent default that only produces a wrong answer downstream when some
  particular record happens to hit the gap.
- **Semantic validation on top of the structural schema, on both sides,
  automatically.** Structural typing alone accepts a well-typed field that is
  still nonsense: `tool_calls[].function.arguments` type-checking as a string
  that isn't actually valid JSON, or a `"tool"`-role message whose
  `tool_call_id` names no tool call anyone made. Both are now checked — every
  "tool" message must carry a `tool_call_id` that resolves to a `tool_calls`
  id from an earlier assistant message in the SAME sample — on brain's read
  side (`chat.rs::sample_from_wire`) and bench's write side
  (`schema.py`'s `ChatRecord`/`ChatRecordV2` `model_validator`,
  automatically on every `datasets build`, not just the separate manual
  `datasets validate` command). This is now AGENTS.md's own hard rule
  ("Validate everything crossing into brain from outside") — see that entry
  for what "structural AND semantic" means in general, not just here.
- **LoRA fine-tuning that actually survives a reload**: `QwenConfig`'s `lora`
  field round-trips through a checkpoint's config JSON (it silently didn't
  before — see `docs/lessons.md`); adapter-only save/load
  (`crates/qwen/src/lora.rs`) and folding into a base for zero-overhead
  serving.
- **Packed SFT export** (bench `benchlib/datasets/segment.py`
  `extract_packed_sample`): one record per trajectory instead of one record
  per decision step, killing the O(n²) nested-prefix token blowup the
  original `extract_samples` path has. Exact for causal-LM training, not an
  approximation, because loss at a trainable position only ever depends on
  tokens strictly before it.
- **A named-adapter model reference grammar**
  (`crates/modelref`/`crates/modelstore`):
  `Qwen/Qwen3-0.6B:owner:name:tag`, with a store layout
  (`adapters/<owner>/<name>/<tag>/`) so a base model and a fine-tune sitting
  side by side are two distinct selectable models.

## Known gaps

- **`render_with_message_boundaries`'s prefix-stability hazard.** Qwen3's own
  template inserts an empty `<think>\n\n</think>\n\n` block for an assistant
  turn only when it is LITERALLY the last message (`loop.last`). A common
  packed-conversation shape — a tool-call turn, its tool result, then a final
  answer (two assistant turns after the last real user turn) — makes the
  tool-call turn look "last" in a truncated prefix when it isn't in the true
  conversation, so `ChatSample::encode` correctly refuses to mask it rather
  than risk a silently wrong boundary (see the method's own doc for why a
  smarter role-preserving probe was tried and reverted: it can pull a
  following message's shared opening boilerplate into the wrong span, a
  smaller but still real silent-mismask risk). **Practical effect: any bench
  trajectory whose packed sample has more than one trainable assistant turn
  after its last real user turn currently fails to encode at all**, not just
  mismasks — this needs a provably exact boundary algorithm (or a documented,
  narrower probe strategy specific to Qwen3's template) before the packed
  export can be trained on end-to-end. Tracked here, not silently absorbed.
- **Tokenization is per-message, not whole-text with offset mapping.** Each
  message's own byte range is encoded separately and concatenated (matching
  `ChatExample`'s existing, accepted trade-off), rather than tokenizing the
  full rendered text once with a real offset map. A BPE merge could in
  principle span a message boundary and tokenize slightly differently than
  the true single-pass encoding would. `data::tokenizer::Tokenizer` has no
  offset-aware `encode` today; adding one is real but separate work (a
  workspace-wide tokenizer API change, several existing callers).
- **Only Qwen3's template is exercised.** GLM (`crates/glm`) is already a
  chat-tuned model with no chat-template code yet; the generic engine should
  handle it once its real `chat_template` string is available to compile and
  cross-validate against — no new Rust needed, but no gate proves it either.
- **DPO / KTO / ORPO / GRPO / full-parameter finetune / continued
  pretraining are not implemented.** For each, what brain already has and
  what's missing:
  - **DPO**: bench already mines branch-retry preference pairs
    (`benchlib/datasets/preference.py`, `dpo`/`kto` export formats). Needs a
    frozen reference-model forward pass and a pairwise loss on top of the
    existing masked-token training loop; no Rust work started.
  - **KTO**: the `kto` exporter already exists (two independent
    `{prompt, completion, label}` rows per pair). Needs an unpaired
    preference loss; no Rust work started.
  - **ORPO / GRPO**: no bench export format, no brain training-loop support.
  - **Full-parameter finetune**: `qwen::finetune::Mode::FullOffload` already
    exists (host-resident AdamW moments) but has no CLI entry point calling
    it — `brain qwen finetune` today does a full finetune by omission
    (`--lora` unset), not through this explicit mode.
  - **Continued pretraining**: no dataset contract, no CLI entry point.
- **Model-hash-based tags** (beyond `:latest`): tracked separately in
  `.todo/model-hashing-and-tags.md`.

## Seams that make each planned item cheap

- `qwen::finetune::Mode` already distinguishes LoRA from full-parameter
  training; a DPO/KTO mode is an enum variant, not a new training loop.
- `LoraCfg.targets` is already a plain list of leaf projection names, not
  hardcoded per-call-site.
- The masked-token dataset layout (`train.u32.bin`/`train.mask.bin`) is
  training-objective-agnostic; a pairwise (DPO) loss needs a second such pair
  (chosen/rejected) in the same layout, not a new format.
- `capability::Provider` is the existing seam for exposing a new training
  action generically over the CLI/D-Bus/event API without a new subcommand.
