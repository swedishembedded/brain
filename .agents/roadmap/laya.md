# laya - roadmap

Laya (`convaiinnovations/laya`, Apache-2.0): a **System-1 decision model** -
state plus a typed question with runtime-supplied options in, one calibrated
probability per option out, in a single forward pass. It never generates
text, which is why it lands beside `crates/decide` (`.agents/roadmap/decide.md`)
rather than a text decoder: both implement the same
`(state, question, allowed answers) -> P(answer | state, question)` contract,
just with different backbones and head shapes. New crate `crates/modernbert`
(package `brain-modernbert`) owns the reusable ModernBERT-large backbone, with
a `laya` module for the head. `brain::DecisionPipeline` dispatches to either
architecture from one directory-shape sniff, so a caller who already has
`triage`/`intents`/`salesagent` pointed at a `crates/decide` checkpoint can
point the same `--encoder` flag at a Laya checkpoint instead.

## What shipped

- [x] **Bidirectional local-window attention kernels** (`0345e9d0f`).
      `attn_scores_cross_win.wgsl`/`attn_scores_cross_kt_win.wgsl` plus
      `block::{CrossWinIds, KeyMinorWin, chunked_bidir_fwd_win}` - the one
      real kernel gap ModernBERT's alternating full/local attention needed.
      Windows the two-rung materialized path only (`crates/decide`'s
      middle `FlashIds` rung is not reproduced, since Laya does not need
      it). The fused `flash_attn_bidir_spans` kernel's own windowing was
      deliberately deferred - see "not yet done" below.
- [x] **`crates/modernbert`'s backbone forward** (`5fd8f72bb`, `19ebcdff9`):
      `ModernBertConfig` (both the nested and flat legacy `config.json`
      schemas), per-span RoPE with per-layer-type theta (160000 full /
      10000 sliding), GeGLU MLP (`gelu(first_half) * second_half`, the
      chunk order verified against the installed `transformers` source),
      no-bias pre-LN residual wiring, alternating full/windowed attention
      through the M1 kernels. Parity-tested against a real `transformers`
      `ModernBertModel` on tiny fp32 weights.
- [x] **The Laya decision head** (`471b17ff2`): `crates/modernbert/src/laya.rs`
      - the per-qtype embedding broadcast, 2 pre-norm self-attention head
      layers (biased, unlike the no-bias trunk), the `[MASK]`-marker gather
      via the existing `embed` kernel, the scorer MLP, and the act/escalate
      head fed `concat(pooled [CLS], 4 host-computed calibration features)`.
      Parity-tested against the real `DecisionModel` reference on both
      option logits and act logits.
- [x] **Real weights, tokenizer, and sequence builder** (`866a1e530`): the
      real 843 MB `convaiinnovations/laya` checkpoint imports (bf16
      safetensors, `hf_to_encoder`/`hf_to_head` name-mapping), the
      byte-level BPE tokenizer resolves through an extended
      `data::qwen_tokenizer::QwenBpe` (a new GPT-NeoX-style pre-tokenizer
      mode, differential-tested against the real `tokenizers` library over
      49 strings), and Rust `build_sequence` reproduces the reference
      packing layout and budget rules. The real checkpoint's argmax
      decisions matched the Python reference on every test case, on both
      the option-logits and act-head argmax, on both backends - the
      behavioral bar the plan set for a bf16 checkpoint (see "not yet
      done" for why a numeric-closeness bar was never the goal).
- [x] **Seeded backward and gradient check** (`a8fc68258`, `3c6ea6aa1`): a
      trainable `ModernBert`/`LayaHead` pair (`new_train_on`,
      `prepare_reverse`, `seed_buf`, `backward_seeded`), gradient-checked
      against finite differences on both backends. This is a **primitive**,
      not a training loop - see "not yet done" below, this is the single
      most important line in this file.
- [x] **SDK dispatch** (`b12e1142c`): `brain::DecisionPipeline` gains a
      private `Backend` enum (`Decide` / `Laya`), routed by a directory-shape
      sniff (`rl_agent_config.json` + `encoder/config.json` with
      `model_type: "modernbert"`), not a shared `config.json` field. `choose`,
      `probability`, `set_question`, `train_choices`, `save_head`, and the
      `Flow`/`Stages` chain are shared honestly across both arms.
      `DecisionPipeline::inner()` became `Option<&mut Decide>` (zero real
      call sites affected).
- [x] **Validation through the three decision samples** (`49c56c87a`):
      `triage`, `intents`, and `salesagent` all run against the real
      checkpoint - see the measured numbers below. Two SDK additions this
      needed and got: `Stages::supports_training`/`Flow::supports_training`
      (a caller checks before calling `.train()` on a pretrained-only
      backend, instead of a `.train()` that either panics or silently
      no-ops) and `DecisionPipeline::set_eval`/`Flow::with_eval` (so a
      caller skipping training still gets a real `Flow::evaluate` number);
      `DecisionPipeline::route(state, proposition) -> RouteVerdict` surfaces
      the act/escalate head, `Err` on the `Decide` arm.

- [x] **The typed request surface, and a JSON endpoint on it**:
      `DecisionPipeline::decide(&State, &[Question]) -> Vec<Answer>` answers
      several typed questions about one state on BOTH arms, reusing
      `decide::primitives`' own `Question`/`Answer`/`Opt` vocabulary (now
      re-exported from the SDK) rather than mirroring it. It is the only SDK
      surface that reaches `Question::Score` at all, the only one where a
      `Choice`'s options carry the descriptions the model reads, and the only
      one taking structured state - `modernbert::OrderedJson` gained a
      key-order-PRESERVING `parse` (the half `write_json` was missing; a
      `serde_json::Value` round-trip sorts, and would tokenize different
      bytes). The Laya arm's `choose`/`probability` are now callers of one
      typed `ask`, so the three question types cannot drift apart in
      calibration. `samples/decision/json` is a JEV-style
      (`{state, questions{type, instructions, criteria}}`) endpoint on stdin/
      stdout built on it, with `--model` now a reusable `appopts::ModelChoice`
      group. Real-weight gated on both checkpoints
      (`crates/sdk/tests/decision_pipeline.rs`); the documented Jev routing
      example answers `billing` at 0.9840 zero-shot.
- [x] **The Laya arm is calibrated the way its own serving reference is.**
      `rl_agent_config.json` carries TWO tables and `rl_agent_api.py` reads
      the finer one first: `temperature_by_options`, keyed
      `(qtype, option-count bucket)` by `rl_common.py::temp_bucket`, then the
      per-qtype `temperature` as a fallback. Only the scalar was applied
      before, which is invisible in argmax (no positive temperature can move
      it) and wrong in every published `choice` distribution - the released
      checkpoint fits `choice:2` at 1.9064, `choice:3-5` at 1.7602,
      `choice:6-10` at 1.0000 and `choice:11+` at 0.1006 against a per-qtype
      1.6369, so a wide choice was being published up to 16x too peaked.
      Found by running 22 fixed questions through BOTH this SDK and the real
      `rl_agent_api.py` on the same checkpoint and diffing every number:
      answers agreed 22/22 but 8 choice distributions differed. Now
      `modernbert::temp_bucket` + `LayaBackend::temperature_for(qtype, k)`,
      gated by `real_laya_choice_probabilities_match_the_reference_serving_
      calibration` (the reference's own printed numbers, not a
      re-derivation). Post-fix the same 22 cases agree to within 3.2e-4, and
      an 11-point temperature sweep agrees to within 4.9e-4.

### Measured numbers (M7, `49c56c87a`)

`decide`/MiniLM trained this session at reduced step counts for time budget;
Laya is zero-shot throughout (no training loop exists - see below). Same data
both arms.

| sample | decide/MiniLM | Laya (zero-shot) | chance |
| --- | --- | --- | --- |
| triage (BANKING77, 8 intents, 160 held-out) | 78.8% acc (300 steps) | 68.8% acc | 12.5% |
| intents seen (65 intents, 300 steps) | 16.0% acc | 78.0% acc | 9.5% |
| intents unseen (12 held-back intents) | 6.0% acc | 71.0% acc | 9.5% |
| intents shuffled-state control | 6.0% acc | 2.0% acc | 9.5% |
| salesagent (final turn, held-out convs) | 0.580 acc / 0.592 AUC-ROC / 0.276 Brier (700 steps) | 0.520 acc / 0.254 Brier (act head: 100% one class) | - |

Read these honestly: the decide arm's `intents` run is undertrained (300 of
the sample's own 2000-step default) and does not itself clear the "reads
options, not positions" bar the sample checks for. Laya's zero-shot `intents`
numbers are the real validation this sample exists to produce - 71.0%
unseen-intent accuracy against 9.5% chance (the model reads option *text*, it
was never brain-side trained on any of these examples), and shuffling the
input state collapses accuracy to 2.0% (it reads the state too, not just the
option list). `salesagent`'s act head predicted the same class for all 100
held-out conversations - a real finding (this checkpoint's act head is
uncalibrated for synthetic B2B SaaS sales dialogue, far from its training
distribution), not a code bug: the option-logit-derived `probability` moved
normally across conversations.

## Not yet done

- [ ] **No training loop.** M5 shipped a gradient-checked, trainable backward
      *primitive* through the full head and trunk - not a reimplementation of
      Laya's own RLCD/REINFORCE training procedure, and no optimizer, loss,
      or training-loop wiring sits on top of that primitive. A
      Laya-pointed `triage`/`intents`/`salesagent` run today gets a clean
      `LAYA_TRAINING_NOT_IMPLEMENTED` error from `.train()`/`train_choices`,
      not a panic and not a silent no-op - callers must check
      `supports_training()` first.
- [ ] **English root checkpoint only.** The multilingual Laya variant is a
      genuinely different backbone (mmBERT-base, not ModernBERT-large), not a
      config variant of what shipped here. The typed-decisions checkpoint is
      untested. Neither is in scope.
- [ ] **512-token trained context, not 8192.** `rl_agent_config.json`'s own
      `max_len` (512) is the checkpoint's real trained context; ModernBERT's
      position table allows up to 8192, but the released checkpoint was
      never trained to use it - `build_sequence` truncates/refuses at 512,
      it does not extrapolate.
- [x] **`brain pull` now works for both checkpoints this work depends on.**
      Found broken during M4 (`convaiinnovations/laya`: no recipe recognized
      its directory layout - `encoder/config.json` + `rl_agent_config.json`,
      no root `config.json` - "no config.json in repo") and M7
      (`sentence-transformers/all-MiniLM-L6-v2`: "unsupported architecture
      BertModel"), worked around by hand at the time, fixed properly as a
      small separate follow-up (pull-side only, `crates/modelstore/src/
      recipe.rs` + `crates/arch/src/lib.rs` + `crates/loader/src/supply.rs`
      - deliberately NOT the serving-side `ArchSpec`/`crates/catalog`
      registry, which would give `crates/decide`/`crates/modernbert` a real
      CLI-servable surface neither had before and this plan never scoped).
      Verified with real, fresh `brain pull` runs for both: Laya now fetches
      exactly the 5 files needed (807 MiB, not the ~1.7 GiB a naive
      whole-listing fetch would have pulled in the out-of-scope
      `multilingual`/`typed-decisions` variant subdirectories - a second real
      bug caught while fixing the first), MiniLM fetches its 4 files (87
      MiB) verbatim with no tensor rewrite (`load_decide` needs the raw HF
      files, so `decide` joined `PASSTHROUGH_TRANSFORMERS_FAMILIES`).
- [ ] **The act/escalate head's class-order semantics are unconfirmed.**
      Only the head's *math* was parity-tested against the real checkpoint
      (M3/M5) - which index means "escalate" vs "continue" was never
      independently checked against the real `rl_common.py`. `RouteVerdict`
      deliberately exposes `act_index`/`act_probabilities` neutrally rather
      than naming a (possibly wrong) `ActDecision::Escalate` variant. A
      caller that wants to print "escalate" instead of "class 1" must
      confirm the order against the reference first.
- [ ] **`samples/decision/salesagent` reaches Laya through a different path
      than `triage`/`intents`.** `salesagent` is built on `ConversionPipeline`
      (`crates/sdk/src/conversion.rs`), which is decide-only by construction
      - it implements a specific policy-gradient/confidence-router recipe
      (`decide::policy`, `decide::routing`) with no Laya equivalent, and
      loads through `crate::decision::load_decide` directly, never through
      `DecisionPipeline`'s `Backend` dispatch. This was a real asymmetry the
      original plan did not anticipate (found in M7): when
      `ConversionPipeline::load` fails, `salesagent` falls back to
      `DecisionPipeline::route` for a zero-shot score, through the same
      architecture-dispatch seam, not directory sniffing in the sample. A
      real Laya equivalent of `ConversionPipeline` (a `repr_snapshot`
      analog for the router's signals, a real training loop) is unbuilt and
      not scoped here.
- [ ] **The fused `flash_attn_bidir_spans` kernel's windowing was
      deliberately deferred** (M1). Every local-attention layer therefore
      always uses the slower materialized `chunked_bidir_fwd_win` rung,
      even on a device where the fused path would otherwise be selected for
      full-attention layers - correct, but not the fast path.
- [ ] **No public docs surface added for the whole decide/Laya decision
      family.** `crates/decide`/`DecisionPipeline` has zero presence in
      `docs/models/`, `docs/manifest.txt`, or `docs/using/sdk.md` today - a
      pre-existing gap, not something this milestone introduced. Laya
      dispatches through the exact same `DecisionPipeline`, so giving Laya
      alone a `docs/models/laya.md` page or catalog row would document the
      new architecture ahead of the established one, the same asymmetry
      `.agents/roadmap/decide.md`'s own plan (see this plan's "explicitly
      not in scope") already refuses to create for the CLI surface.
      Deliberately not done here; a real follow-up is documenting the whole
      decision family at once, not Laya alone.

## Reusable, not Laya-specific

- `Stages::supports_training`/`Flow::supports_training` and
  `DecisionPipeline::set_eval`/`Flow::with_eval` are now the SDK's general
  shape for "this backend arrived pretrained, do not train it, but still
  evaluate it" - any future pretrained-only backend (a quantized/distilled
  export, a hub-imported checkpoint with no brain-side training path) reuses
  the same two seams rather than inventing a new one.
