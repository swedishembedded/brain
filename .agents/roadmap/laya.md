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
      against finite differences on both backends. A **primitive** at the
      time, with no optimizer, loss or loop on top of it - M8 below is what
      closed that.
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
Laya is zero-shot throughout - at M7 no training loop existed for it yet
(M8 below). Same data both arms.

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

- [x] **The training loop (M8).** The single most important line in this
      file's "not yet done" column for six milestones, now closed and gated
      on the real checkpoint (see the measured numbers below).
      `DecisionPipeline::train_choices`/`save_head` work on the Laya arm and
      `Stages::supports_training` is true for it, so `triage`/`intents` train
      against a Laya checkpoint exactly the way they already do against
      MiniLM - they ask the pipeline rather than sniffing the directory, so
      neither sample needed a code change to gain it.

      **The objective is transcribed, not invented.** `rl_common.py` names
      the reward (`proper_reward`) but not the update; the update was found
      in the only PUBLIC Laya training loop, the DDP fine-tuning script
      embedded in the project's own Kaggle notebook
      (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`,
      github.com/NandhaKishorM/laya), cross-read against the model card's
      prose. Two new leaf modules hold it, below both decision model crates:

      * `rlcd::proper` - `proper_reward` verbatim (log score + `w_sph` *
        spherical, minus a ranked-probability term on ordinal questions, with
        the `-9.21` log floor). It returns a REWARD and deliberately no
        gradient, because the reference never differentiates it. Gated
        against golden values produced by RUNNING the real Python under
        torch 2.13 on this box (agreement to 2e-5 on six cases spanning all
        three question types and both hard and soft targets), plus
        strict-propriety and ordinal-ordering property tests.
      * `rlcd::reinforce` - the update: `G` zero-mean-projected Gaussian
        logit perturbations, scored with NO gradient, standardized by a group
        mean and an unbiased std into a GRPO-style advantage, then
        `-mean(adv * log p(sample | logits))` under the exploration density,
        plus a soft cross-entropy term. `explore`/`policy_loss` split on
        exactly the boundary the reference's own `torch.no_grad()` draws,
        which is what makes the differentiable half finite-difference
        checkable at all.

      **What the public record does NOT pin down is recorded rather than
      smoothed over.** `RlcdObjective::default` takes the published
      fine-tuning loop's settings (`G=4`, sigma 0.4 -> 0.1, `w_sph=0.75`,
      `ce_weight=1.0`) because that loop exists in source; the model card's
      prose describes the ORIGINAL pretraining run differently (`G=8`, sigma
      1.0 -> 0.3, `w_sph=0.5`, "pure policy gradient (zero supervised
      cross-entropy loss)") and that script was never published, so it is
      `RlcdObjective::pretrain` instead of the default. `rl_train.py` and
      `evaluate.py` (which fitted `temperature`/`temperature_by_options`) are
      both absent from every public source.

      **The head learning rate is the one number NOT taken from the
      reference**, and it says so: the published loop's `1e-4` is for an
      effective batch of 64 over thousands of updates, while
      `train_choices`'s contract is one example per step over hundreds. The
      SDK uses `3e-4` on a cosine schedule to a `1e-6` floor (the published
      schedule SHAPE, no warmup). Sigma anneals 0.4 -> 0.1 across the run as
      published.

      **`crates/modernbert` gained the composition it was missing**:
      `modernbert::LayaDecision` (trunk + head + tokenizer as one model, with
      `score`/`accumulate`/`adamw_scaled`/`train_step_with`/`save_head`), the
      direct counterpart of `decide::decide::Decide`. Before it, the ONLY
      assembly of a Laya model lived in the SDK, so this crate could not
      train, evaluate or test one without the SDK on top of it. The SDK's
      `LayaBackend` now holds one of these plus the serving calibration, and
      owns nothing else about the architecture. `ModernBert` gained
      `adamw_step_scaled` to match `LayaHead`'s (M6).

      **Trunk training is a construction mode, not a runtime flag**
      (`Training::{Off, HeadOnly, HeadAndTrunk}`), and that is a memory
      decision before a tuning one: a `Role::Trainable` ModernBERT-large
      trunk carries a gradient and two AdamW moments for each of its 395M
      parameters, ~6.3 GB before any activation. `HeadOnly` is what the SDK
      loads - the trunk stays `Role::Frozen` and the head's backward writes
      its hidden-state gradient into a small scratch sink instead of the
      trunk's seed buffer, so the head's reverse pass is identical in both
      modes and only its consumer differs.

      **The act/escalate head is deliberately NOT trained.** Its gradient
      seed is zero on every step, so it comes back exactly as the checkpoint
      shipped it. No public Laya source defines its objective: the published
      loop's only act-head term is a `0.0 * act.sum()` no-op that exists to
      give DDP a gradient path, and the cost matrix behind `act_costs`/
      `cost_wrong_act` appears in prose only.

### Measured numbers (M8)

- **Real checkpoint, held-out accuracy 0.350 -> 0.600** (chance 0.250, 20
  held-out examples), `real_laya_checkpoint_head_training_improves_held_out_
  accuracy` (SDK, slow lane, skip-if-absent). The task is deliberately
  ARBITRARY: four everyday topics (finance / cooking / sport / weather)
  mapped onto four meaningless option names (`alpha`/`beta`/`gamma`/
  `delta`), 16 training sentences, 20 held-out sentences sharing only their
  TOPIC with the training ones. A pretrained decision model cannot guess
  that mapping, so zero-shot sits near chance and there is real headroom;
  getting the held-out ones right needs generalization, not memorization.
  Held-out is scored in the CANONICAL option order while training shuffles a
  sampled subset every step, so a model that learned a POSITION scores at
  chance. 200 steps, head only, ~750 s on this box. The trained head then
  saves, reloads through `DecisionPipelineBuilder::head`, and reproduces
  that 0.600 exactly.

- **The head learning rate was chosen by measurement and the derivation
  alone would have been wrong.** Matching the published run's parameter
  budget (`0.5 * lr * steps`) argued for `3e-3`; three real runs on the same
  task say otherwise:

  | head lr | steps | held-out accuracy |
  | --- | --- | --- |
  | 3e-4 | 120 | 0.375 -> 0.375 (learning, far too slowly) |
  | 3e-3 | 200 | 0.375 -> 0.125 (BELOW chance - it damages the pretrained head) |
  | 1e-3 | 200 | 0.350 -> 0.600 |

  (The first two rows were scored on the earlier 8-example held-out set;
  the last row is the 20-example one this gate now uses, which is why its
  "before" differs.)

- **The training loss is NOT gated, on purpose.** It is printed
  (1.0395 -> 1.7712 on the winning run) and it is not monotone: a REINFORCE
  objective's scalar at a batch of ONE has a mean-zero, high-variance policy
  half over `G` sampled reports, and the option SUBSET drawn each step
  varies in size, so the per-step floor moves too. Gating on it would gate
  this feature on noise. Held-out accuracy is the number that means
  something, so that is what the test asserts.

- **Tiny random model, checkpoint-free** (`crates/modernbert/tests/
  train_convergence.rs`, 4 layers, `d_model` 64, trunk+head, 200 steps, 8
  arbitrary examples over 4 options): **loss 1.1165 -> 0.0000, accuracy
  0.250 -> 1.000** against a chance of 0.250.
- **A saved head reloads and reproduces the same logits** to 1e-4 on both
  the tiny model and the real checkpoint, into a model whose head was seeded
  differently - so what comes back came from the file.
- **The RLCD objective optimizes what it claims to**, checkpoint-free:
  descending it with NO cross-entropy help drives a bare score vector to
  within L1 0.10 of a strictly-proper optimum it is only ever shown sampled
  rewards of (`rlcd::reinforce`'s own unit test).

### A measured architectural fact, recorded because it looks like a bug

A RANDOM ModernBERT trunk cannot be frozen and still learn this task, and
the reason is structural rather than a matter of learning rate: every option
marker is the SAME `[MASK]` token, ModernBERT has no learned position table
(position reaches the model only through RoPE, inside attention), and at
random init attention is near-uniform - so all `k` markers arrive at the
head as nearly the same vector and no head can tell the options apart.
Measured at `d_model` 64: a frozen random trunk returns the four option
logits as `[0.030221, 0.030220, 0.030218, 0.030210]` and 1200 steps move
accuracy from 2/8 to 3/8, while unfreezing the trunk reaches 8/8 within 100.
This is also why `Training::HeadOnly` is nevertheless right for the REAL
checkpoint, whose trunk already separates them.

Separately, a tiny random model trained with SHUFFLED option orders settles
at exactly `ln 4` - a perfectly uniform report, which IS the optimal answer
for a model that cannot read option text. That is a capability limit of a
64-wide 4-layer model on eight examples, not a wiring fault: the permuted
packing itself is now gated against the real Python reference
(`build_sequence`'s `option_order` path, previously uncovered - three new
cases covering a permutation, the same permutation on the option-shrinking
path, and the identity permutation, which must be byte-identical to passing
none).

## Not yet done

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
- [ ] **The act/escalate head is UNTRAINED here, and its class order is now
      documented but still unenforced.** M8's research settled the order:
      index 0 is "act" and index 1 is "escalate". The evidence is three-way -
      the project's own write-up states the head outputs
      `[P(act), P(escalate)]`, its architecture diagram labels the same, and
      the serving reference on this box reads slot 0 for the field it names
      `act_probability` (`rl_agent_api.py`: `ext = {"act_probability":
      float(act[r, 0])}`). The published cost matrix behind the config is
      +1.0 for a correct action, -3.0 for a wrong one (`cost_wrong_act`) and
      -0.5 flat for escalating (`act_costs["escalate"]`), which breaks even
      at `P(correct) > 0.625`. What is still open: `RouteVerdict` continues
      to expose `act_index`/`act_probabilities` neutrally rather than naming
      an `ActDecision::Escalate` variant, because nothing in this workspace
      ENFORCES the order, and a multi-key `act_costs` would have an index
      order resting entirely on Python dict-insertion order with no public
      code to verify it against. Training the head is separately out of
      scope - see M8 above for why (no public source defines its
      objective).
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
      analog for the router's signals) is unbuilt and not scoped here.
      M8 closed the "real training loop" half of that sentence, but NOT the
      `ConversionPipeline` half: `salesagent` still reaches Laya only
      through `DecisionPipeline::route`, zero-shot.
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
