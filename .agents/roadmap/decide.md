# decide - roadmap

`crates/decide` - brain's **decision model**: a bidirectional state encoder plus a
trainable candidate-to-state cross-attention head that returns a calibrated
probability distribution over **developer-defined, runtime-supplied options**
instead of generating text.

brain's own architecture (`Source::Brain`, `Domain::Decision`), clean-room against
a *published interface* only. TypeSafe's Jev is the interface reference - its
architecture, weights and training recipe are not public and nothing here is
derived from them. The wire format is deliberately compatible so existing
System One clients can point at brain unchanged.

---

## 1. What it is

```
(state, question, allowed answers) -> P(answer | state, question)
```

One state encode is shared by every question in a request; each question is scored
independently against it. No answer becomes another question's context.

Three public primitives, one mechanism underneath (a softmax over per-option
scores):

| Primitive | Option space | Answer |
|---|---|---|
| `choice` | up to **255** named options | `choice`, `probabilities`, `confidence` |
| `score`  | **2-10** ordered levels | `score` (0-based `sum i*p_i`), `legend`, `probabilities`, `confidence` |
| `noul`   | implicit `{yes, no}` | `noul` in [0,1] (no `confidence`, matching the reference contract) |

`score`'s levels are scored independently - the model never sees a level's index or
its neighbours - which falls out of per-option scoring for free rather than being
special-cased.

---

## 2. Architecture (decisions, with the reason each one is forced)

| # | Decision | Why |
|---|---|---|
| B1 | **Late fusion.** State encoded once; question+option encoded separately; combined only in the head. | The contract requires one state pass shared by all questions. A uni-encoder that prepends labels to the text scores better but re-encodes per question set, forfeiting the entire latency argument. |
| B2 | **Question instructions live in the slot, never in the state.** Slot text = `{instructions} [SEP] {option name}`. | If instructions entered the state encoding, the state could not be shared across questions. This is what makes independence architectural rather than aspirational. |
| B3 | **One weight-shared encoder, two roles**, separated by BERT's pretrained `token_type_embeddings` (`[2, 384]`): type 0 = state, type 1 = slot. | Fewer params, one latent space, and the role signal is pretrained rather than bolted on. |
| B4 | Slot pooling = **mean-pool** then a learned projection to query space. | MiniLM's sentence-transformer objective *is* mean-pooled; `[CLS]` is the worse-initialised choice here. |
| B5 | Head = **1 multi-head cross-attention layer** (slot query -> state keys/values) + residual + LN + scorer. Depth is a config field, default 1. | Poly-encoder showed one late attention layer recovers most of a cross-encoder's gap at a fraction of the cost. |
| B6 | Scorer = **dot product**; weighted-dot and MLP selectable by config. | GLiNER/GLiClass show late dot-product fusion suffices. The alternatives exist for the ablation, not the default. |
| B7 | Softmax **within a question**. Noul = the same machinery over `{yes,no}`; Score = the same over levels plus the 0-based expectation. | One mechanism, three surfaces. |
| B8 | **Option-embedding cache** at inference, LRU keyed by a digest of (slot-set, weights id). | Turns a 77-option decision into one state pass plus ~15 MFLOP. Survives full fine-tune because inference weights are fixed. |
| B9 | **Windowed state encoding**: state split into <=512-token windows (with overlap), each encoded independently, all token states concatenated into one K/V set the head attends over. 32k = 64 windows. | The only way to reach 32k at linear cost on this hardware (section 5). Works *because* of B1: cross-window integration happens in the head, which is where the decision is made. Same trade ColBERT-style late interaction makes over long documents. |
| B10 | **Global position signal injected at the head**, into the cross-attention K/V - never into the encoder. | Window positions restart at 0, so the head needs global ordering. Injecting at the head keeps the encoder byte-for-byte parity-able against the released checkpoint. |
| B11 | Cross-window mixing (a cheap transformer over window summaries) is **config-gated and decided by ablation**, not assumed. | 64 summaries is computationally nothing; whether it buys accuracy is an open question, so it ships only if measured to. |

### Encoder

`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0): 6 layers, d=384, 12 heads,
ff=1536, exact/erf GELU, learned absolute positions (512), `layer_norm_eps` 1e-12,
vocab 30522, 22.7M params. Tensor names taken from the checkpoint's own safetensors
header.

**Full fine-tune**, with discriminative rates through `ParamStore::set_lr_mult`
(encoder 2e-5, head 1e-3).

Behind a `StateEncoder` seam. The named future alternative is ModernBERT-base
(Apache-2.0, 8k native, RoPE, alternating local/global attention) extended with
`model::yarn` - swapped in only if B11's ablation says encoder-internal long-range
attention beats a 10x cost increase.

### Tokenizer

New `data::wordpiece`: WordPiece (`##` continuing prefix,
`max_input_chars_per_word` 100, `[UNK]`) + `BertNormalizer` (clean_text,
handle_chinese_chars, lowercase, strip_accents follows lowercase) +
`BertPreTokenizer` + the `[CLS] A [SEP]` template. Ids `[PAD]=0 [UNK]=100
[CLS]=101 [SEP]=102 [MASK]=103`.

The workspace's **third tokenizer family**, after byte-level BPE (`bpe`,
`clip_bpe`, `qwen_tokenizer`) and SentencePiece unigram (`unigram`).

---

## 3. Training

| # | Decision |
|---|---|
| D1 | **Random option-subset sampling per example**: k ~ U[2, 77], gold always included, optional "none of the above". This is what makes the model score options rather than learn a disguised 77-way head, and it is what makes the held-out-intent eval mean anything. |
| D2 | Held-out-intent split: **20 of 77 intents** held out by fixed documented seed; train on 57. Both numbers from one code path. |
| D3 | Loss `(1-lambda)*L_focal(gamma) + lambda*L_Brier`, one implementation where `gamma=0, lambda=0` collapses **exactly** to CE. Sweep {CE; focal g=3; CE+Brier l=0.5; focal+Brier}. Winner pinned by ECE at equal accuracy. |
| D4 | Post-hoc **temperature scaling** on a val split, reported before/after. |
| D5 | AdamW, wd 0.01, clip 1.0, `model::train::LrSchedule` warmup->cosine, deterministic for a fixed seed. |
| D6 | Options encoded **once per batch** when the batch shares an option set (always true for Banking77): ~66 s/epoch, ~11 min for 10 epochs on one P40. Per-example option sets are the known slow path - group by option-set there. |

### Expected-cost note, recorded so it is not later misread as a regression

D1 can put our Banking77 number **below** the published RoBERTa-base 93.86% /
ModernBERT-base 93.99% baselines, *by design*: those baselines always see all 77
options at training time and never score an option they have not met. The
held-out-intent number is the one that measures what this model is for.

---

## 4. Evaluation

New `eval::calibration` (none of this exists in the workspace today - `crates/eval`
has detection, MLM and TTS metrics only):

- Accuracy, macro-F1.
- **ECE (15 bins), AdaECE, classwise-ECE, Brier, NLL**, reliability bins.
- Coverage-vs-accuracy, **failure-AUROC**, **AUGRC**.

**Confidence = `1 - H/ln k`.** k-invariant, uses the whole distribution, exactly 1
on a one-hot and 0 on a uniform. `p_max` is disqualified because its floor moves
with a runtime-defined k (0.5 at k=2, 0.0039 at k=255), which makes a *published*
threshold meaningless - and a published threshold is the entire point of exposing
confidence. The selective-prediction literature does not settle the ranking
question (the 13-CSF benchmark in arXiv 2407.01032 found the MSR baseline is never
consistently beaten), so k-invariance is the deciding argument, not a ranking claim.

`probabilities` is **always** returned in full, so a caller preferring MSR or
margin computes it themselves. All four statistics (MSR, margin, normalised
entropy, normalised Gini) are measured by failure-AUROC and AUGRC on held-out data;
the choice is pinned in a test and revisited only with numbers.

### Hard gates

- **Option-permutation invariance** - permuting the option list must not move
  probabilities beyond float noise.
- **Shuffled-state control** - accuracy must collapse toward chance when the state
  is shuffled against its label.
- **Single-window bit-identity** - a state that fits in one window must produce
  token states bit-identical to the unwindowed path.
- Gradient-faithfulness on **both** CPU and GPU backends.

---

## 5. Performance

Measured budget on 2x Tesla P40 (11.76 TFLOPS fp32 peak, sm_61), at a deliberately
conservative 2 TFLOPS effective. MiniLM windowed: 21.2 MFLOP/token of projections
plus 4.7 MFLOP/token of within-window attention.

| State | FLOPs | Latency | Rate |
|---|---|---|---|
| 512 tok | 13 GFLOP | ~7 ms | ~150 Hz |
| 2k | 53 GFLOP | ~27 ms | ~37 Hz |
| 8k | 212 GFLOP | ~106 ms | ~9 Hz |
| 32k | 849 GFLOP | ~425 ms | ~2.4 Hz |

**32k and 10 Hz are not simultaneously available on this hardware with any dense
bidirectional encoder** - ModernBERT-base costs 30.5 TFLOP at 32k (~15 s),
Qwen3-Embedding-0.6B more. This is arithmetic, not a MiniLM limitation. The design
target is therefore *capacity to 32k with cost strictly linear in the state
actually sent, and full speed when the state is small*.

Small states are **dispatch-bound** (~90-100 kernel dispatches per forward), not
FLOP-bound: that is where optimisation starts, beginning with the existing
`flash_attn_bidir*` path that collapses the scores/softmax/apply trio. Large states
are FLOP-bound and batch well (64 windows of 512 is a near-ideal batch), so the
2 TFLOPS figure is pessimistic in the 32k row and optimistic in the 512 row.

No number in this file is inherited from any surveyed project. Every published
latency in this space is single-setup, small-n, and none of it is on Pascal.

---

### Minibatching: one encoder pass carries the whole batch

`Decide::accumulate_batch` / `train_batch` pack `B` examples into ONE encoder
pass - every state's windows first, then every example's option slots, which is
the packer's existing ordering rule - and run the head once per example over
it. The seam that makes it possible is that `Head::set_call` names its state
and its slots by ROW INDEX rather than by a row offset, so example `b` reads
exactly its own rows out of the middle of the pack; the gather is `embed` and
the adjoints are `row_scatter` (state, assigns - the rows are disjoint) and
`emb_bwd` (the `[CLS]` rows, accumulates). Addressing by index rather than by a
bound offset is the same rule commit `e494d47dd` established for the reverse
attention, and for the same reason - see `head.rs`'s module doc.

`Head::clear_seed` is separate from `Head::backward` for this: the seed buffer
is zeroed ONCE per encoder pass, and each example's reverse pass writes only
the rows it owns.

Two things follow that a caller can measure:

* the fixed per-step cost (`zero_grads`, `adamw`, the queue drains between the
  halves) is divided by `B`;
* the encoder's GEMMs see `B` examples' rows at once, which is what a
  dispatch-bound small state needs.

`decide_bench --batch-scan` prints both, per step and per example. It is a
measurement rather than a rule because which of the two dominates is a property
of the card and of the shape.

**The gradient is the same gradient**, and `crates/decide/tests/minibatch.rs`
holds a batched pass against `B` sequential `accumulate` calls buffer by
buffer, in both halves, with the encoder LIVE. A loss curve cannot make that
distinction: a shared pass that dropped every example but the last would still
fall and still converge.

The SDK exposes it as `DecisionPipeline::set_batch_size`, where `steps` keeps
meaning OPTIMIZER steps. The default is one, because AdamW normalizes its step
and so ties the batch size to the learning rate: every accuracy published for
either arm was measured at a batch of one, and the Laya arm's head rate in
particular was fitted by measurement at that batch (see `LAYA_HEAD_LR`'s own
doc for the three runs).

---

## 6. Serving, SDK, sample

- Full serving contract in the same change: capability manifest, resident adapter,
  catalog entry, D-Bus, CLI verb, `examples/` client.
- `run_batch` does a **genuine batched state forward** across concurrent requests.
- New `arch::Domain::Decision`. `Text` does not fit - no vocabulary, a runtime-defined
  output space, and `infer` would mean something else. Required regardless: an SDK
  surface feature name *is* a `Domain` variant.
- SDK `brain::DecisionPipeline`, surface feature `decision`.
- HTTP `POST /v1/systemone` in `apiserve`, wire-compatible: `choice`/`score`/`noul`,
  `criteria`, `instructions`, `probabilities`, `confidence`, `legend`,
  `usage.output_tokens: 0`, and 422s on the 255-option / 2-10-level limits.
- `samples/decision/triage` (package `sample-decision-triage`),
  `brain = { features = ["decision"] }`, declared closure budget, runs without
  weights or exits cleanly saying what to pull.

`Noul` is TypeSafe's coinage, used for wire compatibility -> interface-compatibility
attribution in `/NOTICE.md`.

---

## 7. Data

Banking77 (Casanueva et al. 2020), **CC-BY-4.0**, from the upstream
`PolyAI-LDN/task-specific-datasets` CSVs: 10,003 train / 3,080 test / 77 categories,
mean 11.9 words, p95 29.

- Dataset -> `testdata/` (already fully gitignored, zero tracked files). Needs a
  **URL-download path added to `scripts/data/fetch-testdata.sh`**, which is
  deliberately mirror-only today; the deviation carries one line of justification in
  that script's header.
- MiniLM **weights -> the model store** via `brain pull
  sentence-transformers/all-MiniLM-L6-v2`, not testdata - that script's own stated
  rule, with only two grandfathered exceptions.
- Attribution in `docs/compliance/third-party-models.md`; citation in the dataset
  module's own docs.

Options carry **no descriptions** for run #1: Banking77 ships bare snake_case
categories, humanised to `"card arrival"`. The `description` field stays in the wire
format and the slot template - callers will send it - we simply synthesise none.

---

## 8. Plan

Each numbered item is one self-contained commit. Tests come from the spec and are
verified **red** before the implementation lands.

| # | Deliverable | Gate |
|---|---|---|
| 1 | `data::wordpiece` + `tools/goldens/wordpiece_dump_reference.py` | brain's ids == HF's ids on a corpus covering accents, CJK, punctuation, casing, OOV and >100-char words |
| 2 | `decide::{config,import,model}` - MiniLM encoder | tiny-config smoke, then the parity ladder (embeddings -> layer 0 attention -> layer 0 -> all layers -> pooled) against dumped goldens; `make parity` |
| 3 | Gradcheck for the encoder | FD check green on **CPU and GPU**; no dead gradients |
| 4 | B9/B10 windowed state encoding | single-window bit-identity; token count == input count; overlap correctness |
| 5 | `decide::head` - candidate-to-state cross-attention + scorer | gradcheck fwd+bwd on both backends; option-permutation invariance |
| 6 | `decide::primitives` - Choice/Score/Noul + confidence | limits (255 / 2-10), 0-based score expectation, `1 - H/ln k` on hand-computed cases |
| 7 | `data::banking77` + fetch path + splits + option sampler | 10,003/3,080/77 asserted; held-out split disjoint; sampler always includes gold |
| 8 | `decide::loss` - focal + Brier mix | `gamma=0, lambda=0` bit-identical to CE; loss gradcheck |
| 9 | `decide::train` - self-contained loop, discriminative LR | deterministic for a fixed seed; loss decreases on a fixed micro-run |
| 10 | `eval::calibration` | perfectly-calibrated synthetic -> ECE 0; maximally-overconfident -> ECE -> 1-acc |
| 11 | **Run #1** + the two experiments (4-way loss sweep, 4-way confidence statistic) | measured numbers asserted as floors in the tests that produce them |
| 12 | `decide_bench` + dispatch reduction + `perf gate` baseline | forward *and* backward profiled per kernel kind |
| 13 | Serving contract | manifest test; model reachable over `brain serve --dbus`; `examples/` script runs |
| 14 | `Domain::Decision`, SDK `DecisionPipeline`, `/v1/systemone` | `make check/sdk-features`; wire-format round-trip test |
| 15 | `samples/decision/triage` | `make check/samples`; closure within declared budget |
| 16 | Docs, NOTICE, compliance, AGENTS.md rows, .agents/knowledge/ entries | `make test/full` green, zero warnings |

## 9. Risks

1. **WordPiece normalisation edge cases.** `strip_accents: null` means "follow
   `lowercase`", i.e. *on* for an uncased checkpoint. Getting this wrong produces a
   tokenizer that is right on ASCII and silently wrong on everything else, which no
   accuracy metric would obviously catch. Goldens must include accented and CJK text
   from the start, not as a follow-up.
2. **Backend-specific silent-zero gradients.** A workgroup-barrier reduction with no
   barrier-free sibling can return all-zero gradients on the CPU backend only.
   Gradcheck runs on both backends, every time.
3. **Dispatch-bound small states.** The realtime case is the one the FLOP budget
   does *not* describe. Item 12 is not optional polish.
4. **D1 vs the headline number.** See section 3 - recorded up front.
5. **B9's lost cross-window context.** Real, and measured by B11's ablation rather
   than argued about.

## 10. Found defects

### 10.1 FIXED - `train_choices` then `save_head` on this arm

Fixed by the third option below: the caller chooses.
`DecisionPipeline::set_encoder_frozen(true)` (or `TrainSpec::freeze_encoder`)
trains the head alone and the run's head is writable;
the default is unchanged, still fine-tunes the encoder, and `save_head` still
refuses after it - which is what keeps this arm's published accuracies
meaning what they say. Gated by
`crates/sdk/tests/decision_pipeline.rs::a_frozen_encoder_run_can_save_its_head_and_a_fine_tuned_one_still_cannot`,
which holds BOTH directions: the default run's refusal names the encoder, and
the frozen run's file loads back and reproduces its answer.

The reproduction, kept because it is what the gate is written against:

1. **`train_choices` then `save_head` has never worked on this arm.** FIXED.
   `Decide::new_on` leaves `frozen_encoder = false`, `DecisionPipeline::
   train_choices` trains the encoder at `ENCODER_LR` on every step, and
   `Decide::save_head` refuses (correctly, and for a good reason - see its
   own doc) to write a head-only adapter for a model whose encoder moved. So
   `samples/decision/triage --save` and `samples/decision/intents` both
   reported their accuracy and then failed at the write. Reproduced
   2026-09-22 against the real `sentence-transformers/all-MiniLM-L6-v2`
   checkpoint.

   Three fixes were listed here and none chosen, because two of them change
   what the arm learns. The one taken is the third, which changes nothing
   about training: **when the encoder has moved, the artifact grows to carry
   it.** `Decide::save_model` / `DecisionPipeline::save_model` write a
   complete checkpoint DIRECTORY - `config.json`, `model.safetensors` holding
   the encoder AND head in this crate's own tensor names, and the
   `tokenizer.json` the model was loaded with - which
   `DecisionPipeline::builder(dir).load()` reads back through the same path
   a published checkpoint takes. `load_decide` tells the two apart by
   sniffing the tensor NAMES (`import::sniff_naming`), so a caller passes a
   path and never a format; `EncoderConfig::from_json_strict` refuses a
   config missing any field rather than defaulting it, and the head is
   checked against its manifest both ways.

   Deliberately NOT fixed in the Laya training milestone that found it,
   because every available fix is a real design decision for THIS arm rather
   than a mechanical repair, and two of the three change what the arm learns:

   * freeze the encoder in `train_choices` (cheapest, and what the Laya arm
     does) - but it changes this arm's measured accuracies, which are
     published numbers;
   * keep training the encoder and extend the artifact to carry it;
   * keep both and make the caller choose, so `--save` implies a frozen
     encoder and a trained-encoder run says so. **This is what was done** -
     see 10.1.
   `save_head` keeps its refusal unchanged, and `crates/sdk/tests/
   decision_persistence.rs` pins that it still refuses: a head-only file
   genuinely cannot reproduce a model whose encoder moved, and weakening
   that would trade a loud failure for a silent one. Prefer `save_head`
   when the encoder was frozen - 445k floats against ~23M.

   The Laya arm avoids the defect by construction: its trunk is frozen and
   `LayaDecision::save_head` applies the same refusal to the mode where it
   is not.
