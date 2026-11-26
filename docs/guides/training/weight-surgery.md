# Weight surgery: carving a smaller model out of a trained one

Weight surgery makes a trained checkpoint physically smaller. It removes the channels, layers, residual directions and
vocabulary rows the model needs least, and writes a checkpoint of the **same architecture in the same upstream
layout**, with a config that states every new width. The result is not a masked or factorised copy of the original:
its matrices are smaller, it loads in the upstream tooling unconverted, and brain serves it like any other checkpoint
of that architecture.

Surgery alone costs quality. The second half of the method, **recovery by distillation**, trains the cut model to
predict what the original predicts on the same data, with the original held frozen as the teacher. A cut plus a
recovery is one rung. Rungs can be chained: each new cut starts from the previous rung's recovered model, but is
measured against, and distilled from, the original.

This page explains how the edits work, how they are chosen, and how to run the whole procedure. Qwen3-TTS is the
architecture that declares a surgery graph today. The worked example at the end is a Swedish fine-tune of
Qwen3-TTS-12Hz-1.7B (1,916.7 M parameters) halved to 952.6 M at parity, and quartered to 469.6 M.

- [The pieces](#the-pieces)
- [The edits](#the-edits)
- [Choosing what to remove: calibration and scores](#choosing-what-to-remove-calibration-and-scores)
- [Running a cut](#running-a-cut)
- [Recovery by distillation](#recovery-by-distillation)
- [Validation](#validation)
- [A worked example](#a-worked-example)
- [Declaring a surgery graph for another architecture](#declaring-a-surgery-graph-for-another-architecture)
- [Pitfalls](#pitfalls)

## The pieces

| piece | where | what it does |
|---|---|---|
| surgery graph | `model::surgery::SurgeryGraph`, declared per architecture (`qwen3tts::surgery::graph`) | names every dimension that can be narrowed, every layer stack, every residual stream with the tensors that read and write it, every embedding that folds into a linear map, and every vocabulary with the config keys that hold token ids |
| edits | `modelopt::surgery` | architecture-generic operations on a checkpoint in memory, driven only by the graph |
| calibration | `modelopt::surgery::calibration`, `modelopt::surgery::scores` | statistics and curvature collected on real data, from which the edits rank what to remove |
| plan | `modelopt::surgery::plan::Edit`, a JSON array | which edits to apply, how far, and by which score |
| driver | `brain::SpeechSurgery` (SDK), `brain speech-surgery` (CLI), the served action `speech_surgery` (`brain/train`) | calibrates, applies a plan, measures each edit, exports the checkpoint |
| recovery | `speech_sft` with a `teacher` ([served training](served-training.md)) | distils the cut model from the frozen original |

## The edits

Every edit takes a checkpoint and returns a smaller one plus a `Change` record for the manifest. All edits share three
invariants:

1. **No new tensor kinds.** No edit leaves a factorised pair, an index table or a mask in the output. Whatever is
   written is expressible in the architecture's own config, so the upstream code builds exactly the tensors that are
   stored.
2. **Coupled tensors move together.** The graph lists every tensor indexed by a dimension. Removing channel `c`
   removes its lines from all of them at once.
3. **A masked twin exists for every removal.** Zeroing the right lines (`mask_channels`, `mask_layers`) computes the
   same function at the original shapes. The tests use this to prove that the narrow checkpoint and the masked wide one
   give identical logits.

Within one plan, edits always apply in this order, whatever order the JSON lists them in: trim, fold, select,
rotate, remove.

### `trim_vocabulary`: keep the tokens the domain uses

```json
{"trim_vocabulary": {"vocabulary": "text"}}
```

A text embedding over a multilingual tokenizer is large: 151,936 × 2,048 = 311 M parameters for Qwen3-TTS-1.7B. A
model serving one language needs a fraction of those rows.

- **What is kept.** The trim keeps the union of:
  - the 256 byte tokens, so every string stays encodable;
  - every added or special token;
  - every id the calibration and held-out shards store;
  - every id the shard texts and an optional `--corpus` encode to;
  - the BPE merge closure of all of the above. A kept token whose merge chain passes through a dropped token would
    otherwise become unreachable.
- **What is rewritten.** The kept ids are renumbered densely and the embedding rows sliced. The tokenizer files are
  rewritten to match: `vocab.json`, `merges.txt` and the added tokens in `tokenizer_config.json`. So is every config key
  that holds a token id (`im_start_token_id`, `tts_bos_token_id`, ...) and `text_vocab_size`.
- **Exactness.** The edit is exact for any sequence made of kept tokens. A text that needs a dropped merge still
  encodes, falling back to shorter pieces, but to a different id sequence. The report counts these cases:
  - `corpus_changed` covers the corpus texts and must be 0;
  - the probe set (`--probe`) measures how often unseen text is affected.

  Give the trim a corpus that covers what the model will read. In the worked example, the shard texts alone left 66
  of 300 evaluation sentences tokenised differently. Adding the full text inventory of the data run brought that to 0,
  at the cost of 11 M parameters.

### `fold_embedding`: fold an embedding into the map that reads it

```json
{"fold_embedding": {"fold": "talker.text", "keep": 1024, "score": "wanda"}}
```

When an embedding `E` feeds a square linear map `W1` before any non-linearity, only `E W1ᵀ` matters. The fold keeps
`k` output channels of `W1`. The table becomes `E W1_keepᵀ`, `W1` becomes the `k × k` identity, and the map after the
non-linearity keeps its matching columns.

- At full width the fold is exact.
- Narrower, it removes `(vocab + width) × (width − k)` parameters. It costs only the channels dropped, which are
  ranked by `score`.
- The config key that holds the hidden width (`text_hidden_size`) becomes `k`.

### `select_channels`: keep the most important channels of a dimension

```json
{"select_channels": {"dims": "talker.layers.*.mlp", "keep": 2560, "score": "fisher"}}
{"select_channels": {"dims": "talker.layers.*.kv_heads", "keep": 6, "score": "taylor"}}
```

`dims` is a glob over the graph's dimension names. Every matching dimension keeps the `keep` highest-scoring channels.
Dimensions that share a config key keep the same count, because the config can state only one width.

- **MLP channels.** A SwiGLU channel is one row of the gate projection, one row of the up projection and one column of
  the down projection, removed together.
- **KV groups.** A key/value group is removed with all the query heads that read it: `head_dim` rows of `k_proj` and
  `v_proj`, `heads_per_group × head_dim` rows of `q_proj`, and the matching columns of `o_proj`.
  `num_attention_heads` and `num_key_value_heads` scale together.
- **`head_dim` is never changed.** Rotary sections (mrope) are defined on it.

### `rotate_stream`: narrow the residual stream

```json
{"rotate_stream": {"stream": "talker", "width": 1536, "basis": "pca", "gain_tolerance": 0.25}}
```

This is the most powerful edit and the most delicate one. A residual stream of width `d` is rotated into an orthonormal
basis `Q` and cut to its first `k` directions:

1. Fold every RMSNorm gain on the stream into the linears that read the normed stream. A pure RMSNorm commutes with a
   rotation; a gain does not.
2. Collect the covariance of the stream on calibration data (`sample_rows` rows per point and batch). Take its
   principal directions (`basis: pca`), or the identity (`basis: identity`, which keeps the first `k` coordinates).
3. Multiply every writer of the stream by `Qᵀ` on its output side and every reader by `Q` on its input side. Then keep
   `k` directions.

For Qwen3-TTS the writers and readers include:
- the codec embeddings and codec head;
- the text projection's output and bias;
- the projection into the code predictor;
- the code predictor's codec embeddings, whose width is the Talker's;
- every layer's attention and MLP projections;
- the speaker encoder's output layer, if present.

At full width the rotation is exact, and the tests check that to floating-point tolerance. Cut, it removes about
`2 × d × (d − k)` parameters per layer plus every coupled table.

**Shared norms.** When the output of a norm and raw, un-normed stream vectors enter the same linear map, the norm's
gain cannot be folded away. Qwen3-TTS has this: the Talker's final norm output and the raw codec embeddings both go
through the code predictor's input projection. Then `diag(g)·Q = Q·diag(g′)` only if `Q` mixes channels of equal
gain.

A PCA basis on such a stream therefore mixes only channels whose gain is equal to within `gain_tolerance`, relative,
default 0. At 0 the rotation stays exact at full width. A wider tolerance mixes more channels and approximates the
shared norm. 0.25 measured best for the 1.7B Talker at width 1536: KL 0.84 against 1.78 for exact gain groups.
Knowledge entry 593 has the derivation.

**Equal widths.** Upstream Qwen3-TTS drops the code predictor's input projection when the Talker and the code
predictor are equally wide. The graph refuses a cut that would produce that state with a projection still present.
Narrow both streams together, or keep them unequal.

### `remove_layers`: drop whole layers

```json
{"remove_layers": {"stack": "talker.layers", "count": 4, "by": "kl", "greedy": true}}
{"remove_layers": {"stack": "talker.layers", "layers": [6, 8, 12, 25]}}
```

A layer writes the residual stream only through its output projections, so zeroing those makes it the identity. That
is the masked twin of removing it. Removal renumbers the remaining layers and cuts per-layer config lists such as
`layer_types`.

- `by` ranks layers by the KL from the unedited predictions without the layer (`kl`), by the loss (`loss`), or by
  `1 − cos(input, output)` of the layer (`cosine`, a cheap shortlist).
- **Use `greedy: true`.** Single-layer ablation costs do not add up: eight layers chosen by their individual KL lost
  to eight chosen by cosine. With `greedy`, layers are chosen one at a time, each measured with the previous choices
  already removed (`modelopt::surgery::greedy_removal`). See knowledge entry 596.

## Choosing what to remove: calibration and scores

Every ranking comes from running the model on real data of its domain: the speech shards the model was trained on, or
held-out shards of the same kind. Calibration collects, over `--records` records in batches of `--batch`:

- **Activation statistics.** The summed squares of every channel, and the residual-stream covariances for the
  rotation.
- **Curvature.** One gradient loop, shared with brain's precision search, collects a diagonal Fisher estimate of every
  weight and the group-Taylor gradient of a gate on every channel.

The channel scores (`score`):

| score | importance of channel `c` | notes |
|---|---|---|
| `act_rms` | RMS of the channel's activation | cheap; ignores what reads the channel |
| `wanda` | activation norm × magnitude of the reader's lines | |
| `fisher` | `½ Σ_j F_j w_j²` over the channel's weights | the loss increase of zeroing the channel, to second order |
| `taylor` | `E[(∂L/∂a_c)²]` for a gate `a_c` on the channel | captures interactions between the channel's tensors |

For MLP channels on the 1.7B Talker, keeping 3,072 of 6,144 channels gave a held-out KL of 0.341 with `fisher`, 0.380
with `taylor`, 1.42 with `act_rms` and 3.14 with `wanda`. Prefer `fisher`, then `taylor`.

Every measurement is taken on held-out records, as loss and as KL from the unedited predictions. The KL is reported
per head: Talker codebook 0 and the code predictor's residual codebooks. Measurements use only held-out records made
entirely of kept tokens, so a vocabulary trim does not bias them.

## Running a cut

### 1. Profile the edits alone

Before planning a size, measure what each edit costs on its own:

```bash
brain speech-surgery /models/my-tts \
  --calibration train.bttshrd --held-out validation.bttshrd \
  --plan candidates.json --out profile/ --profile --records 128 --held-out-records 128
```

With `--profile`, every edit in `candidates.json` is applied alone to the base and measured. No checkpoint is written,
and `profile/surgery.json` holds the table. A useful candidate list covers:
- FFN keep counts;
- KV groups;
- stream widths at a few gain tolerances;
- layer counts by each ranking;
- the code predictor's dimensions.

### 2. Search combinations at the target size

Parameter counts are exact and known before anything runs. Pick a few combinations that reach the target, run each
without `--profile`, and compare the reported pre-recovery KL. In the worked example, five combinations of FFN width
and layer count at about 950 M ranged from KL 0.86 to 1.08. Taking fewer layers out and more FFN channels was better.

### 3. Cut

```bash
cat > plan.json <<'EOF'
[
  {"trim_vocabulary": {"vocabulary": "text"}},
  {"select_channels": {"dims": "talker.layers.*.mlp", "keep": 2560, "score": "fisher"}},
  {"remove_layers": {"stack": "talker.layers", "count": 4, "by": "kl", "greedy": true}}
]
EOF
brain speech-surgery /models/my-tts \
  --calibration train.bttshrd --held-out validation.bttshrd \
  --corpus all-texts.txt --probe eval-texts.txt \
  --plan plan.json --out cut/ --records 128 --held-out-records 128 --id local/my-tts-950m-cut
```

`cut/` then holds:
- `checkpoint/`: the upstream-layout release, with tokenizer files, config and speech tokenizer;
- `surgery.json`: the manifest. It records the source and calibration digests, every edit with its method, the kept
  indices, the shapes and parameter counts before and after, and the loss and KL measured after each edit;
- `calibration.safetensors`: the statistics, for reuse.

The same run is available as the SDK builder and as a served action:

```rust
let outcome = brain::SpeechSurgery::new("/models/my-tts", "train.bttshrd")
    .held_out("validation.bttshrd")
    .plan_json(&std::fs::read_to_string("plan.json")?)?
    .corpus(texts)
    .out_dir("cut")
    .records(128)
    .held_out_records(128)
    .run()?;
```

The served action `speech_surgery` (`brain/train`, with `brain serve --training`) takes:
- `base`, `plan` (a JSON string), `dataset_digest` and `held_out_digest` (the shards in the artifact store);
- `records`, `held_out_records`, `batch_utterances`, `sample_rows`, `profile`, `exact` and `install_as`;
- the text blobs `corpus` and `probe`.

It returns the manifest and a `checkpoint_digest`, or installs the result as `local/<install_as>`.

### Chaining cuts

A plan may rotate each stream once. For a second rotation, or to cut a model that was itself cut, chain runs, with
each run's `checkpoint/` as the next run's base. Two flags keep a chained cut honest:

- `--reference DIR` measures every edit against the original model, not against the intermediate base. The ladder is
  steered toward the original's behaviour, and small errors cannot compound unseen.
- `--shard-tokenizer DIR` names the release whose tokenizer wrote the shards, when the base has a trimmed vocabulary.
  The shards' ids are then read through a map into the base's ids.

Starting a cut from the previous recovered rung measured clearly better than cutting the original directly. For the
470 M rung, cutting the recovered 950 M model gave KL 2.86 to 3.2, against 4.8 to 6.0 for fresh cuts of the original.

### Speaker projection

A narrowed Talker can no longer read the original speaker encoder's x-vectors: they are rows of the old width. A cut
that rotates the Talker stream therefore writes `speaker_projection.safetensors` next to the checkpoint, and training
and calibration read x-vectors through it. Named voices are stored as codec-embedding rows and move with the
rotation, so they need nothing extra.

## Recovery by distillation

The cut model is trained with `speech_sft`. Its own checkpoint is its base, and the original is the frozen `teacher`.
The full parameter table is in [served training](served-training.md#speech-fine-tuning); the parameters that matter here:

| param | value used | meaning |
|---|---|---|
| `base` | `local/my-tts-950m-cut` | the cut model, its own base |
| `languages`, `voices` | `{"swedish": {"new": false}, ...}`, every voice `{"new": false}` | rows the student already holds; they map to the teacher's by name |
| `teacher` | `local/my-tts` | the original, frozen, bf16, forward only, on every rank |
| `gt_weight` | 1 | cross-entropy on the recorded speech codes |
| `talker_kd_weight` | 1 | `T² · KL(teacher ‖ student)` on Talker codebook 0 |
| `mtp_kd_weight` | 0.02 | the residual codebooks' KLs, summed |
| `hidden_kd_weight`, `hidden_kd_layers` | 0.1, `"[[6,7],[12,15],[18,21],[24,28]]"` | MSE between student and teacher residual streams through a learned projection per pair; the projections are not exported |
| `kd_temperature` | 1 | |
| `keep_best`, `patience`, `eval_every`, `eval_batches` | true, 10, 100, 16 | the held-out distillation terms select the checkpoint |

Both models are teacher-forced on the same batch and the same histories. A student with a trimmed vocabulary reads the
shards through the text-token map built from both tokenizers. A student token the teacher lacks is refused at job
start. The step metrics and the result report `talker_kl`, `code_predictor_kl` and `hidden_mse`, along with their
held-out values.

What mattered in practice:

- **Learning rate.** Probe it. Short runs of 150 steps at the production batch put the optimum at `5e-5` for the
  first rung, above the `2e-5` of the original fine-tune. A pruned model tolerates and wants a larger step.
- **Patience.** Give it enough held-out batches and patience. With 8 held-out batches and patience 4, noise stopped
  the first recovery early.
- **Hidden-state matching.** The `hidden_kd` term closed the last gap to parity on the first rung. Without it, the
  Talker KL at the same step was 0.130 against 0.108 with it.
- **Teacher.** The teacher is always the original model, never the previous rung.

## Validation

A cut is finished when the recovered model passes the same evaluation the original passed, and it is compared clip
for clip with the original:

- every voice reads the same held-out sentences with the same seeds;
- a recogniser and a speaker encoder the training never touched judge both models.

Parity holds when the compact model is no worse than the original by more than fixed margins, all measured on the
same clips:
- mean CER;
- per-voice similarity;
- per-voice identification;
- failure rate.

A mean can hide one voice losing its timbre, which is why the per-voice checks are separate.

Check that the result loads in the upstream package as well as in brain. `export_edited` refuses a checkpoint that
brain's own readers would not accept, and `qwen3tts::surgery::check_upstream` refuses configs that the upstream code
would build differently from the stored tensors.

## A worked example

The source model is a Swedish fine-tune of Qwen3-TTS-12Hz-1.7B with twelve named voices. Its `model.safetensors`
holds 1,916.7 M parameters, and the 170.6 M speech codec is shared and untouched. Calibration and held-out records
are 128 each from its 420 h training shards.

**Rung 1, about 950 M**, at parity with the original:

| edit | parameters after | held-out KL to the original |
|---|---:|---:|
| `trim_vocabulary` (37,570 of 151,676 ids kept) | 1,682.5 M | 0 |
| `select_channels` Talker MLP keep 2,560 (fisher) | | |
| `remove_layers` Talker 4, by kl, greedy (layers 6, 8, 12, 25) | **952.6 M** | 0.864 |

The recovery ran 6 epochs with hidden-state matching, and the Talker KL fell to 0.078. On 4,800 paired FLEURS clips the
CER difference to the original was −0.0001 (95 % interval −0.003 to +0.003). Similarity and identification were
unchanged for every voice, and there were no failures.

**Rung 2, about 475 M**, cut from rung 1:

| edit | |
|---|---|
| `fold_embedding` text to 1,024 (wanda) | |
| `select_channels` Talker MLP keep 1,152 (fisher) | |
| `select_channels` Talker KV groups keep 6 of 8 (taylor) | |
| `rotate_stream` Talker to 1,536, PCA, `gain_tolerance` 0.25 | |
| `remove_layers` Talker 3, greedy | **469.6 M**, KL 2.86 before recovery |

Four recoveries reached CER +0.0125 at best. That model passes every release gate, but not the parity margin of
+0.0025: the residual-width cut is where the loss became hard to recover.

## Declaring a surgery graph for another architecture

The edits know nothing about Qwen3-TTS. To make another architecture cuttable, write a function from its config to a
`model::surgery::SurgeryGraph`, as `qwen3tts::surgery::graph` does. It declares:

- `ChannelDim`: a named dimension with its `width`; the `Lines` it owns in each tensor (tensor, axis, block size); the
  config keys that count it (`ConfigWidth`, with channels per unit); and `masks`, the tensors whose zeroing removes a
  channel's contribution exactly.
- `LayerStack`: a stack stored as `{prefix}{i}.{leaf}`; its count key; the `outputs` leaves through which a layer
  writes the stream; and the per-layer config lists.
- `Stream`: a residual stream's width and config keys, its writers (tensor and axis), its readers (with the norm in
  front of each), and its norms, marking a norm `shared` when its output shares a map with raw stream vectors.
- `EmbeddingFold`: an embedding, the linear map it feeds, and that map's hidden dimension.
- `Vocabulary`: its size key, its tables, and the config keys holding token ids.

The graph is a function of the config, so an edited checkpoint declares its own narrower graph. Add an upstream check
for any config the architecture's reference code would build differently. Then prove the declaration with the parity
tests in `crates/modelopt/tests/it/surgery.rs`, on a tiny synthetic config, before touching real weights:
- masked versus sliced logits;
- a full-width rotation;
- a vocabulary trim;
- an export/import round trip.

## Pitfalls

- **A tokenizer that disagrees with the reference.** Every component must encode text the way the reference
  tokenizer does, including its pre-tokenizer split. A loader without it once encoded 7.9 % of the training texts
  differently (knowledge entry 594). The trim's report also counts shard texts whose stored ids differ from what the
  tokenizer files encode.
- **Tiny calibration sets for layer choice.** Layer rankings from a few records are noisy. Use the same records for
  every candidate you compare.
- **Width cuts on a shared norm.** See [`rotate_stream`](#rotate_stream-narrow-the-residual-stream). Plain PCA on such
  a stream is lossy even at full width.
- **Expecting speed from width alone.** In Qwen3-TTS single-stream decode, the code predictor's 15 sequential steps per
  frame dominate the time per frame, not the Talker (knowledge entry 600). Halving the Talker barely moves latency, so
  cut the code predictor for that.
- **Recovering from the previous rung.** The teacher is always the original model. A rung that learns only from the
  rung above inherits and compounds its errors.
