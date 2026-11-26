# Sten-TTS compression: halving ladder from 1.9B to ~60M

Scope: Weight surgery, calibration, distillation and validation that carve progressively smaller Sten-TTS variants out of the trained checkpoint.

Goal (user, 2026-11-25): progressively smaller Sten-TTS variants, each about half the previous one's parameters,
carved out of the trained Sten-TTS-1.7B by weight editing plus speech distillation from it. Each variant is
published to `swedishembedded/sten-tts` (its own subfolder) only after full validation. The ladder stops at about
50 M or at diminishing returns (a halving that does not reach parity after its recovery budget).

Brain owns the weight surgery, calibration, training and numerical validation. sten-tts (the recipe repo) owns
the stage that calls brain, the evaluation, the cards and publishing. An orchestrator only spreads the jobs over cards.

## Inventory (phase 0, measured 2026-11-25 on the published checkpoint)

model.safetensors holds 1,916.7 M parameters, all BF16. The speech tokenizer (codec) holds 170.6 M more, is shared and
untouched, and is counted separately.

| Component | Parameters | Shape notes |
|---|---:|---|
| talker.model.layers (28) | 1,409.4 M | 50.3 M/layer: q 2048x2048, k/v 1024x2048, o 2048x2048, MLP 3 x 2048x6144 |
| talker.model.text_embedding | 311.2 M | 151,936 x 2048 (text_hidden_size) |
| code_predictor.model.layers (5) | 78.7 M | hidden 1024, FFN 3072, 16 heads / 8 kv, head_dim 128 |
| code_predictor.model.codec_embedding (15) | 62.9 M | 2048 codes x 2048 (the talker width) |
| code_predictor.lm_head (15) | 31.5 M | 2048 x 1024 |
| talker.codec_head, talker.model.codec_embedding | 12.6 M | 3072 x 2048 each |
| text_projection fc1 + fc2 | 8.4 M | 2048 -> 2048 -> 2048, SiLU between |
| small_to_mtp_projection | 2.1 M | 2048 -> 1024 |

The ladder targets, all in model.safetensors parameters: ≤ 958 M, ≤ 479 M, ≤ 240 M, ≤ 120 M and ≤ 60 M.

## Constraint: upstream layout only

Every variant must load in the upstream `qwen-tts` package and in brain directly from its published files. That
means no remap tables and no factorised linears unknown to the upstream code (see the `feedback-hf-out-of-the-box`
memory). An edit is allowed only if its result is expressible in the upstream config:

* `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads` and `num_key_value_heads` of the
  talker and the code predictor;
* `text_vocab_size` and `text_hidden_size`;
* the tokenizer files (`vocab.json`, `merges.txt` and the special-token ids in `config.json`), rewritten for a
  trimmed vocabulary.

The text projection cannot be folded, because upstream always applies it. Low-rank factorisation is used only to
initialise a narrower dimension, never stored factored. `head_dim` stays at 128 because of the mrope sections.

## Edits (brain `modelopt` surgery, architecture-generic; qwen3tts declares its dependency graph)

1. **Text vocabulary trim.**
   - Keep the 256 byte tokens, every special token, and every token the Swedish corpus and a large Swedish text
     sample produce, plus their BPE merge closure.
   - Renumber the kept tokens densely, rewrite vocab.json, merges.txt and the special ids, and slice the embedding
     rows.
   - Measure how often a token sequence changes.
2. **text_hidden_size reduction.** The embedding goes into fc1, which is linear before the SiLU, so `E W1ᵀ` is
   factorised and fc1 absorbs the basis.
3. **Layer removal** (talker and MTP). Rank layers by measured ablation: teacher-forced loss and KL to the teacher on
   calibration shards, with cosine of residual states as a shortlist. Renumber the layers and rebuild the config.
4. **FFN channel selection.** The candidate scores are activation RMS, Wanda, the existing Fisher and group Taylor.
   Each removed channel goes from gate, up and down together.
5. **Attention head selection**, removing whole KV groups together with their query heads.
6. **Residual width reduction**, SliceGPT style.
   - Fold the RMSNorm weights into the following linears.
   - Rotate by the PCA of calibration residual states and keep the top-k directions.
   - The coupled tensors are the codec embeddings, codec_head, text_projection fc2 out, small_to_mtp in, and the
     MTP codec embeddings (whose width is the talker width).
7. **Same edits for the MTP** (hidden, FFN, layers), measured separately, because acoustic detail lives there.

Each edit writes a manifest. It records:
- the source digest and calibration digest;
- the scoring method and the kept indices;
- the shapes and parameter counts before and after;
- the measured loss and KL before recovery.

Parity first: a masked model and a physically sliced model must produce identical logits on tiny synthetic
configs, and an export/import round trip must be exact.

## Recovery (speech_sft distillation)

`speech_sft` gains a teacher. The frozen Sten-TTS-1.7B runs in the loop on the same batch: no cache, the teacher
is 3.8 GB. The loss is:

  L = λ_gt · CE(ground-truth codes) + λ_t · T² KL(teacher ‖ student) on codebook 0
    + λ_m · Σ_k KL on the 15 residual codebooks + λ_h · optional hidden-state MSE through a learned projection.

- Training covers the v2 shards (420 h) on several cards and servers through an orchestrator, as v2 did.
- The export path handles changed shapes: the student checkpoint is its own base.
- The original Sten is always the teacher. The previous rung only initialises the next student.

### Distillation in `speech_sft` (landed 2026-11-25)

The params, as the served action takes them (flat beside the existing ones): `teacher` (model-store id),
`gt_weight` (1), `talker_kd_weight` (1), `mtp_kd_weight` (0.02: `residual_weight` 0.3 over 15 codebooks; the residual
term is a sum over the codebooks of each one's frame mean), `hidden_kd_weight` (0) with `hidden_kd_layers`
(`[[student, teacher], ...]`, residual stream entering the layer), `kd_temperature` (1). The objective-level config
(`SpeechSftConfig`) takes the same names, `teacher` as a directory. Held-out selection (`patience`, `keep_best`) is the
distillation terms alone when a teacher is named. Design and contracts: `crates/qwen3tts/src/speech_sft/distill.rs`.

- The teacher is a forward-only build at bf16 linears on each rank's card (`Qwen::new_frozen_on`), trimmed to the
  student's rows (#592); student and teacher computed layouts are checked equal each batch.
- Student text tokens map to the teacher's by token string (both tokenizers' `vocab.json` plus added tokens), checked
  at start with every shard token; languages and voices map by name, so a student may hold them at other rows.
- Across widths only named-voice records distil (#591).

Measured 2026-11-25 on n460 (GH200, CUDA backend, NCCL), student = Sten with its first 8 of 28 Talker layers
(0.91 B), synthetic shard (400 records, mean 82 frames), batch 8, 16 steps, mean past step 3:

| run | frames/s | device peak |
|---|---:|---:|
| one card, no teacher | 3,593 | 28.7 GB |
| one card, teacher (KL terms) | 2,557 (0.71x) | 47.8 GB |
| one card, teacher + 2 hidden pairs | 2,509 | 48.2 GB |
| two cards, no teacher | 6,929 | - |
| two cards, teacher | 5,020 (0.72x) | - |

Open:
- The teacher build allocates a training build's activation and backward scratch (inference builds are not sized
  for forward-only batched use); most of the ~19 GB the teacher adds is that scratch. A forward-only batched sizing
  would cut it.
- The student's KD logits are a second head product beside the head's own cross-entropy pass (a few percent of a
  step); a dense soft-target mode of `HeadCe` would fuse them.
- The distillation's forward (`speech_sft.kd_fwd`: gathers, the student's head products, `kd_kl_rows` and the
  read-back of the KLs) took a median 16.8 ms of a ~250 ms step in nsys on the run above (10 steps). `kd_kl_rows` is
  one thread per row (core-compute); a workgroup-per-row variant behind `workgroup_reductions` is the next step if a
  kernel profile puts it first.
- Distilling x-vector (cloning) records across widths needs a rotated speaker encoder (#591).

## Surgery: what landed and what it measured

`modelopt::surgery` edits along the graph `qwen3tts::surgery` declares; `qwen3tts::speech_sft::calibrate` measures;
`brain::SpeechSurgery` composes them (served as `speech_surgery`, CLI `brain speech-surgery BASE --calibration SHARD
--plan FILE --out DIR [--held-out SHARD] [--corpus FILE] [--probe FILE] [--profile] [--exact]`). The plan is a JSON
array of `trim_vocabulary`, `fold_embedding`, `select_channels` (`talker.layers.*.mlp`, `*.kv_heads`, also for
`code_predictor.layers.*`), `rotate_stream` (`talker`, `code_predictor`; `pca` or `identity`, `gain_tolerance`) and
`remove_layers` (`count` + `by: kl|loss|cosine`, or `layers`). Scores: `act_rms`, `wanda`, `fisher`, `taylor`.

Parity on the teacher (n460, CUDA, exact fp32, 64 validation records, unedited loss 2.6726, KL per position):
trimmed vocabulary 0 / 0 (Talker / code predictor), full-width PCA of the Talker stream within equal final-norm gains
1.5e-11 / 4.6e-11, full-width MLP selection, text fold and code-predictor rotation 0. A plain PCA of the Talker stream
was exact for the Talker and 1.476 for the code predictor (#593). The trim keeps 31,789 of 151,676 ids (-246 M
parameters) from the 194,769 training-shard texts; 121 of 2,451 held-out texts (validation shard plus the evaluation
sentences) tokenise differently, and 2,096 of 2,151 validation records hold only kept tokens. A large Swedish text
sample beyond the shard texts was not on disk: the recipe passes it as `corpus`.

Sensitivity of single edits on the unedited teacher (64 train records calibrate, 64 validation records measure,
tensor cores; unedited loss 2.5960; KL = Talker + 0.3 code predictor, before any recovery):

| edit | params removed | loss | KL | Talker KL | CP KL |
|---|---:|---:|---:|---:|---:|
| text fold to 1024 (wanda) | 160.8 M | 2.809 | 0.231 | 0.192 | 0.128 |
| text fold to 512 (wanda) | 240.5 M | 3.907 | 1.357 | 1.207 | 0.502 |
| Talker MLP 75 % (wanda) | 264.2 M | 3.476 | 0.873 | 0.713 | 0.535 |
| Talker MLP 50 % (wanda) | 528.5 M | 5.714 | 3.139 | 2.760 | 1.262 |
| Talker MLP 50 % (act_rms) | 528.5 M | 3.996 | 1.424 | 1.196 | 0.760 |
| Talker MLP 50 % (fisher) | 528.5 M | 2.903 | 0.341 | 0.241 | 0.331 |
| Talker MLP 50 % (taylor) | 528.5 M | 2.943 | 0.380 | 0.258 | 0.409 |
| Talker KV groups 6 of 8 (taylor) | 88.1 M | 2.736 | 0.151 | 0.090 | 0.204 |
| Talker KV groups 4 of 8 (taylor) | 176.2 M | 3.219 | 0.656 | 0.478 | 0.596 |
| Talker width 1792 (pca, equal gains) | 186.4 M | 2.968 | 0.409 | 0.308 | 0.337 |
| Talker width 1536 (pca, equal gains) | 372.8 M | 4.342 | 1.777 | 1.509 | 0.892 |
| Talker width 1536 (pca, gain tolerance 0.05) | 372.8 M | 3.597 | 1.035 | 0.843 | 0.640 |
| Talker width 1536 (pca, gain tolerance 0.25) | 372.8 M | 3.408 | 0.842 | 0.667 | 0.584 |
| Talker width 2048 (pca, gain tolerance 0.25) | 0 | 2.597 | 0.001 | 0.000 | 0.003 |
| Talker width 1536 (identity) | 372.8 M | 4.758 | 2.196 | 1.895 | 1.001 |
| Talker width 1280 (pca, equal gains) | 559.2 M | 6.448 | 3.878 | 3.408 | 1.565 |
| remove 4 Talker layers (ablation kl) | 201.3 M | 2.831 | 0.255 | 0.171 | 0.279 |
| remove 8 Talker layers (ablation kl) | 402.7 M | 4.725 | 2.167 | 1.906 | 0.870 |
| remove 8 Talker layers (cosine) | 402.7 M | 4.233 | 1.676 | 1.421 | 0.851 |
| remove 12 Talker layers (ablation kl) | 604.0 M | 6.040 | 3.487 | 3.074 | 1.380 |
| CP MLP 75 % (wanda) | 11.8 M | 2.957 | 0.372 | 0 | 1.240 |
| CP MLP 50 % (wanda) | 23.6 M | 3.111 | 0.530 | 0 | 1.766 |
| CP KV groups 4 of 8 (taylor) | 15.7 M | 2.995 | 0.411 | 0 | 1.369 |
| CP width 768 (pca) | 28.1 M | 2.816 | 0.226 | 0 | 0.753 |
| CP width 512 (pca) | 56.1 M | 3.134 | 0.549 | 0 | 1.831 |
| remove 1 CP layer (ablation kl) | 15.7 M | 2.958 | 0.367 | 0 | 1.224 |
| remove 2 CP layers (ablation kl) | 31.5 M | 3.791 | 1.207 | 0 | 4.023 |

Read from it: Fisher (then group Taylor) is the channel score to use, the Wanda form here (activation norm times the
down projection's column magnitude) is the worst; per-layer ablation KL does not add up for joint removal (8 layers
by individual KL lose to 8 by cosine); a gain tolerance near 0.25 is the better Talker rotation; the code predictor is
small and sensitive. A chained 1.07 B checkpoint (trim, fold 1024, MLP 4608 by wanda, width 1792, minus 4 layers;
KL 3.27 unrecovered) exported in the upstream layout loads and speaks in brain (`design`, Vulkan) and in the upstream
`qwen-tts` package unmodified; unrecovered, both run long past the sentence.

## Validation (a variant is published only if it passes all of these)

- **The full sten-tts `evaluate` stage** and all six release gates.
- **Parity with Sten-TTS-1.7B** on the same clips (same seeds, FLEURS test, code-switching, numbers):
  - FLEURS CER at most 0.0025 above the teacher's;
  - voice similarity at most 0.01 below, per voice;
  - nearest-voice identity accuracy at least the teacher's minus 0.02;
  - failure rate no higher than the teacher's + 0.005.

  A rung that misses parity after its recovery budget is the point of diminishing returns. It is reported, not
  published as a parity model.
- **Upstream `qwen-tts` loads the variant** from its subfolder and speaks.
- **Card numbers** cover size, speed (tokens/s and real-time factor on a GH200) and the parity table.

## Control

The upstream Qwen3-TTS-12Hz-0.6B-Base, fine-tuned and distilled from Sten with the same budget, is compared against
the ~0.48 B rung. That rung must beat or match it, or the weight-editing path has not paid off.

## State

- 2026-11-25: inventory done. All 12 GH200 (n80, n105, n460) currently host the manual-to-competence brain
  servers. The surgery engine and its parity tests run on CPU first.
- 2026-11-25: weight surgery landed (see "Surgery: what landed and what it measured"): every edit of the plan,
  calibration, the served action and the CLI; teacher parity exact; the single-edit sensitivity profile above.
  Open: scoring the text
  fold and the code predictor by Fisher; a larger Swedish corpus for the trim; `qwen3::slice` duplicates the stream
  rotation for chat models and should become a declaration over `modelopt::surgery`; an SDK-level CPU test of
  `SpeechSurgery` on a tiny release (the pipeline is covered by the modelopt parity tests and the teacher runs).
- 2026-11-26: **rung 1 (≤ 958 M) reached parity: Sten-TTS-950M**, 952,628,480 parameters in model.safetensors
  (codec 170,557,441 apart), recipe `sten-tts/recipe/sten-tts-950m.yaml` (sten-tts `5dc8e4d`), brain `100eeff22`.
  - Cut (`speech_surgery`, 128/128 records): trim to 37,570 ids (training texts, the corpus inventory's texts and both
    shards' stored ids), Talker MLP 2560 by Fisher, 4 Talker layers removed greedily (8, 12, 6, 25). Held-out loss
    2.609 -> 3.435, KL 0.864 (Talker 0.659, code predictor 0.686). Plan sweep and the greedy/one-shot comparison:
    knowledge 596. Narrowing MLPs beat removing layers at this size.
  - Recovery (`speech_sft` distillation, 4 GH200 on one server, 16 utterances a card x 2 accumulation = 128 a step,
    lr 5e-5 cosine; lr probes of 150 steps at batch 128: 1e-5 0.316, 2e-5 0.283, 3e-5 0.272, 5e-5 0.268, 7e-5 0.276,
    1e-4 0.295, 2e-4 0.408, 4e-4 0.839 held-out distillation loss):
    r1 4 epochs, patience 4 on 128 records: stopped by noise at step 3400 (best 3000, held-out 0.179); FLEURS CER
    +0.0107 to the full model (fails parity 0.0025). r2 the same with patience 10 on 256 records: ran 6088 steps,
    held-out 0.149, CER +0.0031 (fails narrowly, CI +0.0006..+0.0053). r3 6 epochs plus hidden-state KD 0.1 on pairs
    [6,7] [12,15] [18,21] [24,28]: best step 9000 of 9132, Talker KL 0.078, code predictor KL 1.75; **every release
    and parity gate passes**: CER -0.0001 (CI -0.0031..+0.0030), worst voice similarity drop 0.0000, identity drop 0,
    failure delta 0, upstream qwen-tts loads. r4 (r3 without hidden KD, same schedule, stopped at 3900 as an
    ablation): Talker KL 0.130 vs r3's 0.108 at step 3000, so the hidden term carries the gain.
  - Not gated and stated in the card: code-switching CER 0.078 against the full model's 0.062; numbers 0.141 vs 0.157.
  - Speed, one request at a time on one GH200 through the API: RTF 0.123 vs 0.134 (101.5 vs 93.0 frames/s), only
    1.09x: single-stream decode is bound by the sequential per-frame steps (24 vs 28 layers, 15 code-predictor steps),
    not by the MLP width. Training ran ~3.3k frames/s a card with the teacher beside it (about 5 % of bf16 peak by a
    FLOP estimate); the teacher's training-sized scratch and the second KD head product (see Distillation, Open)
    remain the obvious next costs.
  - Fixed on the way: builds drew every overwritten weight (knowledge 595), surgery's host curvature and KL sums
    single-threaded, oversized stream results lost (597), shard ids of the teacher's tokenizer read as a trimmed
    student's (598), the trim did not keep the held-out shard's stored ids.
  - Open: the code-switching gap; batched serving throughput of the variant; rung 2 (≤ 479 M) starts from this cut's
    lessons (greedy layers, hidden KD, 256-record selection).
- 2026-11-26: **rung 2 (≤ 476 M) misses parity: the point of diminishing returns.** Recipe
  `sten-tts/recipe/sten-tts-475m.yaml`, brain `f32d7a9c2`.
  - Cut of rung 1 r3 (`compress.from`), measured against the full model (`reference`, `shard_tokenizer`):
    fold the text projection to 1024, Talker MLP 1152 by Fisher, 6 of 8 KV groups by Taylor, Talker width 1536
    (PCA, gain tolerance 0.25), 3 layers removed greedily (6, 15, 19), giving 469,589,504 parameters. Held-out KL to
    the full model 0.118 (r3 itself) -> 2.858 (Talker 2.467, code predictor 1.300); loss 2.596 -> 5.409.
  - Search (KL before recovery, all 457-472 M): fresh cuts of the full model 4.80-5.98; cuts of r3 with 8 KV groups
    3.23-4.76, with 6 groups 2.86-3.16, with 4 groups 3.35-3.42. Width 1664 (fewer layers) 3.22. Gain tolerance 1.0
    3.34 (worse than 0.25). MLP FFN cut in the code predictor 4.91. A fold scored by Fisher is not supported (no
    curvature for the fold).
  - Recoveries (4 GH200 each, 16 x 2 x 4 = 128 a step, hidden KD 0.1 on [5,5] [10,14] [15,20] [21,28] unless said):
    A lr 5e-5, 6 epochs: FLEURS CER +0.0179. B lr 1e-4, 6 epochs: +0.0136 (best held-out Talker KL 0.184).
    C continues B for 6 more epochs at 5e-5: +0.0125 (CI +0.0103..+0.0150). D the same with hidden KD 0.02: +0.0154
    (stopped at 8600). Parity needs ≤ +0.0025. Every other gate holds in all four: the six release gates, voice
    similarity and identity per voice (drops ≤ 0) and failures (0). Twelve epochs instead of six gained 0.0011. The
    loss is in intelligibility, worst in code-switching (CER 0.119 against the full model's 0.062) and numbers.
  - Speed, one request at a time: RTF 0.109-0.111 against 0.132 (113-115 vs 94 frames/s), 1.2x. The code predictor
    holds about 5 ms of each frame and is untouched (knowledge 600).
  - Multi-node: the orchestrator's served job over n105+n460 (the one v2 trained with) built the job on both
    servers (68 GB each card) and then made no step for 15 minutes at 0 % utilisation, with nothing in the fleet log;
    the four recoveries ran as two parallel 4-card runs instead. Not diagnosed further (NCCL logs were not captured).
  - Fixed on the way: cutting a cut (shard tokenizer, reference measurement), x-vectors of a narrowed Talker
    (knowledge 599), a setup that projected every x-vector twice before the first step.
  - Open: a rung-2 recipe that reaches parity would need what the cut lost in intelligibility. Candidates: a longer
    or different-objective recovery (higher `talker_kd_weight`, a ground-truth weight below 1), keeping the Talker at
    2048 wide with stronger FFN/KV cuts, or a looser rung (≈ 600 M).
