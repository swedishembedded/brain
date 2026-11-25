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
