<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 573. A speech route keyed on an action name reached no real model: declare the contract, and check the checkpoint kind

**Lesson:** `/audio/speech` chose its model by an action literally named `speak` with a `text` param. When Qwen3-TTS
moved to `synth`/`clone`/`design` and CosyVoice's one action was `synth`, nothing matched any more, and the only
`speak` left (Qwen3-Omni's) answers a prompt rather than speaking a given text and has no `text` param: the route
served the test double and no real model, while its tests stayed green. A route that maps a generic request onto
model actions needs a typed declaration on the action, not a name: `capability::speech::SpeechContract` says where the
words, the reference recording and its transcript, the instruction, the preset voice, the length cap and the seed go,
and the catalog test `every_speech_contract_names_what_its_action_declares` holds every name to the action's spec and
requires each text-to-speech model to declare one.

The same review found the converse failure inside Qwen3-TTS: each released checkpoint is trained for one way of
choosing the voice (`tts_model_type` base / custom_voice / voice_design in `config.json`), and `design` ran on any of
them. On the Base checkpoint an instruction produced ordinary speech with a 200, as if it had been followed. The
reference package refuses a mismatched kind, except that its 0.6B CustomVoice drops an instruction without a word;
brain refuses that case too (`TtsSpecials::check_steering`, tagged `unsupported-capability`).

## Evidence (2026-11-24)

- Before: no served manifest had an action named `speak` with a `text` param except the test double, so
  `GET /v1/models` listed no real speech model. After: `brain serve --openai` with Qwen3-TTS 0.6B Base lists
  `brain/qwen3tts` with three speech modes.
- Real-weight run over HTTP (CPU, the 0.6B Base, one reference sentence): ECAPA cosine to the reference 0.944 for
  plain synthesis, 0.984 for an x-vector clone, 0.982 for an in-context clone; two requests with `seed` 7 returned
  byte-identical WAVs.
- `design` with `instructions` on the Base checkpoint returned 200 and audio before the check, and is refused now
  (qwen3tts `design_runs_only_where_the_checkpoint_follows_an_instruction`,
  `each_checkpoint_kind_serves_only_the_steering_it_was_trained_for`).
- The rule's source: `qwen_tts` 0.1.1 `inference/qwen3_tts_model.py` (`generate_voice_clone`,
  `generate_voice_design`, `generate_custom_voice` and its `0b6` instruct drop).
