# sample: api/fim-completion

Fill in the middle of code with a DeepSeek-Coder base model over brain's
OpenAI-compatible `POST /v1/completions`: the code before the cursor is the
`prompt`, the code after it the `suffix`, and the completion is what goes
between.

```bash
brain pull deepseek-ai/deepseek-coder-1.3b-base
python3 samples/python/api/fim-completion/fim_completion.py
```

That launches (and stops) its own `brain serve --openai`. Point at a server
you already launched instead:

```bash
brain serve --openai 8788 --api-keys-out /tmp/keys.json &
python3 samples/python/api/fim-completion/fim_completion.py --base-url http://127.0.0.1:8788 --keys-file /tmp/keys.json
```

## What it demonstrates

* `suffix` framed with the checkpoint's own fill-in-the-middle tokens:
  DeepSeek-Coder's `<｜fim▁begin｜>`/`<｜fim▁hole｜>`/`<｜fim▁end｜>`, or the Qwen
  vocabulary's `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`. brain picks
  them from the vocabulary; a model with neither answers `suffix` with a 400.
* A raw completion: no chat template, the text exactly as generated.

The `deepseek-coder-*-base` checkpoints are the ones trained for infilling;
`deepseek-coder-7b-base-v1.5` carries no FIM tokens.

## Options

| flag | default |
|---|---|
| `--model ID` | `deepseek-ai/deepseek-coder-1.3b-base` |
| `--max-tokens N` | `96` |
| `--base-url URL` | unset (self-launches a server instead) |
| `--api-key KEY` / `--keys-file PATH` | required with `--base-url` |
| `--brain PATH` | `./target/release/brain` (self-launch mode only) |
| `--port N` | `8788` (self-launch mode only) |
