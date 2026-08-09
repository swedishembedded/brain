# Kronos (`crates/kronos`)

A from-scratch Rust+WGSL reimplementation of Kronos — a financial
K-line/candlestick foundation model: a BSQ tokenizer (OHLCV bar → hierarchical
discrete tokens) plus an autoregressive decoder with a dual head, imported
exactly from `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base` and
parity-gated to cosine 1.0.

## Model id and weights

- **Id:** `brain/kronos` — reserved vendor `brain/`, never fetched.
- **Weights:** two separate directories, both required:
  - `BRAIN_KRONOS_TOKENIZER` — the `Kronos-Tokenizer-base` HF directory.
  - `BRAIN_KRONOS_DECODER` — the `Kronos-small` HF directory (or a
    `.safetensors` fine-tune checkpoint — see Training below).
- No `import` step: the decoder/tokenizer HF directories load directly.

## Surfaces

D-Bus (via `brain forecast serve`) and the `brain forecast compare|serve|finetune`
CLI only — no HTTP route (the action is named `forecast`, not `generate`, and
it requires an input blob so it isn't `text2image`-shaped either).

## Inference

### CLI

```bash
brain forecast compare --kronos-tokenizer <tok-dir> --kronos-decoder <dec-dir> [--windows 24] [--seed 1337]
brain forecast serve --kronos-tokenizer <tok-dir> --kronos-decoder <dec-dir> [--socket <path> | --listen <addr>]
```

There is no `brain kronos ...` subcommand — `forecast`/`finetune` are the CLI verbs.

### D-Bus

Action `forecast`: input `context` (OHLCV bars `[T,feat]` f32-LE, meta
`{shape}`, required; a univariate `[T]` close series is also accepted and
expanded to bars server-side), optional inputs `ctx_stamp`/`fut_stamp` (u32-LE
calendar stamps), params `horizon` (default 64), `temperature` (default 1.0),
`argmax` (default true), `seed`, `samples` (default 1), `checkpoint` (override
decoder path per request), output `forecast` (f32-LE, meta
`{shape:[horizon,feat], kind:"samples"}`). Runs on CPU/GPU by default; also
NPU-eligible via a two-graph KV-cache rollout.

```bash
BRAIN_KRONOS_TOKENIZER=kronos-tokenizer-base BRAIN_KRONOS_DECODER=kronos-small \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 2
    python3 examples/forecast/forecast_client.py --model brain/kronos --horizon 32
  '
```

Reference client: [`examples/forecast/forecast_client.py`](../../../examples/forecast/forecast_client.py)

## Training / Fine-tune / LoRA

The only forecast model with a CLI-reachable, gated fine-tune
(`crates/cli/src/forecast_cli.rs`, `finetune_universe`): trains the Kronos
decoder over a directory of `<TICKER>.csv` OHLCV files and writes a promoted
checkpoint **only if** it beats the base on a held-out, embargoed split.

Full fine-tune:

```bash
brain forecast finetune --kronos-tokenizer <tok-dir> --kronos-decoder <dec-dir> --data <csv-dir> \
  [--out kronos-decoder-ft.safetensors] [--holdout-data <csv-dir>] \
  [--context 180] [--horizon 5] [--epochs 8] [--lr 4e-5] [--embargo <horizon>] [--batch 1]
```

LoRA (rank-`r` adapters on the attention projections, base weights frozen;
`alpha` is fixed at `2*rank`, not a separate flag):

```bash
brain forecast finetune --kronos-tokenizer <tok-dir> --kronos-decoder <dec-dir> --data <csv-dir> --lora 8
```

## Not supported

QLoRA.

## See also

- Crate: `crates/kronos`
- Workstream ledger: [`status.md`](status.md)
- Umbrella page: [`../forecast/readme.md`](../forecast/readme.md)
