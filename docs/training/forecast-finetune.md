# Forecast fine-tuning (Chronos-2 / Kronos)

brain can adapt a released time-series forecasting model (the Chronos-2 or
Kronos family) to your own market data via LoRA, producing a checkpoint
tuned to recent behavior in the instruments you care about.

Fine-tuning these forecasting models runs on CPU or GPU only — it is not
NPU-capable. The NPU export path is inference-only and can't backpropagate,
so a fine-tune always happens on CPU/GPU first; the resulting checkpoint
remains NPU-exportable afterward for fast serving.

## Input data

Point the fine-tuner at a directory of your own OHLCV (Open/High/Low/Close/
Volume) CSVs, one file per instrument, with columns:

```
Date,open,high,low,close,volume
```

The fine-tuner builds `(context → horizon)` windows across the whole
universe, splits them temporally with an embargo/purge gap so no future bar
leaks into a training window, and normalizes using only past data. You
supply raw daily bars; the leak-safety handling is done for you. More
liquid instruments and more history per instrument make for a better
fine-tune — breadth across many instruments matters as much as depth in any
one of them.

## Running a fine-tune

```
brain forecast finetune \
    --kronos-tokenizer <tokenizer-dir> \
    --kronos-decoder   <decoder-dir> \
    --data <your-ohlcv-dir> \
    --context 180 --horizon 5 --epochs 8 --lr 4e-5 \
    --lora 8 \
    --out <output-checkpoint>.safetensors
```

- `--kronos-tokenizer` / `--kronos-decoder` — local paths to the released
  Kronos tokenizer and decoder checkpoints.
- `--data` — your directory of OHLCV CSVs.
- `--context` / `--horizon` — the window shape (history length / forecast
  length).
- `--epochs` / `--lr` — training length and learning rate.
- `--lora RANK` — LoRA rank; omit for a full-parameter fine-tune.
- `--out` — where to write the resulting checkpoint, if the fine-tune is
  promoted (see below).

The tokenizer stays frozen throughout — only the decoder is fine-tuned.

## The promotion gate

A fine-tune is only kept if it demonstrably beats the un-tuned base model,
out-of-sample, on held-out (embargoed) data. The command prints its
decision:

```
gate: base_val 3.2140 → ft_val 3.1985  (41230 steps)  ⇒  PROMOTE (fine-tune beats base out-of-sample)
promoted checkpoint → <output-checkpoint>.safetensors
```

If the fine-tune does not beat the base, no checkpoint is written and the
base model continues to serve. Many runs will not beat the base — that is
correct, expected behavior, not a failure of the tool.

## Backtesting a checkpoint across your whole universe

`tools/forecast/full_backtest.sh` runs the same walk-forward evaluation the
promotion gate uses, but across every instrument in a directory at once —
useful for checking a checkpoint (promoted or not) more thoroughly than the
gate's own single summary number before you trust it:

```
tools/forecast/full_backtest.sh --data <your-ohlcv-dir> --checkpoint <checkpoint>.safetensors
```

## Honesty notes

- Evaluate strictly on data **after** your training cutoff — contamination
  with the pretraining window is possible on older data.
- Beating the base model in a backtest is not a guarantee of future
  performance. The promotion gate uses a held-out embargoed split, but the
  cleanest read of whether a checkpoint actually helped is realized
  performance going forward, out of sample.
- This is not financial advice. A promoted checkpoint means "beat the base
  model out-of-sample in backtest," not "will make money."
