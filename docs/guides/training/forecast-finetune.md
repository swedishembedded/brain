# Forecast fine-tuning (Kronos / TimesFM-3)

brain can adapt a released time-series forecasting model (Kronos or
TimesFM-3) to your own market data, in full or via LoRA, producing a
checkpoint tuned to recent behavior in the instruments you care about. It is
the `finetune` action of `brain kronos` and `brain timesfm3`.

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

### Validation

Every file is validated before any weights are loaded, with the same checks
the `predict` action applies: the header names the six columns in order,
every row has six fields that all parse, dates are strictly increasing,
values are finite, prices are positive, `high >= max(open, close)`,
`low <= min(open, close)`, and the file is long enough for at least one
`context + horizon` window. Every rejection names the file and the 1-based
line inside it.

**A file that fails stops the run.** A fine-tune is a long job whose whole
output is one promote/keep verdict about one universe, so a universe that
quietly lost names would produce a verdict about an experiment you did not
ask for - and nothing afterwards would tell you which. Validation takes
milliseconds and happens before the checkpoint loads, so the refusal is
immediate.

If refusing the whole run is the wrong trade - a several-hundred-name vendor
dump with one bad row in one file - set `skip_invalid` (`--skip-invalid true`). It trains on the
files that parsed, still lists every rejection with its file and line, and
prints the loaded/rejected split either way:

```
finetune data: BAD.csv: csv: line 118: 5 fields, expected 6
finetune data: excluded QQQ.csv (QQQ is the benchmark index, not one of its constituents)
finetune data: /your/ohlcv-dir: 402 of 403 series loaded, 1 rejected
```

`holdout_data` is validated the same way, up front, so a typo there cannot
cost you the generalization report at the end of a finished run.

## Running a fine-tune

```
brain kronos finetune \
    --data <your-ohlcv-dir> \
    --context 180 --horizon 5 --epochs 8 --lr 4e-5 \
    --lora 8 --seed $RANDOM \
    --save <output-checkpoint>.safetensors
```

The tokenizer and decoder are resolved from the model store (`--tokenizer
DIR --decoder PATH` name them for one run). TimesFM-3 is the same command:
`brain timesfm3 finetune ...` (its `--weights PATH` names a checkpoint).

- `--data` — your directory of OHLCV CSVs.
- `--context` / `--horizon` — the window shape (history length / forecast
  length).
- `--epochs` / `--lr` — training length and learning rate.
- `--lora RANK` — LoRA rank; omit for a full-parameter fine-tune.
- `--seed` - seeds the LoRA adapters' initial factors; drawn when omitted and
  echoed in the result, so a run reproduces by passing it back. A full
  fine-tune draws nothing.
- `--embargo` - bars cleared either side of each calendar cut (default: the
  horizon).
- `--batch` - windows per optimizer step.
- `--activations keep|recompute` - what the backward keeps of the forward:
  every layer's activations (the default), or only each layer's input, rebuilt
  in the backward for the same gradients at lower memory. Kronos and
  TimesFM-3 both take it.
- `--save` - where to write the resulting checkpoint, if the fine-tune is
  promoted (see below).
- `--holdout-data` - a second directory of names never trained on, scored
  after the run to say whether the fine-tune generalizes.
- `--skip-invalid` - train on the files that passed validation instead of
  refusing the run (see [Validation](#validation)).

The action streams one progress line per optimizer step and stops at the next
step when cancelled (Ctrl-C on the CLI, a cancel over the serving surface);
a cancelled run writes nothing.

The tokenizer stays frozen throughout — only the decoder is fine-tuned.

## The promotion gate

A fine-tune is only kept if it demonstrably beats the un-tuned base model,
out-of-sample, on held-out (embargoed) data. The action reports its decision:

```
gate (INCLUDED names, held-out future): base_val 3.2140 -> ft_val 3.1985  (41230 steps)  =>  PROMOTE (fine-tune beats base out-of-sample)
promoted checkpoint -> <output-checkpoint>.safetensors
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
