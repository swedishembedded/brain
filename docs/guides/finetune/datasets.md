# Fine-tuning data — what to train on, and what we actually have

This answers three questions honestly: **is Kronos trained on synthetic or real
data**, **what data do we fine-tune it on**, and **do we have that data** — then gives
the exact commands to fetch fresh data and run a weekly fine-tune.

## Synthetic or real? — real market data, throughout

Kronos is **not** trained on synthetic data. There are three data stages; only the
last is ours, and all three are real:

| Stage | Data | Do we have it? |
|---|---|---|
| **Pretraining** (the released weights) | > 12 billion real K-line records, > 45 global exchanges, 7 granularities (`resources/time-series/docs/kronos.md` §3) | **No — and nobody outside the authors does.** The corpus is *not released*; only the weights are. We do not (and need not) reproduce pretraining. |
| **Reference fine-tune** (`repos/Kronos/finetune`) | Real Chinese A-share **CSI300** daily bars via **Qlib** (`config.py`: `qlib_data_path`, `region=REG_CN`, `instrument='csi300'`) | No — we don't use Qlib/CN. We substitute real **US** equities (below), which is a valid equivalent. |
| **Our fine-tune** (`brain forecast finetune`) | Real **US equity daily OHLCV** from Yahoo Finance, in trademiner's `stocks.db` | **Yes** — see inventory below. |

Where synthetic data *does* appear, and why it's not "training on synthetic":

- **Our gradcheck / from-scratch learning tests** (`crates/kronos/tests/train_gradcheck.rs`,
  `crates/kronos/src/finetune.rs`) use synthetic tokens **only to validate the backward
  pass math** — they prove gradients are correct, they are not the actual training.
- The broader TSFM field leans on synthetic generators (KernelSynth, Chronos-Mixup —
  Moirai 2.0's corpus is ~86% synthetic; `resources/.../datasets.md`). **Kronos does not,
  and neither do we.**
- The paper's "+22% generative fidelity for synthetic data generation" is about *using*
  Kronos to **generate** synthetic bars — the opposite direction.

## What the fine-tune needs

The signal is **cross-sectional** (rank a universe), so the data must be:

- **Breadth** — many liquid names in one universe (dozens to a few hundred). One stock's
  history is not enough.
- **Daily OHLCV per name.** `amount` is synthesized as `(open+high+low+close)/4 × volume`
  — identical to the reference (`qlib_data_preprocess.py`); `volume` and the calendar
  (minute/hour/weekday/day/month) come from the bars/dates. No extra feeds required.
- **Enough history** per name (≥ `context + horizon + a few windows`; default context 180).
  Recent bars are what actually adapt the model.
- **Leak-safe** — the pipeline builds `(context→horizon)` windows and splits them
  **temporally with an embargo/purge gap** (`forecast::train_data`), and normalizes
  past-only. You supply raw OHLCV; leakage is handled for you.

## What we have (inventory)

- **For fine-tuning: yes.** `applications/trademiner/stocks.db` — 560 US tickers of daily
  OHLCV from Yahoo, refreshable to today (the liquid set is fresh to the latest trading
  day). This is real market data and is exactly what the fine-tuner consumes. Plenty of
  breadth and history.
- **For pretraining: no** (undisclosed corpus, above). We **adapt** the released weights;
  we do not retrain from scratch. That is the whole design — a small, gated weekly
  adaptation, not a re-pretrain.
- The `resources/time-series` bundle catalogs large open corpora (LOTSA, GIFT-Eval,
  Time Series Pile, …), but those are **general** time series, **not financial OHLCV**,
  so they are not used for Kronos fine-tuning.

Bottom line: we have everything needed to **fine-tune** on real, recent US market data.
We do not have (and don't need) the original pretraining corpus.

## Commands — fetch the latest data

Refresh the local DB from Yahoo Finance (in trademiner), then export the training
universe to the per-ticker CSV directory the fine-tuner reads:

```bash
# 1. pull the latest bars into trademiner/stocks.db
cd applications/trademiner
make rank/update                 # == python3 dldata.py  (Yahoo → stocks.db)
#   or update just a set:  python3 dldata.py aapl msft nvda ...

# 2. export the most-liquid, fresh, long-enough names to training CSVs
cd ../edgeai/brain
python3 tools/export_ohlcv.py \
    --db ../../trademiner/stocks.db \
    --out out/train-csv \
    --max 150 --min-history 400 --fresh-only
#   → out/train-csv/<TICKER>.csv  (Date,open,high,low,close,volume)
```

`export_ohlcv.py` ranks by recent dollar-volume, keeps the top `--max`, and creates a
`Ticker` index on the DB the first time (safe, one-off, also speeds trademiner).

## Commands — fine-tune (run weekly)

Build the release binary once (`make build` / `cargo build --release`), then:

```bash
cd applications/edgeai/brain
./target/release/brain forecast finetune \
    --kronos-tokenizer /data/workspace/resources/time-series/checkpoints/kronos-tokenizer-base \
    --kronos-decoder   /data/workspace/resources/time-series/checkpoints/kronos-small \
    --data out/train-csv \
    --context 180 --horizon 5 --epochs 8 --lr 4e-5 \
    --lora 8 \
    --out out/kronos-decoder-$(date +%F).weights
```

What it does: enumerates leak-safe windows across the universe, splits them temporally
with an embargo gap, tokenizes with the **frozen** tokenizer, fine-tunes the decoder
(LoRA rank-8 here; drop `--lora` for full-parameter), and prints the gate decision:

```
gate: base_val 3.2140 → ft_val 3.1985  (41230 steps)  ⇒  PROMOTE (fine-tune beats base out-of-sample)
promoted checkpoint → out/kronos-decoder-2026-07-28.weights
```

**A checkpoint is written only if the fine-tune beats the base on the held-out
(embargoed) split** — the anti-overfit gate. Many weeks it will keep the base; that is
correct behaviour, not a failure. Point the ranking tool at the promoted decoder to use
it.

Flags: `--context`/`--horizon` (window shape), `--epochs`/`--lr` (the reference recipe
defaults to LR 4e-5, weight-decay 0.1, grad-clip 3.0), `--lora RANK` (0/omitted = full
fine-tune), `--embargo` (purge gap, default = horizon), `--out` (checkpoint path).

## Honesty notes

- We **adapt** released weights on real, recent US data with a promotion gate — we do not
  claim to reproduce the undisclosed pretraining.
- Evaluate on **post-cutoff** bars; contamination with the pretraining window is possible
  on older data. The gate uses a held-out embargoed split, but the cleanest read of "did
  this week's checkpoint help" is next week's realized RankIC (see the ranking tool).
- Not financial advice; a promoted checkpoint means "beat base out-of-sample in
  backtest," not "will make money."
