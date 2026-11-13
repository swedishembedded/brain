# sample: forecast/forecast-client

Send a context series to `brain serve --dbus`, get a probabilistic forecast
back - end to end, over the generic `Run` method on
`com.swedishembedded.Brain1`, using a file descriptor for the bulk numeric
data (no per-model protocol). `forecast_client.py` works against all four
forecasting foundation models brain serves:

| model | store roles (pins) | input | output |
|---|---|---|---|
| `chronos2` | `chronos2.weights` | univariate series `[T]` | `[levels, horizon]` quantiles (21 levels) |
| `fincast` | `fincast.weights` | univariate series `[T]` (+`freq`) | `[horizon, 1+levels]` - col 0 mean, then 9 quantiles |
| `kronos` | `kronos.tokenizer` + `kronos.decoder` | OHLCV bars `[T, feat]` (or a univariate `[T]` close, expanded server-side) | `[horizon, feat]` sample bars |
| `timesfm3` | `timesfm3.weights` | univariate series `[T]` over this wire (natively multivariate via the library API - see `samples/shell/forecast/cooling-loop/`) | `[horizon, 9]` quantiles |

```bash
tools/dbus-session.sh --serve "--dbus --device cpu --pin chronos2.weights=/path/to/chronos2.safetensors" -- \
  python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/chronos2 --horizon 64
```

Feed a real series instead of the built-in synthetic one:

```bash
python3 samples/python/forecast/forecast-client/forecast_client.py --model brain/chronos2 --series my_series.txt
```

## What it demonstrates

* One wire format for four very different models: raw little-endian float32
  with an explicit `shape` in the blob meta - no per-model protocol. The
  context goes in as one input blob fd, scalar knobs (`horizon`, `freq`)
  ride in the params JSON, and the forecast comes back as an output blob fd
  whose meta carries `{shape, kind, levels}`.
* `chronos2` and `fincast` advertise an NPU footprint, so with an Intel NPU
  budgeted the scheduler **places them on the NPU automatically**
  (`place::pick_device`) - the returned `device` field says where it ran.
  `kronos` and `timesfm3` run on CPU/GPU today.
* Kronos needs both checkpoint dirs, but you do not have to name them: the
  model store resolves `NeoQuasar/Kronos-base` and
  `NeoQuasar/Kronos-Tokenizer-base` (`brain pull` fetches them); pin them
  only to override.

## What it needs

At least one model's checkpoint in the model store (or pinned with `brain
serve --pin`, see the table above) before `brain serve --dbus`. Plus
`jeepney` - `pip install -e brain-py`.

## Options

| flag | default |
|---|---|
| `--model {brain/chronos2,brain/fincast,brain/kronos,brain/timesfm3,brain/mock}` | `brain/chronos2` |
| `--horizon N` | `64` |
| `--context N` | `256` - synthetic context length (ignored with `--series`) |
| `--freq N` | `0` - fincast frequency bucket (0 daily / 1 weekly / 2 monthly) |
| `--series PATH` | unset - file of whitespace/comma-separated f32 (else synthetic) |
| `--bus {SESSION,SYSTEM}` | `SESSION` |
