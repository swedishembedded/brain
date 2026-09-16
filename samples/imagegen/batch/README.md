# sample: imagegen/batch

One pipeline load, many prompts - the shape a service actually has.

```bash
printf 'a whale submarine surfacing at dawn\na lighthouse in heavy fog\n' > prompts.txt
make samples/imagegen/batch/run ARGS="--prompts prompts.txt --out-dir out/batch"
```

## What it demonstrates

* **Load once, generate many.** `ImagePipeline` holds multi-gigabyte device
  memory for its whole lifetime; building one per request is what makes an
  image feature unaffordable. The sample prints the load time separately from
  each per-image time so the difference is visible, not asserted.
* Reproducibility: `--seed N` gives image *i* the seed `N + i`, so a run
  repeats exactly and no two images collide.
* Input hygiene: blank lines and `#` comments in the prompt file are skipped -
  every one of them would otherwise cost a full denoise.

## What it needs

Real weights, same as `imagegen/generate`:

```bash
brain pull black-forest-labs/FLUX.2-klein-4B
```

## Options

| flag | default |
|---|---|
| `--prompts FILE` | *required* |
| `--model ID` | `black-forest-labs/FLUX.2-klein-4B` |
| `--out-dir DIR` | `out/sample-imagegen-batch` |
| `--seed N` | model default |
