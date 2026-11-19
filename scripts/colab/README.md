# TimeSHAP per-variable run on a hosted GPU (Colab)

The explainer game this drives: one player per covariate **column** plus the four
history players (`hist_d1..d4`), payoff = the whole horizon vector, exact
enumeration of `2^N` coalitions, hour-of-day background. Nothing is trained and
nothing refits, so a coalition is one forward pass of a zero-shot checkpoint.

Measured forward cost at `--sampling 25` (London, 4329-point context, horizon 48)
on an RTX 3050 4 GB: Chronos2 0.14 s, Timestral 0.75 s per coalition. Scale from that:

| game | players | coalitions | Chronos2 | Timestral |
|---|---|---|---|---|
| covariates only, per column | 12 | 4 096 | ~10 min | ~51 min |
| full, per column + history | 16 | 65 536 | ~2.5 h | ~13.7 h |
| covariates only, per block (default game) | 5 | 32 | ~5 s | ~25 s |

At `--sampling 100` the context is four times longer, so multiply the per-forward
cost by roughly 20x (17 315-point context).

## Cells, in order

1. Runtime: **Runtime → Change runtime type → GPU**, then check it is really a GPU:

```bash
!nvidia-smi
```

2. Get the code. The `--per-variable` path exists only in the local working tree —
**push it first**, the GitHub copy does not have it:

```bash
!git clone https://github.com/cinnamoll/Forecast-Models-Evaluation.git /content/Forecast-Models-Evaluation
```

3. Persist results in Drive, because `/content` is wiped when the session ends:

```bash
from google.colab import drive; drive.mount('/content/drive')
```

4. Install (never installs torch — Colab already has a CUDA build):

```bash
!cd /content/Forecast-Models-Evaluation && bash scripts/colab/01_setup.sh
```

5. Each later cell must re-source the env file; exports do not survive between cells:

```bash
!source /content/timeshap_env.sh && bash /content/Forecast-Models-Evaluation/scripts/colab/02_run.sh --dry-run
```

6. Real run, one model per invocation, resumable at model granularity:

```bash
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && \
    MODELS=Chronos2 SAMPLING=25 PER_VARIABLE=1 MAX_PLAYERS=16 bash scripts/colab/02_run.sh
```

```bash
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && \
    MODELS=Timestral SAMPLING=25 PER_VARIABLE=1 MAX_PLAYERS=16 bash scripts/colab/02_run.sh
```

If a session dies mid-model, that model restarts from zero — the script writes
`coalitions.json` only after the last coalition. Re-running skips a model whose
`coalitions.json` already exists.

7. Collect the outputs:

```bash
!ls -R /content/drive/MyDrive/timeshap_results/xai_timeshap/london/london_load_COV_25_pervar
```

Each model directory holds `shap_values.csv` (per coalition-free player x step phi),
`base_value.csv` (base, prediction, residual per step), `player_shares.csv`,
`group_shares.csv`, `temporal_pruning.csv`, `coalitions.json` (player list,
coalition count, seconds, additivity residual).

## Before you trust the numbers

- `additivity_residual` in `coalitions.json` must be ~0 (the script already raises
  above 1e-6). It is computed inside one run, so it is a check on the estimator, not
  on the GPU.
- On a T4 (Colab free) `bfloat16` is emulated, so Chronos2 runs slower than the
  numbers above; A100/L4 behave like the local measurements.
- `quarter_hour`, `hour`, `minute` are deterministic functions of each other
  (`quarter_hour = hour*4 + minute//15`), so splitting them into three players does
  not make the game able to tell them apart: their individual credit is arbitrary.
  Same caveat, weaker, for `dayofweek` / `is_weekend`. Report per-column values for
  the weather columns and read the calendar columns as a group.
- Belgium has 21-22 columns (adds `day_ahead_price`, `soc`, eight `static_*`), i.e.
  2M-4M coalitions. The `static_*` columns are dropped by `prepare_inputs` before
  they reach the model, so they carry no attribution; drop them from the frame or
  the game is unaffordable.
- The 4 GB card on the developer machine cannot hold TimesFM3 plus another
  checkpoint; Colab's 16 GB can, but keep one model per invocation anyway, because
  the explainer does not free the previous pipeline.
