# TimeSHAP per-covariate run on a hosted GPU (Colab)

The game: one player per covariate **column** plus the four history players
(`hist_d1..hist_d4`), payoff = the whole horizon vector, exact enumeration of
`2^N` coalitions, hour-of-day background. Nothing is trained and nothing refits,
so one coalition is one forward pass of a zero-shot checkpoint.

Covariates default to five columns — `hour`, `dayofweek`, `humidity`,
`temperature`, `cloud_cover` (`TIMESHAP_COLUMNS`; `temperature` sits where `ghi`
used to be). Belgium wants two more, because the loader scopes them away from the
default set:

```
--covariates hour dayofweek humidity temperature cloud_cover day_ahead_price soc
```

`soc` only exists for `bess_*` series in the loader; on `pv_house_*` the script
attaches it on request (`attach_block`) and logs that the explained model then has
that channel, i.e. is not the benchmark's PV input mapping. Putting `ghi` back
gives eight columns and 12 players (4096 coalitions instead of 2048) - worth it for
PV, where removing `ghi` cost Timestral 3.7 % MAE:

```
--covariates hour dayofweek humidity temperature cloud_cover day_ahead_price soc ghi
```

Measured cost at `--sampling 25` on an RTX 3050 4 GB, Belgium PV (2184 points,
48-step horizon): **Chronos2 0.08 s** and **Timestral 0.27 s** per coalition;
TimesFM3 not measured (and it does 2 forwards per coalition through
`symmetric_averaging`).

| game | players | coalitions | Chronos2 | Timestral |
|---|---|---|---|---|
| 7 columns + history | 11 | 2 048 | ~3 min | ~9 min |
| 8 columns + history (with `ghi`) | 12 | 4 096 | ~6 min | ~18 min |
| 7 columns only (`COVARIATE_ONLY=1`) | 7 | 128 | ~12 s | ~35 s |
| 5 columns + history (London has no price/soc) | 9 | 512 | 36 s | 3.0 min |

At `--sampling 100` the context is much longer (17 315 points for London), so the
per-forward cost grows roughly 20x.

## Never use `conda run` here

`conda run` keeps the child's stdout until the child exits, so a 9-minute game
shows **nothing** and looks frozen. These scripts call `"$PYTHON" -u` and stream.
Verified: a child that prints and then sleeps shows no output through `conda run`
within 4 s.

## Cells, in order

1. **Runtime → Change runtime type → GPU**, then check it:

```bash
!nvidia-smi
```

2. Clone. The current explainer only exists after you push it — the GitHub copy is
behind the local working tree:

```bash
!git clone https://github.com/cinnamoll/Forecast-Models-Evaluation.git /content/Forecast-Models-Evaluation
```

3. Results must live in Drive; `/content` is wiped when the session ends:

```bash
from google.colab import drive; drive.mount('/content/drive')
```

4. Setup (installs the forecasting packages only, never torch). Add
`INSTALL_TIMESFM=1` if this session also runs TimesFM3:

```bash
!cd /content/Forecast-Models-Evaluation && bash scripts/colab/01_setup.sh
```

5. Every later cell re-sources the env file — exports do not survive between cells:

```bash
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && bash scripts/colab/02_run.sh --dry-run
```

6. One series, one model list (the command this project actually uses):

```bash
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && \
  DATASET=belgium SERIES=pv_house_2 MODELS=Chronos2 \
  COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc" \
  SAMPLING=25 bash scripts/colab/02_run.sh
```

```bash
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && \
  DATASET=belgium SERIES=pv_house_2 MODELS=TimesFM3 \
  COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc" \
  SAMPLING=25 bash scripts/colab/02_run.sh
```

7. The whole roster (4 datasets, 13 series, 3 models), one process per model, one
log per model under `$RESULTS/xai_timeshap/logs/`:

```bash
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && bash scripts/colab/03_sweep.sh --dry-run
!source /content/timeshap_env.sh && cd /content/Forecast-Models-Evaluation && bash scripts/colab/03_sweep.sh
```

`03_sweep.sh` skips any model whose `coalitions.json` already exists, so a
disconnect costs the model that was running and nothing else. `TEMPERATURE=1`
switches Belgium to the 8-column game.

8. Collect:

```bash
!ls -R /content/drive/MyDrive/timeshap_results/xai_timeshap
```

## What each output file means

- `shap_values.csv` — `model,player,group,step,phi`, the source of every other file.
- `base_value.csv` — `base = f(∅)`, `prediction = f(X)`, `residual = prediction − base − Σφ` per step.
- `player_shares.csv` — `share = Σ|φ|` normalised to 1, plus `signed_mean` (near zero is sign cancellation across the horizon, not a small effect).
- `group_shares.csv` — `hist_d*` collapsed into `target_history`; the row is **absent** when the game has no history players (a `--covariate-only` run), which is different from "the model ignores the past".
- `temporal_pruning.csv` — per-day `mean_abs_contribution`, to justify splitting the history into four days.
- `coalitions.json` — players, coalition count, estimator, `mode`, `horizon`, `context`, `background_rotation`, `additivity_residual`, `seconds`, `temporal_players`, `per_variable`, `noop_dropped`.
- run level: `group_shares_all_models.csv`, `spearman_matrix.csv` (model × model, constant blocks dropped — with two models it is a single number), `notes.json`.

## Before trusting the numbers

- `additivity_residual` must be ~0 (the script raises above 1e-6). It is a check inside one
  run, not a check of the GPU.
- One device per comparison table: Chronos2 runs bf16 on CUDA and fp32 on CPU, so shares differ in
  the third digit. On a T4 bf16 is emulated, so Chronos2 is slower than the local numbers.
- `hour` is a player in every dataset now. On a 60-minute grid its hour-of-day background would
  reproduce the column exactly, so the profile is **rotated half a day**: switching `hour` off means
  "the time-of-day code shifted 12 hours". Tables produced before that fix (hour = 0) are not
  comparable with new ones.
- An 8-column game is not comparable with a 7-column one: the credit of the dropped column moves
  into `base` and into the columns that stay.
- Belgium `soc` on a `pv_*` series means a model with a channel the benchmark never gives it;
  say so wherever those numbers are reported.
